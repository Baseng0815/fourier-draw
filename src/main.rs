use core::f32;
use std::{
    arch::x86_64::_mm256_permute_pd,
    f32::consts::{E, PI}, time::{Duration, Instant},
};

use egui::{
    CentralPanel, Color32, ComboBox, Frame, Pos2, Rect, ScrollArea, SidePanel, Slider, Stroke, Ui,
    Vec2,
};
use num::{
    complex::{Complex32, ComplexFloat},
    traits::ConstZero,
};
use rand::{
    Rng,
    distr::{Distribution, StandardUniform, Uniform, uniform::SampleUniform},
    random_range,
};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum DrawMode {
    SinglePoints,
    ConnectedPoints,
}

#[derive(Debug)]
struct FourierDrawApp {
    fourier_series: FourierSeries,
    points: Vec<Complex32>,
    num_points: usize,
    stroke_width: f32,
    draw_mode: DrawMode,
    apply_scaling: bool,

    // include some wonky animations B)
    prev_series: FourierSeries,
    animation_start: Instant,
    animation_duration: Duration,
    keep_animating: bool,
}

impl FourierDrawApp {
    fn new(fourier_series: FourierSeries, num_points: usize) -> Self {
        Self {
            fourier_series: fourier_series.clone(),
            points: Vec::new(),
            num_points,
            stroke_width: 1.0,
            draw_mode: DrawMode::ConnectedPoints,
            apply_scaling: true,
            prev_series: fourier_series,
            animation_start: Instant::now(),
            animation_duration: Duration::from_millis(1000),
            keep_animating: false,
        }
    }

    fn draw_points(&self, ui: &mut Ui) {
        let min_dim = ui.available_width().min(ui.available_height());
        let (response, painter) =
            ui.allocate_painter(Vec2::new(min_dim, min_dim), egui::Sense::click_and_drag());

        if self.points.len() == 0 {
            return;
        }

        let min_x = self
            .points
            .iter()
            .min_by(|n0, n1| f32::total_cmp(&n0.re, &n1.re))
            .unwrap()
            .re;
        let max_x = self
            .points
            .iter()
            .max_by(|n0, n1| f32::total_cmp(&n0.re, &n1.re))
            .unwrap()
            .re;

        let min_y = self
            .points
            .iter()
            .min_by(|n0, n1| f32::total_cmp(&n0.im, &n1.im))
            .unwrap()
            .im;
        let max_y = self
            .points
            .iter()
            .max_by(|n0, n1| f32::total_cmp(&n0.im, &n1.im))
            .unwrap()
            .im;

        let range_x = min_x.abs().max(max_x.abs()).max(1.0) * 1.1;
        let range_y = min_y.abs().max(max_y.abs()).max(1.0) * 1.1;

        let to_screen = emath::RectTransform::from_to(
            Rect::from_min_max(Pos2::new(-range_x, -range_y), Pos2::new(range_x, range_y)),
            response.rect,
        );

        match self.draw_mode {
            DrawMode::SinglePoints => {
                for (i, number) in self.points.iter().enumerate() {
                    let pos = Pos2::new(number.re, number.im);
                    let pos_screen = to_screen.transform_pos(pos);
                    let ri = i as f32 / self.points.len() as f32;
                    let color = blend_colors(Color32::RED, Color32::BLUE, ri);
                    painter.circle_filled(pos_screen, self.stroke_width, color);
                }
            }
            DrawMode::ConnectedPoints => {
                for (i, (curr, next)) in self
                    .points
                    .iter()
                    .zip(self.points.iter().skip(1))
                    .enumerate()
                {
                    let pos_curr = Pos2::new(curr.re, curr.im);
                    let pos_next = Pos2::new(next.re, next.im);
                    let pos_screen_curr = to_screen.transform_pos(pos_curr);
                    let pos_screen_next = to_screen.transform_pos(pos_next);
                    let ri = i as f32 / self.points.len() as f32;
                    let color = blend_colors(Color32::RED, Color32::BLUE, ri);
                    painter.line(
                        vec![pos_screen_curr, pos_screen_next],
                        Stroke::new(self.stroke_width, color),
                    );
                }
            }
        }
    }
}

impl eframe::App for FourierDrawApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        SidePanel::right("").show(ctx, |ui| {
            // store previous state to track changes
            let mut should_update_fourier_points = false;
            let mut num_coeff = self.fourier_series.coefficients.len() / 2;
            let mut num_points = self.num_points;

            ui.label("Number of (positive) coefficients");
            ui.add(Slider::new(&mut num_coeff, 0..=50));
            ui.label("Resolution");
            ui.add(Slider::new(&mut num_points, 10..=5000));
            ui.label("Stroke width");
            ui.add(Slider::new(&mut self.stroke_width, 0.1..=10.0));

            ComboBox::from_label("Draw Mode")
                .selected_text(format!("{:?}", self.draw_mode))
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.draw_mode,
                        DrawMode::SinglePoints,
                        "Single Points",
                    );
                    ui.selectable_value(
                        &mut self.draw_mode,
                        DrawMode::ConnectedPoints,
                        "Connected Points",
                    );
                });

            if num_coeff != self.fourier_series.coefficients.len() / 2 {
                self.fourier_series.resize(num_coeff);
                should_update_fourier_points = true;
            }

            if num_points != self.num_points {
                self.num_points = num_points;
                should_update_fourier_points = true;
            }

            ui.checkbox(&mut self.apply_scaling, "Apply scaling");

            if ui.button("Randomize!").clicked() {
                self.prev_series = self.fourier_series.clone();
                self.fourier_series.randomize(self.apply_scaling);
                self.animation_start = Instant::now();
                should_update_fourier_points = true;
            }

            ui.checkbox(&mut self.keep_animating, "Keep animating!");

            ui.separator();

            ScrollArea::vertical().show(ui, |ui| {
                let coeff_len = self.fourier_series.coefficients.len() as i32 / 2;
                for (i, coeff) in self.fourier_series.coefficients.iter_mut().enumerate() {
                    let mut coeff_re = coeff.re();
                    let mut coeff_im = coeff.im();
                    let coeff_i = i as i32 - coeff_len;

                    ui.label(format!("re(c[{coeff_i}])"));
                    ui.add(Slider::new(&mut coeff_re, -10.0..=10.0));
                    ui.label(format!("im(c[{coeff_i}])"));
                    ui.add(Slider::new(&mut coeff_im, -10.0..=10.0));
                    ui.separator();

                    if coeff.re() != coeff_re || coeff.im() != coeff_im {
                        *coeff = Complex32::new(coeff_re, coeff_im);
                        should_update_fourier_points = true;
                    }
                }
            });

            let now = Instant::now();
            let is_animation_running = now < self.animation_start + self.animation_duration;
            if is_animation_running {
                let t = (now - self.animation_start).as_millis() as f32 / self.animation_duration.as_millis() as f32;
                let blended = self.fourier_series.blend(&self.prev_series, t);
                self.points = blended.points(self.num_points);
            } else if self.keep_animating {
                self.prev_series = self.fourier_series.clone();
                self.fourier_series.randomize(self.apply_scaling);
                self.animation_start = Instant::now();
            } else if should_update_fourier_points {
                self.points = self.fourier_series.points(self.num_points);
            }
        });
        CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                Frame::canvas(ui.style()).show(ui, |ui| {
                    self.draw_points(ui);
                });
            });
        });

        ctx.request_repaint_after(Duration::from_millis(10));
    }
}

#[derive(Debug, Clone)]
struct FourierSeries {
    coefficients: Vec<Complex32>,
}

impl FourierSeries {
    fn new(coefficients: Vec<Complex32>) -> Self {
        Self { coefficients }
    }

    fn with_num_coeff(num_coeff: usize) -> Self {
        Self::new(vec![Complex32::ZERO; num_coeff * 2 + 1])
    }

    fn resize(&mut self, num_coeff: usize) {
        let difference = num_coeff as i32 - self.coefficients.len() as i32 / 2;
        self.coefficients.resize(num_coeff * 2 + 1, Complex32::ZERO);

        if difference > 0 {
            self.coefficients.rotate_right(difference as usize);
        } else if difference < 0 {
            self.coefficients.rotate_left(difference.abs() as usize);
        }
    }

    fn randomize(&mut self, apply_scaling: bool) {
        let midpoint = self.coefficients.len() as f32 / 2.0;
        for (i, coefficient) in self.coefficients.iter_mut().enumerate() {
            let scale = if apply_scaling {
                1.0 / (i as f32 - midpoint).abs()
            } else {
                1.0
            };
            let coeff_re = random_range(-10.0..=10.0) * scale;
            let coeff_im = random_range(-10.0..=10.0) * scale;
            *coefficient = Complex32::new(coeff_re, coeff_im);
        }
    }

    fn blend(&self, other: &FourierSeries, t: f32) -> FourierSeries {
        let coefficients = self.coefficients.iter().zip(other.coefficients.iter()).map(|(c0, c1)| {
            t * c0 + (1.0 - t) * c1
        }).collect::<Vec<_>>();
        FourierSeries::new(coefficients)
    }

    fn evaluate(&self, t: f32) -> Complex32 {
        self.coefficients
            .iter()
            .enumerate()
            .map(|(i, ci)| {
                let freq = i as f32 - self.coefficients.len() as f32 / 2.0;
                ci * E.powc(freq * 2.0 * PI * Complex32::I * t)
            })
            .sum()
    }

    fn points(&self, num_points: usize) -> Vec<Complex32> {
        (0..num_points)
            .map(|i| self.evaluate(i as f32 / num_points as f32))
            .collect::<Vec<_>>()
    }
}

fn blend_colors(color_0: Color32, color_1: Color32, t: f32) -> Color32 {
    Color32::from_rgb(
        (t * color_0.r() as f32 + (1.0 - t) * color_1.r() as f32) as u8,
        (t * color_0.g() as f32 + (1.0 - t) * color_1.g() as f32) as u8,
        (t * color_0.b() as f32 + (1.0 - t) * color_1.b() as f32) as u8,
    )
}

fn main() {
    let native_options = eframe::NativeOptions::default();
    let app = FourierDrawApp::new(FourierSeries::with_num_coeff(30), 500);
    eframe::run_native(
        "fourier-draw",
        native_options,
        Box::new(|_| Ok(Box::new(app))),
    )
    .unwrap();
}
