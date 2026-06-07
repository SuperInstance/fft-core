//! # fft-core
//!
//! Fast Fourier Transform: Cooley-Tukey radix-2, Bluestein, real FFT, inverse.
//!
//! ## Modules
//! - `cooley_tukey` — Radix-2 decimation-in-time FFT
//! - `bluestein` — Bluestein's algorithm for arbitrary-length FFT
//! - `real_fft` — Real-valued FFT
//! - `inverse` — Inverse FFT
//! - `window` — Windowing functions (Hamming, Hanning, Blackman)

#![allow(unknown_lints, clippy::needless_range_loop, clippy::ptr_arg, clippy::excessive_precision)]

pub mod cooley_tukey;
pub mod bluestein;
pub mod real_fft;
pub mod inverse;
pub mod window;

/// Complex number.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn mag(&self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    pub fn mag_sq(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn phase(&self) -> f64 {
        self.im.atan2(self.re)
    }

    pub fn conj(&self) -> Self {
        Self { re: self.re, im: -self.im }
    }

    pub fn scale(&self, s: f64) -> Self {
        Self { re: self.re * s, im: self.im * s }
    }
}

impl std::ops::Add for Complex {
    type Output = Complex;
    fn add(self, other: Complex) -> Complex {
        Complex { re: self.re + other.re, im: self.im + other.im }
    }
}

impl std::ops::Sub for Complex {
    type Output = Complex;
    fn sub(self, other: Complex) -> Complex {
        Complex { re: self.re - other.re, im: self.im - other.im }
    }
}

impl std::ops::Mul for Complex {
    type Output = Complex;
    fn mul(self, other: Complex) -> Complex {
        Complex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

pub const PI: f64 = std::f64::consts::PI;
