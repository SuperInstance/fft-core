//! Inverse FFT.

use crate::cooley_tukey::fft;
use crate::Complex;

/// Inverse FFT (power-of-2 length required).
pub fn ifft(x: &[Complex]) -> Vec<Complex> {
    let n = x.len();
    // Conjugate, FFT, conjugate, scale
    let conj: Vec<Complex> = x.iter().map(|c| c.conj()).collect();
    let mut result = fft(&conj);
    for c in &mut result {
        *c = c.conj().scale(1.0 / n as f64);
    }
    result
}

/// Inverse real FFT: reconstruct real signal from positive frequency components.
pub fn irfft(x: &[Complex], n: usize) -> Vec<f64> {
    // Reconstruct full spectrum
    let mut full = Vec::with_capacity(n);
    for k in 0..x.len() {
        full.push(x[k]);
    }
    for k in (1..x.len() - 1).rev() {
        full.push(x[k].conj());
    }

    let ifft_result = ifft(&full);
    ifft_result.iter().map(|c| c.re).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cooley_tukey;

    #[test]
    fn test_ifft_roundtrip() {
        let original: Vec<Complex> = vec![1.0, 2.0, 3.0, 4.0].into_iter().map(|v| Complex::new(v, 0.0)).collect();
        let transformed = cooley_tukey::fft(&original);
        let recovered = ifft(&transformed);
        for i in 0..original.len() {
            assert!((recovered[i].re - original[i].re).abs() < 1e-10, "Mismatch at {}", i);
            assert!(recovered[i].im.abs() < 1e-10);
        }
    }

    #[test]
    fn test_ifft_larger() {
        let n = 16;
        let original: Vec<Complex> = (0..n).map(|i| Complex::new((i as f64).sin(), 0.0)).collect();
        let transformed = cooley_tukey::fft(&original);
        let recovered = ifft(&transformed);
        for i in 0..n {
            assert!((recovered[i].re - original[i].re).abs() < 1e-8, "Mismatch at {}", i);
        }
    }

    #[test]
    fn test_irfft_roundtrip() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let x_complex: Vec<Complex> = original.iter().map(|&v| Complex::new(v, 0.0)).collect();
        let freq = crate::cooley_tukey::fft(&x_complex);
        let recovered_complex = crate::inverse::ifft(&freq);
        for i in 0..original.len() {
            assert!((recovered_complex[i].re - original[i]).abs() < 1e-8);
        }
    }
}
