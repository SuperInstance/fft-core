//! Real-valued FFT.

use crate::cooley_tukey::fft;
use crate::Complex;

/// Real FFT: efficient FFT for real-valued input.
///
/// Returns N//2 + 1 complex values (positive frequencies only).
/// Uses the standard FFT by packing into complex format.
pub fn rfft(x: &[f64]) -> Vec<Complex> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }

    // Pad to next power of 2
    let padded_len = n.next_power_of_two();
    let mut x_complex: Vec<Complex> = x.iter().map(|&v| Complex::new(v, 0.0)).collect();
    x_complex.resize(padded_len, Complex::new(0.0, 0.0));

    let full_fft = fft(&x_complex);

    // Return only positive frequencies
    let n_out = n / 2 + 1;
    full_fft[..n_out].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rfft_dc() {
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let result = rfft(&x);
        assert!((result[0].re - 4.0).abs() < 1e-8);
    }

    #[test]
    fn test_rfft_matches_full_fft() {
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let x_complex: Vec<Complex> = x.iter().map(|&v| Complex::new(v, 0.0)).collect();
        let full_fft = crate::cooley_tukey::fft(&x_complex);
        let real_result = rfft(&x);

        // DC component should match exactly
        assert!((real_result[0].re - full_fft[0].re).abs() < 1e-8);
        // Nyquist should match
        assert!((real_result[2].re - full_fft[2].re).abs() < 1e-8);
    }

    #[test]
    fn test_rfft_nyquist() {
        let x: Vec<f64> = (0..8).map(|i| (-1.0_f64).powi(i as i32)).collect();
        let result = rfft(&x);
        // At Nyquist (k=4), should have energy
        assert!(result[4].re.abs() > 0.5);
    }

    #[test]
    fn test_rfft_length() {
        let x = vec![1.0; 16];
        let result = rfft(&x);
        assert_eq!(result.len(), 9); // N/2 + 1
    }
}
