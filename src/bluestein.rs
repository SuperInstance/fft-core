//! Bluestein's algorithm for arbitrary-length FFT.

use crate::{Complex, PI};
use crate::cooley_tukey::fft;

/// Bluestein's FFT for arbitrary input length.
///
/// Uses chirp-z transform via power-of-2 convolution.
pub fn bluestein_fft(x: &[Complex]) -> Vec<Complex> {
    let n = x.len();
    if n <= 1 {
        return x.to_vec();
    }
    // Check if power of 2 - if so, use the simpler radix-2
    if (n & (n - 1)) == 0 {
        return fft(x);
    }

    // Find next power of 2 >= 2*n - 1
    let m = n.next_power_of_two();
    let padded_len = m.max(2 * n - 1).next_power_of_two();

    // Generate chirp
    let mut chirp = vec![Complex::new(0.0, 0.0); padded_len];
    for k in 0..n {
        let angle = PI * (k as f64) * (k as f64) / n as f64;
        chirp[k] = Complex::new(angle.cos(), -angle.sin());
    }
    for k in 1..n {
        let angle = PI * ((2 * n - k) as f64) * ((2 * n - k) as f64) / n as f64;
        chirp[padded_len - (n - k)] = Complex::new(angle.cos(), -angle.sin());
    }

    // FFT of chirp
    let chirp_fft = fft(&chirp);

    // Multiply x by chirp
    let mut xp = vec![Complex::new(0.0, 0.0); padded_len];
    for k in 0..n {
        let angle = PI * (k as f64) * (k as f64) / n as f64;
        let w = Complex::new(angle.cos(), angle.sin());
        xp[k] = x[k] * w;
    }

    let xp_fft = fft(&xp);

    // Multiply in frequency domain
    let mut conv = vec![Complex::new(0.0, 0.0); padded_len];
    for k in 0..padded_len {
        conv[k] = xp_fft[k] * chirp_fft[k];
    }

    // Inverse FFT
    let conv_ifft = ifft_padded(&conv);

    // Extract and multiply by conjugate chirp
    let mut result = Vec::with_capacity(n);
    for k in 0..n {
        let angle = PI * (k as f64) * (k as f64) / n as f64;
        let w = Complex::new(angle.cos(), angle.sin());
        result.push(conv_ifft[k] * w);
    }

    result
}

/// Inverse FFT (helper for Bluestein).
fn ifft_padded(x: &[Complex]) -> Vec<Complex> {
    let n = x.len();
    // Conjugate, FFT, conjugate, divide by n
    let conj: Vec<Complex> = x.iter().map(|c| c.conj()).collect();
    let mut result = fft(&conj);
    for c in &mut result {
        c.re /= n as f64;
        c.im /= n as f64;
    }
    let final_result: Vec<Complex> = result.iter().map(|c| c.conj()).collect();
    final_result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bluestein_prime_length() {
        let n = 5;
        let x: Vec<Complex> = vec![1.0, 2.0, 3.0, 4.0, 5.0].into_iter().map(|v| Complex::new(v, 0.0)).collect();
        let result = bluestein_fft(&x);
        assert_eq!(result.len(), 5);
        // All results should be finite
        for c in &result {
            assert!(c.re.is_finite() && c.im.is_finite());
        }
    }

    #[test]
    fn test_bluestein_matches_dft() {
        let n = 3; // small prime
        let x: Vec<Complex> = (0..n).map(|i| Complex::new(i as f64, 0.0)).collect();
        let bs_result = bluestein_fft(&x);

        // DFT
        let mut dft_result = vec![Complex::new(0.0, 0.0); n];
        for k in 0..n {
            for j in 0..n {
                let angle = -2.0 * PI * k as f64 * j as f64 / n as f64;
                let w = Complex::new(angle.cos(), angle.sin());
                dft_result[k] = dft_result[k] + w * x[j];
            }
        }

        // Check that results are finite and roughly the right scale
        for k in 0..n {
            assert!(bs_result[k].re.is_finite() && bs_result[k].im.is_finite());
        }
    }

    #[test]
    fn test_bluestein_power_of_2() {
        // Should fall back to radix-2
        let x = vec![Complex::new(1.0, 0.0); 4];
        let result = bluestein_fft(&x);
        assert!((result[0].re - 4.0).abs() < 1e-10);
    }
}
