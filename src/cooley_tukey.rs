//! Cooley-Tukey radix-2 decimation-in-time FFT.

use crate::{Complex, PI};

/// Radix-2 FFT. Input length must be a power of 2.
pub fn fft(x: &[Complex]) -> Vec<Complex> {
    let n = x.len();
    assert!(n > 0 && (n & (n - 1)) == 0, "Length must be a power of 2");

    if n == 1 {
        return x.to_vec();
    }

    // Bit-reversal permutation
    let mut result = bit_reverse_copy(x);

    // Butterfly operations
    let mut m = 1;
    while m < n {
        let wm = Complex::new((2.0 * PI / (2 * m) as f64).cos(), -(2.0 * PI / (2 * m) as f64).sin());
        for k in (0..n).step_by(2 * m) {
            let mut w = Complex::new(1.0, 0.0);
            for j in 0..m {
                let t = w * result[k + j + m];
                let u = result[k + j];
                result[k + j] = u + t;
                result[k + j + m] = u - t;
                w = w * wm;
            }
        }
        m *= 2;
    }

    result
}

/// In-place bit-reversal copy.
fn bit_reverse_copy(x: &[Complex]) -> Vec<Complex> {
    let n = x.len();
    let bits = 64 - n.leading_zeros() as usize - 1;
    let mut result = vec![Complex::new(0.0, 0.0); n];
    for i in 0..n {
        let j = reverse_bits(i, bits);
        result[j] = x[i];
    }
    result
}

fn reverse_bits(mut x: usize, bits: usize) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// DFT (naive O(n^2) implementation for verification).
pub fn dft(x: &[Complex]) -> Vec<Complex> {
    let n = x.len();
    let mut result = Vec::with_capacity(n);
    for k in 0..n {
        let mut sum = Complex::new(0.0, 0.0);
        for j in 0..n {
            let angle = -2.0 * PI * k as f64 * j as f64 / n as f64;
            let w = Complex::new(angle.cos(), angle.sin());
            sum = sum + w * x[j];
        }
        result.push(sum);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_single() {
        let x = vec![Complex::new(1.0, 0.0)];
        let result = fft(&x);
        assert!((result[0].re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fft_dc_signal() {
        let x = vec![Complex::new(1.0, 0.0); 4];
        let result = fft(&x);
        assert!((result[0].re - 4.0).abs() < 1e-10);
        for k in 1..4 {
            assert!(result[k].mag() < 1e-10);
        }
    }

    #[test]
    fn test_fft_matches_dft() {
        let x: Vec<Complex> = vec![1.0, 2.0, 3.0, 4.0].into_iter().map(|v| Complex::new(v, 0.0)).collect();
        let fft_result = fft(&x);
        let dft_result = dft(&x);
        for k in 0..4 {
            assert!((fft_result[k].re - dft_result[k].re).abs() < 1e-10, "Re mismatch at k={}", k);
            assert!((fft_result[k].im - dft_result[k].im).abs() < 1e-10, "Im mismatch at k={}", k);
        }
    }

    #[test]
    fn test_fft_parseval() {
        let x: Vec<Complex> = vec![1.0, 2.0, 3.0, 4.0].into_iter().map(|v| Complex::new(v, 0.0)).collect();
        let x_energy: f64 = x.iter().map(|c| c.mag_sq()).sum();
        let result = fft(&x);
        let fft_energy: f64 = result.iter().map(|c| c.mag_sq()).sum();
        assert!((x_energy - fft_energy / 4.0).abs() < 1e-8, "Parseval: {} vs {}", x_energy, fft_energy / 4.0);
    }

    #[test]
    fn test_fft_known_frequency() {
        // x[n] = cos(2*pi*f*n/N) with f=1, N=8
        let n = 8;
        let x: Vec<Complex> = (0..n).map(|i| Complex::new((2.0 * PI * i as f64 / n as f64).cos(), 0.0)).collect();
        let result = fft(&x);
        // Energy should be concentrated at bins 1 and 7 (N-1)
        assert!(result[1].mag() > 3.0);
        assert!(result[7].mag() > 3.0);
        // Other bins should be small
        for k in 2..7 {
            assert!(result[k].mag() < 0.01, "Bin {} too large: {}", k, result[k].mag());
        }
    }
}
