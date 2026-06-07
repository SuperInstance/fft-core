//! Windowing functions for spectral analysis.

/// Apply a Hamming window to the input signal.
pub fn hamming(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos())
        .collect()
}

/// Apply a Hanning (Hann) window.
pub fn hanning(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()))
        .collect()
}

/// Apply a Blackman window.
pub fn blackman(n: usize) -> Vec<f64> {
    let a0 = 0.42;
    let a1 = 0.5;
    let a2 = 0.08;
    (0..n)
        .map(|i| {
            a0 - a1 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()
                + a2 * (4.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()
        })
        .collect()
}

/// Apply a window to a signal.
pub fn apply_window(signal: &[f64], window: &[f64]) -> Vec<f64> {
    assert_eq!(signal.len(), window.len());
    signal.iter().zip(window.iter()).map(|(s, w)| s * w).collect()
}

/// Rectangular window (no windowing).
pub fn rectangular(n: usize) -> Vec<f64> {
    vec![1.0; n]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_length() {
        let w = hamming(16);
        assert_eq!(w.len(), 16);
    }

    #[test]
    fn test_hamming_endpoints() {
        let w = hamming(16);
        assert!((w[0] - 0.08).abs() < 0.01);
        assert!((w[15] - 0.08).abs() < 0.01);
    }

    #[test]
    fn test_hanning_symmetry() {
        let w = hanning(16);
        for i in 0..8 {
            assert!((w[i] - w[15 - i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_blackman_midpoint() {
        let w = blackman(16);
        // Middle value should be close to 1
        assert!(w[8] > 0.9);
    }

    #[test]
    fn test_apply_window() {
        let signal = vec![1.0; 4];
        let window = vec![0.5; 4];
        let result = apply_window(&signal, &window);
        assert_eq!(result, vec![0.5; 4]);
    }

    #[test]
    fn test_rectangular() {
        let w = rectangular(8);
        assert_eq!(w, vec![1.0; 8]);
    }
}
