# fft-core

Research-grade Fast Fourier Transform in pure Rust.

## Features

- **Cooley-Tukey**: Radix-2 decimation-in-time FFT
- **Bluestein**: Arbitrary-length FFT via chirp-z transform
- **Real FFT**: Efficient FFT for real-valued signals
- **Inverse**: IFFT and inverse real FFT
- **Window**: Hamming, Hanning, Blackman windowing functions

## Usage

```rust
use fft_core::cooley_tukey::fft;
use fft_core::Complex;

fn main() {
    let x: Vec<Complex> = vec![1.0, 2.0, 3.0, 4.0]
        .into_iter()
        .map(|v| Complex::new(v, 0.0))
        .collect();
    let spectrum = fft(&x);
    for (k, c) in spectrum.iter().enumerate() {
        println!("X[{}] = {:.4} + {:.4}j", k, c.re, c.im);
    }
}
```

## No Dependencies

This crate uses only `std`.

## License

MIT OR Apache-2.0
