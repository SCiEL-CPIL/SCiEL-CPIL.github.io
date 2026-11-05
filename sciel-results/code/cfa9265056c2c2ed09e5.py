import numpy as np

N = 400

# EVOLVE-BLOCK-START

def construct():
    # Spectral envelope design via Blaschke product magnitude.
    # The autoconvolution c = h * h has Fourier transform |H(k)|^2.
    # We choose H(k) = |B(e^{ik})|^alpha where B is a Blaschke product
    # with zeros inside the unit disk. The zeros' positions control
    # how flat |H|^2 is, which directly bounds max(c)/sum(h)^2.
    
    # Place zeros at radius r < 1 to create a smooth spectral envelope.
    # Multiple zeros at different angles create a flatter magnitude.
    num_zeros = 12
    r = 0.7  # zero radius (closer to 1 = sharper features)
    
    # Distribute zeros evenly on a circle of radius r in the unit disk
    zero_angles = 2.0 * np.pi * np.arange(num_zeros) / num_zeros
    zeros = r * np.exp(1j * zero_angles)
    
    # Frequency grid for the DFT
    k = np.arange(N)
    omega = 2.0 * np.pi * k / N
    z = np.exp(1j * omega)
    
    # Build the Blaschke product magnitude: |B(z)| = prod |z - a_j| / |1 - conj(a_j) z|
    # For a zero at a, the factor is (z - a)/(1 - conj(a)*z), and its magnitude
    log_mag = np.zeros(N)
    for a in zeros:
        # |z - a| / |1 - conj(a) * z|
        num = np.abs(z - a)
        den = np.abs(1.0 - np.conj(a) * z)
        log_mag += np.log(num / den)
    
    # H(k) = exp(alpha * log_mag), choose alpha to control flatness
    # Larger alpha -> more contrast; we want moderate for flatness
    alpha = 0.5
    H_mag = np.exp(alpha * log_mag)
    
    # Ensure H(0) is positive and normalize so sum(h) is controlled
    # h = IDFT of H (real-valued since H is symmetric magnitude)
    # We need h real: use H(k) with conjugate symmetry
    H = H_mag.copy()
    # Make H conjugate symmetric for real h
    H_sym = np.zeros(N, dtype=complex)
    H_sym[0] = H_mag[0]
    for i in range(1, (N + 1) // 2):
        H_sym[i] = H_mag[i]
        H_sym[N - i] = H_mag[N - i]
    if N % 2 == 0:
        H_sym[N // 2] = H_mag[N // 2]
    
    h = np.fft.ifft(H_sym).real
    
    # Normalize: scale doesn't affect value (ratio is scale-invariant)
    # But ensure sum != 0
    s = np.sum(h)
    if abs(s) < 1e-12:
        # Add a small DC component
        h[0] += 1e-6
        s = np.sum(h)
    if s < 0:
        h = -h
        s = -s
    
    # Optional: iteratively flatten by projecting out the peak
    # Use a few steps of "peak shaving" in spectral domain
    for _ in range(30):
        conv = np.convolve(h, h)
        peak_idx = np.argmax(conv)
        peak_val = conv[peak_idx]
        # If peak is significantly above mean, try to reduce it
        mean_conv = np.mean(conv)
        if peak_val > 1.5 * mean_conv:
            # In spectral domain, reduce the component that creates the peak
            # This is a heuristic: slightly flatten H
            H_sym *= (1.0 - 0.01)
            h_new = np.fft.ifft(H_sym).real
            s_new = np.sum(h_new)
            if abs(s_new) > 1e-10:
                if s_new < 0:
                    h_new = -h_new
                h = h_new
        else:
            break
    
    return h

# EVOLVE-BLOCK-END
