"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Construct a sequence with a shaped power spectrum to minimize peak autocorrelation.
    # We use a decaying amplitude envelope to concentrate energy at low frequencies,
    # which produces a smooth time-domain signal with a dominant DC component and
    # rapidly decaying autocorrelation sidelobes.
    
    # Initialize DFT coefficients
    X = np.zeros(N, dtype=complex)
    
    # Define amplitude envelope
    # A strong DC component and rapid decay for higher frequencies
    # This maximizes (sum x)^2 relative to max |C(k)|
    amplitudes = np.zeros(N)
    for k in range(N):
        # Distance from DC (circular distance)
        dist = min(k, N - k)
        # Exponential decay
        amplitudes[k] = np.exp(-dist / 5.0)
        
    # Generate random phases for the spectrum
    # Ensure conjugate symmetry to produce a real-valued time-domain signal
    for k in range(N // 2 + 1):
        phase = np.random.uniform(0, 2 * np.pi)
        amp = amplitudes[k]
        
        if k == 0:
            # DC component must be real and positive to maximize sum
            X[k] = amp
        elif k == N // 2 and N % 2 == 0:
            # Nyquist frequency must be real for conjugate symmetry
            X[k] = amp * np.cos(phase) # Or just amp, phase doesn't matter much for real constraint if we handle symmetry
            # Actually, for real signal, X[N/2] must be real.
            # We can just set it to a real value.
            X[k] = amp
        else:
            X[k] = amp * np.exp(1j * phase)
            X[N - k] = amp * np.exp(-1j * phase)
            
    # Inverse DFT to get the time-domain signal
    x = np.fft.ifft(X).real
    
    return x
# EVOLVE-BLOCK-END
