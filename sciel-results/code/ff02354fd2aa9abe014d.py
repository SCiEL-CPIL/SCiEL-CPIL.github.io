# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Design a binary step height sequence h on [0, 2] (discretized to N steps)
    bounded in {0, 1} with sum exactly N/2, using a spectral approach to minimize
    the peak sidelobe of the autocorrelation.
    
    The overlap metric is assumed to be related to the autocorrelation peak.
    We construct a binary sequence with flat power spectrum (except DC) to
    minimize peak sidelobes.
    """
    N = 100
    
    # Target: Binary sequence with N/2 ones
    # Spectral design:
    # 1. Generate random phases for Fourier components k=1..N-1
    # 2. Set amplitudes to be constant (flat spectrum)
    # 3. Inverse FFT to get real signal
    # 4. Threshold to binary {0, 1}
    # 5. Adjust to ensure exactly N/2 ones
    
    # We want the autocorrelation to be flat.
    # This corresponds to a flat power spectrum |H[k]|^2.
    # For a binary sequence, this is an approximation (Barker-like).
    
    # Initialize Fourier coefficients
    H = np.zeros(N, dtype=complex)
    H[0] = N / 2.0  # DC component for mean 0.5
    
    # Random phases for non-DC components
    # Amplitudes: We can try to make them equal magnitude
    # |H[k]| = sqrt(N/4) for k != 0? 
    # Sum |H[k]|^2 = N * sum h_i^2 = N * (N/2) = N^2/2.
    # H[0]^2 = N^2/4.
    # Sum_{k!=0} |H[k]|^2 = N^2/2 - N^2/4 = N^2/4.
    # If we set |H[k]| = N / (2 * sqrt(N-1)) for k != 0?
    # (N-1) * (N^2 / (4(N-1))) = N^2/4. Correct.
    
    amp = N / (2.0 * np.sqrt(N - 1))
    for k in range(1, N):
        # Conjugate symmetry for real signal: H[N-k] = conj(H[k])
        # We only need to set k=1..N//2
        pass
        
    # Better: Construct in frequency domain with conjugate symmetry
    H = np.zeros(N, dtype=complex)
    H[0] = N / 2.0
    
    # Set amplitudes for k=1..N-1
    # To ensure real signal, H[N-k] = conj(H[k])
    # We define H[k] for k=1..N//2
    for k in range(1, N // 2 + 1):
        if k == N - k: # Nyquist for even N
            phase = np.random.uniform(0, 2 * np.pi)
            H[k] = amp * np.exp(1j * phase)
        else:
            phase = np.random.uniform(0, 2 * np.pi)
            H[k] = amp * np.exp(1j * phase)
            H[N - k] = np.conj(H[k])
            
    # Inverse FFT
    h_real = np.real(np.fft.ifft(H))
    
    # Threshold to binary
    # We want exactly N/2 ones.
    # Sort indices by h_real value
    indices = np.argsort(h_real)[::-1]
    h_bin = np.zeros(N)
    h_bin[indices[:N//2]] = 1.0
    
    # Verify sum
    # sum(h_bin) should be N/2
    
    return h_bin
# EVOLVE-BLOCK-END
