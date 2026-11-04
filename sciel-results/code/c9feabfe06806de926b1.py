import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_x = None
    best_val = np.inf
    
    # Try to find a sequence with minimal peak sidelobe
    # Using random flat-spectrum sequences (CAZAC-like)
    
    for i in range(5000):
        if time.time() > deadline:
            break
            
        # Generate random phases with Hermitian symmetry
        X = np.zeros(N, dtype=complex)
        X[0] = 1.0  # Real DC component
        
        if N % 2 == 0:
            X[N // 2] = 1.0  # Real Nyquist component
            
        # Random phases for the rest
        # We can generate pairs
        for k in range(1, N // 2):
            phase = np.random.uniform(0, 2 * np.pi)
            X[k] = np.exp(1j * phase)
            X[N - k] = np.conjugate(X[k])
            
        # Inverse FFT to get real sequence x
        x = np.real(np.fft.ifft(X))
        
        # Normalize to unit energy
        energy = np.sum(x**2)
        if energy < 1e-10:
            continue
        x = x / np.sqrt(energy)
        
        # Compute linear autocorrelation
        # Using FFT for speed: Zero pad to 2N
        X_padded = np.fft.fft(x, n=2*N)
        Y_padded = X_padded * np.conjugate(X_padded)
        y = np.real(np.fft.ifft(Y_padded))
        
        # The linear autocorrelation is in y[0:N-1] and y[2N-1:N-1:-1]
        # Actually, np.convolve(x, x, 'full') gives length 2N-1.
        # The center (lag 0) is at index N-1 in the 'full' output.
        # In the zero-padded FFT approach, the circular autocorrelation of length 2N
        # corresponds to the linear autocorrelation if 2N > 2N-1 (which is true).
        # The lags are:
        # y[0] corresponds to lag 0? No.
        # Let's stick to np.convolve for correctness, it's fast enough for N=400.
        y_conv = np.convolve(x, x, mode='full')
        
        # Main lobe is at index N-1
        main_lobe = y_conv[N - 1]
        
        # Sidelobes are all other indices
        # We want to minimize the max absolute value of sidelobes
        # Exclude the main lobe
        sidelobes = np.delete(y_conv, N - 1)
        
        max_sidelobe = np.max(np.abs(sidelobes))
        
        # The metric is likely max_sidelobe / main_lobe
        # Since main_lobe is 1 (normalized energy), val = max_sidelobe
        val = max_sidelobe
        
        if val < best_val:
            best_val = val
            best_x = x
            
    # Fallback if no valid x found
    if best_x is None:
        best_x = np.ones(N)
        
    return best_x
