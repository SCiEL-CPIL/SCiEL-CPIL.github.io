import numpy as np

N = 400

def construct():
    # Initialize with random real vector
    np.random.seed(42)
    x = np.random.randn(N)
    x = x / np.linalg.norm(x)
    
    # Zero-padding length for linear autocorrelation
    M = 2 * N  # Power of 2 for FFT efficiency
    
    best_x = x.copy()
    best_psl = np.inf
    
    for iteration in range(2000):
        # Zero-pad
        x_pad = np.zeros(M)
        x_pad[:N] = x
        # DFT
        X = np.fft.fft(x_pad)
        # Power spectrum
        P = np.abs(X)**2
        # Linear autocorrelation (circular, but zero-padded so it's linear)
        R = np.fft.ifft(P).real
        # Extract linear part: R[0:N-1] are lags 0 to N-1, R[M-N+1:M] are lags -(N-1) to -1
        R_linear = np.concatenate([R[M-N+1:], R[:N]])  # lags -(N-1) to N-1
        # Peak sidelobe
        R_sidelobes = np.abs(R_linear[1:])  # Exclude lag 0 (index 0 is lag 0)
        # Wait, R_linear[0] is lag 0, R_linear[1:N] are lags 1 to N-1, R_linear[N:2N-1] are lags -(N-1) to -1
        # Actually, let me re-extract properly
        # R[0] = lag 0, R[1] = lag 1, ..., R[N-1] = lag N-1, R[N] = lag -(N-1), ..., R[M-1] = lag -1
        R_lag0 = R[0]
        R_sidelobes = np.abs(np.concatenate([R[1:N], R[M-N+1:M]]))
        psl = np.max(R_sidelobes) / R_lag0  # PSL = peak sidelobe / mainlobe
        
        if psl < best_psl:
            best_psl = psl
            best_x = x.copy()
        
        # Flatten the power spectrum
        P_mean = np.mean(P)
        P_new = np.sqrt(P * P_mean)
        # Recover X (preserve phase)
        X_new = np.sqrt(P_new) * np.exp(1j * np.angle(X))
        # Update x
        x_new = np.fft.ifft(X_new).real[:N]
        x_new = x_new / np.linalg.norm(x_new)
        x = x_new
    
    return best_x
