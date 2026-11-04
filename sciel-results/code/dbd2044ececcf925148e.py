import numpy as np

N = 400

def construct():
    # Target size for the support.
    # We explore sizes slightly larger than the max Sidon set size (~20)
    # to allow for sign cancellation of collisions.
    M = 25
    
    # Initialize with a random set of indices
    support = np.random.choice(N, M, replace=False)
    signs = np.random.choice([-1, 1], M)
    
    def compute_peak(x):
        # Circular autoconvolution
        X = np.fft.rfft(x)
        Y = np.fft.irfft(np.abs(X)**2, n=N)
        # Exclude zero lag (index 0)
        # Note: Y[0] is the energy (1.0)
        # We want the max absolute value of the rest
        if N == 1:
            return 0.0
        return np.max(np.abs(Y[1:]))
    
    # Construct initial x
    x = np.zeros(N)
    x[support] = signs / np.sqrt(M)
    
    best_peak = compute_peak(x)
    best_x = x.copy()
    
    # Local search on signs
    # We try flipping signs to reduce the peak
    # This is a greedy approach.
    for _ in range(5000): # Limit iterations
        improved = False
        for i in range(M):
            # Flip sign i
            signs[i] *= -1
            x[support[i]] = signs[i] / np.sqrt(M)
            current_peak = compute_peak(x)
            if current_peak < best_peak:
                best_peak = current_peak
                best_x = x.copy()
                improved = True
            else:
                # Revert
                signs[i] *= -1
                x[support[i]] = signs[i] / np.sqrt(M)
        
        if not improved:
            # If no sign flip helped, try a random restart or break
            # For now, break to save time, or try changing support
            break
    
    return best_x
