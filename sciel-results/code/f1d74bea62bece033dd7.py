import numpy as np

N = 100

def _compute_max_overlap(x):
    # Compute autocorrelation
    # C_k = sum x_i x_{i+k}
    # Using FFT
    n_fft = 2 * N
    X = np.fft.fft(x, n=n_fft)
    corr = np.fft.ifft(X * np.conj(X)).real
    # corr[k] for k=0..N-1 is C_k?
    # np.correlate(x, x, mode='full') gives indices 0..2N-1
    # C_0 is at N-1. C_1 at N. C_{N-1} at 2N-2.
    # Let's just use np.correlate for simplicity if N is small, or FFT if large.
    # For N=100, np.correlate is fine.
    c = np.correlate(x, x, mode='full')
    # c[N-1] is C_0
    # c[N] is C_1
    # ...
    # c[2N-2] is C_{N-1}
    C = c[N : 2*N-1] # C_0 ... C_{N-1}
    # We care about k=1..N-1
    # Overlap O_k = N/2 - C_k
    O = N / 2.0 - C[1:]
    return np.max(O)

def construct():
    # Initialize
    x = np.random.rand(N)
    # Project to sum N/2
    # Simple scaling
    x = x / np.sum(x) * (N / 2.0)
    x = np.clip(x, 0.0, 1.0)
    # Re-project to exact sum
    # Use a simple iterative correction
    for _ in range(100):
        diff = N / 2.0 - np.sum(x)
        if abs(diff) < 1e-12: break
        x += diff / N
        x = np.clip(x, 0.0, 1.0)
    
    best_x = x.copy()
    best_val = _compute_max_overlap(x)
    
    T = 1.0
    T_min = 0.01
    alpha = 0.99
    steps = 5000
    
    current_x = x.copy()
    current_val = best_val
    
    for i in range(steps):
        # Pick two indices
        idx = np.random.choice(N, 2, replace=False)
        i1, i2 = idx[0], idx[1]
        
        # Pick epsilon
        # Max possible transfer is min(x[i1], 1-x[i2])
        # Let's try a random small amount
        eps_max = min(current_x[i1], 1.0 - current_x[i2])
        if eps_max < 1e-6:
            # Try reverse direction
            eps_max = min(current_x[i2], 1.0 - current_x[i1])
            if eps_max < 1e-6:
                continue
            # Swap direction: take from i2, give to i1
            delta = np.random.uniform(0, eps_max)
            new_x = current_x.copy()
            new_x[i2] -= delta
            new_x[i1] += delta
        else:
            delta = np.random.uniform(0, eps_max)
            new_x = current_x.copy()
            new_x[i1] -= delta
            new_x[i2] += delta
            
        new_val = _compute_max_overlap(new_x)
        
        if new_val < current_val or np.random.rand() < np.exp((current_val - new_val) / T):
            current_x = new_x
            current_val = new_val
            if current_val < best_val:
                best_val = current_val
                best_x = new_x.copy()
                
        T *= alpha
        if T < T_min:
            T = T_min

    return best_x
