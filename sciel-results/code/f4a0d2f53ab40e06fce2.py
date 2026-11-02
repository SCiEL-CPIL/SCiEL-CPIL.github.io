import numpy as np

N = 600

def _objective(h):
    """Compute the normalized maximum of the autoconvolution."""
    norm = np.linalg.norm(h)
    if norm < 1e-12:
        return 1e6
    h_norm = h / norm
    # Use FFT for faster convolution
    # Length of conv is 2*N - 1. We need at least 2*N for FFT to avoid circular convolution effects.
    fft_len = 2 * N
    f_h = np.fft.fft(h_norm, n=fft_len)
    ac = np.fft.ifft(f_h * f_h).real
    return np.max(ac)

def construct():
    np.random.seed(42)
    
    # Initial random solution
    h = np.random.rand(N)
    current_score = _objective(h)
    
    # Simulated Annealing parameters
    T_start = 1.0
    T_end = 1e-6
    alpha = 0.995  # Cooling rate
    steps = 20000  # Number of iterations
    
    best_h = h.copy()
    best_score = current_score
    
    for i in range(steps):
        T = T_start * (alpha ** i)
        
        # Generate a neighbor
        # Perturb a random subset of indices
        num_perturb = max(1, int(N * 0.1))
        indices = np.random.choice(N, num_perturb, replace=False)
        noise = np.random.normal(0, 0.5, num_perturb)
        
        h_new = h.copy()
        h_new[indices] += noise
        h_new = np.clip(h_new, 0, None)
        
        new_score = _objective(h_new)
        
        # Acceptance criterion
        if new_score < current_score:
            h = h_new
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
        else:
            # Accept with probability exp(-(delta)/T)
            delta = new_score - current_score
            if delta < 0:
                prob = 1.0
            else:
                prob = np.exp(-delta / T)
            
            if np.random.random() < prob:
                h = h_new
                current_score = new_score
        
        # Optional: Early stopping if temperature is very low and no improvement
        if T < T_end:
            break
            
    return list(best_h)
