import numpy as np

N = 100

def project_to_simplex(h, target_sum):
    """
    Project h onto the set {x | 0 <= x_i <= 1, sum(x) = target_sum}
    using bisection on the Lagrange multiplier.
    """
    def sum_h(lam):
        h_proj = np.clip(h + lam, 0, 1)
        return np.sum(h_proj)
    
    lo = -1.0
    hi = 1.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        s = sum_h(mid)
        if s > target_sum:
            hi = mid
        else:
            lo = mid
            
    lam = (lo + hi) / 2.0
    return np.clip(h + lam, 0, 1)

def calc_max_c_fft(h):
    # Zero pad to 2N for linear autocorrelation
    h_padded = np.zeros(2 * N)
    h_padded[:N] = h
    
    # FFT
    H = np.fft.fft(h_padded)
    # Autocorrelation
    R = np.fft.ifft(np.abs(H) ** 2).real
    
    # R[k] for k=0..N-1 are the sums A_k
    # C_k = 1 - (2/N) * A_k
    # Max C_k corresponds to Min A_k
    # So we want Min R[k] for k in 0..N-1
    min_a = np.min(R[:N])
    max_c = 1.0 - (2.0 / N) * min_a
    return max_c

def construct():
    np.random.seed(42)
    
    # Start with uniform
    h = np.full(N, 0.5)
    
    current_energy = calc_max_c_fft(h)
    
    T = 0.1
    T_min = 1e-6
    alpha = 0.999
    n_iter = 50000
    
    best_h = h.copy()
    best_energy = current_energy
    
    # Adaptive step size
    step_size = 0.5
    
    for i in range(n_iter):
        idx1 = np.random.randint(N)
        idx2 = np.random.randint(N)
        if idx1 == idx2:
            continue
            
        delta = np.random.uniform(-step_size, step_size)
        
        h_new = h.copy()
        h_new[idx1] += delta
        h_new[idx2] -= delta
        
        h_new = project_to_simplex(h_new, N/2.0)
        
        new_energy = calc_max_c_fft(h_new)
        
        if new_energy < current_energy:
            current_energy = new_energy
            h = h_new
            if current_energy < best_energy:
                best_energy = current_energy
                best_h = h.copy()
        else:
            if np.random.rand() < np.exp(-(new_energy - current_energy) / T):
                current_energy = new_energy
                h = h_new
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_h = h.copy()
        
        T *= alpha
        if T < T_min:
            T = T_min
        step_size *= 0.9995
        
    return best_h
