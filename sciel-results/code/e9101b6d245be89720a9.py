import numpy as np
import time

def construct():
    # Initialize with Golay-Rudin-Shapiro sequence
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    current_coeffs = np.array(p[:24], dtype=np.float64)
    
    # Precompute evaluation points
    # Using 4096 points for speed and accuracy balance
    N = 4096
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # M[j, k] = exp(i * k * theta_j)
    M = np.exp(1j * np.outer(theta, np.arange(24)))
    
    def eval_norm(c):
        vals = M @ c
        return np.max(np.abs(vals))
    
    current_energy = eval_norm(current_coeffs)
    best_coeffs = current_coeffs.copy()
    best_energy = current_energy
    
    # Simulated Annealing
    T = 1.0
    alpha = 0.9995 # Slower cooling to explore more
    start_time = time.time()
    deadline = 115.0 # 5s buffer
    
    # Batch processing for speed?
    # No, simple loop is fine if eval is fast.
    
    while time.time() - start_time < deadline:
        idx = np.random.randint(24)
        # Flip
        current_coeffs[idx] *= -1
        new_energy = eval_norm(current_coeffs)
        
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_coeffs = current_coeffs.copy()
        else:
            # Reject, flip back
            current_coeffs[idx] *= -1
            
        T *= alpha
        if T < 1e-5:
            T = 1e-5

    return list(best_coeffs.astype(int))
