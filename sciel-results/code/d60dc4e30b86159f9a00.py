import numpy as np
import time

def construct():
    N = 24
    # Use a dense grid for the search
    M = 2000
    thetas = np.linspace(0, 2 * np.pi, M, endpoint=False)
    # Precompute powers of z
    # Z[k, j] = z_j^k
    # Using np.exp(1j * np.outer(np.arange(N), thetas))
    Z = np.exp(1j * np.outer(np.arange(N), thetas))
    
    def calc_max(coeffs):
        vals = Z @ coeffs
        return np.max(np.abs(vals))
    
    best_coeffs = None
    best_score = np.inf
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initial candidate: Golay-Rudin-Shapiro
    p, q = np.array([1]), np.array([1])
    while len(p) < N:
        p, q = np.concatenate([p, q]), np.concatenate([p, -q])
    grs = p[:N].astype(np.int8)
    
    # List of starting points
    starts = [grs]
    # Add random starts
    for _ in range(20):
        starts.append(np.random.choice([-1, 1], N).astype(np.int8))
        
    for start in starts:
        if time.time() > deadline:
            break
            
        current = start.copy()
        current_val = calc_max(current)
        
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            base_vals = Z @ current
            best_k = -1
            best_new_val = current_val
            
            for k in range(N):
                # Flip c_k
                # Change in P(z) is -2 * c_k * z^k
                delta = -2 * current[k] * Z[k, :]
                new_vals = base_vals + delta
                new_val = np.max(np.abs(new_vals))
                
                if new_val < best_new_val - 1e-12:
                    best_new_val = new_val
                    best_k = k
            
            if best_k != -1:
                current[best_k] *= -1
                current_val = best_new_val
                improved = True
                
        if current_val < best_score:
            best_score = current_val
            best_coeffs = current.copy()
            
    if best_coeffs is None:
        return grs.tolist()
        
    return best_coeffs.tolist()
