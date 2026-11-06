import numpy as np
import time

def construct():
    N = 24
    # Start with Golay-Rudin-Shapiro sequence
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:N])
    
    # Precompute evaluation points
    num_points = 20000
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    z = np.exp(1j * theta)
    
    # Precompute power matrix M[i, k] = z[i]^k
    M = np.ones((num_points, N), dtype=complex)
    for k in range(1, N):
        M[:, k] = M[:, k-1] * z
        
    def get_max_norm(coeffs):
        P_vals = M @ coeffs
        return np.max(np.abs(P_vals))
    
    best_norm = get_max_norm(best_coeffs)
    target_norm = 1.2583758 * np.sqrt(24)
    
    start_time = time.time()
    timeout = 140.0
    
    current_coeffs = best_coeffs.copy()
    current_norm = best_norm
    
    # If we already found the optimum, return
    if best_norm <= target_norm + 1e-9:
        return best_coeffs.tolist()
        
    while time.time() - start_time < timeout:
        # Local Search: Steepest Descent
        improved = True
        while improved:
            improved = False
            best_flip_idx = -1
            best_flip_norm = current_norm
            
            # Compute P(z) for current coeffs
            P_curr = M @ current_coeffs
            
            for k in range(N):
                # Flip bit k
                # P_new = P_curr - 2 * current_coeffs[k] * M[:, k]
                P_new = P_curr - 2 * current_coeffs[k] * M[:, k]
                new_norm = np.max(np.abs(P_new))
                
                if new_norm < best_flip_norm - 1e-12:
                    best_flip_norm = new_norm
                    best_flip_idx = k
            
            if best_flip_idx != -1:
                # Apply the best flip
                current_coeffs[best_flip_idx] = -current_coeffs[best_flip_idx]
                current_norm = best_flip_norm
                improved = True
                
                # Update global best
                if current_norm < best_norm:
                    best_norm = current_norm
                    best_coeffs = current_coeffs.copy()
                    
                    if best_norm <= target_norm + 1e-9:
                        return best_coeffs.tolist()
        
        # Perturbation
        # Flip a random number of bits
        num_flips = np.random.randint(3, 8)
        flip_indices = np.random.choice(N, num_flips, replace=False)
        for idx in flip_indices:
            current_coeffs[idx] = -current_coeffs[idx]
            
        current_norm = get_max_norm(current_coeffs)
        
        # Optional: If perturbed state is much worse than best, maybe revert?
        # But ILS usually continues.
        
    return best_coeffs.tolist()
