import numpy as np
import time
from scipy.optimize import minimize_scalar

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Initial candidate: Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:24], dtype=float)
    
    # Precompute grid for fast evaluation - use higher resolution for better accuracy
    M = 16384  # Higher resolution grid
    theta = np.linspace(0, 2*np.pi, M, endpoint=False)
    z = np.exp(1j * theta)
    # Z[k, j] = z_j^k
    Z = np.vstack([z**k for k in range(24)])
    
    def calc_score(c):
        # c is shape (24,)
        # P_j = sum c_k z_j^k
        P = c @ Z  # shape (M,)
        abs_P = np.abs(P)
        return np.max(abs_P)

    def calc_score_refined(c):
        """High-precision score using golden-section refinement of top peaks"""
        P = c @ Z
        abs_P = np.abs(P)
        
        # Find top N peaks by grid value
        N_peaks = 20
        top_indices = np.argsort(abs_P)[-N_peaks:]
        
        peaks = []
        dt = 2*np.pi / M
        
        for i in top_indices:
            # Refine in neighborhood using golden-section search
            def f(t):
                z_t = np.exp(1j * t)
                val = np.sum(c * z_t ** np.arange(24))
                return -np.abs(val)
            
            theta_i = theta[i]
            res = minimize_scalar(f, bounds=(theta_i - dt, theta_i + dt), 
                                   method='bounded', options={'xatol': 1e-12})
            peaks.append(-res.fun)
        
        if not peaks:
            return np.max(abs_P)
        return max(peaks)

    # Phase 1: Hill climbing from GR with high-resolution grid
    current_coeffs = best_coeffs.copy()
    current_score = calc_score(current_coeffs)
    best_found_score = current_score
    best_found_coeffs = current_coeffs.copy()
    
    improved = True
    while improved and time.time() < deadline - 30.0:
        improved = False
        for i in range(24):
            old_val = current_coeffs[i]
            current_coeffs[i] = -old_val
            
            new_score = calc_score(current_coeffs)
            
            if new_score < current_score - 1e-9:
                current_score = new_score
                improved = True
                if new_score < best_found_score:
                    best_found_score = new_score
                    best_found_coeffs = current_coeffs.copy()
            else:
                current_coeffs[i] = old_val
                
        if not improved:
            # Escape local minimum: random kick
            idx = np.random.randint(24)
            current_coeffs[idx] *= -1
            current_score = calc_score(current_coeffs)
            
    # Phase 2: Simulated annealing to escape deeper local minima
    if time.time() < deadline - 20.0:
        current_coeffs = best_found_coeffs.copy()
        current_score = calc_score(current_coeffs)
        
        T = 0.5  # Initial temperature
        T_min = 0.01
        cooling = 0.995
        
        while T > T_min and time.time() < deadline - 20.0:
            for i in range(24):
                if time.time() >= deadline - 20.0:
                    break
                    
                old_val = current_coeffs[i]
                current_coeffs[i] = -old_val
                
                new_score = calc_score(current_coeffs)
                delta = new_score - current_score
                
                # Accept if better or with probability exp(-delta/T)
                if delta < 0 or np.random.random() < np.exp(-delta / T):
                    current_score = new_score
                    if new_score < best_found_score:
                        best_found_score = new_score
                        best_found_coeffs = current_coeffs.copy()
                else:
                    current_coeffs[i] = old_val
                    
            T *= cooling
            
    # Phase 3: Final high-precision polish on best candidate
    if time.time() < deadline - 10.0:
        # Do one more hill climb from best found
        current_coeffs = best_found_coeffs.copy()
        current_score = calc_score(current_coeffs)
        
        for _ in range(3):  # Multiple passes
            improved = True
            while improved and time.time() < deadline - 10.0:
                improved = False
                for i in range(24):
                    old_val = current_coeffs[i]
                    current_coeffs[i] = -old_val
                    new_score = calc_score(current_coeffs)
                    
                    if new_score < current_score - 1e-9:
                        current_score = new_score
                        improved = True
                        if new_score < best_found_score:
                            best_found_score = new_score
                            best_found_coeffs = current_coeffs.copy()
                    else:
                        current_coeffs[i] = old_val
                        
    # Return the best coefficients found
    return [int(c) for c in best_found_coeffs]
