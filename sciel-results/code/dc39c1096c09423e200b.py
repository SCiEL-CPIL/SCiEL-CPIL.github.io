import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    N = 24
    M = 10000  # Grid size for initial peak detection
    angles = np.linspace(0, 2 * np.pi, M, endpoint=False)
    Z = np.exp(1j * np.outer(np.arange(N), angles))
    
    def eval_poly(c, theta):
        # c: (N,)
        # theta: scalar or array
        # P(z) = sum c_k exp(i k theta)
        # z^k = exp(i k theta)
        # Use matrix multiplication for arrays, direct for scalars
        if np.isscalar(theta):
            z = np.exp(1j * theta)
            # Vectorized power
            zk = np.power(z, np.arange(N))
            return np.abs(np.dot(c, zk))
        else:
            zk = np.exp(1j * np.outer(np.arange(N), theta))
            return np.abs(np.sum(c[:, None] * zk, axis=0))

    def get_peak_refined(c):
        # 1. Dense sampling
        vals = eval_poly(c, angles)
        # Find indices of top peaks
        # We want the max, but also local maxima to ensure we don't miss a sharp peak
        # Simple approach: find the global max index, and its neighbors.
        # Better: find all local maxima on the grid.
        
        # To be efficient, just refine the top 5 peaks from the grid.
        # The sup-norm is determined by the highest peaks.
        sorted_indices = np.argsort(vals)[::-1]
        top_indices = sorted_indices[:5]
        
        max_val = 0.0
        for idx in top_indices:
            # Refine around angles[idx]
            # Golden section search for max of |P(e^{i theta})|
            # Range: [angles[idx] - delta, angles[idx] + delta]
            # delta = 2*pi / M
            delta = 2 * np.pi / M
            a = angles[idx] - delta
            b = angles[idx] + delta
            
            # Handle wrap-around
            # Since the function is periodic, we can just search in [a, b]
            # Golden section constants
            gr = (np.sqrt(5) - 1) / 2
            c1 = b - gr * (b - a)
            c2 = a + gr * (b - a)
            
            f_c1 = eval_poly(c, c1)
            f_c2 = eval_poly(c, c2)
            
            for _ in range(50): # 50 iterations gives ~1e-12 accuracy
                if f_c1 < f_c2:
                    a = c1
                    c1 = c2
                    f_c1 = f_c2
                    c2 = a + gr * (b - a)
                    f_c2 = eval_poly(c, c2)
                else:
                    b = c2
                    c2 = c1
                    f_c2 = f_c1
                    c1 = b - gr * (b - a)
                    f_c1 = eval_poly(c, c1)
            
            peak_val = max(f_c1, f_c2)
            if peak_val > max_val:
                max_val = peak_val
                
        return max_val

    def local_search(c):
        current_peak = get_peak_refined(c)
        improved = True
        while improved:
            improved = False
            for i in range(N):
                old_val = c[i]
                c[i] *= -1
                new_peak = get_peak_refined(c)
                if new_peak < current_peak:
                    current_peak = new_peak
                    improved = True
                else:
                    c[i] = old_val
        return c, current_peak

    # 1. Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    rs_coeffs = np.array(p[:N], dtype=float)
    
    best_coeffs, best_peak = local_search(rs_coeffs.copy())
    
    # 2. Random Restarts
    while time.time() < deadline:
        c = np.random.choice([-1.0, 1.0], size=N)
        curr_coeffs, curr_peak = local_search(c)
        if curr_peak < best_peak:
            best_peak = curr_peak
            best_coeffs = curr_coeffs
            if best_peak < 1.2584:
                break
                
    return [1 if c > 0 else -1 for c in best_coeffs]
