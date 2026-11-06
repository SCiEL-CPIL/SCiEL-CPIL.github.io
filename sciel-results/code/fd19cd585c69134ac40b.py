import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    n = 24
    k = np.arange(n)
    
    # Precompute grid for fast evaluation
    m = 8192
    thetas = np.linspace(0, 2*np.pi, m, endpoint=False)
    # powers_grid[j, i] = exp(i * j * theta_i)
    powers_grid = np.exp(1j * np.outer(k, thetas))
    
    def fast_eval_max(c):
        """Compute max |P(e^{i theta})| on the grid."""
        vals = np.dot(c, powers_grid)
        abs_vals = np.abs(vals)
        return np.max(abs_vals), np.argmax(abs_vals)
    
    def refine_peak(c, theta0, delta):
        """Refine the peak near theta0 using golden section search."""
        def eval_p(theta):
            z = np.exp(1j * theta)
            # Horner's method
            res = c[n-1]
            for i in range(n-2, -1, -1):
                res = res * z + c[i]
            return np.abs(res)
        
        gr = 0.6180339887498949
        a = theta0 - delta
        b = theta0 + delta
        c1 = b - gr * (b - a)
        c2 = a + gr * (b - a)
        f1 = eval_p(c1)
        f2 = eval_p(c2)
        
        for _ in range(30):
            if f1 < f2:
                b = c2
                c2 = c1
                f2 = f1
                c1 = b - gr * (b - a)
                f1 = eval_p(c1)
            else:
                a = c1
                c1 = c2
                f1 = f2
                c2 = a + gr * (b - a)
                f2 = eval_p(c2)
        
        return max(f1, f2, eval_p(theta0))
    
    def get_max_norm(c):
        """Get the max norm with refinement."""
        grid_max, idx = fast_eval_max(c)
        delta = 2 * np.pi / m
        refined = refine_peak(c, thetas[idx], 2*delta)
        return max(grid_max, refined)
    
    def get_max_norm_fast(c):
        """Get the max norm on grid only (for speed during search)."""
        vals = np.dot(c, powers_grid)
        return np.max(np.abs(vals))
    
    # Initial best
    best_c = np.random.choice([-1, 1], size=n)
    best_score = get_max_norm_fast(best_c)
    
    # Vectorized steepest descent
    def vectorized_steepest_descent(c, max_iter=100):
        """Perform steepest descent by trying all single flips and picking the best."""
        current_score = get_max_norm_fast(c)
        improved = True
        iters = 0
        while improved and iters < max_iter:
            improved = False
            best_flip_idx = -1
            best_flip_score = current_score
            
            # Try all single flips
            for j in range(n):
                c[j] *= -1
                new_score = get_max_norm_fast(c)
                if new_score < best_flip_score:
                    best_flip_score = new_score
                    best_flip_idx = j
                c[j] *= -1  # revert
            
            if best_flip_idx >= 0:
                c[best_flip_idx] *= -1
                current_score = best_flip_score
                improved = True
                iters += 1
        
        return c, current_score
    
    # Main search loop with multiple restarts
    restarts = 0
    while time.time() < deadline:
        # Choose start point
        r = random.random()
        if r < 0.3:
            c = np.random.choice([-1, 1], size=n)
        elif r < 0.6:
            c = best_c.copy()
            # Perturb with 1-3 flips
            n_flips = random.randint(1, 3)
            for _ in range(n_flips):
                idx = random.randint(0, n-1)
                c[idx] *= -1
        else:
            c = best_c.copy()
            # Perturb with 4-6 flips
            n_flips = random.randint(4, 6)
            for _ in range(n_flips):
                idx = random.randint(0, n-1)
                c[idx] *= -1
        
        # Run vectorized steepest descent
        c, score = vectorized_steepest_descent(c)
        
        if score < best_score:
            best_score = score
            best_c = c.copy()
        
        restarts += 1
        
        if time.time() > deadline - 10.0:
            break
    
    # Final polish with high-precision evaluation
    # Do a few more rounds of steepest descent with refined evaluation
    for _ in range(5):
        if time.time() > deadline - 2.0:
            break
        current_score = get_max_norm(best_c)
        improved = False
        for j in range(n):
            best_c[j] *= -1
            new_score = get_max_norm(best_c)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                best_c[j] *= -1  # revert
        if not improved:
            break
    
    return best_c.tolist()
