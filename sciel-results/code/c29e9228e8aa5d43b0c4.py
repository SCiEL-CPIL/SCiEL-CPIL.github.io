import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 140.0  # Safety margin before 150s kill

def construct():
    start_time = time.time()
    best_score = 0.0
    best_result = None

    def objective(params):
        """
        Objective function to minimize.
        params: [w, x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
        We maximize sum(r_i) by minimizing -sum(r_i) + penalty.
        """
        w = params[0]
        h = 2.0 - w
        
        # Basic bounds check
        if w < 0.1 or h < 0.1:
            return 1e10
            
        # Extract centers and radii
        # params[1:] is flattened array of (x, y, r) for each circle
        circles = params[1:].reshape(N, 3)
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # 1. Boundary constraints
        # x_i - r_i >= 0, x_i + r_i <= w
        # y_i - r_i >= 0, y_i + r_i <= h
        # r_i >= 0
        
        # Violations
        v_x_min = np.maximum(0, -(x - r))
        v_x_max = np.maximum(0, (x + r) - w)
        v_y_min = np.maximum(0, -(y - r))
        v_y_max = np.maximum(0, (y + r) - h)
        v_r_min = np.maximum(0, -r)
        
        boundary_violation = np.sum(v_x_min**2) + np.sum(v_x_max**2) + np.sum(v_y_min**2) + np.sum(v_y_max**2) + np.sum(v_r_min**2)
        
        # 2. Overlap constraints
        # dist(i,j) >= r_i + r_j
        # dist^2 - (r_i + r_j)^2 >= 0
        # We can use a soft penalty: max(0, (r_i + r_j)^2 - dist^2)^2
        
        overlap_violation = 0.0
        # For N=21, N*(N-1)/2 = 210 pairs. Vectorized is faster.
        # Compute pairwise distances
        # x: (N,), y: (N,), r: (N,)
        
        # Use broadcasting for vectorized distance calculation
        # dx: (N, N), dy: (N, N)
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        
        # Sum of radii: (N, N)
        sum_r = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Constraint: dist_sq >= sum_r^2
        # Violation: max(0, sum_r^2 - dist_sq)
        # We only care about i < j, but squaring the whole matrix and summing is fine, 
        # just double counts. We can divide by 2 or just ignore the factor.
        
        # To save computation, we can take the upper triangle.
        # But vectorized full matrix is simpler and fast enough for N=21.
        # Let's use upper triangle mask to be precise and slightly faster.
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        viol = np.maximum(0, sum_r[mask]**2 - dist_sq[mask])
        overlap_violation = np.sum(viol**2)
        
        # Total penalty
        penalty = 10.0 * (boundary_violation + overlap_violation)
        
        # Objective: -sum(r) + penalty
        # We want to maximize sum(r), so minimize -sum(r)
        score = -np.sum(r)
        
        return score + penalty

    def run_nelder_mead(w_init, centers_init, radii_init, seed):
        """
        Run Nelder-Mead optimization starting from a given configuration.
        Returns (score, w, centers, radii).
        """
        # Initial parameters
        x0 = np.concatenate(([w_init], centers_init.flatten(), radii_init.flatten()))
        
        # Bounds are handled by penalty, but we can provide initial simplex
        # Nelder-Mead doesn't strictly support bounds, but we can clip during evaluation?
        # No, objective handles it. But if it goes crazy, penalty increases.
        
        # Options for Nelder-Mead
        options = {
            'maxiter': 500,
            'xatol': 1e-6,
            'fatol': 1e-8,
            'adaptive': True
        }
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', options=options)
            
            if result.success or (time.time() - start_time) < DEADLINE:
                w_opt = result.x[0]
                circles_opt = result.x[1:].reshape(N, 3)
                x_opt = circles_opt[:, 0]
                y_opt = circles_opt[:, 1]
                r_opt = circles_opt[:, 2]
                
                # Check validity and calculate true score
                h_opt = 2.0 - w_opt
                
                # Validate
                valid = True
                for i in range(N):
                    if r_opt[i] < -1e-9:
                        valid = False
                        break
                    if x_opt[i] - r_opt[i] < -1e-6 or x_opt[i] + r_opt[i] > w_opt + 1e-6:
                        valid = False
                        break
                    if y_opt[i] - r_opt[i] < -1e-6 or y_opt[i] + r_opt[i] > h_opt + 1e-6:
                        valid = False
                        break
                
                if valid:
                    # Check overlaps
                    for i in range(N):
                        for j in range(i + 1, N):
                            dist = np.sqrt((x_opt[i] - x_opt[j])**2 + (y_opt[i] - y_opt[j])**2)
                            if dist < r_opt[i] + r_opt[j] - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    score = np.sum(r_opt)
                    return score, w_opt, x_opt, y_opt, r_opt
                else:
                    return 0.0, w_opt, x_opt, y_opt, r_opt
            else:
                return 0.0, w_init, centers_init[:, 0], centers_init[:, 1], radii_init
        except Exception:
            return 0.0, w_init, centers_init[:, 0], centers_init[:, 1], radii_init

    def generate_initial_config(w, h, seed):
        """
        Generate a random initial configuration of centers and radii.
        """
        rng = np.random.RandomState(seed)
        
        # Random centers
        centers = np.zeros((N, 2))
        centers[:, 0] = rng.uniform(0.1, w - 0.1, N)
        centers[:, 1] = rng.uniform(0.1, h - 0.1, N)
        
        # Initial radii: small random values
        # Estimate max possible radius based on area
        # Area = w * h. Average area per circle = w*h/N.
        # r ~ sqrt(area/pi)
        avg_r = np.sqrt(w * h / (N * np.pi))
        radii = rng.uniform(0.1 * avg_r, 0.5 * avg_r, N)
        
        # Ensure radii don't exceed boundaries initially (rough check)
        radii = np.minimum(radii, np.minimum(centers[:, 0], w - centers[:, 0]))
        radii = np.minimum(radii, np.minimum(centers[:, 1], h - centers[:, 1]))
        radii = np.maximum(radii, 0.01)
        
        return centers, radii

    # Multi-restart strategy
    # Try different aspect ratios
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.6, 2.0, 1.8, 0.7, 1.3, 0.9]
    
    # Shuffle to randomize order of attempts
    rng_global = np.random.RandomState(42)
    rng_global.shuffle(aspect_ratios)
    
    num_restarts = 5  # Number of random restarts per aspect ratio
    
    for ar_idx, ar in enumerate(aspect_ratios):
        if time.time() > DEADLINE:
            break
            
        w = ar * 2.0 / (ar + 1.0)
        h = 2.0 - w
        
        for restart in range(num_restarts):
            if time.time() > DEADLINE:
                break
                
            seed = ar_idx * 100 + restart
            centers, radii = generate_initial_config(w, h, seed)
            
            score, w_opt, x_opt, y_opt, r_opt = run_nelder_mead(w, centers, radii, seed)
            
            if score > best_score:
                best_score = score
                best_result = (w_opt, np.column_stack((x_opt, y_opt)), r_opt)
                
            # If we found a very good score, we might stop early?
            # Theoretical max sum of radii?
            # If square 1x1, 21 circles.
            # Seed score is 1.0. Let's aim for > 1.2.
            # If best_score > 1.5, maybe stop? No, keep searching until deadline.

    # Final polish: If we have a good result, run one more Nelder-Mead with tighter tolerance
    # or just return the best found.
    
    if best_result is not None:
        w, centers, r = best_result
        h = 2.0 - w
        result_array = np.zeros((N, 3))
        for i in range(N):
            x = np.clip(centers[i, 0], r[i], w - r[i])
            y = np.clip(centers[i, 1], r[i], h - r[i])
            r_i = max(0.001, min(r[i], min(w, h) / 2 - 0.001))
            result_array[i] = [x, y, r_i]
        return result_array
    else:
        # Fallback: grid packing
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        cells = []
        for idx in range(N):
            i = idx % side
            j = idx // side
            r = 0.5 / side - 0.001
            x = (i + 0.5) * w / side
            y = (j + 0.5) * h / side
            cells.append([x, y, r])
        return np.array(cells)
