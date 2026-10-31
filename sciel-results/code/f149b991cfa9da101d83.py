"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    deadline = time.time() + 115.0  # Safety margin before 150s limit
    best_result = None
    best_score = 0.0

    # Aspect ratios to try
    # w + h = 2
    # ar = h/w
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.6, 1.3, 0.75, 1.1, 0.9]
    
    # We will adjust the number of restarts based on time
    # Start with a moderate number
    base_restarts = 8

    def validate(x, y, r, w, h):
        # Check if circles are inside rectangle and non-overlapping
        if np.any(r < -1e-9):
            return False
        if np.any(x < r - 1e-9) or np.any(x > w - r + 1e-9):
            return False
        if np.any(y < r - 1e-9) or np.any(y > h - r + 1e-9):
            return False
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-9:
                    return False
        return True

    def hex_init(w, h, est_r, seed):
        """
        Generate a hexagonal-like grid initialization.
        """
        rng = np.random.default_rng(seed)
        
        # Estimate grid dimensions
        n_cols = max(1, int(np.round(np.sqrt(N * w / h))))
        n_rows = max(1, int(np.ceil(N / n_cols)))
        
        # Adjust if grid is too small
        while n_cols * n_rows < N:
            if n_cols <= n_rows:
                n_cols += 1
            else:
                n_rows += 1

        xs = np.zeros(N)
        ys = np.zeros(N)
        
        for i in range(N):
            col = i % n_cols
            row = i // n_cols
            
            # Base position
            x = (col + 0.5) * w / n_cols
            y = (row + 0.5) * h / n_rows
            
            # Hexagonal offset for odd rows
            if row % 2 == 1:
                x += 0.5 * w / n_cols
                # Wrap around or clamp
                if x > w - est_r:
                    x -= w / n_cols
                elif x < est_r:
                    x += w / n_cols

            xs[i] = x
            ys[i] = y

        # Add random jitter to break symmetry
        # Jitter scale relative to est_r
        jitter_scale = 0.2 * est_r
        xs += rng.normal(0, jitter_scale, N)
        ys += rng.normal(0, jitter_scale, N)
        
        # Clip to stay inside (roughly)
        xs = np.clip(xs, est_r * 0.5, w - est_r * 0.5)
        ys = np.clip(ys, est_r * 0.5, h - est_r * 0.5)
        
        return xs, ys

    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break

        w = 2.0 / (1.0 + ar)
        h = 2.0 - w
        
        # Estimate radius based on area packing
        # N * pi * r^2 ~ packing_density * w * h
        # packing_density ~ 0.9 for hexagonal
        est_r = np.sqrt(0.9 * w * h / (N * np.pi))
        
        # We can try to increase est_r slightly to encourage larger circles
        # But start a bit smaller to avoid huge initial penalty
        init_r = est_r * 0.8

        # Determine restarts for this aspect ratio
        # If we have lots of time left, do more restarts
        time_left = deadline - time.time()
        # Assume ~1s per restart (optimistic)
        max_restarts = max(1, int(time_left / 2))
        # Cap restarts to avoid excessive time on one AR
        max_restarts = min(max_restarts, 20)

        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            seed = restart * 1000 + int(ar * 100)
            xs, ys = hex_init(w, h, init_r, seed)
            
            r0 = np.full(N, init_r)
            
            x0 = np.zeros(3 * N)
            x0[0::3] = xs
            x0[1::3] = ys
            x0[2::3] = r0

            def objective(vars_):
                x = vars_[0::3]
                y = vars_[1::3]
                r = vars_[2::3]
                
                # Sum of radii
                sum_r = np.sum(r)
                
                # Overlap penalty
                penalty = 0.0
                # Vectorized calculation for speed?
                # For N=21, loop is fine, but let's try to be efficient
                # d_ij = sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
                # We want max(0, r_i+r_j - d_ij)^2
                
                # Using broadcasting
                # x_diff = x[:, None] - x[None, :]
                # This creates 21x21 matrix
                # It's fast enough
                
                x_diff = x[:, None] - x[None, :]
                y_diff = y[:, None] - y[None, :]
                dist_matrix = np.sqrt(x_diff**2 + y_diff**2)
                
                r_sum_matrix = r[:, None] + r[None, :]
                
                # Overlap = r_sum - dist
                # We only care about upper triangle
                # To avoid double counting and self-overlap (dist=0, r_sum=2r > 0)
                # Mask upper triangle
                mask = np.triu(np.ones((N, N), dtype=bool), k=1)
                
                overlaps = r_sum_matrix - dist_matrix
                # Only consider positive overlaps
                positive_overlaps = np.where(overlaps > 0, overlaps, 0)
                
                # Apply mask
                positive_overlaps = positive_overlaps[mask]
                
                penalty = np.sum(positive_overlaps**2)
                
                return -sum_r + 500.0 * penalty

            def make_constraints(w, h):
                def cons_inside(vars_):
                    x = vars_[0::3]
                    y = vars_[1::3]
                    r = vars_[2::3]
                    return np.concatenate([
                        x - r,
                        w - x - r,
                        y - r,
                        h - y - r
                    ])
                def cons_positive_r(vars_):
                    r = vars_[2::3]
                    # Allow small negative to help optimizer? No, keep positive
                    return r
                return [
                    {'type': 'ineq', 'fun': cons_inside},
                    {'type': 'ineq', 'fun': cons_positive_r}
                ]

            constraints = make_constraints(w, h)
            
            try:
                # Use SLSQP
                # maxiter: allow enough iterations
                result = minimize(
                    objective, x0, method='SLSQP',
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-9, 'disp': False}
                )
                
                x, y, r = result.x[0::3], result.x[1::3], result.x[2::3]
                
                # Clip radii to 0 if negative (shouldn't happen with constraint)
                r = np.maximum(r, 0)
                
                if validate(x, y, r, w, h):
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack([x, y, r])
                        
            except Exception:
                continue

    # Fallback: simple grid in a square of perimeter 4 (w=h=1)
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
        return np.array(cells)

    return best_result
# EVOLVE-BLOCK-END
