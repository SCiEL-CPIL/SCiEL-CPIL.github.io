import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    best_score = 0.0
    best_config = None
    
    def smooth_penalty(params):
        """
        Smooth penalty function using log-sum-exp style softening.
        Returns (objective_value, gradient) implicitly via finite differences or analytical.
        We use a smooth approximation for the max of violations.
        """
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Clip radii to be non-negative for calculation stability
        r = np.maximum(r, 1e-12)
        
        # Boundary violations:
        # Left: r - x
        # Right: x + r - 1
        # Bottom: r - y
        # Top: y + r - 1
        # We want to penalize max(0, violation).
        # Smooth approximation: softplus(violation / sigma) * sigma
        sigma = 0.01
        def softplus(v):
            return sigma * np.log1p(np.exp(v / sigma))
        
        pen_x_left = softplus(r - x)
        pen_x_right = softplus(x + r - 1)
        pen_y_bottom = softplus(r - y)
        pen_y_top = softplus(y + r - 1)
        
        boundary_pen = np.sum(pen_x_left + pen_x_right + pen_y_bottom + pen_y_top)
        
        # Overlap violations:
        # For pair i,j: (r_i + r_j) - dist_ij
        # We want to penalize max(0, overlap).
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                overlap = r[i] + r[j] - d
                if overlap > -sigma * 5: # Only compute softplus if close to violation
                    overlap_pen += softplus(overlap)
        
        # Objective: maximize sum(r) -> minimize -sum(r) + weight * penalties
        # Use adaptive weight? Or fixed high weight?
        # Fixed weight of 100 seems reasonable to force validity.
        w = 100.0
        obj = -np.sum(r) + w * (boundary_pen + overlap_pen)
        
        return obj

    def get_valid_score(x, y, r):
        """
        Check validity with tight tolerance and return sum of radii.
        """
        if np.any(r < 0):
            return -np.inf
        if np.any(x - r < -1e-8) or np.any(y - r < -1e-8):
            return -np.inf
        if np.any(x + r > 1 + 1e-8) or np.any(y + r > 1 + 1e-8):
            return -np.inf
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < r[i] + r[j] - 1e-8:
                    return -np.inf
        return np.sum(r)

    def optimize_from(params0, maxiter=300):
        """
        Run L-BFGS-B from params0.
        Returns (score, config) if valid, else (-inf, None).
        """
        bounds = []
        for i in range(N):
            bounds.append((0, 1))
        for i in range(N):
            bounds.append((0, 1))
        for i in range(N):
            bounds.append((0, 0.5))
            
        try:
            res = minimize(smooth_penalty, params0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
            
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            
            score = get_valid_score(x_opt, y_opt, r_opt)
            
            if score == -np.inf:
                # Try to repair by scaling down radii
                scale = 1.0
                for _ in range(20):
                    r_scaled = r_opt * scale
                    score = get_valid_score(x_opt, y_opt, r_scaled)
                    if score != -np.inf:
                        break
                    scale *= 0.9
                if score == -np.inf:
                    return -np.inf, None
            
            return score, np.column_stack((x_opt, y_opt, r_opt))
            
        except Exception:
            return -np.inf, None

    def init_random():
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.08, 0.12, N)
        return x, y, r

    def init_grid():
        # 5x5 grid
        x = []
        y = []
        r = []
        for i in range(5):
            for j in range(5):
                x.append((j + 0.5) / 5.0)
                y.append((i + 0.5) / 5.0)
                r.append(0.1)
        # Add 26th circle in a gap? Or just replace one?
        # Let's just take 25 and add one at (0.5, 0.5) with small radius
        x.append(0.5)
        y.append(0.5)
        r.append(0.05)
        
        # Shuffle to break symmetry
        idx = np.random.permutation(N)
        return np.array(x)[idx], np.array(y)[idx], np.array(r)[idx]

    def init_hex():
        # Hexagonal packing
        x = []
        y = []
        r = []
        # 5 rows of 5, alternating offset
        for i in range(5):
            y_pos = (i + 0.5) / 5.0
            offset = 0.0 if i % 2 == 0 else 0.1
            for j in range(5):
                x_pos = (j + 0.5) / 5.0 + offset
                # Clip to [0.1, 0.9]
                x_pos = np.clip(x_pos, 0.1, 0.9)
                x.append(x_pos)
                y.append(y_pos)
                r.append(0.1)
        # Add 26th
        x.append(0.5)
        y.append(0.5)
        r.append(0.05)
        
        idx = np.random.permutation(N)
        return np.array(x)[idx], np.array(y)[idx], np.array(r)[idx]

    # Main loop
    num_restarts = 100
    restart = 0
    while time.time() < deadline - 5:
        restart += 1
        if restart % 3 == 1:
            x0, y0, r0 = init_random()
        elif restart % 3 == 2:
            x0, y0, r0 = init_grid()
        else:
            x0, y0, r0 = init_hex()
            
        params0 = np.concatenate([x0, y0, r0])
        score, config = optimize_from(params0)
        
        if score > best_score:
            best_score = score
            if config is not None:
                best_config = config.copy()
                
        # If we have a good config, try a local perturbation search
        if best_config is not None and best_score > 1.0 and time.time() < deadline - 10:
            # Perturb best config
            x_best = best_config[:, 0]
            y_best = best_config[:, 1]
            r_best = best_config[:, 2]
            
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                x_p = x_best + np.random.normal(0, 0.02, N)
                y_p = y_best + np.random.normal(0, 0.02, N)
                r_p = r_best + np.random.normal(0, 0.01, N)
                
                x_p = np.clip(x_p, 0, 1)
                y_p = np.clip(y_p, 0, 1)
                r_p = np.clip(r_p, 0, 0.5)
                
                params_p = np.concatenate([x_p, y_p, r_p])
                score_p, config_p = optimize_from(params_p, maxiter=200)
                
                if score_p > best_score:
                    best_score = score_p
                    if config_p is not None:
                        best_config = config_p.copy()
                        x_best = best_config[:, 0]
                        y_best = best_config[:, 1]
                        r_best = best_config[:, 2]

    # Final fallback
    if best_config is None:
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.full(N, 0.05)
        best_config = np.column_stack((x, y, r))
    
    # Final validity check and repair
    final_score = get_valid_score(best_config[:, 0], best_config[:, 1], best_config[:, 2])
    if final_score == -np.inf:
        x, y, r = best_config[:, 0], best_config[:, 1], best_config[:, 2]
        scale = 1.0
        for _ in range(50):
            r_new = r * scale
            if get_valid_score(x, y, r_new) != -np.inf:
                best_config = np.column_stack((x, y, r_new))
                break
            scale *= 0.95
        else:
            # Extreme fallback
            x = np.random.uniform(0.1, 0.9, N)
            y = np.random.uniform(0.1, 0.9, N)
            r = np.full(N, 0.01)
            best_config = np.column_stack((x, y, r))

    return best_config
