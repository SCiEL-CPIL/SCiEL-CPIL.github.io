"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    best_score = -1.0
    best_result = None

    def compute_rect(aspect):
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        return w, h

    def evaluate(circles, aspect):
        """Return (sum_radii, is_valid). circles: (N,3) array [x, y, r]."""
        w, h = compute_rect(aspect)
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]

        if np.any(xs < rs - 1e-9) or np.any(xs > w - rs + 1e-9) or \
           np.any(ys < rs - 1e-9) or np.any(ys > h - rs + 1e-9):
            return 0.0, False

        n = len(circles)
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < rs[i] + rs[j] - 1e-9:
                    return 0.0, False

        return float(np.sum(rs)), True

    def make_initial(aspect, seed_val=42):
        """Create initial grid layout for given aspect."""
        rng = np.random.RandomState(seed_val)
        w, h = compute_rect(aspect)
        circles = np.zeros((N, 3))
        # Use a 5x5 grid, take first 21
        side = 5
        for i in range(N):
            row = i // side
            col = i % side
            circles[i, 0] = (col + 0.5) / side * w
            circles[i, 1] = (row + 0.5) / side * h
            circles[i, 2] = 0.5 / side * min(w, h) * 0.95
        return circles

    def softplus(x, beta=100.0):
        """Numerically stable softplus."""
        return np.where(x > 0, x + np.log1p(np.exp(-beta * x)) / beta, np.log1p(np.exp(beta * x)) / beta)

    def penalized_objective(params, aspect, beta=100.0):
        """
        Minimize -sum(r) + penalty.
        params: array of shape (N*3,) = [x0,y0,r0, x1,y1,r1, ...]
        """
        n = N
        circles = params.reshape(n, 3)
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        
        w, h = compute_rect(aspect)
        
        # Boundary violations (must be >= 0)
        # x >= r -> x - r >= 0
        # x <= w - r -> w - x - r >= 0
        # y >= r -> y - r >= 0
        # y <= h - r -> h - y - r >= 0
        viol_x_min = rs - xs
        viol_x_max = xs + rs - w
        viol_y_min = rs - ys
        viol_y_max = ys + rs - h
        
        # Pairwise violations
        # dist >= r_i + r_j -> (r_i + r_j)^2 - dist^2 <= 0
        # We want to penalize when (r_i + r_j)^2 - dist^2 > 0
        # Let's use a smooth approximation for max(0, (r_i+r_j)^2 - dist^2)
        # Actually, it's better to penalize the overlap in distance:
        # overlap = (r_i + r_j) - dist
        # We want to penalize max(0, overlap)
        
        penalty = 0.0
        # Boundary penalties
        penalty += np.sum(softplus(viol_x_min, beta))
        penalty += np.sum(softplus(viol_x_max, beta))
        penalty += np.sum(softplus(viol_y_min, beta))
        penalty += np.sum(softplus(viol_y_max, beta))
        
        # Pairwise penalties
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                overlap = rs[i] + rs[j] - dist
                penalty += softplus(overlap, beta)
        
        score = np.sum(rs)
        return -score + penalty

    def penalized_gradient(params, aspect, beta=100.0):
        """
        Gradient of penalized_objective.
        """
        n = N
        circles = params.reshape(n, 3)
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        
        w, h = compute_rect(aspect)
        
        grad = np.zeros_like(params)
        
        # Boundary gradients
        # softplus'(x) = 1 / (1 + exp(-beta*x))
        def sig(x):
            return 1.0 / (1.0 + np.exp(-beta * x))
        
        # x - r >= 0 -> viol = r - x
        # d(viol)/dx = -1, d(viol)/dr = 1
        g_x_min = sig(rs - xs)
        g_x_max = sig(xs + rs - w)
        g_y_min = sig(rs - ys)
        g_y_max = sig(ys + rs - h)
        
        # grad_x += -g_x_min - g_x_max
        # grad_y += -g_y_min - g_y_max
        # grad_r += g_x_min + g_x_max + g_y_min + g_y_max
        
        grad[0::3] -= (g_x_min + g_x_max)  # x
        grad[1::3] -= (g_y_min + g_y_max)  # y
        grad[2::3] += (g_x_min + g_x_max + g_y_min + g_y_max)  # r
        
        # Pairwise gradients
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                overlap = rs[i] + rs[j] - dist
                s = sig(overlap)
                
                # d(overlap)/dx_i = -dx/dist
                # d(overlap)/dx_j = +dx/dist
                # d(overlap)/dr_i = 1
                # d(overlap)/dr_j = 1
                
                inv_dist = 1.0 / dist
                grad[3*i + 0] += s * (-dx * inv_dist)
                grad[3*i + 1] += s * (-dy * inv_dist)
                grad[3*i + 2] += s
                
                grad[3*j + 0] += s * (dx * inv_dist)
                grad[3*j + 1] += s * (dy * inv_dist)
                grad[3*j + 2] += s
                
        # Gradient of -sum(r)
        grad[2::3] -= 1.0
        
        return grad

    def optimize_for_aspect(aspect, circles_init, beta_start=50.0, beta_end=500.0, max_iter=1000):
        """Run L-BFGS-B optimization for a fixed aspect ratio."""
        params_init = circles_init.flatten()
        
        # Bounds: x in [0, w], y in [0, h], r in [0, min(w,h)/2]
        w, h = compute_rect(aspect)
        bounds = []
        for i in range(N):
            bounds.append((0.0, w))  # x
            bounds.append((0.0, h))  # y
            bounds.append((0.01, min(w, h) / 2.0))  # r
            
        # Anneal beta
        n_betas = 5
        betas = np.linspace(beta_start, beta_end, n_betas)
        
        current_params = params_init
        for beta in betas:
            if time.time() > deadline:
                break
            res = minimize(
                penalized_objective,
                current_params,
                args=(aspect, beta),
                method='L-BFGS-B',
                jac=penalized_gradient,
                bounds=bounds,
                options={'maxiter': max_iter // n_betas, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            current_params = res.x
            
        # Evaluate final result
        circles_final = current_params.reshape(N, 3)
        score, valid = evaluate(circles_final, aspect)
        if valid:
            return score, circles_final
        else:
            return -1.0, None

    # Phase 1: Try a few aspect ratios with SA to get good initial guesses
    aspects_to_try = [1.0, 1.2, 0.8, 1.5, 0.7]
    
    # Use SA to warm up
    random.seed(42)
    np.random.seed(42)
    
    for aspect in aspects_to_try:
        if time.time() > deadline:
            break
            
        circles_init = make_initial(aspect, seed_val=42)
        
        # Quick SA to improve initial guess
        T = 0.05
        T_min = 1e-5
        cooling = 0.999
        cur_circles = circles_init.copy()
        cur_aspect = aspect
        cur_score = -1.0
        _, valid = evaluate(cur_circles, cur_aspect)
        if valid:
            cur_score = float(np.sum(cur_circles[:, 2]))
            
        max_sa_iter = 5000
        for it in range(max_sa_iter):
            if time.time() > deadline:
                break
            if T < T_min:
                break
                
            pert_type = random.random()
            new_circles = cur_circles.copy()
            
            if pert_type < 0.4:
                idx = random.randint(0, N - 1)
                new_circles[idx, 0] += random.gauss(0, 0.02)
                new_circles[idx, 1] += random.gauss(0, 0.02)
            elif pert_type < 0.7:
                idx = random.randint(0, N - 1)
                new_circles[idx, 2] = max(0.01, new_circles[idx, 2] + random.gauss(0, 0.01))
            else:
                # Small aspect change
                new_aspect = cur_aspect * (1.0 + random.gauss(0, 0.05))
                new_aspect = max(0.5, min(2.0, new_aspect))
                old_w, old_h = compute_rect(cur_aspect)
                new_w, new_h = compute_rect(new_aspect)
                new_circles[:, 0] *= new_w / old_w
                new_circles[:, 1] *= new_h / old_h
                cur_aspect = new_aspect
            
            new_score, valid = evaluate(new_circles, cur_aspect)
            if valid:
                delta = new_score - cur_score
                if delta > 0 or random.random() < np.exp(delta / T):
                    cur_circles = new_circles
                    cur_score = new_score
            # If invalid, we don't update, effectively penalizing
            
            T *= cooling
            
        if cur_score > best_score:
            best_score = cur_score
            best_result = (cur_circles.copy(), cur_aspect)
            
        # Phase 2: L-BFGS-B polish
        if cur_score > -0.5:  # If we have a reasonable starting point
            score, circles_opt = optimize_for_aspect(cur_aspect, cur_circles)
            if score > best_score:
                best_score = score
                best_result = (circles_opt, cur_aspect)

    # Phase 3: High-precision polish on best result
    if best_result is not None and time.time() < deadline:
        final_circles, final_aspect = best_result[0].copy(), best_result[1]
        # Run one more L-BFGS-B with high beta
        score, circles_opt = optimize_for_aspect(final_aspect, final_circles, beta_start=100.0, beta_end=1000.0, max_iter=500)
        if score > best_score:
            best_score = score
            best_result = (circles_opt, final_aspect)

    if best_result is None:
        # Fallback
        return make_initial(1.0, 42)

    circles, aspect = best_result
    w, h = compute_rect(aspect)
    
    # Final clamp to ensure validity
    for i in range(N):
        r = min(circles[i, 2], circles[i, 0] - 1e-6, w - circles[i, 0] - 1e-6,
                circles[i, 1] - 1e-6, h - circles[i, 1] - 1e-6)
        r = max(r, 0.01)
        circles[i, 2] = r

    return circles
# EVOLVE-BLOCK-END
