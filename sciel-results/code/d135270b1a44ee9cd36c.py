"""Rewritten program for circle packing in perimeter-4 rectangle."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
from scipy.spatial import Delaunay

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    best_value = -np.inf
    best_result = None
    best_w = 1.0
    
    def compute_overlap_penalty_vec(xs, ys, rs):
        """Vectorized overlap penalty using all pairs."""
        # Build distance matrix
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = rs[:, None] + rs[None, :]
        # Only upper triangle
        iu = np.triu_indices(N, 1)
        d = dist[iu]
        md = min_dist[iu]
        viol = np.maximum(0, md - d)
        return np.sum(viol ** 2)

    def boundary_penalty(xs, ys, rs, w, h):
        """Vectorized boundary penalty."""
        pen = 0.0
        # x bounds
        lx = np.maximum(0, rs - xs)
        rx = np.maximum(0, xs - (w - rs))
        # y bounds
        ly = np.maximum(0, rs - ys)
        ry = np.maximum(0, ys - (h - rs))
        pen += np.sum(lx ** 2) + np.sum(rx ** 2) + np.sum(ly ** 2) + np.sum(ry ** 2)
        return pen

    def total_penalty(xs, ys, rs, w, h):
        return boundary_penalty(xs, ys, rs, w, h) + compute_overlap_penalty_vec(xs, ys, rs)

    def check_and_update(x, w, h):
        nonlocal best_value, best_result, best_w
        if time.time() > deadline:
            return
        xs = x[0::3]
        ys = x[1::3]
        rs = x[2::3]
        # Check bounds
        if np.any(rs < 1e-9):
            return
        if np.any(xs - rs < -1e-6) or np.any(xs + rs > w + 1e-6):
            return
        if np.any(ys - rs < -1e-6) or np.any(ys + rs > h + 1e-6):
            return
        # Check disjointness
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx * dx + dy * dy
                min_dist = rs[i] + rs[j]
                if dist_sq < (min_dist - 1e-7) ** 2:
                    return
        val = np.sum(rs)
        if val > best_value:
            best_value = val
            best_result = x.copy()
            best_w = w

    def sa_joint_optimize(x0, w0, max_iters=3000, seed=42):
        """Simulated annealing jointly optimizing w, positions, and radii."""
        xs = x0[0::3].copy()
        ys = x0[1::3].copy()
        rs = x0[2::3].copy()
        w = w0
        h = 2.0 - w
        
        T = 0.08
        cooling = 0.9995
        rng = np.random.default_rng(seed)
        
        def cost(xs, ys, rs, w, h):
            bpen = boundary_penalty(xs, ys, rs, w, h)
            overlap = compute_overlap_penalty_vec(xs, ys, rs)
            return bpen + overlap - np.sum(rs)
        
        current_cost = cost(xs, ys, rs, w, h)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            move_type = rng.random()
            
            if move_type < 0.15:
                # Perturb aspect ratio
                old_w = w
                old_xs, old_ys, old_rs = xs.copy(), ys.copy(), rs.copy()
                w += rng.normal(0, T * 0.5)
                w = np.clip(w, 0.4, 1.6)
                h = 2.0 - w
                # Re-clamp circles to new bounds
                for i in range(N):
                    rs[i] = np.clip(rs[i], 1e-4, min(w, h) / 2.0)
                    xs[i] = np.clip(xs[i], rs[i], w - rs[i])
                    ys[i] = np.clip(ys[i], rs[i], h - rs[i])
                new_cost = cost(xs, ys, rs, w, h)
                delta = new_cost - current_cost
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_cost = new_cost
                else:
                    w = old_w
                    h = 2.0 - w
                    xs, ys, rs = old_xs, old_ys, old_rs
                    current_cost = cost(xs, ys, rs, w, h)
                    
            else:
                # Perturb a random circle
                idx = rng.integers(0, N)
                old_x, old_y, old_r = xs[idx], ys[idx], rs[idx]
                scale = T * 0.3
                xs[idx] += rng.normal(0, scale)
                ys[idx] += rng.normal(0, scale)
                rs[idx] += rng.normal(0, scale * 0.5)
                
                # Clamp to bounds
                rs[idx] = np.clip(rs[idx], 1e-4, min(w, h) / 2.0)
                xs[idx] = np.clip(xs[idx], rs[idx], w - rs[idx])
                ys[idx] = np.clip(ys[idx], rs[idx], h - rs[idx])
                
                new_cost = cost(xs, ys, rs, w, h)
                delta = new_cost - current_cost
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_cost = new_cost
                else:
                    xs[idx], ys[idx], rs[idx] = old_x, old_y, old_r
                
            T *= cooling
        
        x_final = np.zeros(3 * N)
        x_final[0::3] = xs
        x_final[1::3] = ys
        x_final[2::3] = rs
        return x_final, w

    def refine_with_slsqp(x0, w0, max_iter=200):
        """Use SLSQP to refine the solution with proper constraints."""
        x0 = x0.copy()
        w = w0
        h = 2.0 - w
        
        def objective(v):
            xs = v[0::3]
            ys = v[1::3]
            rs = v[2::3]
            # Soft penalty for violations to guide optimizer
            bpen = boundary_penalty(xs, ys, rs, w, h)
            overlap = compute_overlap_penalty_vec(xs, ys, rs)
            return bpen + overlap - np.sum(rs)
        
        # Constraints: non-overlapping
        cons = []
        for i in range(N):
            for j in range(i + 1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2
                })
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((1e-4, min(w, h)/2))  # x
            bounds.append((1e-4, min(w, h)/2))  # y
            bounds.append((1e-4, min(w, h)/2))  # r
        
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': max_iter, 'ftol': 1e-10})
        return res.x

    def refine_positions_fixed_radii(rs, xs_init, ys_init, w, h, max_iter=100):
        """Refine positions only, keeping radii fixed."""
        def obj(v):
            xs = v[0::2]
            ys = v[1::2]
            bpen = boundary_penalty(xs, ys, rs, w, h)
            overlap = compute_overlap_penalty_vec(xs, ys, rs)
            return bpen + overlap
        
        cons = []
        for i in range(N):
            for j in range(i + 1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: (v[2*i] - v[2*j])**2 + (v[2*i+1] - v[2*j+1])**2 - (rs[i] + rs[j])**2
                })
        
        bounds = []
        for i in range(N):
            bounds.append((rs[i], w - rs[i]))  # x
            bounds.append((rs[i], h - rs[i]))  # y
        
        v0 = np.zeros(2 * N)
        v0[0::2] = xs_init
        v0[1::2] = ys_init
        
        res = minimize(obj, v0, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': max_iter, 'ftol': 1e-10})
        return res.x

    # Phase 1: Joint SA optimization with multiple initial aspect ratios
    init_widths = [0.7, 0.85, 1.0, 1.15, 1.3, 1.45]
    
    for idx_w, w_init in enumerate(init_widths):
        if time.time() > deadline - 15:
            break
        h_init = 2.0 - w_init
        
        # Initialize with grid
        a = int(np.ceil(np.sqrt(N * w_init / h_init)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
            
        x0 = np.zeros(3 * N)
        for i in range(N):
            col = i % a
            row = i // a
            x0[3 * i] = (col + 0.5) * w_init / a
            x0[3 * i + 1] = (row + 0.5) * h_init / b
            x0[3 * i + 2] = min(w_init / a, h_init / b) * 0.45
        
        # Joint SA optimization
        x_j, w_j = sa_joint_optimize(x0, w_init, max_iters=2500, seed=42 + idx_w)
        h_j = 2.0 - w_j
        check_and_update(x_j, w_j, h_j)
        
        # SLSQP refinement
        if time.time() < deadline - 20:
            x_ref = refine_with_slsqp(x_j, w_j, max_iter=150)
            check_and_update(x_ref, w_j, h_j)
            
            # Second pass with perturbation
            if time.time() < deadline - 15:
                rng2 = np.random.default_rng(123 + idx_w * 7)
                x0b = x_ref.copy()
                for i in range(N):
                    x0b[3*i] += rng2.normal(0, 0.02)
                    x0b[3*i+1] += rng2.normal(0, 0.02)
                    x0b[3*i+2] += rng2.normal(0, 0.01)
                x_j2, w_j2 = sa_joint_optimize(x0b, w_j, max_iters=1200, seed=200 + idx_w)
                h_j2 = 2.0 - w_j2
                check_and_update(x_j2, w_j2, h_j2)
                
                # Final SLSQP polish
                if time.time() < deadline - 10:
                    x_pol = refine_with_slsqp(x_j2, w_j2, max_iter=100)
                    check_and_update(x_pol, w_j2, h_j2)

    # Phase 2: Fixed-w refinement on best aspect ratio found
    if best_result is not None and time.time() < deadline - 25:
        for w_refine in [best_w - 0.1, best_w, best_w + 0.1]:
            if time.time() > deadline - 5:
                break
            if w_refine < 0.4 or w_refine > 1.6:
                continue
            h_refine = 2.0 - w_refine
            a = int(np.ceil(np.sqrt(N * w_refine / h_refine)))
            b = int(np.ceil(N / a))
            while a * b < N:
                b += 1
            x0r = np.zeros(3 * N)
            for i in range(N):
                col = i % a
                row = i // a
                x0r[3 * i] = (col + 0.5) * w_refine / a
                x0r[3 * i + 1] = (row + 0.5) * h_refine / b
                x0r[3 * i + 2] = min(w_refine / a, h_refine / b) * 0.45
            
            x_rf, _ = sa_joint_optimize(x0r, w_refine, max_iters=800, seed=300)
            h_rf = 2.0 - w_refine
            check_and_update(x_rf, w_refine, h_rf)
            
            if time.time() < deadline - 8:
                x_pol = refine_with_slsqp(x_rf, w_refine, max_iter=120)
                check_and_update(x_pol, w_refine, h_rf)

    # Fallback if no valid result found
    if best_result is None:
        a, b = 7, 3
        w, h = 1.4, 0.6
        r = 0.1 - 1e-4
        best_result = np.zeros(3 * N)
        for i in range(a):
            for j in range(b):
                idx = i * b + j
                if idx >= N:
                    break
                best_result[3 * idx] = (i + 0.5) * w / a
                best_result[3 * idx + 1] = (j + 0.5) * h / b
                best_result[3 * idx + 2] = r
    
    # Format output: (x, y, r) for each circle
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = best_result[3 * i:3 * i + 3]
    
    return result
# EVOLVE-BLOCK-END
