"""Circle packing in unit square n=32: maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit
    best_score = -np.inf
    best_result = None

    # --- Helper Functions ---

    def _make_hex_init():
        """Create hexagonal lattice initialization."""
        points = []
        for row in range(10):
            for col in range(10):
                x = col + (0.5 if row % 2 else 0)
                y = row * np.sqrt(3) / 2
                points.append((x, y))
        pts = np.array(points)
        center = pts.mean(axis=0)
        pts = pts - center
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:N]
        init_pts = pts[idx].copy()
        min_x, min_y = init_pts.min(axis=0)
        max_x, max_y = init_pts.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y
        scale = 0.9 / max(width, height)
        init_pts[:, 0] = (init_pts[:, 0] - min_x) * scale
        init_pts[:, 1] = (init_pts[:, 1] - min_y) * scale
        current_w = init_pts.max(axis=0) - init_pts.min(axis=0)
        offset_x = (1.0 - current_w[0]) / 2
        offset_y = (1.0 - current_w[1]) / 2
        init_pts[:, 0] += offset_x
        init_pts[:, 1] += offset_y
        return init_pts

    def _make_grid_init():
        """Create grid initialization."""
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    centers.append([0.5/side + i/side, 0.5/side + j/side])
        return np.array(centers[:N])

    def _make_random_init(seed):
        """Create random initialization."""
        rng = np.random.RandomState(seed)
        return rng.uniform(0.1, 0.9, size=(N, 2))

    def _compute_initial_radius(pts):
        """Compute initial radius based on minimum pairwise distance."""
        min_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist:
                    min_dist = d
        return min_dist / 2 * 0.85

    def _objective(vars_, penalty_weight=500.0):
        """Objective: minimize -sum(radii) + penalty for violations (vectorized)."""
        X = vars_[0::3]
        Y = vars_[1::3]
        R = vars_[2::3]
        obj = -np.sum(R)
        # Vectorized pairwise distances
        dx = X[:, None] - X[None, :]
        dy = Y[:, None] - Y[None, :]
        d = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(d, np.inf)
        overlap = R[:, None] + R[None, :] - d
        overlap = np.maximum(overlap, 0)
        pen = np.sum(overlap ** 2)
        # Boundary violations
        pen += np.sum(np.maximum(X - R, 0) ** 2)
        pen += np.sum(np.maximum(1.0 - X - R, 0) ** 2)
        pen += np.sum(np.maximum(Y - R, 0) ** 2)
        pen += np.sum(np.maximum(1.0 - Y - R, 0) ** 2)
        return obj + penalty_weight * pen

    def _objective_grad(vars_, penalty_weight=500.0):
        """Analytical gradient of the objective (vectorized)."""
        X = vars_[0::3]
        Y = vars_[1::3]
        R = vars_[2::3]
        grad = np.zeros_like(vars_)
        grad[2::3] = -1.0
        weight = penalty_weight
        # Vectorized pairwise
        dx = X[:, None] - X[None, :]
        dy = Y[:, None] - Y[None, :]
        d = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(d, np.inf)
        overlap = R[:, None] + R[None, :] - d
        mask = overlap > 0
        factor = 2.0 * overlap * mask
        # Gradient w.r.t. X
        gx = -weight * np.sum(factor * dx / d, axis=1)
        gy = -weight * np.sum(factor * dy / d, axis=1)
        gr = weight * np.sum(factor, axis=1)
        grad[0::3] += gx
        grad[1::3] += gy
        grad[2::3] += gr
        # Boundary gradients
        mask_x0 = X - R < 0
        grad[0::3][mask_x0] += weight * 2.0 * (X[mask_x0] - R[mask_x0])
        grad[2::3][mask_x0] += weight * (-2.0) * (X[mask_x0] - R[mask_x0])
        mask_x1 = 1.0 - X - R < 0
        grad[0::3][mask_x1] += weight * (-2.0) * (1.0 - X[mask_x1] - R[mask_x1])
        grad[2::3][mask_x1] += weight * (-2.0) * (1.0 - X[mask_x1] - R[mask_x1])
        mask_y0 = Y - R < 0
        grad[1::3][mask_y0] += weight * 2.0 * (Y[mask_y0] - R[mask_y0])
        grad[2::3][mask_y0] += weight * (-2.0) * (Y[mask_y0] - R[mask_y0])
        mask_y1 = 1.0 - Y - R < 0
        grad[1::3][mask_y1] += weight * (-2.0) * (1.0 - Y[mask_y1] - R[mask_y1])
        grad[2::3][mask_y1] += weight * (-2.0) * (1.0 - Y[mask_y1] - R[mask_y1])
        return grad

    def _evaluate(centers, radii):
        """Compute sum of radii if valid, else -inf."""
        for i in range(N):
            x, y = centers[i]
            r = radii[i]
            if x - r < -1e-9 or x + r > 1 + 1e-9:
                return -np.inf
            if y - r < -1e-9 or y + r > 1 + 1e-9:
                return -np.inf
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < radii[i] + radii[j] - 1e-9:
                    return -np.inf
        return np.sum(radii)

    def _polish(pos, rad, n_iters=20):
        """Iteratively recompute max radii for given positions (vectorized)."""
        rad = rad.copy()
        for _ in range(n_iters):
            # Compute all pairwise distances
            dx = pos[:, 0][:, None] - pos[:, 0][None, :]
            dy = pos[:, 1][:, None] - pos[:, 1][None, :]
            d = np.sqrt(dx * dx + dy * dy)
            np.fill_diagonal(d, np.inf)
            # For each i, min over j of (d[i,j] - rad[j])
            constraint_matrix = d - rad[None, :]
            min_others = np.min(constraint_matrix, axis=1)
            min_bound = np.minimum(np.minimum(pos[:, 0], 1 - pos[:, 0]),
                                    np.minimum(pos[:, 1], 1 - pos[:, 1]))
            new_rad = np.maximum(1e-6, np.minimum(min_others, min_bound) - 1e-6)
            if np.max(np.abs(new_rad - rad)) < 1e-10:
                rad = new_rad
                break
            rad = new_rad
        return rad

    def _validity_check(arr):
        """Check if circles are valid: inside [0,1]^2 and non-overlapping."""
        if arr.shape != (N, 3):
            return False
        x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
        if np.any(r <= 0):
            return False
        if np.any(x < -1e-9) or np.any(x > 1 + 1e-9):
            return False
        if np.any(y < -1e-9) or np.any(y > 1 + 1e-9):
            return False
        if np.any(x < r - 1e-9) or np.any(x > 1 - r + 1e-9):
            return False
        if np.any(y < r - 1e-9) or np.any(y > 1 - r + 1e-9):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if dist < r[i] + r[j] - 1e-7:
                    return False
        return True

    def _local_search_anneal(pos, rad, time_budget=2.0):
        """
        Simulated Annealing style local search with adaptive perturbation.
        """
        current_pos = pos.copy()
        current_rad = rad.copy()
        current_score = _evaluate(current_pos, current_rad)
        if np.isneginf(current_score):
             return current_pos, current_rad, current_score

        best_pos = current_pos.copy()
        best_rad = current_rad.copy()
        best_score = current_score

        rng = np.random.RandomState(int(time.time() * 1000) % 10000)
        
        # Cooling schedule
        T = 0.03
        T_min = 0.0005
        cooling_rate = 0.97
        max_steps = 3000
        
        steps_done = 0
        start_ls = time.time()
        no_improve_count = 0
        
        while steps_done < max_steps and (time.time() - start_ls) < time_budget:
            # Adaptive perturbation size based on current radii
            mean_r = np.mean(current_rad)
            sigma = 0.01 + 0.02 * mean_r
            
            # Perturb a random subset of circles
            n_perturb = rng.randint(1, 5)
            indices = rng.choice(N, n_perturb, replace=False)
            
            new_pos = current_pos.copy()
            # Gaussian perturbation with adaptive scale
            noise = rng.normal(0, sigma, size=(n_perturb, 2))
            new_pos[indices] += noise
            
            # Keep inside bounds with margin
            margin = 0.02
            new_pos[:, 0] = np.clip(new_pos[:, 0], margin, 1 - margin)
            new_pos[:, 1] = np.clip(new_pos[:, 1], margin, 1 - margin)
            
            # Recalculate radii for new positions
            new_rad = _polish(new_pos, current_rad.copy(), n_iters=8)
            
            new_score = _evaluate(new_pos, new_rad)
            
            if np.isneginf(new_score):
                T *= cooling_rate
                steps_done += 1
                no_improve_count += 1
                continue
            
            delta = new_score - current_score
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta > 0 or (T > T_min and rng.rand() < np.exp(delta / T)):
                current_pos = new_pos
                current_rad = new_rad
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_pos = current_pos.copy()
                    best_rad = current_rad.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            # Early termination if stuck for too long
            if no_improve_count > 500 and steps_done > 500:
                break
                
            T *= cooling_rate
            if T < T_min:
                T = T_min
            steps_done += 1
            
        return best_pos, best_rad, best_score

    # --- Main Execution Loop ---

    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((1e-5, 0.5))

    # Define initializations: hex, grid, and several random
    inits = []
    inits.append(('hex', _make_hex_init()))
    inits.append(('grid', _make_grid_init()))
    for seed in range(15): # Reduced random seeds to allow time for local search
        inits.append(('rand', _make_random_init(seed + 42)))

    for idx, (init_type, init_pts) in enumerate(inits):
        if time.time() > deadline - 10:
            break

        init_r = _compute_initial_radius(init_pts)

        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3 * i] = init_pts[i, 0]
            x0[3 * i + 1] = init_pts[i, 1]
            x0[3 * i + 2] = init_r

        # Phase 1: Global Optimization (L-BFGS-B)
        # Use fewer iterations for random inits to save time for local search
        maxiter = 200 if idx < 2 else 100
        
        try:
            res = minimize(_objective, x0, method='L-BFGS-B', jac=_objective_grad,
                           bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-8, 'gtol': 1e-6})
            x_opt = res.x
        except Exception:
            x_opt = x0

        X = x_opt[0::3].copy()
        Y = x_opt[1::3].copy()
        R = x_opt[2::3].copy()
        pos = np.column_stack((X, Y))

        # Initial Polish
        R = _polish(pos, R, n_iters=10)
        score = _evaluate(pos, R)
        
        # Phase 2: Local Search (Annealing) - The Structural Improvement
        # Only run local search if we have time and the current solution is valid
        if score > -np.inf and time.time() < deadline - 5:
            # Allocate time for local search: more for promising starts (hex/grid)
            ls_time = 4.0 if idx < 2 else 2.0
            pos_ls, R_ls, score_ls = _local_search_anneal(pos, R, time_budget=ls_time)
            
            if score_ls > score:
                pos = pos_ls
                R = R_ls
                score = score_ls

        # Final Polish on the locally searched result
        R = _polish(pos, R, n_iters=30)
        score = _evaluate(pos, R)

        if score > best_score:
            best_score = score
            best_result = np.column_stack([pos, R])

    # If no valid result found, return grid fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final high-precision polish on the best result
    if time.time() < deadline - 2:
        pos = best_result[:, :2].copy()
        rad = best_result[:, 2].copy()
        
        # One more tight L-BFGS-B pass with tight bounds and higher penalty
        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3 * i] = pos[i, 0]
            x0[3 * i + 1] = pos[i, 1]
            x0[3 * i + 2] = rad[i]
        
        tight_bounds = []
        for i in range(N):
            tight_bounds.append((max(0, pos[i,0] - 0.03), min(1, pos[i,0] + 0.03)))
            tight_bounds.append((max(0, pos[i,1] - 0.03), min(1, pos[i,1] + 0.03)))
            tight_bounds.append((max(1e-5, rad[i] * 0.9), min(0.5, rad[i] * 1.1)))
        
        try:
            # Use higher penalty weight for final polish to enforce constraints more strictly
            def _final_objective(vars_):
                return _objective(vars_, penalty_weight=2000.0)
            
            def _final_grad(vars_):
                return _objective_grad(vars_, penalty_weight=2000.0)
            
            res = minimize(_final_objective, x0, method='L-BFGS-B', jac=_final_grad,
                           bounds=tight_bounds, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            x_opt = res.x
            X = x_opt[0::3].copy()
            Y = x_opt[1::3].copy()
            R = x_opt[2::3].copy()
            pos_p = np.column_stack((X, Y))
            R = _polish(pos_p, R, n_iters=50)
            score_p = _evaluate(pos_p, R)
            if score_p > best_score:
                best_result = np.column_stack([pos_p, R])
                best_score = score_p
        except Exception:
            pass
        
        # Additional local search pass on the polished result
        if time.time() < deadline - 3:
            pos = best_result[:, :2].copy()
            rad = best_result[:, 2].copy()
            pos_ls, rad_ls, score_ls = _local_search_anneal(pos, rad, time_budget=2.0)
            if score_ls > best_score:
                best_result = np.column_stack([pos_ls, rad_ls])
                best_score = score_ls
            # Final polish
            rad_final = _polish(best_result[:, :2], best_result[:, 2], n_iters=50)
            score_final = _evaluate(best_result[:, :2], rad_final)
            if score_final > best_score:
                best_result = np.column_stack([best_result[:, :2], rad_final])
                best_score = score_final
    
    # Final validity check and fix if needed
    if not _validity_check(best_result):
        # Reduce radii until valid
        for _ in range(50):
            best_result[:, 2] *= 0.95
            if _validity_check(best_result):
                break
    
    # Ensure all radii are positive
    best_result[:, 2] = np.maximum(best_result[:, 2], 1e-6)
    
    return best_result
