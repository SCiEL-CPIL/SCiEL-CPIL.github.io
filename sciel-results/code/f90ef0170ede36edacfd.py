import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s out of 150s, safety margin

    best_result = None
    best_score = -np.inf

    # Pre-compute indices for upper triangle
    iu = np.triu_indices(N, k=1)
    
    def penalty_function(params, penalty_weight=1000.0):
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]
        
        # Boundary penalty
        # x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
        d_left = xy[:, 0] - r
        d_right = 1 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1 - xy[:, 1] - r
        boundary_viol = np.sum(np.maximum(0, -d_left)**2) + np.sum(np.maximum(0, -d_right)**2) + \
                        np.sum(np.maximum(0, -d_bottom)**2) + np.sum(np.maximum(0, -d_top)**2)

        # Overlap penalty
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        
        # Only consider upper triangle to avoid double counting
        overlap = np.maximum(0, min_dist[iu] - dist[iu])
        overlap_viol = np.sum(overlap**2)

        return -np.sum(r) + penalty_weight * boundary_viol + penalty_weight * overlap_viol

    def is_valid(result, tol=1e-8):
        n = N
        xy = result[:2*n].reshape(n, 2)
        r = result[2*n:]

        if np.any(r < -tol):
            return False
        if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
            return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
            return False

        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        if np.any(dist[iu] < min_dist[iu] - tol):
            return False

        return True

    def get_score(result):
        n = N
        r = result[2*n:]
        return np.sum(r)

    def push_apart(params, n_iters=200):
        """
        Vectorized collision resolution.
        Iteratively pushes overlapping circles apart and clamps to boundaries.
        """
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()
        
        # Precompute pairwise differences
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        # Avoid division by zero
        dist[dist < 1e-10] = 1e-10
        
        min_dist = r[:, None] + r[None, :]
        
        for _ in range(n_iters):
            # Recalculate distances
            dx = xy[:, 0][:, None] - xy[:, 0][None, :]
            dy = xy[:, 1][:, None] - xy[:, 1][None, :]
            dist = np.sqrt(dx**2 + dy**2)
            dist[dist < 1e-10] = 1e-10
            
            # Calculate overlap
            # overlap[i,j] = max(0, (r[i] + r[j]) - dist[i,j])
            overlap = np.maximum(0, min_dist - dist)
            
            # If no overlap, break
            if np.sum(overlap) < 1e-12:
                break
                
            # Calculate push vectors
            # direction[i,j] = (x_j - x_i) / dist[i,j]
            # We want to push i away from j and j away from i.
            # force on i from j is -overlap * dir_ij
            # force on j from i is +overlap * dir_ij
            
            # Unit vectors
            # ux[i,j] = dx[i,j] / dist[i,j]  (points from i to j)
            ux = dx / dist
            uy = dy / dist
            
            # Displacement for each circle
            # For circle i, it is pushed by all j where overlap[i,j] > 0
            # displacement_i = - sum_j overlap[i,j] * u_ij
            # Note: overlap is symmetric, u_ij = -u_ji
            
            # Sum of forces on x and y
            # We can use matrix multiplication or summing over axes
            # disp_x[i] = - sum_j overlap[i,j] * ux[i,j]
            disp_x = -np.sum(overlap * ux, axis=1)
            disp_y = -np.sum(overlap * uy, axis=1)
            
            # Apply damping to prevent oscillation
            damping = 0.5
            xy[:, 0] += disp_x * damping
            xy[:, 1] += disp_y * damping
            
            # Clamp to boundaries
            xy[:, 0] = np.clip(xy[:, 0], r, 1 - r)
            xy[:, 1] = np.clip(xy[:, 1], r, 1 - r)

        return np.concatenate([xy.flatten(), r])

    def refine_solution(params, max_iter=500, tol=1e-14, penalty_weight=1000.0):
        try:
            result = minimize(penalty_function, params, args=(penalty_weight,), method='L-BFGS-B',
                             options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-12, 'maxls': 20})
            return result.x
        except Exception:
            return params

    def get_initial_config(restart, rng):
        n = N
        side = int(np.ceil(np.sqrt(N))) # 6
        
        if restart % 10 == 0:
            # 6x6 Grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.001
        elif restart % 10 == 1:
            # Perturbed 6x6 Grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.001
            xy += rng.uniform(-0.02, 0.02, size=(n, 2))
        elif restart % 10 == 2:
            # Hexagonal Packing (6 cols, 6 rows, staggered)
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 6
                col = i % 6
                x = (col + 0.5)/6 + (0.25 if row % 2 == 1 else 0)
                y = (row + 0.5)/6
                xy[i] = [x, y]
                r[i] = 0.08
        elif restart % 10 == 3:
            # Random Uniform
            xy = rng.uniform(0.08, 0.92, size=(n, 2))
            r = rng.uniform(0.05, 0.1, size=n)
        elif restart % 10 == 4:
            # 8x4 Grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 8
                col = i % 8
                xy[i] = [(col + 0.5)/8, (row + 0.5)/4]
                r[i] = 0.5/8 - 0.001
            xy += rng.uniform(-0.01, 0.01, size=(n, 2))
        elif restart % 10 == 5:
            # 4x8 Grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 4
                col = i % 4
                xy[i] = [(col + 0.5)/4, (row + 0.5)/8]
                r[i] = 0.5/4 - 0.001
            xy += rng.uniform(-0.01, 0.01, size=(n, 2))
        elif restart % 10 == 6:
            # Hexagonal Variant (different stagger)
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 6
                col = i % 6
                x = (col + 0.5)/6 + (0.2 if row % 2 == 0 else 0)
                y = (row + 0.5)/6
                xy[i] = [x, y]
                r[i] = 0.075
        elif restart % 10 == 7:
            # Clustered: 4 groups of 8
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
            for i in range(n):
                group = i // 8
                cx, cy = centers[group]
                sub_i = i % 8
                # 3x3 grid within cluster (minus 1 to make 8)
                sx = (sub_i % 3) / 3.0 + 0.5/3
                sy = (sub_i // 3) / 3.0 + 0.5/3
                xy[i] = [cx + (sx - 0.5)*0.4, cy + (sy - 0.5)*0.4]
                r[i] = 0.06
        elif restart % 10 == 8:
            # Random Medium
            xy = rng.uniform(0.08, 0.92, size=(n, 2))
            r = rng.uniform(0.06, 0.09, size=n)
        else:
            # Random Small
            xy = rng.uniform(0.05, 0.95, size=(n, 2))
            r = rng.uniform(0.03, 0.06, size=n)

        params = np.concatenate([xy.flatten(), r])
        return params

    rng = np.random.RandomState(12345)

    # Phase 1: Many restarts with diverse initializations
    # Vectorized push_apart is fast, so we can afford more restarts
    n_restarts = 150
    for restart in range(n_restarts):
        if time.time() > deadline - 10: # Leave 10s for polish
            break

        params = get_initial_config(restart, rng)
        # Quick push to resolve gross overlaps
        params = push_apart(params, n_iters=100)

        if is_valid(params):
            score = get_score(params)
            if score > best_score:
                best_score = score
                best_result = params.copy()

        # Refine with L-BFGS-B
        # Use moderate maxiter for speed, high penalty weight
        refined = refine_solution(params, max_iter=300, tol=1e-12, penalty_weight=1000.0)

        if is_valid(refined):
            score = get_score(refined)
            if score > best_score:
                best_score = score
                best_result = refined.copy()
        else:
            # Try to repair
            repaired = push_apart(refined, n_iters=100)
            if is_valid(repaired):
                score = get_score(repaired)
                if score > best_score:
                    best_score = score
                    best_result = repaired.copy()
            else:
                # Shrink radii until valid
                xy = refined[:2*N].reshape(N, 2).copy()
                r = refined[2*N:].copy()
                for _ in range(100):
                    if is_valid(np.concatenate([xy.flatten(), r])):
                        break
                    r *= 0.95
                final_params = np.concatenate([xy.flatten(), r])
                if is_valid(final_params):
                    score = get_score(final_params)
                    if score > best_score:
                        best_score = score
                        best_result = final_params.copy()

    # Phase 2: High-precision polish on best solution
    if best_result is not None and time.time() < deadline - 2:
        # Tighten tolerance and increase iterations
        polished = refine_solution(best_result, max_iter=1000, tol=1e-15, penalty_weight=10000.0)
        if is_valid(polished):
            score = get_score(polished)
            if score > best_score:
                best_score = score
                best_result = polished.copy()

        # Try perturbations and re-refine
        for polish_iter in range(20):
            if time.time() > deadline - 1:
                break
            perturbed = best_result.copy()
            xy_p = perturbed[:2*N].reshape(N, 2).copy()
            r_p = perturbed[2*N:].copy()
            # Small perturbation
            xy_p += rng.uniform(-0.002, 0.002, size=(N, 2))
            perturbed = np.concatenate([xy_p.flatten(), r_p])
            perturbed = push_apart(perturbed, n_iters=50)
            polished2 = refine_solution(perturbed, max_iter=500, tol=1e-15, penalty_weight=10000.0)
            if is_valid(polished2):
                score = get_score(polished2)
                if score > best_score:
                    best_score = score
                    best_result = polished2.copy()

    if best_result is not None:
        n = N
        xy = best_result[:2*n].reshape(n, 2).copy()
        r = best_result[2*n:].copy()
        r = np.clip(r, 0, 0.5)
        xy[:, 0] = np.clip(xy[:, 0], r, 1-r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1-r)
        final_params = np.concatenate([xy.flatten(), r])
        if is_valid(final_params):
            return np.column_stack([xy, r])

    # Fallback: Grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
