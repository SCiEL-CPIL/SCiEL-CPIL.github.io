import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

N = 50
DEADLINE = 145.0  # Safety margin before 150s limit


def _true_score(centers):
    """Compute the true packing radius for a given configuration."""
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5):
        return 0.0
    boundary = (0.5 - np.abs(centers)).min()
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = (diffs ** 2).sum(axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_dist = np.sqrt(dists_sq.min())
    return min(boundary, min_dist / 2.0)


def _init_grid():
    """Grid initialization trying multiple configurations."""
    best_pts = None
    best_r = -1

    for rows, cols in [(7, 8), (8, 7), (5, 10), (10, 5), (6, 9), (9, 6), (7, 7)]:
        total = rows * cols
        if total < N:
            continue

        xs = (np.arange(cols) + 0.5) / cols - 0.5
        ys = (np.arange(rows) + 0.5) / rows - 0.5
        grid = np.array([(x, y) for y in ys for x in xs])

        if total == N:
            pts = grid
        else:
            dists_to_center = np.sqrt((grid ** 2).sum(axis=1))
            indices = np.argsort(dists_to_center)[:N]
            pts = grid[indices]

        r = _true_score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    return best_pts if best_pts is not None else (np.random.rand(N, 2) - 0.5) * 0.9


def _init_hexagonal():
    """Hexagonal initialization with proper spacing."""
    best_pts = None
    best_r = -1

    for s in [0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]:
        pts = []
        row_height = s * np.sqrt(3) / 2
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = j * s + (i % 2) * s / 2
                y = i * row_height
                if abs(x) <= 0.5 and abs(y) <= 0.5:
                    pts.append([x, y])

        pts = np.array(pts)
        if len(pts) >= N:
            # Select N points that maximize minimum distance
            selected = []
            remaining = pts.copy()
            for _ in range(N):
                if len(remaining) == 0:
                    break
                if len(selected) == 0:
                    idx = 0
                else:
                    sel = np.array(selected)
                    dists = np.min(np.linalg.norm(remaining[:, np.newaxis, :] - sel[np.newaxis, :, :], axis=2), axis=1)
                    idx = np.argmax(dists)
                selected.append(remaining[idx])
                remaining = np.delete(remaining, idx, axis=0)

            if len(selected) == N:
                pts_sel = np.array(selected)
                r = _true_score(pts_sel)
                if r > best_r:
                    best_r = r
                    best_pts = pts_sel.copy()

    if best_pts is not None:
        return best_pts
    return _init_grid()


def _force_relax(centers, n_steps=300, lr_init=0.01, lr_decay=0.999):
    """Force-directed relaxation to resolve overlaps and keep in bounds."""
    centers = centers.copy()
    n = len(centers)
    lr = lr_init

    for step in range(n_steps):
        forces = np.zeros_like(centers)

        for k in range(2):
            mask_r = centers[:, k] > 0.45
            forces[mask_r, k] -= (centers[mask_r, k] - 0.45) * 20
            mask_l = centers[:, k] < -0.45
            forces[mask_l, k] += ((-0.45) - centers[mask_l, k]) * 20

        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)

        threshold = 0.12
        mask = dists < threshold
        np.fill_diagonal(mask, False)

        force_mag = np.zeros_like(dists)
        force_mag[mask] = (threshold - dists[mask]) / (dists[mask] + 1e-10) ** 2
        force_mag = np.minimum(force_mag, 50)

        forces += (force_mag[:, :, np.newaxis] * diffs).sum(axis=1)

        centers += forces * lr
        centers = np.clip(centers, -0.5, 0.5)

        lr *= lr_decay

    return centers


def _delaunay_sparse_objective(x, beta=100.0):
    """
    Sparse soft-min objective using Delaunay triangulation to identify
    critical (close) pairs. This focuses optimization on active constraints.
    """
    centers = x.reshape(-1, 2)
    boundary_dists = 0.5 - np.abs(centers)
    
    # Get all pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = (diffs ** 2).sum(axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Use Delaunay to find sparse neighbor set
    try:
        tri = Delaunay(centers)
        # Get unique edges from triangulation
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a,b), max(a,b)))
        # Also add a few more neighbors for robustness
        # K-nearest neighbors with k=6
        for i in range(N):
            d_i = dists[i].copy()
            d_i[i] = np.inf
            nn = np.argsort(d_i)[:6]
            for j in nn:
                edges.add((min(i,j), max(i,j)))
    except Exception:
        edges = set()
        for i in range(N):
            for j in range(i+1, N):
                edges.add((i, j))
    
    # Build sparse constraint set
    pair_dists = []
    for (i, j) in edges:
        pair_dists.append(dists[i, j] / 2.0)
    pair_dists = np.array(pair_dists) if pair_dists else np.array([1.0])
    
    all_dists = np.concatenate([boundary_dists.ravel(), pair_dists])
    
    # Soft min
    v_min = all_dists.min()
    shifted = all_dists - v_min
    shifted = np.clip(shifted, 0, 100.0 / beta)
    log_sum = np.log(np.sum(np.exp(-beta * shifted))) / beta
    soft_min = v_min - log_sum
    
    return -soft_min


def _slsqp_epigraph_polish(centers, time_budget=30.0):
    """
    SLSQP epigraph formulation: maximize r subject to
    - pairwise distances >= 2r
    - wall distances >= r
    
    Uses Delaunay-based sparse constraints for efficiency.
    """
    c = centers.copy()
    r_init = _true_score(c)
    
    # Build sparse constraint set using Delaunay
    try:
        tri = Delaunay(c)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a,b), max(a,b)))
        # Add K-nearest neighbors
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        for i in range(N):
            nn = np.argsort(d[i])[:5]
            for j in nn:
                edges.add((min(i,j), max(i,j)))
    except Exception:
        edges = set()
        for i in range(N):
            for j in range(i+1, N):
                edges.add((i,j))
    
    edge_list = sorted(edges)
    
    def objective(x):
        return -x[-1]  # maximize r = x[-1]
    
    # Build constraint functions
    constraints = []
    for (i, j) in edge_list:
        def _pair_constraint(x, i=i, j=j):
            ci = x[2*i:2*i+2]
            cj = x[2*j:2*j+2]
            r = x[-1]
            return np.sqrt(np.sum((ci - cj)**2)) - 2*r
        constraints.append({'type': 'ineq', 'fun': _pair_constraint})
    
    for i in range(N):
        for k in range(2):
            def _wall_constraint(x, i=i, k=k):
                c_ik = x[2*i + k]
                r = x[-1]
                return 0.5 - np.abs(c_ik) - r
            constraints.append({'type': 'ineq', 'fun': _wall_constraint})
    
    x0 = np.concatenate([c.ravel(), [r_init]])
    bounds = [(-0.5, 0.5)] * (2*N) + [(1e-6, 0.5)]
    
    best_x = x0.copy()
    best_r = r_init
    start_time = time.time()
    
    # Run SLSQP with progressive maxiter and tighter tolerances
    maxiters_list = [150, 300, 500, 800]
    ftols = [1e-10, 1e-11, 1e-12, 1e-12]
    for idx, maxiter in enumerate(maxiters_list):
        if (time.time() - start_time) > time_budget:
            break
        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': ftols[idx]}
            )
            r_new = res.x[-1]
            c_new = res.x[:2*N].reshape(N, 2)
            r_verified = _true_score(c_new)
            if r_verified > best_r:
                best_r = r_verified
                best_x = res.x.copy()
                x0 = res.x.copy()
        except Exception:
            pass
    
    return best_x[:2*N].reshape(N, 2), best_r


def _voronoi_active_set_refine(centers, max_iter=50, tol=1e-9):
    """
    Voronoi-style active-set refinement: identify kissing graph and
    nudge circles to increase local minimum radius.
    """
    c = centers.copy()
    r_cur = _true_score(c)
    if r_cur <= 0:
        return c, r_cur

    step_sizes = [0.002, 0.001, 0.0005, 0.0002]
    current_step = 0

    for it in range(max_iter):
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        wall_dist = 0.5 - np.abs(c)
        
        tol_d = max(1e-7, 1e-4 * r_cur)
        
        # Find active pairs (distance close to 2*r)
        active_pairs = []
        for i in range(N):
            for j in range(i+1, N):
                if d[i,j] < 2*r_cur + tol_d:
                    active_pairs.append((i,j))
        
        # Find active walls
        active_walls = []
        for i in range(N):
            for k in range(2):
                if wall_dist[i,k] < r_cur + tol_d:
                    active_walls.append((i,k))
        
        if len(active_pairs) == 0 and len(active_walls) == 0:
            break
        
        # Build displacement: push circles away from active constraints
        disp = np.zeros_like(c)
        
        for (i, j) in active_pairs:
            dij = c[i] - c[j]
            norm = np.linalg.norm(dij)
            if norm > 1e-12:
                direction = dij / norm
                # Push i away from j and j away from i
                disp[i] += direction * 0.5
                disp[j] -= direction * 0.5
        
        for (i, k) in active_walls:
            # Push circle i inward from wall k
            if k == 0:  # x-direction
                if c[i,0] > 0:
                    disp[i, 0] -= 0.5
                else:
                    disp[i, 0] += 0.5
            else:  # y-direction
                if c[i,1] > 0:
                    disp[i, 1] -= 0.5
                else:
                    disp[i, 1] += 0.5
        
        # Normalize and apply step
        disp_norm = np.linalg.norm(disp)
        if disp_norm > 1e-12:
            step_size = step_sizes[current_step]
            c += disp / disp_norm * step_size
            c = np.clip(c, -0.5, 0.5)
        
        new_r = _true_score(c)
        if new_r > r_cur + tol:
            r_cur = new_r
            # Reset step size counter on improvement
            current_step = 0
        else:
            # Try smaller step
            current_step = min(current_step + 1, len(step_sizes) - 1)
            if current_step == len(step_sizes) - 1 and new_r <= r_cur + tol:
                break
    
    return c, r_cur


def _local_refine(centers, time_budget=20.0):
    """Local refinement with small perturbations (hill climbing)."""
    start_time = time.time()
    best_centers = centers.copy()
    best_score = _true_score(best_centers)

    step_sizes = [0.001, 0.0005, 0.0002, 0.0001]

    for step in step_sizes:
        if time.time() - start_time > time_budget:
            break

        improved = True
        while improved and time.time() - start_time < time_budget:
            improved = False
            for i in range(N):
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        new_c = best_centers.copy()
                        new_c[i, 0] += dx
                        new_c[i, 1] += dy
                        if np.any(np.abs(new_c) > 0.5):
                            continue
                        s = _true_score(new_c)
                        if s > best_score + 1e-12:
                            best_centers = new_c
                            best_score = s
                            improved = True
            if time.time() - start_time > time_budget:
                break

    return best_centers, best_score


def _soft_min_radius(c, t):
    """Differentiable soft-min of all distances."""
    wall = 0.5 - np.abs(c)
    wall_flat = wall.ravel()
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_terms = d / 2.0
    pair_flat = pair_terms.ravel()
    all_terms = np.concatenate([wall_flat, pair_flat])
    m = all_terms.min()
    s = np.exp(-(all_terms - m) / t)
    soft = m - t * np.log(s.sum())
    return soft


def _negative_soft_radius(x, t):
    c = x.reshape(N, 2)
    return -_soft_min_radius(c, t)


def _optimize_lbfgsb(centers, time_budget=15.0):
    """Optimize using L-BFGS-B with soft-min surrogate and progressive beta."""
    start_time = time.time()
    x0 = centers.ravel()
    best_x = x0.copy()
    best_score = _true_score(centers)

    # Progressive beta schedule - tighter approximations to true min
    betas = [40, 80, 160, 320, 640]

    for beta in betas:
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 2:
            break

        def objective(x):
            return _negative_soft_radius(x, 1.0/beta)

        try:
            maxiter = 400 if beta < 100 else 600
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': maxiter, 'ftol': 1e-13, 'gtol': 1e-11}
            )
            new_centers = result.x.reshape(-1, 2)
            new_score = _true_score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_x = result.x.copy()
            x0 = result.x.copy()
        except Exception:
            pass

    # Final high-precision polish with very tight soft-min
    elapsed = time.time() - start_time
    if elapsed < time_budget - 3:
        try:
            def objective_sharp(x):
                return _negative_soft_radius(x, 0.0005)

            result = minimize(
                objective_sharp,
                x0,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            new_centers = result.x.reshape(-1, 2)
            new_score = _true_score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_x = result.x.copy()
        except Exception:
            pass

    return best_x.reshape(-1, 2), best_score


def construct():
    """Main entry point combining multiple optimization strategies."""
    start_time = time.time()
    np.random.seed(42)

    # 1. Generate multiple initial configurations
    inits = []
    for _ in range(2):
        init = _init_hexagonal()
        inits.append(init)
    inits.append(_init_grid())
    inits.append((np.random.rand(N, 2) - 0.5) * 0.85)

    best_centers = None
    best_score = -1

    # 2. For each init, do force relax + L-BFGS-B + SLSQP + active-set + local refine
    for init in inits:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 40:
            break

        # Force relax
        relaxed = _force_relax(init, n_steps=200)
        score_relaxed = _true_score(relaxed)
        if score_relaxed > best_score:
            best_score = score_relaxed
            best_centers = relaxed.copy()

        # L-BFGS-B optimization
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 25:
            t_lbfgs = min(remaining * 0.35, 35.0)
            centers_lbfgs, score_lbfgs = _optimize_lbfgsb(relaxed, time_budget=t_lbfgs)
            if score_lbfgs > best_score:
                best_score = score_lbfgs
                best_centers = centers_lbfgs.copy()

        # SLSQP epigraph polish (direct max-min optimization)
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 20 and best_centers is not None:
            t_slsqp = min(remaining * 0.35, 40.0)
            centers_slsqp, score_slsqp = _slsqp_epigraph_polish(best_centers, time_budget=t_slsqp)
            if score_slsqp > best_score:
                best_score = score_slsqp
                best_centers = centers_slsqp.copy()

        # Voronoi active-set refinement
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 10 and best_centers is not None:
            centers_vor, score_vor = _voronoi_active_set_refine(best_centers, max_iter=40)
            if score_vor > best_score:
                best_score = score_vor
                best_centers = centers_vor.copy()

        # Local refinement
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 8 and best_centers is not None:
            t_lr = min(remaining * 0.4, 15.0)
            centers_lr, score_lr = _local_refine(best_centers, time_budget=t_lr)
            if score_lr > best_score:
                best_score = score_lr
                best_centers = centers_lr.copy()

    # 3. Final SLSQP polish on best solution with more time
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 15 and best_centers is not None:
        t_final_slsqp = min(remaining * 0.6, 45.0)
        centers_final, score_final = _slsqp_epigraph_polish(best_centers, time_budget=t_final_slsqp)
        if score_final > best_score:
            best_score = score_final
            best_centers = centers_final.copy()

    # 4. Final L-BFGS-B polish with tight soft-min
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 10 and best_centers is not None:
        t_final_lbfgs = min(remaining * 0.4, 20.0)
        centers_final_lbfgs, score_final_lbfgs = _optimize_lbfgsb(best_centers, time_budget=t_final_lbfgs)
        if score_final_lbfgs > best_score:
            best_score = score_final_lbfgs
            best_centers = centers_final_lbfgs.copy()

    # 5. Final active-set + local refinement
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 8 and best_centers is not None:
        centers_vor2, score_vor2 = _voronoi_active_set_refine(best_centers, max_iter=50)
        if score_vor2 > best_score:
            best_score = score_vor2
            best_centers = centers_vor2.copy()

    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 5 and best_centers is not None:
        t_final_lr = min(remaining * 0.6, 10.0)
        centers_final_lr, score_final_lr = _local_refine(best_centers, time_budget=t_final_lr)
        if score_final_lr > best_score:
            best_score = score_final_lr
            best_centers = centers_final_lr.copy()

    # Ensure validity
    if best_centers is None:
        best_centers = _init_grid()
    best_centers = np.clip(best_centers, -0.5, 0.5)

    return best_centers


if __name__ == "__main__":
    t0 = time.time()
    c = construct()
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Score: {_true_score(c)}")
    print(f"Shape: {c.shape}")
