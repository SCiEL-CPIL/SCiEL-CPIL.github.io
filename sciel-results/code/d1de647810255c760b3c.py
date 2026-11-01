"""Improved packer: dense distance matrix + fast simulated annealing + soft-min polish."""
import numpy as np
import time

N = 300
SEED = 12345
T_START = time.time()
DEADLINE = 110.0  # safety margin before the 120 s wall-clock limit


def _init_dist(points):
    """Compute initial pairwise distance matrix."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    return dists


def _update_dist_row(dist, i, new_pt, points):
    """Update row and column i of the distance matrix."""
    # dist[i, :] = norm(points[i] - points)
    # dist[:, i] = norm(points - points[i])
    diff = points - new_pt
    dists = np.sqrt(np.sum(diff * diff, axis=1))
    dist[i, :] = dists
    dist[:, i] = dists


def _score_fast(points, dist):
    """Return the feasible common radius (0.0 if invalid)."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # distance to container boundary
    norm = np.linalg.norm(points, axis=1)
    r_bound = 1.0 - np.max(norm)
    if r_bound < 0.0:
        return 0.0
    # minimum pairwise half-distance
    # Note: dist has 0 on diagonal
    r_pair = 0.5 * np.min(dist)
    r = min(r_bound, r_pair)
    return float(r) if r > 0.0 else 0.0


def _soft_min_radius(points, dist, beta=50.0):
    """
    Smooth surrogate for the radius objective using log-sum-exp soft-min.
    Returns a smooth value that approximates min(r_bound, r_pair) but
    gives gradient signal from ALL constraints, not just the bottleneck.
    """
    norm = np.linalg.norm(points, axis=1)
    r_bound = 1.0 - norm  # shape (N,)
    
    # Pairwise half-distances, exclude diagonal
    # dist[i,j] for i != j
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    r_pair = 0.5 * d  # shape (N, N)
    
    # Soft-min over all boundary margins
    # log-sum-exp trick: soft_min(x) = -log(sum(exp(-beta*x))) / beta
    # For stability, subtract min first
    rb_min = np.min(r_bound)
    rb_exp = np.exp(-beta * (r_bound - rb_min))
    soft_r_bound = rb_min - np.log(np.sum(rb_exp)) / beta
    
    # Soft-min over all pairwise half-distances (upper triangle to avoid double counting)
    iu = np.triu_indices(N, k=1)
    rp_vals = r_pair[iu]
    rp_min = np.min(rp_vals)
    rp_exp = np.exp(-beta * (rp_vals - rp_min))
    soft_r_pair = rp_min - np.log(np.sum(rp_exp)) / beta
    
    # Combine: soft-min of the two
    m = min(soft_r_bound, soft_r_pair)
    exp_vals = np.exp(-beta * np.array([soft_r_bound - m, soft_r_pair - m]))
    soft_r = m - np.log(np.sum(exp_vals)) / beta
    
    return float(soft_r)


def _lattice_seed(spacing):
    """Generate a hexagonal lattice and pick N points inside the disk."""
    while True:
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        keep = rad <= 1.0 - spacing / 2.0
        pts = pts[keep]
        if len(pts) >= N:
            # prefer inner points (smaller radius) to maximise margin
            order = np.argsort(np.hypot(pts[:, 0], pts[:, 1]), kind="stable")
            return pts[order[:N]].copy()
        spacing *= 0.97


def _anneal(points, dist, t_budget, rng):
    """Simulated annealing with dense distance matrix updates, using soft-min for acceptance."""
    start = time.time()
    best = points.copy()
    best_dist = dist.copy()
    best_r = _score_fast(best, best_dist)
    cur = points.copy()
    cur_dist = dist.copy()
    cur_r = best_r
    cur_soft = _soft_min_radius(cur, cur_dist, beta=50.0)

    # initial temperature from typical pair distance
    d0 = np.min(cur_dist)
    T0 = 0.02 * d0
    T_min = 1e-5 * d0
    step0 = 0.05 * d0
    step_min = 1e-4 * d0

    # Estimate total iterations for linear cooling
    # Rough estimate: ~2000 iterations per second
    est_iters = int(t_budget * 2000)
    est_iters = max(est_iters, 1000)

    for it in range(est_iters):
        if time.time() - start >= t_budget:
            break

        # Linear temperature schedule
        frac = it / est_iters
        T = T0 * (1.0 - frac) + T_min * frac
        step = step0 * (1.0 - frac) + step_min * frac

        # propose a random move on a random point
        idx = rng.integers(N)
        ang = rng.random() * 2.0 * np.pi
        mag = rng.random() * step
        dx = mag * np.cos(ang)
        dy = mag * np.sin(ang)

        new_pt = cur[idx].copy()
        new_pt[0] += dx
        new_pt[1] += dy

        # reject if it leaves the container (with margin)
        if np.hypot(new_pt[0], new_pt[1]) > 1.0 - 1e-9:
            continue

        old_pt = cur[idx].copy()
        old_row = cur_dist[idx].copy()
        cur[idx] = new_pt
        _update_dist_row(cur_dist, idx, new_pt, cur)
        new_r = _score_fast(cur, cur_dist)
        new_soft = _soft_min_radius(cur, cur_dist, beta=50.0)

        # Accept if hard score improves, or with probability based on soft-min change
        if new_r >= cur_r:
            accept = True
        else:
            delta_soft = new_soft - cur_soft
            if delta_soft < 0:
                # Soft-min decreased, but maybe hard score is same or slightly worse
                # Use soft-min for acceptance when hard scores are close
                if new_r == cur_r:
                    accept = rng.random() < np.exp(delta_soft / max(T, 1e-12))
                else:
                    accept = False
            else:
                accept = True

        if accept:
            cur_r = new_r
            cur_soft = new_soft
            if new_r > best_r:
                best_r = new_r
                best = cur.copy()
                best_dist = cur_dist.copy()
        else:
            cur[idx] = old_pt
            cur_dist[idx, :] = old_row
            cur_dist[:, idx] = old_row

    return best, best_dist, best_r


def _polish(points, dist, t_budget, rng):
    """
    High-precision polish pass using sub-gradient steps on the soft-min objective.
    This is a deterministic local optimization that exploits the smooth surrogate.
    """
    start = time.time()
    best = points.copy()
    best_dist = dist.copy()
    best_r = _score_fast(best, best_dist)
    cur = points.copy()
    cur_dist = dist.copy()
    cur_r = best_r

    d0 = np.min(cur_dist)
    base_step = 0.005 * d0

    # Sub-gradient polish: move each point in the direction that increases soft-min radius
    beta = 50.0
    n_iter = 0

    while time.time() - start < t_budget:
        # Compute sub-gradient of soft-min w.r.t. each point
        norm = np.linalg.norm(cur, axis=1)
        r_bound = 1.0 - norm  # (N,)
        
        d = cur_dist.copy()
        np.fill_diagonal(d, np.inf)
        r_pair = 0.5 * d  # (N, N)
        
        # Soft-min weights
        rb_min = np.min(r_bound)
        rb_w = np.exp(-beta * (r_bound - rb_min))
        rb_w /= np.sum(rb_w)
        
        iu = np.triu_indices(N, k=1)
        rp_vals = r_pair[iu]
        rp_min = np.min(rp_vals)
        rp_w = np.exp(-beta * (rp_vals - rp_min))
        rp_w /= np.sum(rp_w)
        
        # Combine
        soft_rb = rb_min - np.log(np.sum(np.exp(-beta * (r_bound - rb_min)))) / beta
        soft_rp = rp_min - np.log(np.sum(np.exp(-beta * (rp_vals - rp_min)))) / beta
        m = min(soft_rb, soft_rp)
        w_rb = np.exp(-beta * (soft_rb - m))
        w_rp = np.exp(-beta * (soft_rp - m))
        w_tot = w_rb + w_rp
        w_rb /= w_tot
        w_rp /= w_tot
        
        # Gradient of soft-min w.r.t. point positions
        # d(soft_r_bound)/dp_i = w_rb * rb_w_i * (-p_i / |p_i|)  (for |p_i| > 0)
        # d(soft_r_pair)/dp_i = w_rp * sum_j rp_w_{ij} * (p_i - p_j) / |p_i - p_j|
        
        grad = np.zeros_like(cur)
        
        # Boundary contribution
        for i in range(N):
            if norm[i] > 1e-10:
                grad[i] -= w_rb * rb_w[i] * cur[i] / norm[i]
        
        # Pairwise contribution
        # For each pair (i,j) with i<j, weight rp_w for that pair
        # Contribution to grad[i]: +w_rp * rp_w_{ij} * (p_i - p_j) / |p_i - p_j|
        # Contribution to grad[j]: -w_rp * rp_w_{ij} * (p_i - p_j) / |p_i - p_j|
        
        # Vectorized: for each i, sum over j != i of w_rp * rp_w_{ij} * (p_i - p_j) / dist_ij
        # This is O(N^2) but we can do it in a loop over i with vectorized inner ops
        
        # Build the weighted pairwise gradient
        # For point i: grad_pair[i] = w_rp * sum_j (rp_w_{ij} * (p_i - p_j) / dist_ij)
        # where rp_w_{ij} is the normalized soft-min weight for pair (i,j)
        
        # First build the full rp_w matrix (symmetric)
        rp_w_full = np.zeros((N, N))
        for k in range(len(iu[0])):
            i, j = iu[0][k], iu[1][k]
            rp_w_full[i, j] = rp_w[k]
            rp_w_full[j, i] = rp_w[k]
        
        # For each point i, compute sum_j rp_w_full[i,j] * (p_i - p_j) / dist_ij
        for i in range(N):
            diff = cur[i] - cur  # (N, 2)
            dists = np.linalg.norm(diff, axis=1)  # (N,)
            # Avoid division by zero
            valid = dists > 1e-10
            if np.any(valid):
                # Weighted average of unit vectors from i to j
                unit = np.zeros_like(diff)
                unit[valid] = diff[valid] / dists[valid, None]
                grad[i] += w_rp * np.sum(rp_w_full[i, valid, None] * unit[valid], axis=0)
        
        # Now do a line search / step
        # Try moving each point in the gradient direction with a small step
        step = base_step * (1.0 - 0.5 * (n_iter % 100) / 100.0)
        
        # Move all points simultaneously (or one at a time for safety)
        # One at a time is safer for constraint satisfaction
        improved = False
        for i in range(N):
            if time.time() - start >= t_budget:
                break
            
            old_pt = cur[i].copy()
            old_row = cur_dist[i].copy()
            
            # Move in gradient direction
            move = step * grad[i]
            # Normalize the move to have magnitude at most step
            move_norm = np.linalg.norm(move)
            if move_norm > 1e-15:
                move = move / move_norm * min(move_norm, step)
            
            new_pt = old_pt + move
            
            # Check container constraint
            if np.hypot(new_pt[0], new_pt[1]) > 1.0 - 1e-9:
                # Project back
                r = np.linalg.norm(new_pt)
                new_pt = new_pt / r * (1.0 - 1e-6)
            
            cur[i] = new_pt
            _update_dist_row(cur_dist, i, new_pt, cur)
            new_r = _score_fast(cur, cur_dist)
            
            if new_r >= cur_r:
                cur_r = new_r
                improved = True
                if new_r > best_r:
                    best_r = new_r
                    best = cur.copy()
                    best_dist = cur_dist.copy()
            else:
                # Revert
                cur[i] = old_pt
                cur_dist[i, :] = old_row
                cur_dist[:, i] = old_row
        
        n_iter += 1
        
        # Reduce step size if no improvement
        if not improved:
            base_step *= 0.95
            if base_step < 1e-8 * d0:
                break

    return best, best_dist, best_r


def construct():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # initial hex-lattice seed
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    pts = _lattice_seed(spacing)
    dist = _init_dist(pts)
    best = pts.copy()
    best_dist = dist.copy()
    best_r = _score_fast(best, best_dist)

    # Phase 1: Multi-start annealing within the time budget
    # Reserve 25s for polishing
    polish_reserve = 25.0
    while time.time() - t0 < DEADLINE - polish_reserve:
        # small perturbation of the seed for diversity
        jitter = 0.01 * rng.standard_normal((N, 2))
        cand = best + jitter
        # clip inside container
        rad = np.linalg.norm(cand, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            cand[mask] = cand[mask] / rad[mask, None] * (1.0 - 1e-6)
        
        cand_dist = _init_dist(cand)

        t_left = DEADLINE - polish_reserve - (time.time() - t0)
        if t_left < 5.0:
            break
        t_alloc = min(t_left - 2.0, 15.0)  # cap per-anneal budget
        cand, cand_dist, r = _anneal(cand, cand_dist, t_alloc, rng)
        if r > best_r:
            best_r = r
            best = cand
            best_dist = cand_dist

    # Phase 2: High-precision polish pass with sub-gradient optimization
    t_left = DEADLINE - (time.time() - t0)
    if t_left > 5.0:
        best, best_dist, best_r = _polish(best, best_dist, t_left - 2.0, rng)

    # final safety: ensure valid output
    final_r = _score_fast(best, best_dist)
    if final_r <= 0.0:
        # fall back to a guaranteed-valid lattice
        best = _lattice_seed(2.0 / (np.sqrt(N) + 1.0))

    return best
