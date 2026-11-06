"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _score(pts):
    """Largest common radius for a set of centres."""
    if pts is None or len(pts) == 0:
        return 0.0
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    wall = np.minimum(
        np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
        np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5),
    )
    wall_min = wall.min()
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min()
    return min(wall_min, 0.5 * nn)


def _make_hex(rows, spacing, dy, xoff):
    """Build a hex lattice with the given row lengths."""
    pts = []
    h = spacing * np.sqrt(3.0) / 2.0
    for i, n in enumerate(rows):
        y = -dy + i * h
        base = (0.5 if (n % 2 == 1) else 0.0) * spacing
        xs = np.arange(n) * spacing + base
        xs = xs - (xs.max() + xs.min()) / 2.0 + xoff
        for x in xs:
            pts.append([x, y])
    return np.array(pts)


def _soft_min_neg(pts, t):
    """
    Smooth surrogate for the negative min-distance objective.
    Uses log-sum-exp of squared (2t - d) terms for pairs where 2t > d,
    plus wall terms. We want to maximize t such that all distances >= 2t
    and walls >= t.
    
    We define a smooth penalty that is 0 when feasible and grows when not.
    Using soft-min: f(t) = t - softmin(distances)
    where softmin is a smooth approximation of min.
    """
    n = len(pts)
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    
    # Wall distances
    wall = np.minimum(
        np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
        np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5),
    )
    
    # All constraints: pairwise distances/2 and wall distances
    # We want min of all these to be >= t
    # Smooth: use log-sum-exp of exp(-beta * (constraint - t))
    # If constraint > t, exp(-beta*(constraint-t)) is small
    # If constraint < t, it's large -> penalty
    
    beta = 50.0
    # Pairwise: d/2
    half_d = d / 2.0
    # Only consider finite entries
    mask = np.isfinite(half_d)
    half_d_flat = half_d[mask]
    wall_flat = wall
    
    # Combine all constraints
    all_constraints = np.concatenate([half_d_flat, wall_flat])
    
    # softmin approximation: -log(sum(exp(-beta * c))) / beta
    # = t - softmin_penalty
    shifted = all_constraints - t
    # Clip to avoid overflow
    shifted = np.clip(shifted, -50.0, 50.0)
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    softmin_val = t - log_sum / beta
    
    # We want to maximize softmin_val, i.e., minimize -softmin_val
    return -softmin_val


def _neg_score(params, rows):
    spacing, dy, xoff = params
    if spacing <= 0:
        return 1.0
    pts = _make_hex(rows, spacing, dy, xoff)
    return -_score(pts)


def _full_optimize(pts, max_iter=200, deadline=None):
    """
    Optimize free positions to maximize the minimum distance.
    Uses smooth surrogate objective with log-sum-exp.
    """
    pts = pts.copy()
    n = len(pts)
    
    # We'll use scipy's minimize on the flattened position vector
    x0 = pts.flatten()
    
    def objective(x):
        p = x.reshape(n, 2)
        # Smooth penalty for violations
        diff = p[:, None, :] - p[None, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(d, np.inf)
        
        wall = np.minimum(
            np.minimum(0.5 - p[:, 0], p[:, 0] + 0.5),
            np.minimum(0.5 - p[:, 1], p[:, 1] + 0.5),
        )
        
        half_d = d / 2.0
        mask = np.isfinite(half_d)
        half_d_flat = half_d[mask]
        all_c = np.concatenate([half_d_flat, wall])
        
        # softmin: maximize this
        # Use log-sum-exp trick for numerical stability
        beta = 30.0
        min_c = np.min(all_c)
        # softmin = -log(sum(exp(-beta * (c - min_c)))) / beta + min_c
        shifted = all_c - min_c
        shifted = np.clip(shifted, 0, 50.0)
        log_sum = np.log(np.sum(np.exp(-beta * shifted)))
        softmin_val = min_c - log_sum / beta
        
        # Also add a small barrier to keep points in bounds
        barrier = 0.0
        for i in range(n):
            for k in range(2):
                if p[i][k] < -0.49:
                    barrier += 100.0 * (-0.49 - p[i][k])**2
                elif p[i][k] > 0.49:
                    barrier += 100.0 * (p[i][k] - 0.49)**2
        
        return -softmin_val + barrier
    
    def jac(x):
        # Numerical gradient
        eps = 1e-7
        g = np.zeros_like(x)
        for i in range(len(x)):
            xp = x.copy()
            xm = x.copy()
            xp[i] += eps
            xm[i] -= eps
            g[i] = (objective(xp) - objective(xm)) / (2 * eps)
        return g
    
    best_x = x0.copy()
    best_f = objective(x0)
    
    for restart in range(3):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        if restart == 0:
            x_init = x0
        else:
            x_init = x0 + np.random.normal(0, 0.005, len(x0))
            x_init = np.clip(x_init, -0.49, 0.49)
        
        try:
            res = minimize(objective, x_init, method="L-BFGS-B",
                           jac=jac,
                           options={"maxiter": max_iter, "gtol": 1e-10,
                                    "ftol": 1e-14, "maxls": 50})
            f = res.fun
            if f < best_f:
                best_f = f
                best_x = res.x
        except Exception:
            pass
    
    return best_x.reshape(n, 2)


def _refine_free(pts, deadline=None):
    """
    Refine positions freely (not constrained to hex lattice) to maximize min distance.
    Multiple restarts with different smoothing parameters.
    """
    best_pts = pts.copy()
    best_score = _score(pts)
    
    betas = [20.0, 30.0, 50.0, 80.0]
    for beta in betas:
        if deadline is not None and time.time() > deadline - 5.0:
            break
        # Optimize with this beta
        res_pts = _full_optimize(best_pts, max_iter=150, deadline=deadline)
        s = _score(res_pts)
        if s > best_score:
            best_score = s
            best_pts = res_pts
    
    return best_pts, best_score


def _hill_climb(pts, max_iter=400, step_size=0.001, deadline=None):
    """Local hill-climbing: perturb one circle at a time to improve min distance."""
    pts = pts.copy()
    current_score = _score(pts)
    no_improve = 0
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 3.0:
            break
        
        idx = np.random.randint(len(pts))
        orig = pts[idx].copy()
        improved = False
        
        # Adaptive step size
        for attempt in range(30):
            ss = step_size * (1.0 if attempt < 20 else 0.5)
            new_pos = orig + np.random.normal(0, ss, 2)
            if np.any(new_pos < -0.5) or np.any(new_pos > 0.5):
                continue
            pts[idx] = new_pos
            new_score = _score(pts)
            if new_score > current_score + 1e-10:
                current_score = new_score
                improved = True
                no_improve = 0
                break
            else:
                pts[idx] = orig
        
        if not improved:
            no_improve += 1
            if no_improve > 50:
                step_size *= 0.85
                no_improve = 0
                if step_size < 1e-5:
                    break
        else:
            # Occasionally increase step
            if it % 200 == 0:
                step_size *= 1.1
    
    return pts, current_score


def _soft_repulsion_force(pts, r_target, k_rep=100.0, k_wall=50.0):
    """Compute repulsion forces between circles and wall forces."""
    n = len(pts)
    forces = np.zeros_like(pts)
    
    for i in range(n):
        for j in range(i + 1, n):
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                forces[i] += np.random.normal(0, 1e-4, 2)
                forces[j] -= forces[i]
                continue
            if dist < 2 * r_target:
                overlap = 2 * r_target - dist
                push = k_rep * overlap / dist
                forces[i] += push * d
                forces[j] -= push * d
            elif dist < 2.5 * r_target:
                overlap = 2.5 * r_target - dist
                push = k_rep * 0.1 * overlap / dist
                forces[i] += push * d
                forces[j] -= push * d
    
    for i in range(n):
        for k in range(2):
            if pts[i][k] < -0.5 + r_target:
                forces[i][k] += k_wall * (-0.5 + r_target - pts[i][k])
            elif pts[i][k] > 0.5 - r_target:
                forces[i][k] -= k_wall * (pts[i][k] - (0.5 - r_target))
    
    return forces


def _relax(pts, r_target, max_iter=1500, dt=0.001, deadline=None):
    """Relax positions to eliminate overlaps at given radius."""
    pts = pts.copy()
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 3.0:
            break
        forces = _soft_repulsion_force(pts, r_target)
        pts += dt * forces
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
        
        max_overlap = 0.0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dist = np.linalg.norm(pts[i] - pts[j])
                if dist < 2 * r_target:
                    max_overlap = max(max_overlap, 2 * r_target - dist)
            for k in range(2):
                if pts[i][k] < -0.5 + r_target:
                    max_overlap = max(max_overlap, -0.5 + r_target - pts[i][k])
                elif pts[i][k] > 0.5 - r_target:
                    max_overlap = max(max_overlap, pts[i][k] - (0.5 - r_target))
        
        if max_overlap < 1e-7:
            break
    
    return pts


def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin under 150s

    best_pts = None
    best_score = -1.0

    # Candidate row-length patterns that sum to 50.
    patterns = [
        [7, 6, 7, 6, 7, 6, 7, 4],
        [6, 7, 6, 7, 6, 7, 6, 5],
        [8, 6, 7, 6, 7, 6, 7, 3],
        [7, 7, 6, 7, 6, 7, 6, 4],
        [5, 7, 6, 7, 6, 7, 6, 6],
        [6, 6, 7, 6, 7, 6, 7, 5],
        [7, 6, 6, 7, 6, 7, 6, 5],
        [6, 7, 7, 6, 6, 7, 6, 5],
    ]

    for rows in patterns:
        if sum(rows) != N:
            continue
        if time.time() > deadline - 15.0:
            break

        max_n = max(rows)
        init = [1.0 / max_n, 0.0, 0.0]

        def obj(p, rows=rows):
            return _neg_score(p, rows)

        res = minimize(obj, init, method="Nelder-Mead",
                       options={"maxiter": 1500, "xatol": 1e-8, "fatol": 1e-10,
                                "adaptive": True})
        if time.time() > deadline - 15.0:
            break

        s = -res.fun
        if s > best_score:
            best_score = s
            best_pts = _make_hex(rows, *res.x)

        # Local polish with L-BFGS-B
        for xoff0 in (-0.02, 0.0, 0.02):
            for dy0 in (-0.01, 0.0, 0.01):
                if time.time() > deadline - 15.0:
                    break
                init2 = [res.x[0], dy0, xoff0]
                try:
                    res2 = minimize(obj, init2, method="L-BFGS-B",
                                    options={"maxiter": 400, "gtol": 1e-9})
                    s2 = -res2.fun
                    if s2 > best_score:
                        best_score = s2
                        best_pts = _make_hex(rows, *res2.x)
                except Exception:
                    pass

    # Free-position refinement with smooth surrogate
    if best_pts is not None and time.time() < deadline - 25.0:
        refined_pts, refined_score = _refine_free(best_pts, deadline=deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined_pts

    # Physics-based radius growth
    if best_pts is not None and time.time() < deadline - 20.0:
        r = best_score
        pts = best_pts.copy()
        
        for grow_iter in range(40):
            if time.time() > deadline - 15.0:
                break
            r_new = r * 1.002 + 1e-8
            pts_relaxed = _relax(pts, r_new, max_iter=800, dt=0.0005, deadline=deadline)
            s_new = _score(pts_relaxed)
            if s_new >= r_new - 1e-8:
                r = r_new
                pts = pts_relaxed
                if r > best_score:
                    best_score = r
                    best_pts = pts.copy()
            else:
                r_new = r * 1.001 + 1e-9
                pts_relaxed = _relax(pts, r_new, max_iter=400, dt=0.0005, deadline=deadline)
                s_new = _score(pts_relaxed)
                if s_new >= r_new - 1e-9:
                    r = r_new
                    pts = pts_relaxed
                    if r > best_score:
                        best_score = r
                        best_pts = pts.copy()
                else:
                    break

    # Final hill-climb polish
    if best_pts is not None and time.time() < deadline - 8.0:
        pts_polished, s_polished = _hill_climb(best_pts, max_iter=500, step_size=0.001, deadline=deadline)
        if s_polished > best_score:
            best_score = s_polished
            best_pts = pts_polished

    # One more round of smooth optimization on the best result
    if best_pts is not None and time.time() < deadline - 5.0:
        final_pts, final_score = _refine_free(best_pts, deadline=deadline)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts

    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    return best_pts
# EVOLVE-BLOCK-END
