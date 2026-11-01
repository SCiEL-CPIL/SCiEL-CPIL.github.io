import numpy as np
from scipy.optimize import minimize
import time

N = 14
D = 3
TIME_LIMIT = 145.0  # Safety margin before 150s limit

def _pairwise_sq_dists(pts):
    """Compute squared pairwise distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sum(diff**2, axis=-1)

def _true_score(pts):
    """Calculate the actual score: (max_dist / min_dist)^2."""
    sqd = _pairwise_sq_dists(pts)
    mask = ~np.eye(N, dtype=bool)
    d_sq = sqd[mask]
    mn = np.min(d_sq)
    mx = np.max(d_sq)
    if mn < 1e-20:
        return np.inf
    return mx / mn

def _normalize(pts):
    """Center and normalize points."""
    pts = pts - np.mean(pts, axis=0)
    norm_sq = np.sum(pts**2)
    if norm_sq < 1e-10:
        return pts
    pts = pts * np.sqrt(N / norm_sq)
    return pts

def _soft_log_ratio_sq(pts, tau):
    """
    Smooth approximation of log(max_d^2) - log(min_d^2) using soft min/max on squared distances.
    """
    sqd = _pairwise_sq_dists(pts)
    mask = ~np.eye(N, dtype=bool)
    d_sq = sqd[mask]
    
    # Stable soft max: tau * log(sum(exp(d_sq/tau)))
    max_d_sq = np.max(d_sq)
    soft_max = max_d_sq + tau * np.log(np.sum(np.exp((d_sq - max_d_sq) / tau)))
    
    # Stable soft min: -tau * log(sum(exp(-d_sq/tau)))
    min_d_sq = np.min(d_sq)
    soft_min = min_d_sq - tau * np.log(np.sum(np.exp((min_d_sq - d_sq) / tau)))
    
    return soft_max - soft_min

def _objective_coords(x, tau):
    """
    Objective in terms of coordinates. x is flat array of size N*D.
    """
    pts = x.reshape(N, D)
    pts = _normalize(pts)
    
    sqd = _pairwise_sq_dists(pts)
    mask = ~np.eye(N, dtype=bool)
    d_sq = sqd[mask]
    
    # Penalize too-close points
    min_d_sq = np.min(d_sq)
    if min_d_sq < 1e-8:
        return 1e10
    
    return _soft_log_ratio_sq(pts, tau)

def _generate_icosahedron():
    """Generate vertices of a regular icosahedron (12 points)."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

def _generate_fib_sphere(n):
    """Fibonacci sphere distribution of n points."""
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    theta = np.pi * (1 + np.sqrt(5)) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack((x, y, z))

def _generate_initial_guesses():
    """Generate diverse initial configurations."""
    guesses = []
    rng = np.random.default_rng(42)
    
    # 1. Icosahedron + 2 points on z-axis (symmetric)
    ico = _generate_icosahedron()
    for h in [0.3, 0.5, 0.7, 1.0, 1.3]:
        extra = np.array([[0, 0, h], [0, 0, -h]])
        guesses.append(np.vstack([ico, extra]))
    
    # 2. Icosahedron + center + 1 point
    ico = _generate_icosahedron()
    for h in [0.5, 1.0, 1.5]:
        extra = np.array([[0, 0, 0], [0, 0, h]])
        guesses.append(np.vstack([ico, extra]))
    
    # 3. Fibonacci sphere
    guesses.append(_generate_fib_sphere(N))
    
    # 4. Random perturbations of icosahedron + 2
    ico = _generate_icosahedron()
    for _ in range(10):
        pts = ico.copy()
        extra = np.array([[0, 0, 0.5], [0, 0, -0.5]])
        pts = np.vstack([pts, extra])
        pts = pts + rng.normal(0, 0.15, size=pts.shape)
        guesses.append(pts)
    
    # 5. Random points
    for _ in range(8):
        pts = rng.normal(0, 1, size=(N, D))
        guesses.append(pts)
    
    # 6. Two rings (top/bottom) + 2 poles
    for top_h, bot_h, r_top, r_bot in [(0.7, -0.7, 0.7, 0.7), (0.5, -0.5, 0.9, 0.9), (0.8, -0.8, 0.5, 0.5)]:
        angles_top = 2 * np.pi * np.arange(6) / 6
        angles_bot = 2 * np.pi * np.arange(6) / 6 + np.pi / 6
        top_pts = np.column_stack([r_top * np.cos(angles_top), r_top * np.sin(angles_top), np.full(6, top_h)])
        bot_pts = np.column_stack([r_bot * np.cos(angles_bot), r_bot * np.sin(angles_bot), np.full(6, bot_h)])
        poles = np.array([[0, 0, 1.2], [0, 0, -1.2]])
        guesses.append(np.vstack([top_pts, bot_pts, poles]))
    
    # 7. Cuboctahedron-like (12 points) + 2
    # Cuboctahedron vertices are all permutations of (0, ±1, ±1)
    phi = (1 + np.sqrt(5)) / 2
    cubo = np.array([
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1]
    ])
    norms = np.linalg.norm(cubo, axis=1, keepdims=True)
    cubo = cubo / norms
    for h in [0.4, 0.8, 1.2]:
        extra = np.array([[0, 0, h], [0, 0, -h]])
        guesses.append(np.vstack([cubo, extra]))
    
    return guesses

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_score = np.inf
    best_pts = None
    
    guesses = _generate_initial_guesses()
    
    # Two-stage optimization: coarse (large tau) then fine (small tau)
    tau_stages = [0.2, 0.08, 0.03, 0.01, 0.005]
    
    # First pass: coarse optimization from all starts
    candidates = []
    for guess in guesses:
        if time.time() > deadline - 30:
            break
        
        x0 = guess.flatten()
        
        try:
            res = minimize(
                _objective_coords,
                x0,
                args=(tau_stages[0],),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            if res.fun < 1e5:
                candidates.append(res.x)
        except Exception:
            continue
    
    # Keep top candidates by objective value
    if candidates:
        cand_scores = []
        for cx in candidates:
            pts = cx.reshape(N, D)
            pts = _normalize(pts)
            score = _true_score(pts)
            cand_scores.append((score, cx))
        
        cand_scores.sort(key=lambda x: x[0])
        # Keep top 10 for fine-tuning
        top_candidates = [c[1] for c in cand_scores[:10]]
    else:
        top_candidates = [g.flatten() for g in guesses[:5]]
    
    # Second pass: fine-tune top candidates with decreasing tau
    for cx in top_candidates:
        if time.time() > deadline - 20:
            break
        
        x_curr = cx.copy()
        
        for tau in tau_stages[1:]:
            if time.time() > deadline - 20:
                break
            
            try:
                res = minimize(
                    _objective_coords,
                    x_curr,
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < _objective_coords(x_curr, tau) + 1e-6:
                    x_curr = res.x
            except Exception:
                continue
        
        # Evaluate final score
        pts = x_curr.reshape(N, D)
        pts = _normalize(pts)
        
        score = _true_score(pts)
        if score < best_score:
            best_score = score
            best_pts = pts
    
    # If we still have time, try a few more random starts with fine tau
    if time.time() < deadline - 15:
        rng = np.random.default_rng(123)
        for _ in range(15):
            if time.time() > deadline - 10:
                break
            
            x0 = rng.normal(0, 1, size=(N, D)).flatten()
            
            try:
                # Quick coarse
                res1 = minimize(
                    _objective_coords, x0, args=(0.15,),
                    method='L-BFGS-B', options={'maxiter': 150, 'ftol': 1e-9}
                )
                if res1.fun > 1e5:
                    continue
                
                # Fine
                res2 = minimize(
                    _objective_coords, res1.x, args=(0.03,),
                    method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-11, 'gtol': 1e-7}
                )
                
                pts = res2.x.reshape(N, D)
                pts = _normalize(pts)
                
                score = _true_score(pts)
                if score < best_score:
                    best_score = score
                    best_pts = pts
                    
            except Exception:
                continue
    
    # Final high-precision polish on the best candidate
    if best_pts is not None and time.time() < deadline - 5:
        x_best = best_pts.flatten()
        try:
            res_final = minimize(
                _objective_coords,
                x_best,
                args=(0.002,),
                method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res_final.fun < _objective_coords(x_best, 0.002) + 1e-8:
                pts_final = res_final.x.reshape(N, D)
                pts_final = _normalize(pts_final)
                
                score_final = _true_score(pts_final)
                if score_final < best_score:
                    best_score = score_final
                    best_pts = pts_final
        except Exception:
            pass
    
    # If no valid points found, fallback to grid (should not happen)
    if best_pts is None:
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_pts = np.array(grid, dtype=float)
        
    return best_pts
