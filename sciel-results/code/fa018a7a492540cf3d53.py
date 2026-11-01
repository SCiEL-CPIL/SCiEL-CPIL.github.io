"""Combined approach: Gram-matrix SDP relaxation with eigenvalue penalties,
plus soft-min/soft-max surrogate for smooth optimization, and final coordinate polish."""
import numpy as np
import scipy.optimize
import time
from scipy.spatial.distance import cdist

N = 14
D = 3
DEADLINE = 130.0  # Safety margin before 150s limit

# EVOLVE-BLOCK-START

def _true_score(pts):
    """Compute the actual score: (max_dist / min_dist)^2."""
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-12:
        return np.inf
    return (mx / mn) ** 2

def _pairwise_dists(pts):
    """Return pairwise distance matrix (N, N)."""
    return cdist(pts, pts)

def _soft_log_ratio(dist, tau):
    """
    Smooth approximation of log(max_d) - log(min_d) using soft min/max via log-sum-exp.
    dist: (N, N) distance matrix
    tau: temperature parameter
    Returns scalar objective.
    """
    mask = ~np.eye(N, dtype=bool)
    d = dist[mask]
    
    # Stable soft max
    max_d = np.max(d)
    soft_max = max_d + tau * np.log(np.sum(np.exp((d - max_d) / tau)))
    
    # Stable soft min
    min_d = np.min(d)
    soft_min = min_d - tau * np.log(np.sum(np.exp((min_d - d) / tau)))
    
    return soft_max - soft_min

def _objective_coords(x, tau):
    """
    Objective in terms of coordinates. x is flat array of size N*D.
    Centers and normalizes to fix gauge freedom.
    """
    pts = x.reshape(N, D)
    pts = pts - np.mean(pts, axis=0)
    
    norm_sq = np.sum(pts**2)
    if norm_sq < 1e-10:
        return 1e10
    pts = pts * np.sqrt(N / norm_sq)
    
    dist = _pairwise_dists(pts)
    mask = ~np.eye(N, dtype=bool)
    d = dist[mask]
    
    min_d = np.min(d)
    if min_d < 1e-5:
        return 1e10
    
    return _soft_log_ratio(dist, tau)

def _gram_objective(G_flat, lambda_rank, tau):
    """
    Objective for Gram matrix optimization.
    G_flat: flattened (N, N) Gram matrix.
    We enforce PSD via eigenvalue penalty and rank-3 via eigenvalue penalty.
    The distance matrix is D_ij = sqrt(G_ii + G_jj - 2*G_ij).
    We minimize a soft log ratio of distances plus penalties.
    """
    G = G_flat.reshape(N, N)
    
    # Symmetrize
    G = 0.5 * (G + G.T)
    
    # Eigenvalue decomposition
    eigvals = np.linalg.eigvalsh(G)
    
    # PSD penalty: penalize negative eigenvalues
    neg_eigs = eigvals[eigvals < 0]
    psd_penalty = np.sum(neg_eigs**2)
    
    # Rank-3 penalty: penalize eigenvalues beyond the top 3
    # Sort eigenvalues descending
    sorted_eigs = np.sort(eigvals)[::-1]
    if len(sorted_eigs) > 3:
        rank_penalty = np.sum(sorted_eigs[3:]**2)
    else:
        rank_penalty = 0.0
    
    # Compute distances from Gram matrix
    G_ii = np.diag(G)
    G_jj = G_ii[np.newaxis, :]
    dist_sq = G_ii[:, np.newaxis] + G_jj - 2 * G
    
    # Clip to avoid sqrt of negative
    dist_sq = np.maximum(dist_sq, 0)
    dist = np.sqrt(dist_sq)
    
    mask = ~np.eye(N, dtype=bool)
    d = dist[mask]
    
    min_d = np.min(d)
    if min_d < 1e-5:
        return 1e10
    
    soft_ratio = _soft_log_ratio(dist, tau)
    
    return soft_ratio + lambda_rank * (psd_penalty + rank_penalty)

def _extract_points_from_gram(G):
    """Extract points from a Gram matrix via eigendecomposition."""
    G = 0.5 * (G + G.T)
    eigvals, eigvecs = np.linalg.eigh(G)
    
    # Take top 3 eigenvalues and their eigenvectors
    idx = np.argsort(eigvals)[::-1][:D]
    eigvals_top = eigvals[idx]
    eigvecs_top = eigvecs[:, idx]
    
    # Ensure non-negative eigenvalues
    eigvals_top = np.maximum(eigvals_top, 0)
    
    pts = eigvecs_top * np.sqrt(eigvals_top)
    
    # Center
    pts = pts - np.mean(pts, axis=0)
    
    return pts

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
    
    # 1. Icosahedron + 2 points on z-axis
    ico = _generate_icosahedron()
    for h in [0.3, 0.5, 0.7, 1.0]:
        extra = np.array([[0, 0, h], [0, 0, -h]])
        guesses.append(np.vstack([ico, extra]))
    
    # 2. Icosahedron + center + 1 point
    for h in [0.5, 1.0, 1.5]:
        extra = np.array([[0, 0, 0], [0, 0, h]])
        guesses.append(np.vstack([ico, extra]))
    
    # 3. Fibonacci sphere
    guesses.append(_generate_fib_sphere(N))
    
    # 4. Random perturbations of icosahedron + 2
    rng = np.random.default_rng(42)
    ico = _generate_icosahedron()
    for _ in range(8):
        pts = ico.copy()
        extra = np.array([[0, 0, 0.5], [0, 0, -0.5]])
        pts = np.vstack([pts, extra])
        pts = pts + rng.normal(0, 0.1, size=pts.shape)
        guesses.append(pts)
    
    # 5. Random points
    for _ in range(5):
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
    
    return guesses

def _polish_coordinates(pts, max_iters=500, tol=1e-10):
    """
    Final high-precision polish pass using L-BFGS-B with tight tolerances.
    Uses the true (nonsmooth) objective for final refinement.
    """
    def _true_obj(x):
        p = x.reshape(N, D)
        p = p - np.mean(p, axis=0)
        norm_sq = np.sum(p**2)
        if norm_sq < 1e-10:
            return 1e10
        p = p * np.sqrt(N / norm_sq)
        return _true_score(p)
    
    x0 = pts.flatten()
    
    try:
        res = scipy.optimize.minimize(
            _true_obj,
            x0,
            method='L-BFGS-B',
            options={'maxiter': max_iters, 'ftol': tol, 'gtol': 1e-8, 'maxfun': 10000}
        )
        if res.fun < 1e5:
            p = res.x.reshape(N, D)
            p = p - np.mean(p, axis=0)
            norm_sq = np.sum(p**2)
            if norm_sq > 1e-10:
                p = p * np.sqrt(N / norm_sq)
            return p
    except Exception:
        pass
    
    return pts

def construct():
    start_time = time.time()
    
    best_score = np.inf
    best_pts = None
    
    guesses = _generate_initial_guesses()
    
    # Stage 1: Gram matrix optimization (SDP relaxation)
    # Initialize Gram matrices from initial guesses
    gram_candidates = []
    for guess in guesses[:10]:  # Limit to 10 for speed
        if time.time() - start_time > DEADLINE - 40:
            break
        
        pts = guess - np.mean(guess, axis=0)
        norm_sq = np.sum(pts**2)
        if norm_sq > 1e-10:
            pts = pts * np.sqrt(N / norm_sq)
        
        G = pts @ pts.T
        G_flat = G.flatten()
        
        # Optimize Gram matrix
        lambda_rank = 10.0
        tau = 0.1
        
        try:
            res = scipy.optimize.minimize(
                _gram_objective,
                G_flat,
                args=(lambda_rank, tau),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            if res.fun < 1e5:
                G_opt = res.x.reshape(N, N)
                pts_opt = _extract_points_from_gram(G_opt)
                score = _true_score(pts_opt)
                if score < best_score:
                    best_score = score
                    best_pts = pts_opt
                gram_candidates.append((score, pts_opt))
        except Exception:
            continue
    
    # Stage 2: Coordinate-based soft objective optimization
    tau_stages = [0.3, 0.1, 0.05, 0.02, 0.01]
    
    coord_candidates = []
    for guess in guesses:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        x0 = guess.flatten()
        
        try:
            res = scipy.optimize.minimize(
                _objective_coords,
                x0,
                args=(tau_stages[0],),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            if res.fun < 1e5:
                pts = res.x.reshape(N, D)
                pts = pts - np.mean(pts, axis=0)
                norm_sq = np.sum(pts**2)
                if norm_sq > 1e-10:
                    pts = pts * np.sqrt(N / norm_sq)
                score = _true_score(pts)
                coord_candidates.append((score, pts))
        except Exception:
            continue
    
    # Combine candidates from both stages
    all_candidates = gram_candidates + coord_candidates
    
    # Sort by score and keep top candidates
    if all_candidates:
        all_candidates.sort(key=lambda x: x[0])
        top_candidates = [c[1] for c in all_candidates[:8]]
    else:
        top_candidates = [g for g in guesses[:3]]
    
    # Stage 3: Fine-tune with decreasing tau
    for pts in top_candidates:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        for tau in tau_stages[1:]:
            if time.time() - start_time > DEADLINE - 15:
                break
            
            x_curr = pts.flatten()
            try:
                res = scipy.optimize.minimize(
                    _objective_coords,
                    x_curr,
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < 1e5:
                    pts = res.x.reshape(N, D)
                    pts = pts - np.mean(pts, axis=0)
                    norm_sq = np.sum(pts**2)
                    if norm_sq > 1e-10:
                        pts = pts * np.sqrt(N / norm_sq)
            except Exception:
                continue
    
    # Stage 4: Final polish with true objective
    if time.time() - start_time < DEADLINE - 5:
        if best_pts is not None:
            polished = _polish_coordinates(best_pts, max_iters=300, tol=1e-12)
            score = _true_score(polished)
            if score < best_score:
                best_score = score
                best_pts = polished
    
    # Also polish a few other top candidates if time permits
    if time.time() - start_time < DEADLINE - 10:
        for pts in top_candidates[:3]:
            if time.time() - start_time > DEADLINE - 5:
                break
            polished = _polish_coordinates(pts, max_iters=200, tol=1e-12)
            score = _true_score(polished)
            if score < best_score:
                best_score = score
                best_pts = polished
    
    # Fallback
    if best_pts is None:
        best_pts = _generate_fib_sphere(N)
    
    return best_pts

# EVOLVE-BLOCK-END
