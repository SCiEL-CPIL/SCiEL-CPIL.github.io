"""Packing 50 equal circles in a unit circle using dynamics + soft-min polishing."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _score(pts):
    """Compute the feasible radius for a set of points."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    # Boundary constraint
    d_boundary = 1.0 - np.linalg.norm(pts, axis=1)
    r_boundary = np.min(d_boundary)
    # Pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    r_pair = 0.5 * np.min(dists)
    return min(r_boundary, r_pair)


def _hex_lattice_init(n=N):
    """Generate a hexagonal lattice packing inside the unit disk."""
    # Use a dense hex lattice and pick the best n points
    best_pts = None
    best_score = -1
    for scale in np.linspace(0.8, 2.5, 40):
        spacing = 2.0 / (scale + 1.0)
        k = int(1.0 / spacing) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        pts = np.array(rows)
        r = np.hypot(pts[:, 0], pts[:, 1])
        mask = r <= 1.0 - spacing / 2.0
        pts = pts[mask]
        if len(pts) >= n:
            # Sort by distance from center, take closest n
            order = np.argsort(r[mask], kind="stable")[:n]
            candidate = pts[order]
            s = _score(candidate)
            if s > best_score:
                best_score = s
                best_pts = candidate.copy()
    return best_pts


def _soft_min_objective(pts, beta=50.0):
    """
    Negative of the soft-min of the minimum radius.
    We want to maximize r = min(r_boundary, r_pair).
    We minimize -r using a soft-min surrogate.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    # Boundary terms
    d_boundary = 1.0 - np.linalg.norm(pts, axis=1)
    # Pairwise terms (half distance)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    r_pair = 0.5 * dists
    
    # Soft min of all constraints
    # Use log-sum-exp for stability
    all_r = np.concatenate([d_boundary, r_pair[r_pair < np.inf].ravel()])
    if len(all_r) == 0:
        return 0.0
    
    # Soft min: -logsumexp(-beta * all_r) / beta
    # For large beta, this approximates min(all_r)
    # We want to maximize min, so minimize -min = logsumexp(-beta*all_r)/beta
    # But logsumexp can overflow, so shift
    shifted = -beta * all_r
    shifted = shifted - np.max(shifted)
    logsumexp_val = np.max(shifted) + np.log(np.sum(np.exp(shifted)))
    soft_min = -logsumexp_val / beta
    
    return -soft_min  # We minimize this, so it's negative of soft_min


def _soft_min_grad(pts, beta=50.0):
    """Gradient of the soft-min objective."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    
    # Boundary terms
    d_boundary = 1.0 - np.linalg.norm(pts, axis=1)
    # Pairwise terms
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    r_pair = 0.5 * dists
    
    all_r = np.concatenate([d_boundary, r_pair[r_pair < np.inf].ravel()])
    if len(all_r) == 0:
        return np.zeros_like(pts)
    
    # Soft min weights
    shifted = -beta * all_r
    shifted = shifted - np.max(shifted)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    
    grad = np.zeros_like(pts)
    
    # Boundary gradient contribution
    # d_boundary_i = 1 - ||p_i||, grad = -p_i / ||p_i||
    for i in range(n):
        norm_i = np.linalg.norm(pts[i])
        if norm_i > 1e-12:
            grad[i] += weights[i] * (-pts[i] / norm_i) * beta
    
    # Pairwise gradient contribution
    # r_pair_ij = 0.5 * ||p_i - p_j||, grad wrt p_i = 0.5 * (p_i - p_j) / ||p_i - p_j||
    pair_idx = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist > 1e-12:
                w = weights[n + pair_idx]
                grad[i] += w * (0.5 * d / dist) * beta
                grad[j] += w * (-0.5 * d / dist) * beta
            pair_idx += 1
    
    return grad


def _dynamics_solve(pts, n_steps=5000, dt=0.001, damping=0.95, repulsion_k=100.0, boundary_k=50.0):
    """
    Simple dynamics: circles repel each other and are pushed back inside the boundary.
    """
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    
    for step in range(n_steps):
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 1e-12:
                    # Random nudge
                    d = np.random.randn(2) * 1e-6
                    dist = np.linalg.norm(d)
                # Repulsion force ~ 1/dist^2 or linear
                force_mag = repulsion_k / (dist + 0.01)
                force = force_mag * d / dist
                forces[i] += force
                forces[j] -= force
        
        # Boundary force
        for i in range(n):
            norm_i = np.linalg.norm(pts[i])
            if norm_i > 1.0:
                # Push back toward center
                forces[i] -= boundary_k * (norm_i - 1.0) * pts[i] / norm_i
            elif norm_i > 0.9:
                # Gentle push inward near boundary
                forces[i] -= boundary_k * 0.1 * (norm_i - 0.9) * pts[i] / norm_i
        
        # Update positions
        pts += dt * forces
        pts *= damping  # Simple damping
        
        # Hard constraint: keep inside unit disk
        norms = np.linalg.norm(pts, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / norms[mask][:, np.newaxis]
    
    return pts


def _polish(pts, beta_start=20.0, beta_end=500.0, n_iter=200):
    """
    Progressive beta polishing with L-BFGS-B.
    """
    current_pts = pts.copy()
    n_betas = min(n_iter, 50)
    betas = np.geomspace(beta_start, beta_end, n_betas)
    
    for beta in betas:
        result = minimize(
            _soft_min_objective,
            current_pts,
            args=(beta,),
            method='L-BFGS-B',
            jac=_soft_min_grad,
            options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.success or result.fun < _soft_min_objective(current_pts, beta):
            current_pts = result.x
    
    return current_pts


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Step 1: Initial hexagonal lattice
    pts = _hex_lattice_init(N)
    if pts is None:
        # Fallback: random
        pts = np.random.randn(N, 2) * 0.8
    
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    # Step 2: Dynamics to improve
    if time.time() < deadline - 30.0:
        pts_dyn = _dynamics_solve(pts, n_steps=3000, dt=0.0005, damping=0.98, 
                                   repulsion_k=50.0, boundary_k=30.0)
        score_dyn = _score(pts_dyn)
        if score_dyn > best_score:
            best_pts = pts_dyn.copy()
            best_score = score_dyn
    
    # Step 3: Soft-min polishing with increasing beta
    if time.time() < deadline - 15.0:
        pts_polished = _polish(best_pts, beta_start=20.0, beta_end=500.0, n_iter=30)
        score_polished = _score(pts_polished)
        if score_polished > best_score:
            best_pts = pts_polished.copy()
            best_score = score_polished
    
    # Step 4: Final high-precision polish
    if time.time() < deadline - 5.0:
        pts_final = _polish(best_pts, beta_start=100.0, beta_end=1000.0, n_iter=20)
        score_final = _score(pts_final)
        if score_final > best_score:
            best_pts = pts_final.copy()
            best_score = score_final
    
    # Ensure validity
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms > 1.0):
        best_pts[norms > 1.0] = best_pts[norms > 1.0] / norms[norms > 1.0][:, np.newaxis]
    
    if not np.all(np.isfinite(best_pts)):
        best_pts = _hex_lattice_init(N)
    
    return best_pts
# EVOLVE-BLOCK-END
