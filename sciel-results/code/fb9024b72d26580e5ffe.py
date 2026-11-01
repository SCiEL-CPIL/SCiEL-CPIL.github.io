import numpy as np
import scipy.optimize as so
import time
import itertools
from scipy.spatial.transform import Rotation

N = 14
D = 3

def _pairwise_distances_squared(X):
    """Compute squared pairwise distances for points X (N x D)."""
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist2 = np.sum(diff**2, axis=2)
    return dist2

def _raw_score(X):
    """
    Calculate the actual score: (max_dist / min_dist)^2.
    X is an array of shape (N, D).
    """
    dist2 = _pairwise_distances_squared(X)
    mask = ~np.eye(N, dtype=bool)
    d2 = dist2[mask]
    if len(d2) == 0:
        return 1e10
    d_min = np.min(d2)
    d_max = np.max(d2)
    if d_min < 1e-12:
        return 1e10
    return (d_max / d_min)

def _objective_smooth(x, tau=50.0):
    """
    Smooth approximation of log(max_sq_dist) - log(min_sq_dist).
    Minimizing this minimizes the ratio of max to min distance.
    Uses log-sum-exp for stability.
    x is a flattened array of length N*D.
    """
    X = x.reshape(N, D)
    
    # Center the points to remove translation invariance
    X = X - np.mean(X, axis=0, keepdims=True)
    
    # Compute squared distances
    dist2 = _pairwise_distances_squared(X)
    mask = ~np.eye(N, dtype=bool)
    d2 = dist2[mask]
    
    # Handle degenerate cases
    if np.any(d2 < 1e-15):
        return 1e10
    
    # Normalize to fix scale: make the mean squared distance = 1
    mean_sq_dist = np.mean(d2)
    if mean_sq_dist < 1e-12:
        return 1e10
    
    X = X / np.sqrt(mean_sq_dist)
    dist2 = _pairwise_distances_squared(X)
    mask = ~np.eye(N, dtype=bool)
    d2 = dist2[mask]
    
    # Log squared distances
    log_sq_dists = np.log(d2)
    
    # Soft max of log(squared distances)
    m_max = np.max(log_sq_dists)
    exp_max_terms = np.exp(tau * (log_sq_dists - m_max))
    soft_max_log = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    # Soft min of log(squared distances)
    m_min = np.min(log_sq_dists)
    exp_min_terms = np.exp(-tau * (log_sq_dists - m_min))
    soft_min_log = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    # Objective is the difference
    obj = soft_max_log - soft_min_log
    
    return obj

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = 1e10
    best_X = None
    
    def update_best(X):
        nonlocal best_score, best_X
        if X is None:
            return
        X = np.asarray(X)
        if X.shape != (N, D):
            return
        # Check for duplicate points
        score = _raw_score(X)
        if score < best_score:
            best_score = score
            best_X = X.copy()

    rng = np.random.default_rng(42)
    
    def generate_tetrahedral_structure():
        # Vertices of a regular tetrahedron
        v1 = np.array([1, 1, 1])
        v2 = np.array([1, -1, -1])
        v3 = np.array([-1, 1, -1])
        v4 = np.array([-1, -1, 1])
        
        # Midpoints of edges
        m12 = (v1 + v2) / 2
        m13 = (v1 + v3) / 2
        m14 = (v1 + v4) / 2
        m23 = (v2 + v3) / 2
        m24 = (v2 + v4) / 2
        m34 = (v3 + v4) / 2
        
        # Centers of faces
        f123 = (v1 + v2 + v3) / 3
        f124 = (v1 + v2 + v4) / 3
        f134 = (v1 + v3 + v4) / 3
        f234 = (v2 + v3 + v4) / 3
        
        pts = np.vstack([
            v1, v2, v3, v4,
            m12, m13, m14, m23, m24, m34,
            f123, f124, f134, f234
        ])
        
        # Apply random rotation
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        
        # Center
        pts -= np.mean(pts, axis=0)
        return pts

    def generate_fibonacci_sphere():
        N_pts = N
        golden_angle = np.pi * (3. - np.sqrt(5.))
        X = np.zeros((N_pts, 3))
        for i in range(N_pts):
            z = 1. - (i / (N_pts - 1.)) * 2.
            radius = np.sqrt(1. - z**2)
            theta = golden_angle * i
            x = np.cos(theta) * radius
            y = np.sin(theta) * radius
            X[i] = [x, y, z]
        # Center
        X -= np.mean(X, axis=0)
        return X

    def generate_random_sphere():
        X = rng.normal(size=(N, D))
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        rot = Rotation.from_quat(rng.normal(size=4))
        X = rot.apply(X)
        X -= np.mean(X, axis=0)
        return X

    def generate_rings():
        # Two rings of 7 points each
        ring1_angles = np.linspace(0, 2*np.pi, 7, endpoint=False) + rng.uniform(0, 2*np.pi)
        ring2_angles = ring1_angles + np.pi/7
        z_offset = rng.uniform(-0.5, 0.5)
        ring1 = np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles), np.full(7, z_offset)])
        ring2 = np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles), np.full(7, -z_offset)])
        pts = np.vstack([ring1, ring2])
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        pts -= np.mean(pts, axis=0)
        return pts

    inits = []
    for _ in range(8):
        inits.append(generate_tetrahedral_structure())
    for _ in range(4):
        inits.append(generate_fibonacci_sphere())
    for _ in range(4):
        inits.append(generate_random_sphere())
    for _ in range(4):
        inits.append(generate_rings())
    
    # Optimization Loop
    # We use a multi-stage tau schedule to anneal the objective.
    tau_schedule = [10.0, 50.0, 200.0, 1000.0]
    
    n_restarts = len(inits)
    candidates = [X.flatten() for X in inits]
    
    # Phase 1: Coarse optimization with L-BFGS-B
    for i in range(n_restarts):
        if time.time() > deadline - 30:
            break
        x0 = candidates[i]
        try:
            res = so.minimize(
                _objective_smooth, 
                x0, 
                method='L-BFGS-B', 
                args=(tau_schedule[0],),
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                candidates[i] = res.x
        except Exception:
            pass

    # Phase 2: Medium optimization
    for i in range(n_restarts):
        if time.time() > deadline - 20:
            break
        x0 = candidates[i]
        try:
            res = so.minimize(
                _objective_smooth, 
                x0, 
                method='L-BFGS-B', 
                args=(tau_schedule[1],),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                candidates[i] = res.x
        except Exception:
            pass

    # Phase 3: Fine optimization with higher tau
    for i in range(n_restarts):
        if time.time() > deadline - 10:
            break
        x0 = candidates[i]
        try:
            res = so.minimize(
                _objective_smooth, 
                x0, 
                method='L-BFGS-B', 
                args=(tau_schedule[2],),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                candidates[i] = res.x
        except Exception:
            pass

    # Phase 4: Polish the best candidate with Nelder-Mead on the raw objective
    if best_X is not None and time.time() < deadline - 2:
        for _ in range(5):
            if time.time() > deadline - 1:
                break
            X_pert = best_X + rng.normal(size=(N, D), scale=0.01)
            X_pert -= np.mean(X_pert, axis=0)
            try:
                res = so.minimize(
                    lambda x: _objective_smooth(x, tau=5000.0),
                    X_pert.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, D)
                    update_best(pts)
            except Exception:
                pass

    if best_X is None:
        # Fallback: return a grid
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        return np.array(grid, dtype=float)
        
    return best_X
