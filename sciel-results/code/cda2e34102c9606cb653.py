import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 14
D = 3
TIME_LIMIT = 145.0  # Keep 5s margin from 150s limit

def _pairwise_sq_dists(pts):
    """Calculate squared pairwise distances between all pairs of points."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sum(diff**2, axis=-1)

def _true_score(pts):
    """Calculate the actual score: (max_dist / min_dist)^2"""
    sq_d = _pairwise_sq_dists(pts)
    mask = ~np.eye(N, dtype=bool)
    d_sq = sq_d[mask]
    mn_sq = np.min(d_sq)
    mx_sq = np.max(d_sq)
    if mn_sq < 1e-20:
        return np.inf
    return mx_sq / mn_sq

def _normalize_points(pts):
    """Center and normalize points to have sum of squared norms = N."""
    pts = pts - np.mean(pts, axis=0)
    norm_sq = np.sum(pts**2)
    if norm_sq < 1e-10:
        return pts
    return pts * np.sqrt(N / norm_sq)

def _objective_qcqp(x, upper_bound_sq):
    """
    Objective for the QCQP formulation:
    Maximize min squared distance, subject to max squared distance <= upper_bound_sq.
    
    We minimize: -min_sq_dist + penalty * max(0, max_sq_dist - upper_bound_sq)^2
    
    x: flat array of size N*D
    """
    pts = x.reshape(N, D)
    pts = _normalize_points(pts)
    
    sq_d = _pairwise_sq_dists(pts)
    mask = ~np.eye(N, dtype=bool)
    d_sq = sq_d[mask]
    
    mn_sq = np.min(d_sq)
    mx_sq = np.max(d_sq)
    
    if mn_sq < 1e-12:
        return 1e10
    
    # Penalty for violating upper bound
    violation = np.maximum(0, mx_sq - upper_bound_sq)
    penalty = 1000.0 * violation**2
    
    # We want to maximize mn_sq, so minimize -mn_sq
    return -mn_sq + penalty

def _objective_ratio(x):
    """
    Direct ratio objective: (max_dist / min_dist)^2
    """
    pts = x.reshape(N, D)
    pts = _normalize_points(pts)
    
    sq_d = _pairwise_sq_dists(pts)
    mask = ~np.eye(N, dtype=bool)
    d_sq = sq_d[mask]
    
    mn_sq = np.min(d_sq)
    mx_sq = np.max(d_sq)
    
    if mn_sq < 1e-12:
        return 1e10
    
    return mx_sq / mn_sq

def _generate_icosahedron():
    """Generate vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    return verts

def _generate_fib_sphere(n):
    """Generate points on a sphere using Fibonacci spiral."""
    i = np.arange(n)
    phi = np.arccos(1 - 2*(i+0.5)/n)
    theta = np.pi * (1 + np.sqrt(5)) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack((x, y, z))

def _generate_initial_guesses():
    """Generate a list of initial configurations."""
    guesses = []
    
    ico = _generate_icosahedron()
    ico_norm = np.linalg.norm(ico, axis=1, keepdims=True)
    ico = ico / ico_norm
    
    # Guess A: Icosahedron + Center + 1 point
    p14_a = np.vstack([ico, np.array([[0,0,0]]), np.array([[0,0,0.5]])])
    guesses.append(p14_a)
    
    # Guess B: Icosahedron + 2 points on Z axis (symmetric)
    p14_b = np.vstack([ico, np.array([[0,0,0.5]]), np.array([[0,0,-0.5]])])
    guesses.append(p14_b)
    
    # Guess C: Fibonacci Sphere
    fib_pts = _generate_fib_sphere(N)
    guesses.append(fib_pts)
    
    # Guess D: Random points on sphere
    for _ in range(5):
        rand_pts = np.random.randn(N, D)
        rand_pts = rand_pts / np.linalg.norm(rand_pts, axis=1, keepdims=True)
        guesses.append(rand_pts)
        
    # Guess E: Ring configurations
    # 7 points on a ring in xy-plane, 7 on a ring in xz-plane (offset)
    angles1 = 2*np.pi*np.arange(7)/7
    ring1 = np.column_stack([np.cos(angles1), np.sin(angles1), np.zeros(7)])
    angles2 = 2*np.pi*np.arange(7)/7 + np.pi/7
    ring2 = np.column_stack([np.cos(angles2), np.zeros(7), np.sin(angles2)])
    guesses.append(np.vstack([ring1, ring2]))
    
    return guesses

def _local_optimize_ratio(x0, maxiter=500):
    """Run local optimization using ratio objective."""
    try:
        res = minimize(
            _objective_ratio,
            x0,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        pts = res.x.reshape(N, D)
        pts = _normalize_points(pts)
        score = _true_score(pts)
        if score < np.inf:
            return pts, score
    except Exception:
        pass
    return None, np.inf

def _qcqp_solve(x0, upper_bound_sq, maxiter=500):
    """
    Solve the QCQP: maximize min_sq_dist subject to max_sq_dist <= upper_bound_sq.
    Returns the optimized points and the achieved min_sq_dist.
    """
    try:
        res = minimize(
            lambda x: _objective_qcqp(x, upper_bound_sq),
            x0,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        pts = res.x.reshape(N, D)
        pts = _normalize_points(pts)
        sq_d = _pairwise_sq_dists(pts)
        mask = ~np.eye(N, dtype=bool)
        d_sq = sq_d[mask]
        mn_sq = np.min(d_sq)
        mx_sq = np.max(d_sq)
        return pts, mn_sq, mx_sq
    except Exception:
        return None, 0, np.inf

def _bisection_qcqp(x0, start_upper=None, n_iter=15):
    """
    Use bisection on the upper bound to find the optimal configuration.
    We want to find the largest upper_bound_sq such that we can achieve a good min_sq_dist.
    """
    if start_upper is None:
        # Estimate initial upper bound
        pts = x0.reshape(N, D)
        pts = _normalize_points(pts)
        sq_d = _pairwise_sq_dists(pts)
        mask = ~np.eye(N, dtype=bool)
        d_sq = sq_d[mask]
        start_upper = np.max(d_sq) * 1.5
    
    low = 0.1
    high = start_upper
    
    best_pts = None
    best_score = np.inf
    
    for i in range(n_iter):
        mid = (low + high) / 2
        
        pts, mn_sq, mx_sq = _qcqp_solve(x0, mid, maxiter=300)
        
        if pts is None:
            high = mid
            continue
        
        # Check if constraint is satisfied
        if mx_sq <= mid * 1.01:
            # Constraint satisfied, try to increase upper bound
            low = mid
            score = _true_score(pts)
            if score < best_score:
                best_score = score
                best_pts = pts
        else:
            # Constraint violated, decrease upper bound
            high = mid
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    
    def time_remaining():
        return TIME_LIMIT - (time.time() - start_time)
    
    best_score = np.inf
    best_pts = None
    
    def update_best(pts, score):
        nonlocal best_score, best_pts
        if score < best_score:
            best_score = score
            best_pts = pts
            return True
        return False
    
    # Phase 1: Local optimization from multiple starting points using ratio objective
    guesses = _generate_initial_guesses()
    
    for guess in guesses:
        if time_remaining() < 60:
            break
        x0 = guess.flatten()
        pts, score = _local_optimize_ratio(x0, maxiter=400)
        if pts is not None:
            update_best(pts, score)
    
    # Phase 2: QCQP with bisection from best solution
    if best_pts is not None and time_remaining() > 30:
        x0 = best_pts.flatten()
        pts_qcqp, score_qcqp = _bisection_qcqp(x0, n_iter=12)
        if pts_qcqp is not None:
            update_best(pts_qcqp, score_qcqp)
    
    # Phase 3: More local optimizations from perturbed best solution
    if best_pts is not None and time_remaining() > 10:
        for _ in range(8):
            if time_remaining() < 10:
                break
            noise = np.random.randn(N, D) * 0.1
            perturbed = best_pts + noise
            x0 = perturbed.flatten()
            pts, score = _local_optimize_ratio(x0, maxiter=300)
            if pts is not None:
                update_best(pts, score)
    
    # Phase 4: Final polish with QCQP from best solution
    if best_pts is not None and time_remaining() > 5:
        x0 = best_pts.flatten()
        pts_qcqp, score_qcqp = _bisection_qcqp(x0, n_iter=8)
        if pts_qcqp is not None:
            update_best(pts_qcqp, score_qcqp)
    
    # Fallback
    if best_pts is None:
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_pts = np.array(grid, dtype=float)
    
    return best_pts
