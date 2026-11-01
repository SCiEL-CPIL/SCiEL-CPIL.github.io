"""Rewritten program: hexagonal lattice init + gradient-based local search with SA restarts."""
import numpy as np
import time

N = 50
DEADLINE = 135.0  # safety margin before 150s kill


def _score(pts):
    """Compute the minimum radius r for which circles are disjoint and inside unit circle."""
    # Distance from origin to each center
    dists = np.sqrt(np.sum(pts ** 2, axis=1))
    # Distance from each center to boundary
    boundary_dists = 1.0 - dists
    min_boundary = np.min(boundary_dists)
    
    if min_boundary < 0:
        return -1.0  # invalid
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    pair_dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(pair_dists, np.inf)
    min_pair = np.min(pair_dists) / 2.0
    
    return min(min_pair, min_boundary)


def _gradient(pts, r):
    """
    Compute gradient of the objective w.r.t. each point.
    Objective: maximize r = min(min_pair_dist/2, min_boundary_dist)
    
    We use a soft-min approach: for each constraint, compute gradient.
    The active constraints are those closest to the minimum.
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    
    # Boundary constraints
    dists = np.sqrt(np.sum(pts ** 2, axis=1))
    boundary_dists = 1.0 - dists
    min_boundary = np.min(boundary_dists)
    
    # Pairwise constraints
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    pair_dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(pair_dists, np.inf)
    min_pair_half = np.min(pair_dists) / 2.0
    
    r_val = min(min_pair_half, min_boundary)
    
    # Determine which constraints are active (close to minimum)
    # Use a threshold to include near-active constraints
    eps = 1e-6
    
    # Boundary gradient: for each point, gradient of (1 - |p_i|) w.r.t. p_i is -p_i/|p_i|
    # If boundary constraint is active, we want to push point inward
    for i in range(n):
        if boundary_dists[i] < min_boundary + eps:
            d = dists[i]
            if d > 1e-10:
                grad[i] += -pts[i] / d  # push toward center
    
    # Pairwise gradient: for each pair, gradient of |p_i - p_j|/2
    # For point i: (p_i - p_j) / (2 * |p_i - p_j|)
    # For point j: (p_j - p_i) / (2 * |p_i - p_j|)
    for i in range(n):
        for j in range(i + 1, n):
            d = pair_dists[i, j]
            if d < 2 * r_val + 2 * eps and d > 1e-10:
                direction = (pts[i] - pts[j]) / d
                grad[i] += direction / 2.0
                grad[j] += -direction / 2.0
    
    return grad


def _hex_lattice_init():
    """Generate hexagonal lattice points and select N closest to center."""
    spacing = 2.0 / (np.sqrt(N) + 1.5)
    while True:
        k = int(1.0 / spacing) + 3
        points = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing / 2.0:
                    points.append((x, y, r))
        if len(points) >= N:
            points.sort(key=lambda p: p[2])
            return np.array([p[:2] for p in points[:N]])
        spacing *= 0.97


def _local_search(pts, max_iter=500, tol=1e-10):
    """Gradient-based local search to maximize the minimum radius."""
    best_pts = pts.copy()
    best_r = _score(best_pts)
    
    for iteration in range(max_iter):
        r = _score(pts)
        if r <= best_r + 1e-12:
            break
        best_r = r
        best_pts = pts.copy()
        
        # Compute gradient
        grad = _gradient(pts, r)
        
        # Adaptive step size
        norm_grad = np.sqrt(np.sum(grad ** 2))
        if norm_grad < 1e-12:
            break
        
        # Line search
        step = 0.01
        improved = False
        for _ in range(30):
            new_pts = pts + step * grad
            new_r = _score(new_pts)
            if new_r > r + 1e-14:
                pts = new_pts
                improved = True
                break
            step *= 0.5
        
        if not improved:
            break
    
    return best_pts, best_r


def _sa_search(pts, start_temp=0.01, end_temp=1e-6, max_iter=2000):
    """Simulated annealing with gradient steps."""
    best_pts = pts.copy()
    best_r = _score(best_pts)
    current_pts = pts.copy()
    current_r = _score(current_pts)
    
    for iteration in range(max_iter):
        # Temperature schedule
        t = iteration / max_iter
        temp = start_temp * (end_temp / start_temp) ** t
        
        # Gradient step
        grad = _gradient(current_pts, current_r)
        norm_grad = np.sqrt(np.sum(grad ** 2))
        
        if norm_grad < 1e-12:
            break
        
        # Step with noise
        step_size = 0.005
        noise = np.random.randn(*current_pts.shape) * temp * 0.1
        new_pts = current_pts + step_size * (grad / norm_grad) + noise
        
        new_r = _score(new_pts)
        
        # Accept if better or with probability exp((new_r - current_r) / temp)
        if new_r > current_r:
            current_pts = new_pts
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
        else:
            prob = np.exp((new_r - current_r) / (temp + 1e-15))
            if np.random.random() < prob:
                current_pts = new_pts
                current_r = new_r
    
    return best_pts, best_r


def construct():
    start_time = time.time()
    
    # Initial solution from hexagonal lattice
    pts = _hex_lattice_init()
    best_pts = pts.copy()
    best_r = _score(best_pts)
    
    # Local search to refine
    best_pts, best_r = _local_search(best_pts, max_iter=300)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 30:
        return best_pts
    
    # Simulated annealing with multiple restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Perturb from best solution
        noise_scale = 0.02 * (1 - restart / n_restarts)
        perturbed = best_pts + np.random.randn(N, 2) * noise_scale
        
        # SA search
        sa_pts, sa_r = _sa_search(perturbed, start_temp=0.005, end_temp=1e-7, max_iter=500)
        
        if sa_r > best_r:
            best_r = sa_r
            best_pts = sa_pts.copy()
        
        # Local search on SA result
        ls_pts, ls_r = _local_search(sa_pts, max_iter=200)
        if ls_r > best_r:
            best_r = ls_r
            best_pts = ls_pts.copy()
    
    # Final high-precision polish
    if time.time() - start_time < DEADLINE - 5:
        final_pts, final_r = _local_search(best_pts, max_iter=1000, tol=1e-12)
        if final_r > best_r:
            best_r = final_r
            best_pts = final_pts.copy()
    
    return best_pts
