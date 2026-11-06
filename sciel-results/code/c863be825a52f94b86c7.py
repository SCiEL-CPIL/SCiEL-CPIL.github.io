"""Pack 50 equal circles in the unit circle using a multi-restart local search with repulsion forces."""
import numpy as np
import time

N = 50

def _score(pts):
    """Compute the packing radius for a set of points."""
    if pts.shape != (N, 2):
        return -1.0
    if not np.all(np.isfinite(pts)):
        return -1.0
    # Check container constraint
    dists_center = np.linalg.norm(pts, axis=1)
    if np.any(dists_center >= 1.0):
        return -1.0
    r_container = 1.0 - np.max(dists_center)
    # Check pairwise distances
    # Compute all pairwise distances
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dist_mat = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist_mat, np.inf)
    min_dist = np.min(dist_mat)
    r_pair = min_dist / 2.0
    return min(r_container, r_pair)

def _repulsion_force(pts, r_target=None):
    """
    Compute a repulsion force on each point.
    Points are pushed apart and kept inside the container.
    """
    n = len(pts)
    force = np.zeros_like(pts)
    
    # Pairwise repulsion: push apart points that are too close
    for i in range(n):
        for j in range(i + 1, n):
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                # Random small perturbation
                d = np.array([np.random.randn(), np.random.randn()])
                dist = np.linalg.norm(d)
            # Strong repulsion when close, weaker when far
            # Use a smooth potential
            if dist < 0.5:
                # Linear repulsion that increases as points get closer
                strength = 1.0 / (dist + 1e-6)
                force[i] += d / dist * strength
                force[j] -= d / dist * strength
    
    # Container constraint: push inward if too close to boundary
    for i in range(n):
        r = np.linalg.norm(pts[i])
        if r > 0.95:
            # Push toward center
            force[i] -= pts[i] / r * (r - 0.95) * 5.0
    
    return force

def _local_search(pts, max_iters=5000, lr=0.01, tol=1e-10):
    """
    Local search using gradient-free optimization with repulsion.
    """
    best_pts = pts.copy()
    best_score = _score(pts)
    current_score = best_score
    
    for it in range(max_iters):
        # Compute forces
        f = _repulsion_force(pts)
        
        # Update positions
        pts_new = pts + lr * f
        
        # Project inside unit circle
        r = np.linalg.norm(pts_new, axis=1)
        mask = r > 1.0
        if np.any(mask):
            pts_new[mask] = pts_new[mask] / r[mask] * 0.999
        
        new_score = _score(pts_new)
        
        if new_score > current_score - 1e-12:
            pts = pts_new
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
            # Decrease learning rate
            lr *= 0.999
        else:
            # Revert and decrease learning rate more aggressively
            lr *= 0.95
        
        if lr < 1e-10:
            break
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_pts = None
    best_score = -1.0
    
    # Multiple restarts with different initializations
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Hexagonal lattice
            pts = _init_hexagonal()
        elif restart == 1:
            # Random uniform in disk
            pts = _init_random()
        elif restart == 2:
            # Ring-based initialization
            pts = _init_rings()
        elif restart == 3:
            # Fibonacci spiral
            pts = _init_fibonacci()
        else:
            # Perturbed versions of best found so far
            if best_pts is not None:
                pts = best_pts + np.random.randn(N, 2) * 0.05
                # Project inside unit circle
                r = np.linalg.norm(pts, axis=1)
                mask = r > 1.0
                if np.any(mask):
                    pts[mask] = pts[mask] / r[mask] * 0.999
            else:
                pts = _init_random()
        
        # Run local search with decreasing max iterations as time runs out
        time_remaining = deadline - time.time()
        if time_remaining < 10.0:
            max_iters = 1000
        elif time_remaining < 30.0:
            max_iters = 3000
        else:
            max_iters = 5000
        
        pts_opt, score = _local_search(pts, max_iters=max_iters)
        
        if score > best_score:
            best_score = score
            best_pts = pts_opt
    
    # Final polish if time allows
    if best_pts is not None and time.time() < deadline - 2.0:
        pts_polish, score_polish = _local_search(best_pts, max_iters=10000, lr=0.001, tol=1e-12)
        if score_polish > best_score:
            best_pts = pts_polish
            best_score = score_polish
    
    if best_pts is None:
        # Fallback: simple hexagonal
        best_pts = _init_hexagonal()
    
    return best_pts

def _init_hexagonal():
    """Hexagonal lattice initialization."""
    spacing = 0.24
    pts = []
    k = 5
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) < 0.9:
                pts.append([x, y])
    pts = np.array(pts)
    # Take the first N points closest to center
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:N]
    return pts[idx]

def _init_random():
    """Random uniform initialization in the disk."""
    while True:
        pts = np.random.randn(N, 2)
        r = np.linalg.norm(pts, axis=1)
        mask = r > 1.0
        if not np.any(mask):
            break
        pts[mask] = pts[mask] / r[mask] * 0.99
    return pts

def _init_rings():
    """Ring-based initialization."""
    pts = []
    # Center
    pts.append([0.0, 0.0])
    # Rings
    ring_radii = [0.2, 0.4, 0.6, 0.8]
    counts = [6, 12, 18, 24]
    for r, count in zip(ring_radii, counts):
        for i in range(count):
            angle = 2 * np.pi * i / count
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            pts.append([x, y])
    pts = np.array(pts[:N])
    return pts

def _init_fibonacci():
    """Fibonacci spiral initialization."""
    pts = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(N):
        r = 0.95 * np.sqrt((i + 0.5) / N)
        theta = i * golden_angle
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        pts.append([x, y])
    return np.array(pts)
