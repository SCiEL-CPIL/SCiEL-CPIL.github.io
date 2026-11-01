"""Physics-based packing algorithm for 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def _hex_init(N, seed=42):
    """Generate hexagonal lattice points with random perturbation, select N closest to origin."""
    rng = np.random.default_rng(seed)
    s = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
    s *= 0.92
    
    best_points = None
    for attempt in range(20):
        k = int(1.0 / s) + 5
        pts = []
        for j in range(-k, k + 1):
            y = j * s * np.sqrt(3.0) / 2.0
            if abs(y) > 1.0:
                continue
            x_offset = (j % 2) * s / 2.0
            for i in range(-k, k + 1):
                x = (i + x_offset) * s
                r = np.hypot(x, y)
                if r <= 1.0 - s * 0.5:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            dists = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(dists, kind='stable')[:N]
            selected = pts[idx].copy()
            # Add small random perturbation to break symmetry
            selected += rng.normal(0, s * 0.1, size=selected.shape)
            if best_points is None:
                best_points = selected
        s *= 0.97
    return best_points if best_points is not None else np.random.randn(N, 2) * 0.5

def _compute_radius(pts):
    """Compute the feasible radius for a set of points."""
    n = len(pts)
    if n < 2:
        min_pair = 1.0
    else:
        tree = cKDTree(pts)
        pairs = tree.query_pairs(r=1.0)
        if len(pairs) == 0:
            min_pair = 1.0
        else:
            idx1 = np.array([p[0] for p in pairs])
            idx2 = np.array([p[1] for p in pairs])
            dists = np.linalg.norm(pts[idx1] - pts[idx2], axis=1)
            min_pair = np.min(dists) / 2.0
    
    dists_boundary = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(dists_boundary)
    
    return min(min_pair, min_boundary)

def _physics_simulate(pts, max_iter=5000, dt=0.001, cooling=0.999, deadline=None):
    """
    Physics-based simulation where circles repel each other and the container wall.
    Uses a spring-like force model to maximize minimum distance.
    """
    n = len(pts)
    pts = pts.copy()
    
    # Find initial minimum distance to set force scale
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=0.3)
    if len(pairs) > 0:
        idx1 = np.array([p[0] for p in pairs])
        idx2 = np.array([p[1] for p in pairs])
        dists = np.linalg.norm(pts[idx1] - pts[idx2], axis=1)
        min_d = np.min(dists)
    else:
        min_d = 0.05
    
    # Target spacing
    target = 0.055
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        tree = cKDTree(pts)
        pairs = tree.query_pairs(r=target * 3)
        
        for i, j in pairs:
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                d = np.array([1e-10, 0.0])
                dist = 1e-10
            # Spring force: push apart if too close, weak attraction if far
            if dist < target:
                force_mag = (target - dist) / target * 10.0
                forces[i] += (d / dist) * force_mag
                forces[j] -= (d / dist) * force_mag
            elif dist < target * 2:
                # Weak repulsion for stability
                force_mag = (target * 2 - dist) / target * 0.5
                forces[i] += (d / dist) * force_mag
                forces[j] -= (d / dist) * force_mag
        
        # Wall repulsion
        for i in range(n):
            r = np.linalg.norm(pts[i])
            if r > 1e-10:
                dist_to_wall = 1.0 - r
                if dist_to_wall < target * 0.5:
                    force_mag = (target * 0.5 - dist_to_wall) / target * 20.0
                    forces[i] -= (pts[i] / r) * force_mag
        
        # Apply forces with damping
        pts += forces * dt
        dt *= cooling
        
        # Project inside unit circle
        dists = np.hypot(pts[:, 0], pts[:, 1])
        mask = dists > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / dists[mask] * 0.9999
    
    return pts

def _local_refine(pts, max_iter=2000, deadline=None):
    """
    Local refinement using gradient ascent on minimum distance.
    Focus on the most constrained pairs.
    """
    n = len(pts)
    pts = pts.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Find minimum distance pair
        tree = cKDTree(pts)
        pairs = tree.query_pairs(r=0.15)
        
        if len(pairs) == 0:
            break
        
        idx1 = np.array([p[0] for p in pairs])
        idx2 = np.array([p[1] for p in pairs])
        dists = np.linalg.norm(pts[idx1] - pts[idx2], axis=1)
        min_idx = np.argmin(dists)
        
        i, j = pairs[min_idx]
        d = pts[i] - pts[j]
        dist = np.linalg.norm(d)
        
        if dist < 1e-10:
            continue
        
        # Move apart
        step = 0.0005
        pts[i] += (d / dist) * step
        pts[j] -= (d / dist) * step
        
        # Check boundary
        for idx in [i, j]:
            r = np.linalg.norm(pts[idx])
            if r > 1.0:
                pts[idx] = pts[idx] / r * 0.9999
        
        # Also check all points for boundary
        dists_all = np.hypot(pts[:, 0], pts[:, 1])
        mask = dists_all > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / dists_all[mask] * 0.9999
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # Try multiple random seeds for hex initialization
    best_pts = None
    best_r = -np.inf
    
    for seed in [42, 123, 777, 2021, 999]:
        if time.time() > deadline - 20.0:
            break
        
        pts = _hex_init(N, seed=seed)
        if pts is None or len(pts) != N:
            continue
        
        # Physics simulation
        pts_sim = _physics_simulate(pts, max_iter=3000, dt=0.002, cooling=0.9995, deadline=deadline - 15.0)
        pts_sim = _local_refine(pts_sim, max_iter=1000, deadline=deadline - 10.0)
        
        r = _compute_radius(pts_sim)
        if r > best_r:
            best_r = r
            best_pts = pts_sim.copy()
    
    if best_pts is None:
        # Fallback
        best_pts = _hex_init(N, seed=42)
    
    # Final local refinement
    best_pts = _local_refine(best_pts, max_iter=2000, deadline=deadline - 5.0)
    final_r = _compute_radius(best_pts)
    
    # Ensure validity
    dists = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = dists > 1.0
    if np.any(mask):
        best_pts[mask] = best_pts[mask] / dists[mask] * 0.9999
    
    return best_pts
# EVOLVE-BLOCK-END
