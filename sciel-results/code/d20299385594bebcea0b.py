"""
Strategy: Physics-based relaxation (Simulated Annealing) + Local Pairwise Search.
Instead of smooth approximations (soft-min) and L-BFGS, we model the circles as 
particles with strong repulsive forces. We use Simulated Annealing to escape 
local minima and find a dense packing, then refine with local moves.
"""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # Safety margin before 150s kill


def _elapsed():
    return time.time() - _START_TIME


def _remaining():
    return max(0.0, _DEADLINE - time.time())


def compute_radius(pts):
    """Compute the maximum common radius for given centers."""
    if len(pts) < 2:
        return 0.0
    # Efficient distance calculation
    # d[i, j] is distance between i and j
    d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    min_dist_to_edge = (1.0 - np.linalg.norm(pts, axis=1)).min()
    return min(min_pair, min_dist_to_edge)


def seed_hexagonal(n_points):
    """Generate hexagonal lattice points within unit circle as initial guess."""
    spacing = 2.0 / (np.sqrt(n_points) + 1.5)
    points = []
    k = int(1.0 / spacing) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) < n_points:
        # Fallback: random points
        rng = np.random.RandomState(42)
        angles = rng.uniform(0, 2 * np.pi, n_points)
        radii = rng.sqrt(rng.uniform(0, 0.9, n_points))
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return points
    
    # Sort by distance from center, take closest n_points
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n_points]
    return points[idx]


def simulate_annealing(pts, max_time=100.0):
    """
    Physics-based relaxation using Simulated Annealing.
    Energy is defined to penalize overlaps (repulsion) and boundary violations.
    """
    n = len(pts)
    pts = pts.copy()
    rng = np.random.RandomState(42)
    
    # Parameters
    # We want to maximize the minimum distance.
    # Let's minimize an energy function E = sum(repulsion) + sum(wall_penalty)
    # Repulsion: 1/d^2 or similar. To avoid singularity, clamp d.
    # Wall: (r_i - 1 + r)^2 if r_i > 1-r? No, we don't know r yet.
    # Actually, we can just push them inside and repel each other.
    # A common approach: minimize sum_{i<j} (1/d_ij)^k + lambda * sum_i max(0, |c_i| - (1-r_est))^2
    # But r_est changes. 
    # Better: Just maximize min distance directly? No, SA needs a scalar.
    # Let's use a potential energy that is low for well-separated points.
    # E = sum_{i<j} exp(-beta * d_ij) + wall_penalty
    # Or simply: E = sum_{i<j} (1/d_ij)^2 + wall_penalty
    
    # Let's try: E = sum_{i<j} (1/d_ij)^2 + 100 * sum_i max(0, norm(c_i) - 0.99)^2
    # This pushes points apart and keeps them inside.
    # As we anneal, we can increase the power of repulsion to make it "harder".
    
    T = 1.0
    T_min = 1e-6
    cooling_rate = 0.9995
    max_steps = int(500000) # Cap steps to avoid infinite loop if time check fails
    step_size = 0.05
    
    start_time = time.time()
    
    # Initial energy calculation
    def get_energy(p):
        # Pairwise distances
        d = np.linalg.norm(p[:, np.newaxis, :] - p[np.newaxis, :, :], axis=-1)
        np.fill_diagonal(d, 1e-9) # Avoid div by zero
        # Repulsion: sum of 1/d^4 (stronger repulsion)
        # We only care about close pairs? No, sum all.
        # To be efficient, we can just sum all. N=50 is small.
        repulsion = np.sum(1.0 / (d * d)) # O(N^2)
        
        # Wall penalty: points should be inside r=1.
        # We don't know the final radius, but let's keep them well inside.
        # If we push them to the boundary, the min distance to edge will be small.
        # The objective is min(min_pair/2, min_dist_to_edge).
        # A good heuristic is to keep them centered and packed.
        dist_to_center = np.linalg.norm(p, axis=1)
        # Penalty for being too close to edge (say > 0.95)
        # Actually, we want them to fill the space.
        # Let's use a soft wall at 1.0.
        wall_violation = np.maximum(0.0, dist_to_center - 0.98)
        wall_penalty = np.sum(wall_violation * wall_violation)
        
        return repulsion + 100.0 * wall_penalty

    E_current = get_energy(pts)
    best_pts = pts.copy()
    best_E = E_current
    
    for step in range(max_steps):
        if time.time() - start_time > max_time:
            break
        
        # Pick a random point
        i = rng.randint(n)
        
        # Propose a move: random perturbation
        # Adaptive step size?
        delta = rng.normal(0, step_size, size=2)
        
        # Check if new position is valid (inside container roughly)
        new_pos = pts[i] + delta
        if np.linalg.norm(new_pos) > 1.0:
            continue # Reject moves outside
            
        pts_new = pts.copy()
        pts_new[i] = new_pos
        
        E_new = get_energy(pts_new)
        dE = E_new - E_current
        
        if dE < 0 or rng.random() < np.exp(-dE / T):
            pts = pts_new
            E_current = E_new
            
            if E_current < best_E:
                best_E = E_current
                best_pts = pts.copy()
                
        # Cool down
        T *= cooling_rate
        if T < T_min:
            break
            
    return best_pts


def local_refinement(pts, max_time=20.0):
    """
    Local search: try to move each point slightly to increase the minimum distance.
    """
    n = len(pts)
    pts = pts.copy()
    rng = np.random.RandomState(42)
    
    def get_min_dist(p):
        d = np.linalg.norm(p[:, np.newaxis, :] - p[np.newaxis, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_dist_to_edge = (1.0 - np.linalg.norm(p, axis=1)).min()
        return min(min_pair, min_dist_to_edge)
    
    current_r = get_min_dist(pts)
    best_r = current_r
    best_pts = pts.copy()
    
    start_time = time.time()
    step_size = 0.01
    
    while time.time() - start_time < max_time:
        improved = False
        for i in range(n):
            if time.time() - start_time > max_time:
                break
                
            # Try a few random directions
            for _ in range(10):
                delta = rng.normal(0, step_size, size=2)
                new_pos = pts[i] + delta
                if np.linalg.norm(new_pos) > 1.0:
                    continue
                    
                pts_new = pts.copy()
                pts_new[i] = new_pos
                r_new = get_min_dist(pts_new)
                
                if r_new > best_r + 1e-10:
                    pts = pts_new
                    best_r = r_new
                    best_pts = pts.copy()
                    improved = True
                    break # Restart search for this point? Or continue?
                    # Let's restart the loop for this point to keep it simple
                elif r_new > current_r:
                    # Accept if it improves current, even if not best
                    pts = pts_new
                    current_r = r_new
                    improved = True
                    break
        
        if not improved:
            # Reduce step size
            step_size *= 0.8
            if step_size < 1e-6:
                break
                
    return best_pts


def construct():
    """Construct centers of N equal circles inside unit circle."""
    if _remaining() < 10.0:
        # Not enough time, return hexagonal seed
        return seed_hexagonal(N)
    
    # 1. Generate initial seed
    pts = seed_hexagonal(N)
    
    # 2. Physics-based relaxation (Simulated Annealing)
    # Allocate most time here
    sa_time = min(100.0, _remaining() - 20.0)
    if sa_time > 10.0:
        pts = simulate_annealing(pts, max_time=sa_time)
    
    # 3. Local refinement
    if _remaining() > 15.0:
        local_time = min(20.0, _remaining() - 5.0)
        if local_time > 5.0:
            pts = local_refinement(pts, max_time=local_time)
    
    # Ensure validity: check all points are inside unit circle
    dists = np.linalg.norm(pts, axis=1)
    if np.any(dists > 1.0):
        mask = dists > 1.0
        pts[mask] = pts[mask] / dists[mask, np.newaxis] * 0.999
    
    return pts
# EVOLVE-BLOCK-END
