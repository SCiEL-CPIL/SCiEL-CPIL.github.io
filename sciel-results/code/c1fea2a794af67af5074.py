"""
Strategy: Simulated Annealing with k-d tree acceleration + Gradient Polish.
Instead of pure gradient ascent on a smooth soft-min objective (which is prone to
local minima in the high-dimensional landscape of 300 points), we use a stochastic
approach:
1. Initialize with a hexagonal lattice.
2. Use Simulated Annealing (SA) to explore the configuration space.
   - Randomly select a point and a random number of neighbors (via k-d tree).
   - Move them together by a random displacement.
   - Accept moves based on the change in the packing radius (with probability 
     exp(delta/T) for worsening moves).
   - This allows escaping local minima by accepting "bad" moves early on.
3. After SA converges, use a few iterations of gradient ascent on the soft-min 
   objective to polish the result to high precision.
"""
import numpy as np
import time
import math
from scipy.spatial import cKDTree

N = 300
DEADLINE = 145.0  # Safety margin from 150s limit

def _hex_lattice_init():
    """Generate a hexagonal lattice of points inside the unit disk, return first N."""
    best_points = None
    # Search for a lattice spacing that fits at least N points
    # Optimal packing density is pi/sqrt(12) ~ 0.9069
    # Area = pi. 300 circles of radius r. Area = 300 * pi * r^2.
    # 300 * r^2 ~ 0.9069 -> r^2 ~ 0.00302 -> r ~ 0.055
    # Diameter ~ 0.11. Lattice spacing ~ 0.11.
    for s in np.linspace(0.09, 0.13, 20):
        k = int(1.0 / s) + 3
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * math.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    points.append((x, y))
        points = np.array(points)
        if len(points) >= N:
            # Keep the N closest to center for a compact packing
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind='stable')[:N]
            pts = points[idx]
            if best_points is None:
                best_points = pts
            # If we found a good density, stop
            if s > 0.10:
                break
    if best_points is not None and len(best_points) == N:
        return best_points
    
    # Fallback
    s = 0.05
    k = int(1.0 / s) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * math.sqrt(3.0) / 2.0
            if x * x + y * y <= 1.0:
                points.append((x, y))
    points = np.array(points)
    r = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(r, kind='stable')[:N]
    return points[idx]

def _score(points):
    """Compute the maximum radius r for given centers."""
    # Distance to boundary
    dist_boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(dist_boundary)
    
    # Pairwise distances
    # For speed, we can use cKDTree here too, but for 300 points, O(N^2) is ~90k ops, which is fast.
    # However, since we are calling this frequently in SA, let's optimize.
    # Actually, 300x300 distance matrix is 90k entries. Computing it is very fast in numpy.
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_boundary, min_pair / 2.0)

def _sa_optimize(points, time_limit):
    """
    Simulated Annealing optimization.
    """
    start_time = time.time()
    best_points = points.copy()
    best_score = _score(points)
    
    # Initial temperature
    # Typical improvement is small. Let's start with T such that exp(-1/T) is not too small.
    # If delta_r is ~0.001, T ~ 0.001 gives 37% acceptance.
    T = 0.005
    T_min = 1e-6
    
    # Cooling schedule
    # Linear or exponential? Exponential is standard.
    # Let's estimate number of iterations we can do.
    # Each iter: KDTree query (fast), score calc (O(N^2) ~ 90k ops), random gen.
    # Let's assume ~1000 iters/sec. In 100s, ~100k iters.
    # We will just run until time limit.
    
    current_points = points.copy()
    current_score = best_score
    
    # Pre-allocate random numbers? No, numpy is fast enough.
    
    # We need a KDTree for neighbor selection.
    # Rebuilding KDTree every iteration is too slow.
    # We can update it, but scipy's KDTree doesn't support efficient updates.
    # Alternative: Just pick a random point and its k nearest neighbors from the *current* set?
    # That requires building a tree or O(N^2) search.
    # For N=300, O(N^2) search for k=10 neighbors is 90k distance calcs. That's fine.
    # But doing it every iter adds overhead.
    # Let's stick to picking a random point and moving it, or a random point and a few random others?
    # Moving a single point is often too local.
    # Moving a cluster is better.
    
    # Strategy:
    # 1. Pick a random point i.
    # 2. Find its k nearest neighbors (k=5-10).
    # 3. Move i and its neighbors by a random vector.
    # To find neighbors efficiently without rebuilding tree:
    # Since N is small, we can just compute distances to all other points and pick the smallest k.
    # This is O(N) per iteration (plus the score calc O(N^2)).
    
    k_neighbors = 5
    
    while time.time() - start_time < time_limit:
        # Select a random point
        i = np.random.randint(N)
        
        # Find k nearest neighbors to i
        # dists_to_i = np.hypot(current_points[:,0] - current_points[i,0], current_points[:,1] - current_points[i,1])
        # mask = np.ones(N, dtype=bool); mask[i]=False
        # idx = np.argpartition(dists_to_i, k_neighbors)[:k_neighbors]
        # neighbors = np.where(dists_to_i < dists_to_i[idx][-1])[0] # might be more than k
        
        # Simpler: just pick i and maybe 1-2 random others?
        # Let's try moving just i first. If it stagnates, we might need clusters.
        # Actually, for packing, moving a single point is often sufficient to relieve local pressure.
        # Let's start with single point moves to keep it simple and fast.
        
        idx_to_move = np.array([i])
        
        # Random displacement
        # Scale of displacement should be related to current radius.
        # r ~ 0.05. Displacement ~ 0.01 - 0.05.
        # Let's use a scale that decays with T? Or fixed?
        # Fixed scale is simpler.
        scale = 0.02
        dx = np.random.normal(0, scale, size=len(idx_to_move))
        dy = np.random.normal(0, scale, size=len(idx_to_move))
        
        new_points = current_points.copy()
        new_points[idx_to_move, 0] += dx
        new_points[idx_to_move, 1] += dy
        
        # Check boundary constraint strictly? 
        # If a point goes outside, the score drops significantly.
        # We can just let the score function handle it.
        
        new_score = _score(new_points)
        
        delta = new_score - current_score
        
        if delta > 0:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            # Metropolis criterion
            # Accept with probability exp(delta / T)
            if np.random.random() < np.exp(delta / T):
                current_points = new_points
                current_score = new_score
        
        # Cool down
        # Exponential cooling
        T *= 0.9995
        if T < T_min:
            T = T_min
            # If T is too low, we are stuck in local minimum.
            # We can restart with higher T? Or just continue.
            # Let's add a small probability to restart T.
            if np.random.random() < 0.01:
                T = 0.001

    return best_points

def _polish(points, time_limit):
    """
    Gradient polish using soft-min objective.
    """
    start_time = time.time()
    best_points = points.copy()
    best_score = _score(points)
    
    beta = 500.0
    lr = 1e-4
    
    current_points = points.copy()
    
    # Simple gradient ascent
    # We reuse the logic from the previous program but simplified.
    
    def _objective_and_grad(pts, beta):
        n = len(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        mask = ~np.eye(n, dtype=bool)
        dists_safe = np.where(dists < 1e-10, 1e-10, dists)
        
        w_pair = np.exp(-beta * dists_safe / 2.0)
        w_pair[~mask] = 0.0
        
        Z_pair = np.sum(w_pair) / 2.0
        
        r = np.hypot(pts[:, 0], pts[:, 1])
        r_safe = np.where(r < 1e-10, 1e-10, r)
        g = 1.0 - r
        
        w_bound = np.exp(-beta * g)
        Z_bound = np.sum(w_bound)
        
        Z = Z_pair + Z_bound
        
        if Z < 1e-15:
            return 0.0, np.zeros_like(pts)
        
        unit_vec = diff / dists_safe[..., np.newaxis]
        grad_pair = np.einsum('ij,ijc->c', w_pair, unit_vec) * 0.5
        
        unit_r = pts / r_safe[:, np.newaxis]
        grad_bound = w_bound[:, np.newaxis] * unit_r
        
        grad = (grad_pair + grad_bound) / Z
        
        obj = -1.0 / beta * np.log(Z)
        
        return obj, grad

    for it in range(1000):
        if time.time() - start_time > time_limit:
            break
        
        obj, grad = _objective_and_grad(current_points, beta)
        
        current_points += lr * grad
        
        # Project inside
        r = np.hypot(current_points[:, 0], current_points[:, 1])
        mask = r > 1.0
        if np.any(mask):
            current_points[mask] = current_points[mask] / r[mask, np.newaxis] * (1.0 - 1e-10)
        
        if it % 10 == 0:
            s = _score(current_points)
            if s > best_score:
                best_score = s
                best_points = current_points.copy()
                
        # Decrease learning rate
        lr *= 0.99

    return best_points

def construct():
    start_time = time.time()
    
    # Generate initial hexagonal lattice
    points = _hex_lattice_init()
    if points is None or len(points) != N:
        points = np.zeros((N, 2))
        for i in range(N):
            angle = 2 * np.pi * i / N
            r = 0.5
            points[i, 0] = r * np.cos(angle)
            points[i, 1] = r * np.sin(angle)
    
    # Phase 1: Simulated Annealing
    # Give it 120 seconds
    sa_time = 120.0
    points = _sa_optimize(points, sa_time)
    
    # Phase 2: Gradient Polish
    # Give it the remaining time, capped at 20s
    remaining = DEADLINE - (time.time() - start_time)
    polish_time = min(20.0, max(0.0, remaining))
    
    if polish_time > 0:
        points = _polish(points, polish_time)
    
    # Final validity check
    r = np.hypot(points[:, 0], points[:, 1])
    if np.any(r > 1.0):
        mask = r > 1.0
        points[mask] = points[mask] / r[mask, np.newaxis] * (1.0 - 1e-10)
        
    return points
