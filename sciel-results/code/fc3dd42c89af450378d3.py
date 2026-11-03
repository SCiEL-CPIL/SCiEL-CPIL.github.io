"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import itertools
import math

N = 13

# EVOLVE-BLOCK-START

def _min_triangle_area_fast(pts):
    """
    Fast vectorized computation of the minimum triangle area.
    Uses broadcasting to compute all C(N,3) areas at once.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
        
    # Generate indices for all unique triples (i, j, k) with i < j < k
    # We can use itertools.combinations for indices, then vectorize the math
    # Or better, use np.triu_indices on a 3D structure? 
    # Simplest robust vectorization for N=13:
    # Precompute index combinations outside? No, let's just do it inside for simplicity, 
    # but the math will be vectorized.
    
    # Actually, for N=13, 286 triples is very small. 
    # We can create a (286, 3) array of indices.
    indices = list(itertools.combinations(range(N), 3))
    if not indices:
        return 0.0
        
    # Create an array of shape (num_triples, 3, 2)
    idx_array = np.array(indices, dtype=np.int32) # (286, 3)
    p = pts[idx_array] # (286, 3, 2)
    
    # p[:, 0] is A, p[:, 1] is B, p[:, 2] is C
    A = p[:, 0, :]
    B = p[:, 1, :]
    C = p[:, 2, :]
    
    # Area = 0.5 * | (B-A) x (C-A) |
    # In 2D, cross product is scalar: (Bx-Ax)*(Cy-Ay) - (By-Ay)*(Cx-Ax)
    AB = B - A
    AC = C - A
    cross = AB[:, 0] * AC[:, 1] - AB[:, 1] * AC[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return float(np.min(areas))

def _hull_area(pts):
    """Compute the area of the convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(pts):
    """Compute the score: min triangle area / hull area."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 0.0
    min_tri = _min_triangle_area_fast(pts)
    return min_tri / hull_a

def _neg_score(flat_pts):
    """Negative score for minimization."""
    pts = flat_pts.reshape(N, 2)
    # Clip to a reasonable box to avoid degenerate hulls or huge coordinates
    # The problem is invariant to translation and scaling, but we normalize by hull area.
    # However, DE needs bounded domains. Let's bound to [-1, 1] or [0, 1].
    # Since we normalize by hull area, the absolute scale doesn't matter for the score,
    # but numerical stability might. [0, 1] is safe.
    pts = np.clip(pts, -1.0, 1.0)
    s = _score(pts)
    return -s

def _differential_evolution_search(bounds, deadline, max_iter=100, popsize=15, seed=42):
    """
    Run Differential Evolution to find a good global candidate.
    """
    # Define bounds for each coordinate
    # We can use a tighter bound if we know the optimal configuration is "compact"
    # but [-1, 1] is safe.
    
    try:
        result = differential_evolution(
            _neg_score,
            bounds=bounds,
            maxiter=max_iter,
            popsize=popsize,
            tol=1e-5,
            seed=seed,
            polish=False, # We will do our own polish
            callback=None, # We'll check time manually if needed, but DE is fast enough for small N
            init='latinhypercube'
        )
        
        # DE does not have a direct deadline parameter in older scipy versions, 
        # but it stops on convergence. 
        # To be safe with time, we rely on the fact that N=13 is small.
        # If it takes too long, we might need to wrap it, but usually it's fast.
        
        best_flat = result.x
        best_score = -result.fun
        return best_flat.reshape(N, 2), best_score
        
    except Exception as e:
        print(f"DE failed: {e}")
        return None, 0.0

def _polish_sa(pts_init, deadline, max_iter=20000, T0=0.01, cooling=0.999):
    """
    Fast Simulated Annealing for local polishing.
    """
    pts = pts_init.copy().astype(np.float64)
    best_pts = pts.copy()
    best_score = _score(pts)
    current_score = best_score
    T = T0
    
    rng = np.random.RandomState(12345)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Move one point
        idx = rng.randint(N)
        # Small step size
        step = rng.normal(0, T, size=2)
        new_pts = pts.copy()
        new_pts[idx] += step
        new_pts[idx] = np.clip(new_pts[idx], -1.0, 1.0)
        
        new_score = _score(new_pts)
        
        # Accept if better or with probability exp(delta/T)
        if new_score >= current_score:
            pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
        else:
            delta = new_score - current_score
            if delta > 0: # Should not happen if new_score < current_score
                pass
            else:
                prob = np.exp(delta / max(T, 1e-10))
                if rng.random() < prob:
                    pts = new_pts
                    current_score = new_score
        
        T *= cooling
        if T < 1e-8:
            T = 1e-8
            
    return best_pts, best_score

def _initial_grid(n, seed=42):
    """Generate a reasonable initial grid."""
    rng = np.random.RandomState(seed)
    # 4x4 grid = 16 points, take 13
    side = 4
    points = []
    for i in range(side):
        for j in range(side):
            if len(points) >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            # Add small jitter
            x += rng.uniform(-0.05, 0.05)
            y += rng.uniform(-0.05, 0.05)
            points.append([x, y])
    return np.array(points[:n])

def construct():
    start_time = time.time()
    # Safety margin: 150s limit, use 140s
    deadline = start_time + 140.0
    
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Differential Evolution
    # Bounds: [-1, 1] for x and y
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    try:
        # Run DE with a moderate population
        de_pts, de_score = _differential_evolution_search(bounds, deadline, max_iter=50, popsize=20, seed=42)
        if de_pts is not None and de_score > best_score:
            best_score = de_score
            best_pts = de_pts
    except Exception:
        pass
    
    if time.time() > deadline:
        if best_pts is not None:
            return best_pts
        return _initial_grid(N)
    
    # Phase 2: Polish with Simulated Annealing
    if best_pts is not None:
        try:
            polished_pts, polished_score = _polish_sa(best_pts, deadline, max_iter=50000, T0=0.05, cooling=0.9995)
            if polished_score > best_score:
                best_score = polished_score
                best_pts = polished_pts
        except Exception:
            pass
    
    # Phase 3: If time remains, try another DE restart or SA restart
    if time.time() < deadline - 10.0:
        # Try a different initial guess for SA
        init_pts = _initial_grid(N, seed=123)
        try:
            sa_pts, sa_score = _polish_sa(init_pts, deadline, max_iter=50000, T0=0.1, cooling=0.999)
            if sa_score > best_score:
                best_score = sa_score
                best_pts = sa_pts
        except Exception:
            pass
            
    if best_pts is not None:
        return best_pts
    
    return _initial_grid(N)

# EVOLVE-BLOCK-END
