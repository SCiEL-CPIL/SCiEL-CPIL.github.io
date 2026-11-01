"""Rewritten packing program for 50 equal circles in a unit square."""
import time
import numpy as np
from scipy.spatial.distance import cdist

N = 50
SIDE = 1.0
HALF = 0.5


def _compute_score(centers):
    """Return the packing radius for a set of centers."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary distance
    d_boundary = np.min(np.min(
        np.array([HALF - centers[:, 0], HALF - centers[:, 1],
                  centers[:, 0] + HALF, centers[:, 1] + HALF]), axis=1))
    if d_boundary <= 0:
        return 0.0
    # Pairwise distances
    D = cdist(centers, centers)
    np.fill_diagonal(D, np.inf)
    d_pair = np.min(D) / 2.0
    return min(d_boundary, d_pair)


def _initial_grid():
    """Create a 7x7 grid, take 50 points, and jitter slightly."""
    m = 7
    # Use 7x7 = 49, need 50, so use a different approach
    # Actually let's use a more structured initial guess
    # 7 columns x 7 rows = 49, add 1 more
    # Better: use a hexagonal-like packing initial guess
    
    # Start with a 7x7 grid (49 points) plus 1 extra
    # Actually, let's use a smarter initial layout
    # 50 points: 7 rows, some with 7, some with 6
    # 7*7 = 49, need 1 more. Use 8 rows? 8*7=56, too many.
    # Let's use 7 columns: 50/7 ≈ 7.14, so 7 rows of 7 = 49, plus 1
    
    # Simpler: use a hexagonal packing pattern
    # For 50 circles, a good initial guess is a hexagonal arrangement
    
    # Use a grid with slight hexagonal offset
    m = 7
    xs = (np.arange(m) + 0.5) / m - 0.5
    ys = (np.arange(m) + 0.5) / m - 0.5
    
    # Create 7x7 grid
    grid = np.array([(x, y) for y in ys for x in xs])
    
    # We need 50 points, have 49. Add one more by splitting or using a different layout
    # Let's use a 8x7 grid and take the first 50
    m_cols = 7
    m_rows = 8
    xs2 = (np.arange(m_cols) + 0.5) / m_cols - 0.5
    ys2 = (np.arange(m_rows) + 0.5) / m_rows - 0.5
    grid2 = np.array([(x, y) for y in ys2 for x in xs2])
    
    # Take first 50
    return grid2[:N].copy()


def construct():
    """Construct centers for 50 equal circles in unit square."""
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s limit
    
    # Initial guess: 8x7 grid, take 50
    centers = _initial_grid()
    best_score = _compute_score(centers)
    best_centers = centers.copy()
    
    # Simulated annealing / local optimization
    # We'll use a simple approach: iteratively move points to increase the minimum distance
    
    # Compute current radius
    r = best_score
    if r <= 0:
        r = 0.01
    
    # Optimization loop
    np.random.seed(42)
    
    # Temperature schedule
    T_start = 0.05
    T_end = 0.001
    n_iter = 50000  # Will be limited by time
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        # Temperature decay
        frac = it / n_iter
        T = T_start * (T_end / T_start) ** frac
        
        # Randomly pick a point and try to move it
        i = np.random.randint(N)
        old_pos = centers[i].copy()
        
        # Propose a move: random perturbation
        scale = T * 0.1
        new_pos = old_pos + np.random.randn(2) * scale
        
        # Keep within bounds (with some margin)
        new_pos[0] = np.clip(new_pos[0], -HALF + r * 0.5, HALF - r * 0.5)
        new_pos[1] = np.clip(new_pos[1], -HALF + r * 0.5, HALF - r * 0.5)
        
        centers[i] = new_pos
        
        new_score = _compute_score(centers)
        
        if new_score >= best_score:
            best_score = new_score
            best_centers = centers.copy()
            # Accept
            if new_score > r:
                r = new_score
        else:
            # Metropolis acceptance
            delta = new_score - best_score
            if delta < 0 and np.random.rand() < np.exp(delta / max(T, 1e-10)):
                # Accept worse move
                pass
            else:
                # Reject
                centers[i] = old_pos
    
    # Final polish: try to improve by moving each point optimally
    for polish_iter in range(5):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            
            # Try small moves in different directions
            for dx, dy in [(0.001, 0), (-0.001, 0), (0, 0.001), (0, -0.001),
                           (0.001, 0.001), (-0.001, -0.001), (0.001, -0.001), (-0.001, 0.001)]:
                old_pos = best_centers[i].copy()
                new_pos = old_pos + np.array([dx, dy])
                new_pos[0] = np.clip(new_pos[0], -HALF + 1e-6, HALF - 1e-6)
                new_pos[1] = np.clip(new_pos[1], -HALF + 1e-6, HALF - 1e-6)
                
                best_centers[i] = new_pos
                s = _compute_score(best_centers)
                
                if s > best_score:
                    best_score = s
                    improved = True
                else:
                    best_centers[i] = old_pos
    
        if not improved:
            break
    
    return best_centers
