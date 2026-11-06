"""
Packing 200 equal circles in a unit square.
Combines smooth global optimization with bottleneck-driven local refinement.
"""
import numpy as np
import time
from scipy.spatial.distance import pdist

N = 200
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin for 150s limit
SAFETY_MARGIN = 5.0

def _remaining_time():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)

def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Wall distances
    # Distance to left wall: x + 0.5
    # Distance to right wall: 0.5 - x
    # Distance to bottom wall: y + 0.5
    # Distance to top wall: 0.5 - y
    dw = np.minimum(
        np.minimum(0.5 + centers[:, 0], 0.5 - centers[:, 0]),
        np.minimum(0.5 + centers[:, 1], 0.5 - centers[:, 1])
    )
    wall_term = np.min(dw)
    
    if np.any(dw < 0):
        return 0.0

    # Pairwise distances
    if N < 2:
        return float(wall_term)
    
    d = pdist(centers)
    min_pair = np.min(d)
    pair_term = min_pair / 2.0
    
    return min(float(wall_term), float(pair_term))

def _pdist_index_to_pair(k, n):
    """Convert pdist condensed index to (i, j) with i < j."""
    # Vectorized approach for speed
    # The k-th element in condensed distance matrix corresponds to:
    # i = floor((n - 1 - sqrt(8*k + 1))/2)
    # j = i + 1 + (k - (n-1)*i + i*(i-1)/2) ... wait, simpler logic:
    
    # Let's use a lookup or simple loop if N is small, but N=200 is medium.
    # Formula:
    # k = (n-1)*i + i*(i-1)/2 + (j - i - 1)
    # This is tricky to invert directly without search.
    # For N=200, a linear search or binary search on i is fast enough.
    
    # Approximate i:
    # k ~ n*i - i^2/2
    # i^2/2 - n*i + k = 0
    # i = (n - sqrt(n^2 - 2k))
    i_approx = int(np.floor((n - np.sqrt(n*n - 2*k))))
    
    # Verify and adjust
    for i in range(max(0, i_approx-2), min(n-1, i_approx+2)):
        start_idx = (n-1)*i + i*(i-1)//2
        end_idx = start_idx + (n-1-i)
        if start_idx <= k < end_idx:
            j_offset = k - start_idx
            j = i + 1 + j_offset
            return i, j
            
    # Fallback
    return 0, 1

def _initial_hex_grid():
    """Create a hexagonal-like initial grid."""
    # Approximate rows and cols
    # Area per circle ~ 1/200. Side ~ 1/sqrt(200) ~ 0.07
    # Hex packing density ~ 0.9069
    # Let's just use a square grid with offset for simplicity and robustness
    m = int(np.ceil(np.sqrt(N)))
    # Adjust m to get close to N
    # Try different m values to fill N
    best_centers = None
    best_error = float('inf')
    
    for m in range(12, 20):
        rows = m
        cols = m
        total = rows * cols
        if total < N:
            continue
            
        x_coords = (np.arange(cols) + 0.5) / cols - 0.5
        y_coords = (np.arange(rows) + 0.5) / rows - 0.5
        
        centers = []
        for r in range(rows):
            offset = 0.0 if r % 2 == 0 else 0.5 / cols
            for c in range(cols):
                x = (c + offset + 0.5) / (cols + (1 if r%2==1 else 0)) - 0.5
                # Simpler: just standard grid with jitter
                pass
        
        # Simpler standard grid
        x = (np.arange(m) + 0.5) / m - 0.5
        y = (np.arange(m) + 0.5) / m - 0.5
        xx, yy = np.meshgrid(x, y)
        centers = np.column_stack((xx.ravel(), yy.ravel()))[:N]
        
        # Add slight jitter to break symmetry
        rng = np.random.RandomState(42)
        centers += rng.normal(0, 0.001, centers.shape)
        centers = np.clip(centers, -0.49, 0.49)
        
        r = _compute_radius(centers)
        if r > best_error:
            best_error = r
            best_centers = centers.copy()
            
    return best_centers

def _optimize_repulsion(centers, time_limit=20.0):
    """
    Global optimization using repulsion forces.
    This helps spread out the circles globally.
    """
    start_time = time.time()
    centers = centers.copy().astype(np.float64)
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Learning rate schedule
    lr = 0.005
    decay = 0.999
    repulsion_power = 4
    
    while time.time() - start_time < time_limit and lr > 1e-6:
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, 1e-10)
        
        # Force magnitude ~ 1/d^power
        repulsion = 1.0 / (dist ** repulsion_power)
        np.fill_diagonal(repulsion, 0)
        
        force_mag = repulsion * lr
        forces += np.sum(force_mag[:, :, np.newaxis] * diff, axis=1)
        
        # Wall repulsion
        # Push away from walls
        wall_strength = 1.0
        dist_left = centers[:, 0] + 0.5
        dist_right = 0.5 - centers[:, 0]
        dist_bottom = centers[:, 1] + 0.5
        dist_top = 0.5 - centers[:, 1]
        
        # Force is proportional to 1/d^2 to be strong near walls
        forces[:, 0] += wall_strength * lr / (dist_left**2 + 1e-10)
        forces[:, 0] -= wall_strength * lr / (dist_right**2 + 1e-10)
        forces[:, 1] += wall_strength * lr / (dist_bottom**2 + 1e-10)
        forces[:, 1] -= wall_strength * lr / (dist_top**2 + 1e-10)
        
        centers += forces
        centers = np.clip(centers, -0.499, 0.499)
        
        current_r = _compute_radius(centers)
        if current_r > best_r:
            best_r = current_r
            best_centers = centers.copy()
            
        lr *= decay
        
    return best_centers, best_r

def _bottleneck_refinement(centers, time_limit=60.0):
    """
    Local optimization driven by the bottleneck constraints.
    This is the key improvement: directly addressing the limiting factors.
    """
    start_time = time.time()
    centers = centers.copy().astype(np.float64)
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    step_size = 0.002
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        # 1. Identify Bottlenecks
        dw = np.minimum(
            np.minimum(0.5 + centers[:, 0], 0.5 - centers[:, 0]),
            np.minimum(0.5 + centers[:, 1], 0.5 - centers[:, 1])
        )
        min_wall_idx = np.argmin(dw)
        min_wall_dist = dw[min_wall_idx]
        
        d = pdist(centers)
        min_pair_idx = np.argmin(d)
        min_pair_dist = d[min_pair_idx]
        
        # 2. Determine which constraint is limiting
        if min_pair_dist / 2.0 <= min_wall_dist:
            # Pair constraint is limiting
            i, j = _pdist_index_to_pair(min_pair_idx, N)
            vec = centers[i] - centers[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-9:
                # Random kick if coincident
                vec = np.random.randn(2)
                norm = np.linalg.norm(vec)
            direction = vec / norm
            move = step_size * direction
            centers[i] += move
            centers[j] -= move
        else:
            # Wall constraint is limiting
            idx = min_wall_idx
            x, y = centers[idx]
            dl = x + 0.5
            dr = 0.5 - x
            db = y + 0.5
            dt = 0.5 - y
            
            min_d = min(dl, dr, db, dt)
            if min_d == dl:
                move = np.array([step_size, 0])
            elif min_d == dr:
                move = np.array([-step_size, 0])
            elif min_d == db:
                move = np.array([0, step_size])
            else:
                move = np.array([0, -step_size])
            centers[idx] += move
        
        # 3. Clip to bounds
        centers = np.clip(centers, -0.4999, 0.4999)
        
        # 4. Evaluate
        new_r = _compute_radius(centers)
        if new_r > best_r:
            best_r = new_r
            best_centers = centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Decay step size if stuck
            if no_improve_count > 50:
                step_size *= 0.9
                if step_size < 1e-5:
                    # Try a random perturbation to escape local minimum
                    idx = np.random.randint(N)
                    centers[idx] += np.random.randn(2) * 0.005
                    centers[idx] = np.clip(centers[idx], -0.4999, 0.4999)
                    no_improve_count = 0
                    step_size = 0.001 # Reset to small step

    return best_centers, best_r

def construct():
    start_time = time.time()
    
    # Phase 1: Initial Grid
    centers = _initial_hex_grid()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Phase 2: Global Repulsion Optimization
    # Allocate ~30s for this phase
    time_left = _remaining_time()
    if time_left > 35:
        centers, r = _optimize_repulsion(centers, time_limit=min(30, time_left - 5))
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    # Phase 3: Bottleneck Refinement
    # Allocate remaining time
    time_left = _remaining_time()
    if time_left > 10:
        centers, r = _bottleneck_refinement(centers, time_limit=time_left - 5)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
            
    # Final Polish: Very small steps on bottleneck
    time_left = _remaining_time()
    if time_left > 5:
        # Run bottleneck refinement again with smaller initial step size
        # We can just call the same function, it decays step size anyway
        centers, r = _bottleneck_refinement(centers, time_limit=time_left - 2)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()

    # Ensure validity
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers

if __name__ == "__main__":
    # Quick test
    t0 = time.time()
    res = construct()
    t1 = time.time()
    r = _compute_radius(res)
    print(f"Time: {t1-t0:.2f}s, Radius: {r:.10f}")
