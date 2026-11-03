"""Pack 50 equal circles in a unit square using hard-disk expansion + SLSQP with contact-graph refinement."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 50
DEADLINE = 135.0  # safety margin before 150s kill

def _true_radius(centers):
    """Compute the true max common radius for given centers."""
    # Wall radius = min of (0.5 - |coord|)
    wall_r = np.min(0.5 - np.abs(centers))
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _expand_radius(centers, r_target, max_iter=2000, tol=1e-10):
    """
    Hard-disk radius expansion: try to expand all circles to radius r_target.
    If collisions occur, resolve by pushing apart.
    Returns updated centers and success flag.
    """
    centers = centers.copy()
    for it in range(max_iter):
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Find collisions: dist < 2 * r_target
        collision_mask = dist < 2 * r_target - tol
        if not np.any(collision_mask):
            # Check wall constraints
            wall_violation = np.any(0.5 - np.abs(centers) < r_target - tol)
            if not wall_violation:
                return centers, True
            # Push away from walls
            for i in range(N):
                for k in range(2):
                    if centers[i, k] > 0 and 0.5 - centers[i, k] < r_target - tol:
                        centers[i, k] = 0.5 - r_target
                    elif centers[i, k] < 0 and 0.5 + centers[i, k] < r_target - tol:
                        centers[i, k] = -0.5 + r_target
            continue
        
        # Resolve collisions: push apart along the collision axis
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                if collision_mask[i, j]:
                    d = centers[j] - centers[i]
                    dist_ij = np.linalg.norm(d)
                    if dist_ij < 1e-12:
                        d = np.random.uniform(-1, 1, 2)
                        dist_ij = np.linalg.norm(d)
                    d = d / dist_ij
                    overlap = 2 * r_target - dist_ij
                    # Push both apart by half the overlap
                    centers[i] -= 0.5 * overlap * d
                    centers[j] += 0.5 * overlap * d
                    moved = True
        
        # Keep centers within bounds
        centers = np.clip(centers, -0.5, 0.5)
        
        if not moved:
            break
    
    # Final check
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    wall_r = np.min(0.5 - np.abs(centers))
    success = (min_pair >= 2 * r_target - tol) and (wall_r >= r_target - tol)
    return centers, success

def _slsqp_polish(centers, time_limit=30.0):
    """
    SLSQP polish: maximize min(wall_r, min_pair/2) with explicit constraints.
    """
    x0 = centers.ravel()
    start_time = time.time()
    
    def objective(x):
        c = x.reshape(N, 2)
        # Wall distances
        wall_dists = 0.5 - np.abs(c)
        min_wall = np.min(wall_dists)
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist)
        r = min(min_wall, min_pair / 2.0)
        return -r
    
    def constraint_func(x):
        c = x.reshape(N, 2)
        # Constraint: min distance >= 2 * r_current (but we don't know r in advance)
        # Instead, use a soft constraint: ensure all pairs are at least some min distance
        # Actually, for SLSQP, we can use the epigraph form:
        # Maximize t subject to:
        #   0.5 - |c_i| >= t for all i, k
        #   |c_i - c_j| >= 2*t for all i < j
        #   t >= 0
        # But SLSQP doesn't handle t as a variable easily in this form.
        # Alternative: use penalty or just maximize the min directly.
        # Let's use a different approach: minimize -min(...) with bounds on t.
        # Actually, let's just use the objective as is and rely on SLSQP's ability
        # to handle the non-smoothness.
        pass
    
    # Use SLSQP with bounds to keep centers in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def callback(xk):
        if time.time() - start_time > time_limit:
            raise RuntimeError("Time limit exceeded")
        return False
    
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False},
            callback=callback
        )
        if res.x is not None:
            return res.x.reshape(N, 2)
    except RuntimeError:
        pass
    except Exception:
        pass
    
    return centers

def _delaunay_refine(centers, r_current, time_limit=10.0):
    """
    Identify active contacts via Delaunay triangulation and refine locally.
    """
    start_time = time.time()
    centers = centers.copy()
    
    for _ in range(5):
        if time.time() - start_time > time_limit:
            break
        
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(centers)
            simplices = tri.simplices
        except Exception:
            break
        
        # Find pairs with distance close to 2*r_current (active contacts)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Active contacts: dist < 2*r_current * 1.01
        active_pairs = np.argwhere(dist < 2 * r_current * 1.01)
        active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]]
        
        if len(active_pairs) == 0:
            break
        
        # For each active pair, ensure they are exactly at distance 2*r_current
        # and move them apart if they overlap
        for i, j in active_pairs:
            d = centers[j] - centers[i]
            dist_ij = np.linalg.norm(d)
            if dist_ij < 1e-12:
                d = np.random.uniform(-1, 1, 2)
                dist_ij = np.linalg.norm(d)
            d = d / dist_ij
            if dist_ij < 2 * r_current:
                overlap = 2 * r_current - dist_ij
                centers[i] -= 0.5 * overlap * d
                centers[j] += 0.5 * overlap * d
        
        # Keep in bounds
        centers = np.clip(centers, -0.5, 0.5)
    
    return centers

def _make_hexagonal_init():
    """Create a hexagonal lattice initialization."""
    # Approximate spacing for 50 points
    # Area per point ~ 1/50 = 0.02
    # Hex area ~ spacing^2 * sqrt(3)/2
    # spacing^2 ~ 0.02 * 2 / 1.732 ~ 0.023, spacing ~ 0.15
    spacing = 0.15
    dy = spacing * np.sqrt(3) / 2
    points = []
    for j in range(10):
        for i in range(10):
            x = i * spacing - 0.5 + spacing / 2
            y = j * dy - 0.5
            if j % 2 != 0:
                x += spacing / 2
            points.append((x, y))
    points = np.array(points)
    # Take first N and center
    points = points[:N]
    points = points - np.mean(points, axis=0)
    # Scale to fit
    max_ext = np.max(np.abs(points))
    if max_ext > 0.45:
        points = points * (0.45 / max_ext)
    return points

def _make_grid_init(n_x, n_y):
    """Create a rectangular grid."""
    ticks_x = (np.arange(n_x) + 0.5) / n_x - 0.5
    ticks_y = (np.arange(n_y) + 0.5) / n_y - 0.5
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
    return grid[:N].copy()

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    # Generate initial configurations
    inits = []
    
    # 1. Hexagonal
    c_hex = _make_hexagonal_init()
    inits.append(c_hex)
    
    # 2. Grid 7x8
    c_grid1 = _make_grid_init(7, 8)
    inits.append(c_grid1)
    
    # 3. Grid 8x7
    c_grid2 = _make_grid_init(8, 7)
    inits.append(c_grid2)
    
    # 4. Random
    np.random.seed(42)
    c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
    inits.append(c_rand)
    
    # Process each initial configuration
    for i, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        time_left = DEADLINE - 15 - (time.time() - start_time)
        
        # Step 1: Hard-disk expansion to find a good radius
        # Estimate initial radius
        r_est = 0.07  # start with known best
        
        # Try to expand from a smaller radius
        r_small = 0.05
        centers_expanded, success = _expand_radius(init, r_small, max_iter=500)
        
        if not success:
            # If expansion failed, just use the initial
            centers_expanded = init.copy()
        
        # Now try to increase radius incrementally
        r_target = r_small
        max_r_attempts = 5
        for _ in range(max_r_attempts):
            if time.time() - start_time > DEADLINE - 15:
                break
            r_target += 0.005
            centers_new, success = _expand_radius(centers_expanded, r_target, max_iter=1000)
            if success:
                centers_expanded = centers_new
            else:
                # Try to recover
                centers_expanded, _ = _expand_radius(centers_expanded, r_target - 0.002, max_iter=500)
                break
        
        # Step 2: SLSQP polish
        refined = _slsqp_polish(centers_expanded, time_limit=min(20.0, time_left * 0.6))
        
        # Step 3: Delaunay refinement
        r_current = _true_radius(refined)
        refined = _delaunay_refine(refined, r_current, time_limit=min(5.0, time_left * 0.2))
        
        r = _true_radius(refined)
        
        if r > best_r:
            best_r = r
            best_centers = refined.copy()
    
    # If time remains, do additional random restarts
    while time.time() - start_time < DEADLINE - 5:
        seed = int(time.time() * 1000) % 100000
        np.random.seed(seed)
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        time_left = DEADLINE - 5 - (time.time() - start_time)
        if time_left < 10:
            break
        
        # Quick expansion
        r_small = 0.05
        centers_expanded, _ = _expand_radius(c_rand, r_small, max_iter=300)
        
        # SLSQP polish
        refined = _slsqp_polish(centers_expanded, time_limit=time_left * 0.7)
        
        r = _true_radius(refined)
        
        if r > best_r:
            best_r = r
            best_centers = refined.copy()
    
    if best_centers is None:
        # Fallback: return a grid
        best_centers = _make_grid_init(7, 8)
    
    return best_centers
