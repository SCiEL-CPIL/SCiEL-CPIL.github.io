"""Improved packing: Lloyd's Algorithm (Voronoi) + Smooth Surrogate Polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 50

def _score(centers):
    """Compute the maximum common radius for given centers (exact, for evaluation)."""
    centers = np.asarray(centers)
    wall_min = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_min)
    n = len(centers)
    if n < 2:
        return r_wall
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    return min(r_wall, r_pair)

def _soft_min(values, beta=50.0):
    """Smooth approximation to min via log-sum-exp (negative log-sum-exp)."""
    values = np.asarray(values)
    m = np.min(values)
    exponent = -beta * (values - m)
    exponent = np.clip(exponent, -700, 700)
    log_sum = np.log(np.sum(np.exp(exponent)))
    return m - log_sum / beta

def _smooth_objective(x, beta=50.0):
    """
    Smooth surrogate: minimize -soft_min(all constraints).
    """
    c = x.reshape(-1, 3)
    wall_constraints = 0.5 - np.abs(c).ravel()
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_constraints = dists / 2.0
    pair_constraints = pair_constraints[np.triu_indices_from(pair_constraints, k=1)]
    all_constraints = np.concatenate([wall_constraints, pair_constraints])
    s = _soft_min(all_constraints, beta)
    out_of_bounds = np.maximum(0, np.abs(c) - 0.4999)
    penalty = 10.0 * np.sum(out_of_bounds ** 2)
    return -s + penalty

def _voronoi_cell_centroid(clipped_points, cell_vertices, cube_min=-0.5, cube_max=0.5):
    """
    Compute the centroid of a Voronoi cell clipped to the cube.
    cell_vertices: array of vertices defining the cell (may include points outside cube).
    We clip to cube and compute centroid.
    """
    # Filter vertices to those inside or on the cube boundary
    inside = (clipped_points >= cube_min - 1e-10) & (clipped_points <= cube_max + 1e-10)
    if not np.all(inside):
        # Clip vertices to cube
        clipped = np.clip(clipped_points, cube_min, cube_max)
        # For simplicity, use the clipped points; a proper clipping would require
        # polygon clipping, but for centroid approximation this works well
        vertices = clipped
    else:
        vertices = clipped_points
    
    if len(vertices) < 3:
        return None
    
    # Compute centroid as mean of vertices (approximation)
    # A more accurate method would use polygon area-weighted centroid,
    # but mean of vertices is a good approximation for near-convex cells
    centroid = np.mean(vertices, axis=0)
    
    # Ensure centroid is inside cube
    centroid = np.clip(centroid, cube_min + 1e-10, cube_max - 1e-10)
    return centroid

def _lloyd_iteration(centers, max_iter=100, deadline=None, alpha=1.0):
    """
    Lloyd's algorithm: iteratively move each center to the centroid of its
    Voronoi cell (clipped to the cube). This equalizes clearance distances.
    """
    c = centers.copy()
    cube_min = -0.5
    cube_max = 0.5
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        try:
            # Compute Voronoi diagram
            vor = Voronoi(c)
        except Exception:
            break
        
        new_c = c.copy()
        valid_count = 0
        
        for i in range(len(c)):
            # Get the region for point i
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if len(region) == 0:
                continue
            
            # Get vertices of this cell
            vertex_indices = region
            # Filter out -1 indices (unbounded regions)
            vertex_indices = [v for v in vertex_indices if v >= 0]
            
            if len(vertex_indices) < 3:
                continue
            
            vertices = vor.vertices[vertex_indices]
            
            # Compute centroid of clipped cell
            centroid = _voronoi_cell_centroid(vertices, vertices, cube_min, cube_max)
            
            if centroid is not None:
                # Move toward centroid with damping
                new_c[i] = (1 - alpha) * c[i] + alpha * centroid
                valid_count += 1
        
        # Clip to valid range
        new_c = np.clip(new_c, cube_min + 1e-6, cube_max - 1e-6)
        
        # Check for convergence
        displacement = np.max(np.linalg.norm(new_c - c, axis=1))
        c = new_c
        
        if displacement < 1e-10:
            break
    
    return c

def _initial_grid_jittered():
    """Jittered grid initialization to break symmetry."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(grid))
    centers = grid[perm[:N]].copy()
    # Add small jitter
    jitter = rng.uniform(-0.01, 0.01, size=centers.shape)
    centers += jitter
    # Clip to valid range
    centers = np.clip(centers, -0.49, 0.49)
    return centers

def _initial_fcc_like():
    """FCC-inspired initialization: place points in a face-centered cubic pattern."""
    # FCC lattice in a cube: corners + face centers
    # Scale to fit in [-0.5, 0.5]^3
    # Use a 3x3x3 grid with FCC offsets
    
    centers = []
    # Grid spacing
    spacing = 1.0 / 3.0
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                x = -0.5 + i * spacing
                y = -0.5 + j * spacing
                z = -0.5 + k * spacing
                centers.append([x, y, z])
    
    # Add face-center-like offsets
    rng = np.random.RandomState(42)
    centers = np.array(centers[:N])
    jitter = rng.uniform(-0.02, 0.02, size=centers.shape)
    centers += jitter
    centers = np.clip(centers, -0.48, 0.48)
    return centers

def construct():
    t_start = time.time()
    deadline = t_start + 135  # Safety margin before 150s limit
    
    best_centers = None
    best_score = -np.inf
    
    # 1. Generate initial candidates
    inits = [
        _initial_grid_jittered(),
        _initial_fcc_like()
    ]
    
    # Add a few random inits for diversity
    rng = np.random.RandomState(123)
    for _ in range(2):
        rand_centers = rng.uniform(-0.4, 0.4, size=(N, 3))
        inits.append(rand_centers)
        
    for init_idx, init in enumerate(inits):
        if time.time() > deadline - 25:
            break
            
        cur = init.copy()
        cur_score = _score(cur)
        
        if best_centers is None or cur_score > best_score:
            best_centers = cur.copy()
            best_score = cur_score
            
        # 2. Lloyd's Algorithm (Voronoi-based) - Structural Change
        # This directly optimizes minimum clearance by equalizing influence regions
        if time.time() > deadline - 20:
            continue
        
        lloyd_result = _lloyd_iteration(cur, max_iter=150, deadline=deadline - 15, alpha=0.8)
        lloyd_score = _score(lloyd_result)
        
        if lloyd_score > best_score:
            best_centers = lloyd_result.copy()
            best_score = lloyd_score
            cur = lloyd_result.copy()
        
        # 3. Refinement: Smooth Surrogate with L-BFGS-B
        if time.time() > deadline - 12:
            continue
        
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 8:
                break
            try:
                res = minimize(
                    _smooth_objective,
                    cur.ravel(),
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * 3 * N,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
                )
                new_centers = res.x.reshape(-1, 3)
                new_score = _score(new_centers)
                if new_score > best_score:
                    best_centers = new_centers.copy()
                    best_score = new_score
                    cur = new_centers.copy()
            except Exception:
                pass
        
        # 4. Final Polish: Exact Coordinate Descent
        if time.time() > deadline - 3:
            continue
        
        cur = best_centers.copy()
        improved = True
        step = 0.001
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(N):
                if time.time() > deadline - 1:
                    break
                for coord in range(3):
                    for delta in [step, -step, step*0.5, -step*0.5, step*0.25, -step*0.25]:
                        new_c = cur.copy()
                        new_c[i, coord] += delta
                        if np.all(np.abs(new_c) <= 0.5):
                            s = _score(new_c)
                            if s > best_score + 1e-12:
                                cur = new_c
                                best_centers = new_c.copy()
                                best_score = s
                                improved = True
            if not improved:
                step *= 0.5
                if step < 1e-7:
                    break
    
    if best_centers is None:
        # Fallback: return grid init
        best_centers = _initial_grid_jittered()
    
    return best_centers
