"""Pack 50 equal circles in a unit square, maximize common radius."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import Delaunay, Voronoi

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _true_score(centers):
    """Compute the exact max common radius for given centers."""
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    # Wall radius
    wall_r = np.min(0.5 - np.abs(centers))
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _compute_inradii(centers):
    """
    Compute the inradius of each Voronoi cell for the given centers.
    The inradius of a Voronoi cell is the maximum distance from the center 
    to the boundary of the cell. This is the maximum radius a circle centered 
    at the point can have without crossing a Voronoi edge.
    Note: This is an upper bound on the feasible radius if we assume the 
    Voronoi tessellation is fixed, but for optimization, we use it as a 
    proxy for the local "bottleneck" constraint.
    """
    try:
        vor = Voronoi(centers)
    except Exception:
        return np.zeros(N)
    
    inradii = np.zeros(N)
    for i in range(N):
        # Get the vertices of the Voronoi cell for point i
        # vor.regions[i] contains indices of vertices in vor.vertices
        region = vor.regions[i]
        if len(region) == 0:
            # Unbounded region? Should not happen for points inside a box 
            # if we treat the box boundary as constraints, but Voronoi 
            # considers infinite plane. We must handle walls separately.
            # For now, if unbounded, it means the point is on the convex hull.
            # The constraint is the wall distance.
            inradii[i] = 0.5 - np.max(np.abs(centers[i]))
            continue
        
        verts = vor.vertices[region]
        # The inradius is the minimum distance from the center to the edges of the polygon
        # defined by verts.
        # For a convex polygon, the inradius is the minimum distance to any edge.
        min_dist_to_edge = np.inf
        center = centers[i]
        
        for k in range(len(verts)):
            p1 = verts[k]
            p2 = verts[(k + 1) % len(verts)]
            # Distance from point to line segment p1-p2
            # Vector p1->p2
            v = p2 - p1
            v_len_sq = np.dot(v, v)
            if v_len_sq < 1e-12:
                continue
            # Projection of (center - p1) onto v
            t = np.dot(center - p1, v) / v_len_sq
            t = np.clip(t, 0, 1)
            proj = p1 + t * v
            dist = np.linalg.norm(center - proj)
            if dist < min_dist_to_edge:
                min_dist_to_edge = dist
        
        inradii[i] = min_dist_to_edge
    
    # The actual feasible radius is limited by the min of:
    # 1. Inradius of Voronoi cell (distance to neighbor)
    # 2. Distance to wall
    # However, the Voronoi inradius *is* the distance to the nearest neighbor 
    # (or neighbor's Voronoi edge). The wall constraint is separate.
    # So the bottleneck for point i is min(inradii[i], wall_dist_i)
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    return np.minimum(inradii, wall_dists)

def _objective_min_inradius(centers_flat):
    """
    Objective function for global search: Maximize the minimum inradius.
    Returns negative of the minimum inradius for minimization.
    """
    centers = centers_flat.reshape(N, 2)
    # Ensure centers are within bounds to avoid Voronoi issues or infinite values
    centers = np.clip(centers, -0.5, 0.5)
    
    inradii = _compute_inradii(centers)
    min_inradius = np.min(inradii)
    
    # Add a penalty if any center is outside the square (shouldn't happen due to bounds)
    if np.any(np.abs(centers) > 0.5):
        return 100.0
        
    return -min_inradius

def _slsqp_active_polish(centers, r_init, time_limit=10.0):
    """
    Polish the configuration using SLSQP, but only enforcing constraints 
    that are close to being active (i.e., distance < r_init * 1.1 or wall dist < r_init * 1.1).
    This reduces the number of constraints significantly.
    """
    start_time = time.time()
    
    # Identify active pairs
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Threshold for active constraints
    thresh = r_init * 1.2
    
    active_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            if dist[i, j] < 2 * thresh:
                active_pairs.append((i, j))
        
    # Wall constraints are always active if close to wall
    # We'll include all wall constraints as they are linear and cheap
    
    x0 = np.concatenate([centers.ravel(), [r_init]])
    
    def objective(x):
        return -x[-1]
    
    def constraints(x):
        c = x[:N*2].reshape(N, 2)
        r = x[-1]
        cons = []
        
        # Wall constraints (linear)
        for i in range(N):
            cons.append(0.5 - c[i, 0] - r)
            cons.append(0.5 + c[i, 0] - r)
            cons.append(0.5 - c[i, 1] - r)
            cons.append(0.5 + c[i, 1] - r)
            
        # Active pairwise constraints (non-linear)
        for i, j in active_pairs:
            dx = c[i, 0] - c[j, 0]
            dy = c[i, 1] - c[j, 1]
            cons.append(dx*dx + dy*dy - 4*r*r)
            
        return cons
    
    def jac_obj(x):
        grad = np.zeros_like(x)
        grad[-1] = -1.0
        return grad
    
    # Bounds
    bounds = [(-0.5, 0.5)] * (2*N) + [(0.0, 0.5)]
    
    try:
        # Use SLSQP with active constraints
        # We might need to re-evaluate active pairs if r increases significantly,
        # but for a polish step, this is usually stable.
        res = minimize(
            objective, 
            x0, 
            method='SLSQP',
            bounds=bounds,
            constraints=({'type': 'ineq', 'fun': constraints},),
            options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
        )
        
        if res.success or res.x is not None:
            new_centers = res.x[:N*2].reshape(N, 2)
            new_r = res.x[-1]
            # Verify
            actual_r = _true_score(new_centers)
            if actual_r > r_init * 0.99: # Accept if not significantly worse
                return new_centers, actual_r
                
    except Exception:
        pass
        
    return centers, r_init

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    # 1. Generate a good initial guess using a hexagonal grid
    # Hex grid is a known good packing for dense circles
    def _init_hex():
        # Approximate hex packing for 50 circles
        # 7x7 grid is 49, add 1? Or 8x8 cut?
        # Let's use a 7x7 grid with hex offset, then add one in a gap or just use 50 points from a larger grid
        points = []
        dy = 0.14 # Spacing
        dx = 0.14
        for j in range(8):
            for i in range(8):
                x = i * dx - 0.5 + dx/2
                y = j * dy * np.sqrt(3)/2 - 0.5 + dy*np.sqrt(3)/4
                if j % 2 != 0:
                    x += dx / 2
                points.append([x, y])
        
        points = np.array(points[:N])
        # Center and scale to fit
        points -= np.mean(points, axis=0)
        max_ext = np.max(np.abs(points))
        if max_ext > 0.45:
            points *= 0.45 / max_ext
        return points

    init_centers = _init_hex()
    # Jitter to break symmetry
    np.random.seed(42)
    init_centers += np.random.normal(0, 0.005, init_centers.shape)
    
    # 2. Global Search: Differential Evolution on "Min Inradius"
    # DE is good for finding a configuration where the *minimum* inradius is large.
    # We run it for a short time to find a promising basin.
    
    bounds_de = [(-0.5, 0.5)] * (2 * N)
    
    # We will run DE for a limited number of generations
    # To save time, we run a few short DE runs with different seeds
    
    de_runs = 3
    for run_idx in range(de_runs):
        if time.time() - start_time > DEADLINE - 30:
            break
            
        # Define a custom objective that also checks validity quickly
        def obj_de(x):
            c = x.reshape(N, 2)
            # Quick check: if any point is too close to another or wall, penalize
            # But _compute_inradii is expensive. 
            # Let's use a cheaper proxy for DE:
            # Min pairwise dist / 2 and Min wall dist
            # This is O(N^2) but fast with numpy.
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dist, np.inf)
            min_pair = np.min(dist) / 2.0
            min_wall = np.min(0.5 - np.abs(c))
            return -min(min_pair, min_wall)
        
        # Use the cheap objective for DE to find a good configuration
        # Then use the expensive Voronoi logic for the final polish?
        # Actually, the "Min Inradius" strategy is better applied via the polish.
        # Let's use DE to maximize the *exact* score (cheap version) to get a good start.
        
        try:
            result = differential_evolution(
                obj_de, 
                bounds_de, 
                seed=42 + run_idx,
                maxiter=50, 
                popsize=15,
                tol=1e-5,
                polish=False, # We will polish manually
                init='latinhypercube'
            )
            
            de_centers = result.x.reshape(N, 2)
            de_r = -result.fun
            
            # Polish this candidate
            polished_centers, polished_r = _slsqp_active_polish(de_centers, de_r, time_limit=10)
            
            if polished_r > best_r:
                best_r = polished_r
                best_centers = polished_centers.copy()
                
        except Exception:
            continue

    # 3. If DE didn't find anything good, fall back to the initial hex grid
    if best_centers is None:
        best_centers = init_centers
        best_r = _true_score(best_centers)
        
    # 4. Final Polish on the best candidate
    if time.time() - start_time < DEADLINE - 5:
        time_left = DEADLINE - 5 - (time.time() - start_time)
        # Run a more aggressive polish
        polished_centers, polished_r = _slsqp_active_polish(best_centers, best_r, time_limit=time_left)
        if polished_r > best_r:
            best_r = polished_r
            best_centers = polished_centers.copy()
            
    # 5. One last check
    final_r = _true_score(best_centers)
    if final_r > best_r:
        best_r = final_r
        
    return best_centers

# Ensure the function signature matches
# The task requires construct() -> array of shape (N, 2)
