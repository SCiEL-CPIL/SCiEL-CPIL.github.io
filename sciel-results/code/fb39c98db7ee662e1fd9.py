import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
import time

N = 32

def voronoi_inradii(points):
    """
    Given N points in [0,1]^2, compute the inradius of each Voronoi cell
    clipped to [0,1]^2. This is the maximum radius of a circle centered at
    the point that fits within its Voronoi cell and the unit square.
    
    For each point, the inradius is the minimum distance to:
    1. The boundaries of the unit square
    2. The edges of its Voronoi cell (perpendicular bisectors to neighbors)
    
    We use the fact that for a Voronoi cell, the inradius from a vertex
    is the distance to the nearest edge. We approximate by checking distances
    to all other points (divided by 2) and to the walls.
    """
    p = points.reshape(N, 2)
    
    # Distance to walls
    dist_walls = np.minimum(p, 1 - p)
    min_wall_dist = np.min(dist_walls, axis=1)
    
    # Distance to nearest neighbor / 2 (approximation of Voronoi edge distance)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    
    radii = np.minimum(min_wall_dist, min_neighbor_dist)
    return np.sum(radii)

def exact_voronoi_inradii(points):
    """
    More accurate computation using actual Voronoi vertices.
    For each site, find the minimum distance from the site to any edge
    of its Voronoi cell (including the bounding box edges).
    """
    p = points.reshape(N, 2)
    
    # Compute Voronoi diagram
    try:
        vor = Voronoi(p)
    except Exception:
        # Fallback to approximation
        return voronoi_inradii(points)
    
    radii = np.zeros(N)
    
    for i in range(N):
        center = p[i]
        
        # Distance to walls
        min_wall = min(center[0], 1-center[0], center[1], 1-center[1])
        
        # Get Voronoi region vertices
        region_indices = vor.regions[vor.point_region[i]]
        
        if len(region_indices) == 0:
            radii[i] = min_wall
            continue
        
        # Get vertices of this region
        vertices = []
        for idx in region_indices:
            if idx != -1:
                vertices.append(vor.vertices[idx])
        
        vertices = np.array(vertices)
        
        if len(vertices) == 0:
            radii[i] = min_wall
            continue
        
        # The inradius is the minimum distance from center to any edge of the polygon
        # An edge is between consecutive vertices
        n_verts = len(vertices)
        min_dist = min_wall
        
        for j in range(n_verts):
            v1 = vertices[j]
            v2 = vertices[(j+1) % n_verts]
            
            # Distance from center to line segment v1-v2
            # Project center onto the line
            edge_vec = v2 - v1
            edge_len_sq = np.dot(edge_vec, edge_vec)
            
            if edge_len_sq < 1e-12:
                dist = np.linalg.norm(center - v1)
            else:
                t = np.dot(center - v1, edge_vec) / edge_len_sq
                t = np.clip(t, 0, 1)
                closest_point = v1 + t * edge_vec
                dist = np.linalg.norm(center - closest_point)
            
            min_dist = min(min_dist, dist)
        
        radii[i] = min_dist
    
    return np.sum(radii)

def create_grid_init(rows, cols, n):
    """Create a grid initialization with specified rows and cols, taking first n points"""
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) < n:
                points.append([xs[j], ys[i]])
    return np.array(points).flatten()

def construct():
    start_time = time.time()
    timeout = 110  # Leave 10s margin
    
    best_c = None
    best_val = -np.inf
    
    # Multiple initialization strategies
    init_variants = [
        create_grid_init(6, 6, N),
        create_grid_init(5, 7, N),
        create_grid_init(7, 5, N),
        create_grid_init(4, 8, N),
        create_grid_init(8, 4, N),
        create_grid_init(5, 6, N),
        create_grid_init(6, 5, N),
    ]
    
    # Add random perturbations
    all_inits = []
    for init in init_variants:
        all_inits.append(init.copy())
        for noise_scale in [0.01, 0.03, 0.08]:
            noise = np.random.normal(0, noise_scale, size=init.shape)
            perturbed = np.clip(init + noise, 0.02, 0.98)
            all_inits.append(perturbed)
    
    # Add pure random inits
    for _ in range(8):
        random_c = np.random.uniform(0.05, 0.95, size=N*2)
        all_inits.append(random_c)
    
    # Optimize each initialization
    for idx, c_init in enumerate(all_inits):
        if time.time() - start_time > timeout:
            break
        
        # Try Powell first (robust for non-smooth objectives)
        try:
            res = minimize(lambda c: -voronoi_inradii(c), c_init, method='Powell',
                          options={'maxiter': 500, 'disp': False, 'ftol': 1e-8})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_c = res.x.copy()
        except:
            pass
        
        if time.time() - start_time > timeout:
            break
        
        # Try Nelder-Mead if we have time
        if time.time() - start_time < timeout - 20:
            try:
                res = minimize(lambda c: -voronoi_inradii(c), c_init, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_c = res.x.copy()
            except:
                pass
    
    # If no valid result, fall back to grid
    if best_c is None:
        best_c = create_grid_init(6, 6, N)
        best_val = voronoi_inradii(best_c)
    
    # Final refinement
    if time.time() - start_time < timeout - 15:
        try:
            res = minimize(lambda c: -voronoi_inradii(c), best_c, method='Powell',
                          options={'maxiter': 300, 'disp': False, 'ftol': 1e-10})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_c = res.x.copy()
        except:
            pass
    
    # One more refinement with Nelder-Mead on best
    if time.time() - start_time < timeout - 5:
        try:
            res = minimize(lambda c: -voronoi_inradii(c), best_c, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_c = res.x.copy()
        except:
            pass
    
    # Convert to (N, 3) array using exact Voronoi inradii for final result
    p = best_c.reshape(N, 2)
    
    # Use exact Voronoi inradii for the final radii calculation
    try:
        vor = Voronoi(p)
        radii = np.zeros(N)
        for i in range(N):
            center = p[i]
            min_wall = min(center[0], 1-center[0], center[1], 1-center[1])
            region_indices = vor.regions[vor.point_region[i]]
            
            if len(region_indices) == 0:
                radii[i] = min_wall
                continue
            
            vertices = []
            for idx in region_indices:
                if idx != -1:
                    vertices.append(vor.vertices[idx])
            
            vertices = np.array(vertices)
            if len(vertices) == 0:
                radii[i] = min_wall
                continue
            
            n_verts = len(vertices)
            min_dist = min_wall
            
            for j in range(n_verts):
                v1 = vertices[j]
                v2 = vertices[(j+1) % n_verts]
                edge_vec = v2 - v1
                edge_len_sq = np.dot(edge_vec, edge_vec)
                
                if edge_len_sq < 1e-12:
                    dist = np.linalg.norm(center - v1)
                else:
                    t = np.dot(center - v1, edge_vec) / edge_len_sq
                    t = np.clip(t, 0, 1)
                    closest_point = v1 + t * edge_vec
                    dist = np.linalg.norm(center - closest_point)
                
                min_dist = min(min_dist, dist)
            
            radii[i] = min_dist
    except:
        # Fallback to approximation
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_neighbor_dist = np.min(dists, axis=1) / 2.0
        dist_walls = np.minimum(p, 1 - p)
        min_wall_dist = np.min(dist_walls, axis=1)
        radii = np.minimum(min_wall_dist, min_neighbor_dist)
    
    result = np.column_stack((p, radii))
    return result
