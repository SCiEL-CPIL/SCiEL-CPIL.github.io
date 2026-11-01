import numpy as np
from scipy.optimize import minimize
import time
from scipy.spatial import Voronoi

N = 32

def compute_voronoi_inradii(points):
    """
    Compute the inradius of each Voronoi cell for the given points.
    Returns array of inradii (one per point) and the incenter positions.
    """
    n = len(points)
    inradii = np.zeros(n)
    incenters = points.copy()
    
    # Add boundary points to ensure all cells are bounded
    # Add points far outside the unit square to create bounded cells
    eps = 0.1
    boundary_points = np.array([
        [-eps, -eps], [1+eps, -eps], [-eps, 1+eps], [1+eps, 1+eps],
        [-eps, 0.5], [1+eps, 0.5], [0.5, -eps], [0.5, 1+eps],
        [0.25, -eps], [0.75, -eps], [0.25, 1+eps], [0.75, 1+eps],
        [-eps, 0.25], [-eps, 0.75], [1+eps, 0.25], [1+eps, 0.75]
    ])
    
    all_points = np.vstack([points, boundary_points])
    
    try:
        vor = Voronoi(all_points)
    except Exception:
        # Fallback: use simple distance-based approach
        for i in range(n):
            px, py = points[i]
            min_wall = min(px, 1-px, py, 1-py)
            diffs = points - points[i]
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            dists[i] = np.inf
            min_neigh = np.min(dists) / 2.0
            inradii[i] = min(min_wall, min_neigh)
        return inradii, incenters
    
    for i in range(n):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if -1 in region or len(region) == 0:
            # Unbounded region - use wall distance
            px, py = points[i]
            inradii[i] = min(px, 1-px, py, 1-py)
            incenters[i] = points[i]
            continue
        
        # Get vertices of this Voronoi cell
        vertices = vor.vertices[region]
        
        if len(vertices) < 3:
            px, py = points[i]
            inradii[i] = min(px, 1-px, py, 1-py)
            incenters[i] = points[i]
            continue
        
        # Compute incenter: point equidistant from all edges
        # For a polygon, the incenter maximizes the minimum distance to edges
        # We'll use a simple iterative approach
        
        # Initialize at the centroid
        center = np.mean(vertices, axis=0)
        
        # Iteratively move toward the direction that increases min distance
        for _ in range(50):
            # Compute distances from center to each edge
            min_dist = np.inf
            best_direction = np.zeros(2)
            
            for j in range(len(vertices)):
                v1 = vertices[j]
                v2 = vertices[(j + 1) % len(vertices)]
                
                # Edge vector
                edge_vec = v2 - v1
                edge_len = np.linalg.norm(edge_vec)
                if edge_len < 1e-10:
                    continue
                edge_unit = edge_vec / edge_len
                
                # Normal to edge (pointing inward)
                normal = np.array([-edge_unit[1], edge_unit[0]])
                
                # Check which direction points inward
                mid_edge = (v1 + v2) / 2
                if np.dot(center - mid_edge, normal) < 0:
                    normal = -normal
                
                # Distance from center to this edge
                dist = np.dot(center - v1, normal)
                
                if dist < min_dist:
                    min_dist = dist
                    best_direction = normal
            
            if min_dist < 1e-10:
                break
            
            # Move in the direction of the closest edge
            step = 0.1 * min_dist
            new_center = center + step * best_direction / np.linalg.norm(best_direction)
            
            # Check if this improves the minimum distance
            new_min_dist = np.inf
            for j in range(len(vertices)):
                v1 = vertices[j]
                v2 = vertices[(j + 1) % len(vertices)]
                edge_vec = v2 - v1
                edge_len = np.linalg.norm(edge_vec)
                if edge_len < 1e-10:
                    continue
                edge_unit = edge_vec / edge_len
                normal = np.array([-edge_unit[1], edge_unit[0]])
                mid_edge = (v1 + v2) / 2
                if np.dot(new_center - mid_edge, normal) < 0:
                    normal = -normal
                dist = np.dot(new_center - v1, normal)
                new_min_dist = min(new_min_dist, dist)
            
            if new_min_dist > min_dist:
                center = new_center
            else:
                break
        
        # Also consider wall constraints
        wall_dist = min(center[0], 1-center[0], center[1], 1-center[1])
        inradii[i] = min(min_dist, wall_dist)
        
        # The incenter should be where the circle touches all constraints
        # For now, use the computed center
        incenters[i] = center
    
    return inradii, incenters

def sum_radii_voronoi(points):
    """
    Sum of inradii of Voronoi cells for the given points.
    """
    inradii, _ = compute_voronoi_inradii(points)
    return np.sum(inradii)

def voronoi_optimize(points, max_iter=200, step_decay=0.99):
    """
    Optimize points by moving them toward their Voronoi cell incenters.
    """
    p = points.copy()
    step = 0.1
    
    for it in range(max_iter):
        inradii, incenters = compute_voronoi_inradii(p)
        old_sum = np.sum(inradii)
        
        # Move each point toward its incenter
        for i in range(N):
            direction = incenters[i] - p[i]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                p[i] += step * direction / norm
        
        # Keep points in bounds
        p = np.clip(p, 0.001, 0.999)
        
        # Decay step size
        step *= step_decay
        
        # Check convergence
        new_inradii, _ = compute_voronoi_inradii(p)
        new_sum = np.sum(new_inradii)
        
        if abs(new_sum - old_sum) < 1e-10:
            break
    
    return p

def create_grid_init(rows, cols, n):
    """Create a grid initialization with specified rows and cols, taking first n points"""
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) < n:
                points.append([xs[j], ys[i]])
    return np.array(points[:n]).flatten()

def halton_sequence(n, dims=2):
    """Generate n points in the unit square using Halton sequences"""
    def halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    
    points = []
    for i in range(1, n + 1):
        points.append([halton(i, 2), halton(i, 3)])
    return np.array(points)

def construct():
    start_time = time.time()
    timeout = 110
    
    best_p = None
    best_val = -np.inf
    
    # Initializations
    inits = []
    
    # 1. Standard Grids
    inits.append(create_grid_init(6, 6, N).reshape(N, 2))
    inits.append(create_grid_init(5, 7, N).reshape(N, 2))
    inits.append(create_grid_init(7, 5, N).reshape(N, 2))
    inits.append(create_grid_init(4, 8, N).reshape(N, 2))
    inits.append(create_grid_init(8, 4, N).reshape(N, 2))
    
    # 2. Halton Sequence (Low Discrepancy)
    halton_points = halton_sequence(N)
    inits.append(halton_points)
    
    # 3. Jittered Grids
    np.random.seed(42)
    base = create_grid_init(6, 6, N).reshape(N, 2)
    for jitter_scale in [0.01, 0.02, 0.03, 0.05]:
        for _ in range(3):
            jitter = np.random.uniform(-jitter_scale, jitter_scale, size=(N, 2))
            p = np.clip(base + jitter, 0.01, 0.99)
            inits.append(p)
    
    # 4. Random
    for _ in range(5):
        p = np.random.uniform(0.05, 0.95, size=(N, 2))
        inits.append(p)
    
    # 5. Hexagonal-like pattern
    hex_points = []
    for row in range(6):
        for col in range(6):
            if len(hex_points) < N:
                x = (col + 0.5 + (row % 2) * 0.5) / 6.0
                y = (row + 0.5) / 6.0
                hex_points.append([x, y])
    inits.append(np.array(hex_points[:N]))
    
    for idx, p_init in enumerate(inits):
        if time.time() - start_time > timeout - 20:
            break
        
        # Voronoi-based optimization
        try:
            optimized = voronoi_optimize(p_init, max_iter=100, step_decay=0.995)
            inradii, _ = compute_voronoi_inradii(optimized)
            val = np.sum(inradii)
            if val > best_val:
                best_val = val
                best_p = optimized.copy()
        except Exception:
            pass
        
        # Fine-tune with local search
        if time.time() - start_time < timeout - 10:
            try:
                # Small perturbations to escape local optima
                for _ in range(10):
                    perturbed = optimized + np.random.uniform(-0.005, 0.005, size=(N, 2))
                    perturbed = np.clip(perturbed, 0.001, 0.999)
                    refined = voronoi_optimize(perturbed, max_iter=50, step_decay=0.99)
                    inradii, _ = compute_voronoi_inradii(refined)
                    val = np.sum(inradii)
                    if val > best_val:
                        best_val = val
                        best_p = refined.copy()
            except Exception:
                pass
        
        if time.time() - start_time > timeout - 8:
            break
    
    if best_p is None:
        best_p = create_grid_init(6, 6, N).reshape(N, 2)
    
    # Compute final radii
    p = best_p
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    dist_walls = np.minimum(p, 1 - p)
    min_wall_dist = np.min(dist_walls, axis=1)
    radii = np.minimum(min_wall_dist, min_neighbor_dist)
    
    result = np.column_stack((p, radii))
    return result
