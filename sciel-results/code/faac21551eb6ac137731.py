"""Circle packing in unit square - Voronoi-based structural approach."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
import time
import random

def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return np.sum(r), True

def _compute_voronoi_inradii(xy):
    """
    Compute the inradius of each Voronoi cell defined by the centers xy.
    The inradius is the maximum radius of a circle centered at xy[i] that fits
    entirely within the Voronoi cell V_i.
    """
    n = len(xy)
    if n < 3:
        # Degenerate case: just use wall distances
        r = np.zeros(n)
        for i in range(n):
            wall_dist = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
            r[i] = wall_dist
        return r
    
    try:
        vor = Voronoi(xy)
    except Exception:
        # Fallback: use nearest neighbor distances
        r = np.zeros(n)
        for i in range(n):
            wall_dist = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
            min_dist = np.inf
            for j in range(n):
                if i == j:
                    continue
                d = np.sqrt((xy[i, 0] - xy[j, 0])**2 + (xy[i, 1] - xy[j, 1])**2)
                min_dist = min(min_dist, d)
            r[i] = min(wall_dist, min_dist * 0.5)
        return r
    
    r = np.zeros(n)
    
    for i in range(n):
        # Wall distances are always constraints
        min_r = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
        
        # Find the Voronoi cell for point i
        # vor.regions contains lists of vertex indices. -1 indicates unbounded.
        # We need to find which region corresponds to point i.
        # vor.point_region[i] gives the index of the region for point i.
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if len(region) == 0:
            # This shouldn't happen for interior points, but handle just in case
            continue
            
        # Get the vertices of this cell
        # Note: vertices with index -1 are at infinity
        cell_vertices = []
        for v_idx in region:
            if v_idx != -1:
                cell_vertices.append(vor.vertices[v_idx])
                
        if not cell_vertices:
            continue
            
        cell_vertices = np.array(cell_vertices)
        
        # The inradius is the minimum distance from the center to any edge of the cell.
        # The edges of the Voronoi cell are line segments connecting pairs of consecutive vertices.
        # However, a more robust way is to check the distance to the lines defined by the edges.
        # Actually, the distance to the edge is the perpendicular distance to the line segment,
        # but since the center is inside the convex polygon, the closest point on the boundary
        # will be on one of the edges.
        
        # Iterate over edges of the polygon
        m = len(cell_vertices)
        for k in range(m):
            p1 = cell_vertices[k]
            p2 = cell_vertices[(k + 1) % m]
            
            # Vector for the edge
            edge_vec = p2 - p1
            # Vector from p1 to center
            vec_to_center = xy[i] - p1
            
            # Project vec_to_center onto edge_vec
            # dist = |vec_to_center x edge_vec| / |edge_vec| (in 2D, cross product is scalar)
            edge_len_sq = np.dot(edge_vec, edge_vec)
            if edge_len_sq < 1e-12:
                continue
                
            # 2D cross product (z-component)
            cross = vec_to_center[0] * edge_vec[1] - vec_to_center[1] * edge_vec[0]
            dist = abs(cross) / np.sqrt(edge_len_sq)
            
            # Check if the projection falls within the segment
            # t = dot(vec_to_center, edge_vec) / dot(edge_vec, edge_vec)
            t = np.dot(vec_to_center, edge_vec) / edge_len_sq
            if 0 <= t <= 1:
                min_r = min(min_r, dist)
            else:
                # If projection is outside, the closest point is a vertex
                # But we are iterating over edges, so vertices are covered by adjacent edges?
                # Actually, checking distance to vertices is safer if we just take min distance to all vertices.
                # But for a convex polygon containing the point, the min distance to the boundary
                # is the min of the perpendicular distances to the edges (where projection is inside)
                # or distances to vertices.
                # Let's just compute distance to p1 and p2 to be safe, although perpendicular
                # distance is usually the binding constraint for Voronoi cells which are convex.
                d1 = np.linalg.norm(xy[i] - p1)
                d2 = np.linalg.norm(xy[i] - p2)
                min_r = min(min_r, d1, d2)

    return r

def _voronoi_objective(xy):
    """
    Objective function to maximize: sum of Voronoi inradii.
    We will minimize the negative of this.
    """
    r = _compute_voronoi_inradii(xy)
    return -np.sum(r)

def _gradient_ascent_voronoi(xy_init, deadline, lr=0.01, max_iter=5000):
    """
    Gradient ascent on the sum of Voronoi inradii.
    Uses numerical gradients.
    """
    n = len(xy_init)
    params = xy_init.ravel().copy()
    best_val = _voronoi_objective(params.reshape(-1, 2))
    best_params = params.copy()
    
    step_size = lr
    eps = 1e-5
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        # Compute numerical gradient
        grad = np.zeros_like(params)
        for k in range(len(params)):
            params_plus = params.copy()
            params_plus[k] += eps
            val_plus = _voronoi_objective(params_plus.reshape(-1, 2))
            grad[k] = (val_plus - best_val) / eps
            
        # Gradient step (ascent, so add gradient)
        params = params + step_size * grad
        
        # Clip to valid range (keep centers inside [0.01, 0.99] to avoid boundary issues with Voronoi)
        for i in range(n):
            params[2*i] = np.clip(params[2*i], 0.01, 0.99)
            params[2*i+1] = np.clip(params[2*i+1], 0.01, 0.99)
            
        new_val = _voronoi_objective(params.reshape(-1, 2))
        
        if new_val < best_val: # Remember we are minimizing negative sum, so lower is better
            best_val = new_val
            best_params = params.copy()
            step_size *= 0.999
        else:
            step_size *= 0.95
            params = best_params.copy()
            
        if step_size < 1e-7:
            break
            
    return best_params.reshape(-1, 2)

def _simulated_annealing_voronoi(xy_init, deadline, t0=0.05, cooling=0.999, max_iter=10000):
    """
    Simulated annealing on center positions, optimizing Voronoi inradii sum.
    """
    n = len(xy_init)
    current_params = xy_init.ravel().copy()
    current_val = _voronoi_objective(current_params.reshape(-1, 2))
    best_params = current_params.copy()
    best_val = current_val
    t = t0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        # Propose random perturbation
        new_params = current_params.copy()
        idx = random.randint(0, len(new_params) - 1)
        delta = random.gauss(0, 0.02)
        new_params[idx] += delta
        new_params[idx] = np.clip(new_params[idx], 0.01, 0.99)
        
        new_val = _voronoi_objective(new_params.reshape(-1, 2))
        delta_e = new_val - current_val
        
        if delta_e < 0 or random.random() < np.exp(-delta_e / max(t, 1e-10)):
            current_params = new_params
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_params = new_params.copy()
                
        t *= cooling
        
    return best_params.reshape(-1, 2)

def _grid_init(side):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                positions.append([x, y])
    # Pad if needed
    while len(positions) < N:
        positions.append([0.5, 0.5])
    return np.array(positions[:N])

def _random_init():
    """Generate random positions."""
    positions = []
    for i in range(N):
        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.95)
        positions.append([x, y])
    return np.array(positions[:N])

def _best_fit_local_search(xy, r, deadline, grid_size=80):
    """
    Iteratively update each circle to the best position (maximizing its radius)
    given all other circles fixed.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = np.sum(best_r)
    else:
        best_total = 0.0

    # Candidate grid
    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    # Precompute candidate wall clearances
    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    max_sweeps = 25
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            # Other circles
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]

            # Compute clearance to others for all candidates
            if len(others_xy) > 0:
                dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)  # (M, n-1)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)  # (M,)
                r_candidates = np.minimum(wall_clear, circle_clear)
            else:
                r_candidates = wall_clear.copy()

            # Find best candidate
            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]

            # Only accept if it improves total radius
            old_r = r[i]
            if new_r > old_r + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True

        # After full sweep, evaluate and update best
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total, best_valid

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False
    
    # Strategy 1: Voronoi-based gradient ascent
    if time.time() < deadline - 40.0:
        for side in [5, 6]:
            if time.time() > deadline - 40.0:
                break
            xy_init = _grid_init(side)
            
            # Run gradient ascent
            ga_xy = _gradient_ascent_voronoi(xy_init, deadline - 40.0, lr=0.005, max_iter=2000)
            
            # Compute radii from Voronoi cells
            r = _compute_voronoi_inradii(ga_xy)
            
            # Validate and evaluate
            total, valid = _eval_packing(ga_xy, r)
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_xy = ga_xy.copy()
                best_r = r.copy()
                
    # Strategy 2: Voronoi-based simulated annealing
    if time.time() < deadline - 20.0:
        for side in [5, 6]:
            if time.time() > deadline - 20.0:
                break
            xy_init = _grid_init(side)
            
            # Run simulated annealing
            sa_xy = _simulated_annealing_voronoi(xy_init, deadline - 20.0, t0=0.05, cooling=0.9995, max_iter=5000)
            
            # Compute radii
            r = _compute_voronoi_inradii(sa_xy)
            
            total, valid = _eval_packing(sa_xy, r)
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_xy = sa_xy.copy()
                best_r = r.copy()
                
    # Strategy 3: Best-fit local search on the best result found so far
    if best_xy is not None and time.time() < deadline - 10.0:
        out_xy, out_r, total, valid = _best_fit_local_search(best_xy, best_r, deadline - 10.0, grid_size=100)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r
            
    # If no valid result, fall back to simple grid
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        xy_init = _grid_init(side)
        r = np.full(N, 0.4 / side)
        total, valid = _eval_packing(xy_init, r)
        if valid:
            best_total = total
            best_valid = True
            best_xy = xy_init
            best_r = r
        else:
            # Last resort
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.35 / side
            return result
            
    # Final best-fit local search polish
    if time.time() < deadline - 3.0:
        p_xy, p_r, p_total, p_valid = _best_fit_local_search(best_xy, best_r, deadline - 2.0, grid_size=120)
        if p_valid and p_total > best_total:
            best_total = p_total
            best_valid = True
            best_xy = p_xy
            best_r = p_r
            
    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
        
    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.35 / side
            
    return result
# EVOLVE-BLOCK-END
