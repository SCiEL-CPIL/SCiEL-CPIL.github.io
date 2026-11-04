import numpy as np
from scipy.spatial import Voronoi, QhullError
import time
from scipy.optimize import minimize

N = 26
DEADLINE = 140.0  # Safety margin from 150s limit

def construct():
    start_time = time.time()

    def check_validity(x, y, r):
        if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
            return False
        if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
            return False
        if np.any(r < -1e-9):
            return False

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(min_dist, 0)

        if np.any(dist < min_dist - 1e-9):
            return False
        return True

    def compute_radii(x, y):
        """Compute radii from centers: min of half-distance to neighbors and boundary."""
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        min_dist = np.min(dist, axis=1)
        r = np.minimum(r_bound, 0.5 * min_dist)
        r = np.maximum(r, 1e-8)
        return r

    def lloyd_iteration(x, y, n_iter=50):
        """
        Perform Lloyd's algorithm iterations.
        Moves each circle to the centroid of its Voronoi cell (clipped to unit square).
        This is a discrete local search that often escapes shallow local minima of gradient methods.
        """
        x_curr = x.copy()
        y_curr = y.copy()
        
        for _ in range(n_iter):
            if time.time() - start_time > DEADLINE - 10:
                break
                
            # Add boundary points to ensure Voronoi works correctly for boundary-adjacent cells
            # We create a slightly larger set of points including boundary midpoints/corners
            # to help define the cells of points near the edge, but we only update the N points.
            # Actually, standard Voronoi on just the N points works, but cells near boundary are unbounded.
            # We need to intersect with the square.
            
            try:
                points = np.column_stack((x_curr, y_curr))
                vor = Voronoi(points)
                
                new_x = x_curr.copy()
                new_y = y_curr.copy()
                
                for i in range(N):
                    # Get the region for point i
                    region_index = vor.point_region[i]
                    region = vor.regions[region_index]
                    
                    if -1 in region:
                        # Unbounded region (touching boundary)
                        # For unbounded regions, a simple heuristic is to keep the point or 
                        # project the centroid of the bounded part. 
                        # A more robust approach for packing is to treat the boundary as a constraint.
                        # If a point is on the boundary, moving it to the centroid of its Voronoi cell
                        # (intersected with the square) is the ideal move.
                        
                        # Let's compute the centroid of the Voronoi cell intersected with [0,1]^2.
                        # This is complex to do analytically. 
                        # Approximation: If unbounded, we can try to move it towards the center 
                        # of the square slightly, or just leave it if it's already near the boundary.
                        # Better: Use the centroid of the vertices that are inside the square.
                        
                        verts = vor.vertices[region]
                        # Filter vertices inside [0,1]^2 (with small tolerance)
                        mask = (verts[:, 0] >= -1e-9) & (verts[:, 0] <= 1 + 1e-9) & \
                               (verts[:, 1] >= -1e-9) & (verts[:, 1] <= 1 + 1e-9)
                        
                        if np.sum(mask) > 0:
                            # Centroid of the visible part
                            cx = np.mean(verts[mask, 0])
                            cy = np.mean(verts[mask, 1])
                            # Clip to square
                            cx = np.clip(cx, 0, 1)
                            cy = np.clip(cy, 0, 1)
                            new_x[i] = cx
                            new_y[i] = cy
                        else:
                            # No vertices inside? This shouldn't happen for a point inside the square
                            # unless it's a degenerate case. Keep current position.
                            pass
                    else:
                        # Bounded region
                        verts = vor.vertices[region]
                        cx = np.mean(verts[:, 0])
                        cy = np.mean(verts[:, 1])
                        new_x[i] = cx
                        new_y[i] = cy
                        
                # Update positions
                x_curr = new_x
                y_curr = new_y
                
            except QhullError:
                # Qhull can fail if points are in general position issues.
                # Break and return current best
                break
                
        return x_curr, y_curr

    def greedy_construct():
        """
        Construct a packing using a greedy algorithm.
        Place circles one by one at the location that maximizes the minimum distance to existing circles and boundaries.
        """
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        # We will use a grid of candidate positions to speed up the search for the next best spot
        # Grid size 50x50 = 2500 candidates.
        grid_size = 50
        xs = np.linspace(0.01, 0.99, grid_size)
        ys = np.linspace(0.01, 0.99, grid_size)
        
        for i in range(N):
            best_score = -1
            best_x = 0.5
            best_y = 0.5
            
            # Vectorized scoring for all grid points
            # dist to existing circles
            # We need min distance to any existing circle i < current
            if i > 0:
                # Compute distance from all grid points to all existing circles
                # Grid shape: (G, G, 2)
                gx, gy = np.meshgrid(xs, ys)
                gx_flat = gx.flatten()
                gy_flat = gy.flatten()
                
                # Distances to existing circles
                # x[i] shape (i,), gx_flat shape (G*G,)
                dx = gx_flat[:, None] - x[:i][None, :]
                dy = gy_flat[:, None] - y[:i][None, :]
                dists = np.sqrt(dx**2 + dy**2)
                min_dist_to_circles = np.min(dists, axis=1)
                
                # Distance to boundaries
                dist_to_bound = np.minimum(
                    np.minimum(gx_flat, 1.0 - gx_flat),
                    np.minimum(gy_flat, 1.0 - gy_flat)
                )
                
                # The radius of the new circle will be half the min distance to neighbors and boundary
                # r_new = 0.5 * min(min_dist_to_circles, dist_to_bound)
                # Wait, the radius is limited by the distance to the *nearest* neighbor.
                # If we place a circle at p, its radius is min( dist(p, boundary), 0.5 * min(dist(p, q_j)) )
                # We want to maximize this radius.
                
                potential_r = np.minimum(min_dist_to_circles * 0.5, dist_to_bound)
                
                # Find the max potential radius
                max_idx = np.argmax(potential_r)
                best_score = potential_r[max_idx]
                best_x = gx_flat[max_idx]
                best_y = gy_flat[max_idx]
            else:
                # First circle: center of square
                best_x = 0.5
                best_y = 0.5
                best_score = 0.5

            x[i] = best_x
            y[i] = best_y
            
        return x, y

    # 1. Greedy Construction
    x, y = greedy_construct()
    
    # 2. Local Search (Lloyd's Algorithm)
    # Lloyd's tends to equalize radii. For maximizing sum, we might want to allow some large circles.
    # However, Lloyd's is a good way to "settle" the configuration.
    # Let's run Lloyd's, then compute radii.
    
    x_lloyd, y_lloyd = lloyd_iteration(x, y, n_iter=100)
    
    r_lloyd = compute_radii(x_lloyd, y_lloyd)
    score_lloyd = np.sum(r_lloyd)
    
    best_x = x_lloyd
    best_y = y_lloyd
    best_r = r_lloyd
    best_score = score_lloyd

    # 3. Refinement with Gradient Descent (L-BFGS)
    # Use the Lloyd's result as a starting point for a more precise gradient-based optimization.
    # Objective: Maximize sum of radii.
    # Radii are defined by the constraints.
    
    def objective(params):
        x = params[:N]
        y = params[N:2*N]
        
        if np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1):
            return 1e6
            
        r = compute_radii(x, y)
        
        # Check overlaps
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(min_dist, 0)
        
        overlaps = np.maximum(0, min_dist - dist)
        penalty = np.sum(overlaps) * 1000
        
        return -np.sum(r) + penalty

    if time.time() - start_time < DEADLINE - 10:
        try:
            bounds_opt = [(0.0, 1.0)] * (2 * N)
            res = minimize(
                objective, 
                np.concatenate([best_x, best_y]),
                method='L-BFGS-B', 
                bounds=bounds_opt,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            x_res = res.x[:N]
            y_res = res.x[N:]
            r_res = compute_radii(x_res, y_res)
            
            if check_validity(x_res, y_res, r_res):
                score_res = np.sum(r_res)
                if score_res > best_score:
                    best_score = score_res
                    best_x = x_res
                    best_y = y_res
                    best_r = r_res
        except Exception:
            pass

    # 4. Multi-start with different random seeds for Lloyd's?
    # Lloyd's is deterministic given the start. 
    # We can try perturbing the greedy start slightly before Lloyd's.
    
    for seed in range(5):
        if time.time() - start_time > DEADLINE - 15:
            break
            
        # Perturb the greedy solution
        np.random.seed(seed)
        x_pert = best_x + np.random.normal(0, 0.01, N)
        y_pert = best_y + np.random.normal(0, 0.01, N)
        x_pert = np.clip(x_pert, 0, 1)
        y_pert = np.clip(y_pert, 0, 1)
        
        x_lloyd_p, y_lloyd_p = lloyd_iteration(x_pert, y_pert, n_iter=50)
        r_lloyd_p = compute_radii(x_lloyd_p, y_lloyd_p)
        
        if check_validity(x_lloyd_p, y_lloyd_p, r_lloyd_p):
            score_lloyd_p = np.sum(r_lloyd_p)
            if score_lloyd_p > best_score:
                best_score = score_lloyd_p
                best_x = x_lloyd_p
                best_y = y_lloyd_p
                best_r = r_lloyd_p
                
                # Polish this new best
                if time.time() - start_time < DEADLINE - 5:
                    try:
                        res = minimize(
                            objective, 
                            np.concatenate([best_x, best_y]),
                            method='L-BFGS-B', 
                            bounds=[(0.0, 1.0)] * (2 * N),
                            options={'maxiter': 100, 'ftol': 1e-12}
                        )
                        x_res = res.x[:N]
                        y_res = res.x[N:]
                        r_res = compute_radii(x_res, y_res)
                        
                        if check_validity(x_res, y_res, r_res):
                            score_res = np.sum(r_res)
                            if score_res > best_score:
                                best_score = score_res
                                best_x = x_res
                                best_y = y_res
                                best_r = r_res
                    except Exception:
                        pass

    # Final validity check and scaling if necessary
    if not check_validity(best_x, best_y, best_r):
        for scale in np.linspace(0.999, 0.95, 20):
            r_test = best_r * scale
            if check_validity(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            best_r = compute_radii(best_x, best_y)

    return np.column_stack((best_x, best_y, best_r))
