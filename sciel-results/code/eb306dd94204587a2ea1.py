import numpy as np
from scipy.spatial import Voronoi, HalfspaceIntersection
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Initialization: Jittered Grid
    # Use a 4x4x4 grid (64 points) and pick 50? Or 3x3x3 (27) + 23 random?
    # Let's use a 3x3x3 grid for 27 points, then add 23 random points in the gaps.
    # Or better: Start with a random configuration with some repulsion.
    # A simple jittered 4x4x4 grid subset is good.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Shuffle and take N
    np.random.seed(42)
    indices = np.random.choice(len(grid), N, replace=False)
    centers = grid[indices].copy()
    # Add small jitter to break symmetry
    centers += np.random.uniform(-0.01, 0.01, centers.shape)
    centers = np.clip(centers, -0.5, 0.5)

    best_centers = centers.copy()
    best_radius = -1.0

    def calc_radius(c):
        # c is (N, 3)
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists) / 2.0
        
        # Wall distances
        wall_dists = np.min(0.5 - np.abs(c), axis=1)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist, min_wall_dist)

    best_radius = calc_radius(best_centers)

    # 2. Lloyd's Algorithm
    # We use HalfspaceIntersection to find the centroid of the Voronoi cell clipped by the cube.
    
    # Pre-compute cube half-spaces
    # Inequality form: a^T x <= b
    # x >= -0.5  =>  -x <= 0.5
    # x <= 0.5   =>   x <= 0.5
    cube_hs = np.array([
        [-1, 0, 0, 0.5],
        [ 1, 0, 0, 0.5],
        [ 0, -1, 0, 0.5],
        [ 0, 1, 0, 0.5],
        [ 0, 0, -1, 0.5],
        [ 0, 0, 1, 0.5]
    ])

    def lloyd_step(c):
        new_c = np.zeros_like(c)
        for i in range(N):
            # Construct half-spaces for cell i
            # Voronoi constraints: ||x - c_i|| <= ||x - c_j||
            # 2 x . (c_j - c_i) <= ||c_j||^2 - ||c_i||^2
            # a = 2 * (c_j - c_i), b = ||c_j||^2 - ||c_i||^2
            
            hs_list = [cube_hs]
            for j in range(N):
                if i == j:
                    continue
                a = 2.0 * (c[j] - c[i])
                b = np.dot(c[j], c[j]) - np.dot(c[i], c[i])
                hs_list.append(np.array([[a[0], a[1], a[2], b]]))
            
            hs = np.vstack(hs_list)
            
            try:
                # HalfspaceIntersection expects (M, 4) where M is number of half-spaces
                # The last column is the offset b
                # The intersection is computed for the point 0? No, it handles general.
                # Actually, HalfspaceIntersection(halfspaces, origin)
                # The origin should be inside the intersection.
                # c[i] is inside its own Voronoi cell.
                inter = HalfspaceIntersection(hs, c[i])
                
                # Get vertices
                verts = inter.intersections
                
                if len(verts) == 0:
                    new_c[i] = c[i]
                else:
                    # Centroid of the polytope
                    # Approximate by average of vertices? No, that's not the centroid.
                    # But for convex polytopes, the average of vertices is not the centroid.
                    # However, for Lloyd's algorithm, moving to the vertex average is a common approximation 
                    # if exact centroid is hard. 
                    # Better: Compute the exact centroid of the polytope?
                    # SciPy doesn't provide centroid directly.
                    # We can triangulate the faces and sum the tetrahedra centroids?
                    # Or, simpler: The centroid of the Voronoi cell is the point that minimizes the 
                    # variance.
                    # Let's use the average of the vertices as a heuristic. 
                    # Wait, `scipy.spatial` has `ConvexHull`. We can compute the centroid of the convex hull 
                    # by decomposing it into tetrahedra from the origin?
                    # Actually, for small N, we can just use the average of the vertices. 
                    # It is not exact, but it is a valid point in the cell and often close enough 
                    # to drive convergence.
                    # To be more accurate, we can compute the centroid of the convex hull 
                    # using `scipy.spatial.ConvexHull` and its `volume` and `centroid`?
                    # No, `ConvexHull` doesn't have centroid.
                    # Let's stick to vertex average for speed and simplicity, or 
                    # compute the integral over the volume?
                    # Given N=50, vertex average is likely sufficient for the "Lloyd" part.
                    # The L-BFGS polish will fix the exact position.
                    new_c[i] = np.mean(verts, axis=0)
            except Exception:
                new_c[i] = c[i]
        
        # Clip to cube
        new_c = np.clip(new_c, -0.5, 0.5)
        return new_c

    iterations = 0
    while time.time() < deadline - 20.0: # Leave 20s for polish
        old_c = centers.copy()
        centers = lloyd_step(centers)
        iterations += 1
        
        r = calc_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
            
        # Check convergence
        if np.linalg.norm(centers - old_c) < 1e-5:
            break

    # 3. Polish with L-BFGS-B
    if time.time() < deadline - 5.0:
        def soft_min_neg(c):
            # c is (N*3,)
            c = c.reshape(N, 3)
            # Pairwise distances
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            # Wall distances
            wall_dists = 0.5 - np.abs(c)
            wall_dists = np.min(wall_dists, axis=1)
            
            # Combine all distances
            # We want to maximize the minimum.
            # Minimize -softmin.
            # softmin(x) = -1/beta * log(sum(exp(-beta*x)))
            # We have a set of values: pair_dists/2 and wall_dists.
            # Let's flatten them.
            # Number of pairs: N*(N-1)/2
            # Number of walls: N*3? No, min over k. So N values.
            
            beta = 100.0 # Sharper is better for final polish
            
            vals = []
            # Pairs
            for i in range(N):
                for j in range(i+1, N):
                    vals.append(dists[i, j] / 2.0)
            # Walls
            vals.extend(wall_dists)
            vals = np.array(vals)
            
            # Soft min
            # Handle numerical stability
            max_val = np.max(vals)
            exp_vals = np.exp(-beta * (vals - max_val))
            sum_exp = np.sum(exp_vals)
            log_sum_exp = np.log(sum_exp) - beta * max_val
            soft_min_val = -log_sum_exp / beta
            
            return -soft_min_val

        def soft_min_neg_grad(c):
            c = c.reshape(N, 3)
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            wall_dists = 0.5 - np.abs(c)
            min_wall_idx = np.argmin(np.abs(c), axis=1)
            wall_dists_flat = np.min(wall_dists, axis=1)
            
            beta = 100.0
            
            # Collect values and gradients
            vals = np.zeros(N*(N-1)//2 + N)
            grads = np.zeros((N, 3))
            
            idx = 0
            for i in range(N):
                for j in range(i+1, N):
                    d = dists[i, j]
                    if d > 1e-8:
                        v = d / 2.0
                        vals[idx] = v
                        # Gradient of v w.r.t c_i and c_j
                        # v = 0.5 * ||c_i - c_j||
                        # dv/dc_i = 0.5 * (c_i - c_j) / d
                        # dv/dc_j = 0.5 * (c_j - c_i) / d
                        vec = (c[i] - c[j]) / d
                        grads[i] += 0.5 * vec
                        grads[j] += -0.5 * vec
                    else:
                        vals[idx] = 0.0
                    idx += 1
            
            for i in range(N):
                v = wall_dists_flat[i]
                vals[idx] = v
                # Gradient of (0.5 - |c_ik|) w.r.t c_i
                # d/dc_ik (0.5 - |c_ik|) = -sign(c_ik)
                k = min_wall_idx[i]
                grads[i, k] += -np.sign(c[i, k])
                idx += 1
            
            # Compute weights
            max_val = np.max(vals)
            exp_vals = np.exp(-beta * (vals - max_val))
            sum_exp = np.sum(exp_vals)
            weights = exp_vals / sum_exp
            
            # Gradient of -softmin = sum(weights * grad(val))
            # Wait, softmin(x) = -1/beta log(sum(exp(-beta x)))
            # d/dx softmin = (1/beta) * (sum(exp(-beta x) * beta) / sum(exp(-beta x))) * (-1)?
            # Let's derive:
            # f(x) = -1/beta log(S), S = sum(exp(-beta x_k))
            # df/dx_k = -1/beta * 1/S * dS/dx_k
            # dS/dx_k = -beta exp(-beta x_k)
            # df/dx_k = -1/beta * 1/S * (-beta exp(-beta x_k)) = exp(-beta x_k) / S = w_k
            # So gradient of softmin is w_k.
            # We are minimizing -softmin.
            # So gradient of objective is -w_k.
            # Wait.
            # Objective O = -softmin.
            # dO/dx_k = - d(softmin)/dx_k = -w_k.
            # So grad(O) = - sum(w_k * grad(x_k)).
            
            grad_obj = -np.sum(weights[:, np.newaxis] * grads, axis=0)
            
            return grad_obj.flatten()

        # Initial guess for polish
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                soft_min_neg, 
                x0, 
                method='L-BFGS-B', 
                jac=soft_min_neg_grad,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < soft_min_neg(x0):
                new_centers = res.x.reshape(N, 3)
                new_centers = np.clip(new_centers, -0.5, 0.5)
                r = calc_radius(new_centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = new_centers
        except Exception:
            pass

    return best_centers
