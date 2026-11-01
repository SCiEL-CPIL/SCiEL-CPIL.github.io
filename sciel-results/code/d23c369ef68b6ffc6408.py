import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit

    def is_valid(result, tol=1e-9):
        n = N
        xy = result[:2*n].reshape(n, 2)
        r = result[2*n:]

        if np.any(r < -tol):
            return False
        if np.any(xy[:, 0] - r < -tol) or np.any(1.0 - xy[:, 0] - r < -tol):
            return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1.0 - xy[:, 1] - r < -tol):
            return False

        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist[iu] < min_dist[iu] - tol):
            return False
        return True

    def get_score(result):
        return np.sum(result[2*N:])

    def push_apart(params, n_iters=100):
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()

        for _ in range(n_iters):
            moved = False
            for i in range(n):
                for j in range(i+1, n):
                    dx = xy[j, 0] - xy[i, 0]
                    dy = xy[j, 1] - xy[i, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = r[i] + r[j]
                    if dist < min_dist and dist > 1e-12:
                        overlap = min_dist - dist
                        wi = r[i] / (r[i] + r[j]) if r[i] + r[j] > 1e-15 else 0.5
                        wj = 1.0 - wi
                        push_x = (dx / dist) * overlap
                        push_y = (dy / dist) * overlap
                        xy[i, 0] -= push_x * wi
                        xy[i, 1] -= push_y * wi
                        xy[j, 0] += push_x * wj
                        xy[j, 1] += push_y * wj
                        moved = True
                    elif dist <= 1e-12:
                        angle = np.random.uniform(0, 2*np.pi)
                        xy[i, 0] -= 0.01 * np.cos(angle)
                        xy[i, 1] -= 0.01 * np.sin(angle)
                        xy[j, 0] += 0.01 * np.cos(angle)
                        xy[j, 1] += 0.01 * np.sin(angle)
                        moved = True
            for i in range(n):
                if xy[i, 0] < r[i]:
                    xy[i, 0] = r[i]; moved = True
                if xy[i, 0] > 1.0 - r[i]:
                    xy[i, 0] = 1.0 - r[i]; moved = True
                if xy[i, 1] < r[i]:
                    xy[i, 1] = r[i]; moved = True
                if xy[i, 1] > 1.0 - r[i]:
                    xy[i, 1] = 1.0 - r[i]; moved = True
            if not moved:
                break
        return np.concatenate([xy.flatten(), r])

    def voronoi_inradius(xy, n=N):
        """Compute the inradius of each Voronoi cell (distance to nearest edge)."""
        # Add boundary points to make Voronoi cells bounded
        # We'll compute the distance from each point to the nearest other point
        # and to the boundary, then use the minimum as a conservative estimate
        # Actually, let's compute proper Voronoi cells
        
        # For efficiency, we'll compute the inradius as the minimum distance
        # to any other point (half the distance) and to the boundary
        # This is a lower bound on the true Voronoi inradius
        
        # Better approach: compute actual Voronoi diagram
        try:
            vor = Voronoi(xy)
        except Exception:
            # Fallback: use simple distance-based estimate
            r = np.zeros(n)
            for i in range(n):
                # Distance to boundary
                d_bound = min(xy[i,0], 1-xy[i,0], xy[i,1], 1-xy[i,1])
                # Distance to nearest other point
                d_min = np.inf
                for j in range(n):
                    if i != j:
                        d = np.sqrt((xy[i,0]-xy[j,0])**2 + (xy[i,1]-xy[j,1])**2)
                        d_min = min(d_min, d)
                r[i] = min(d_bound, d_min/2)
            return r
        
        r = np.zeros(n)
        for i in range(n):
            # Get the Voronoi region for point i
            region = vor.regions[vor.point_region[i]]
            if len(region) == 0:
                # Unbounded cell - use distance to boundary
                d_bound = min(xy[i,0], 1-xy[i,0], xy[i,1], 1-xy[i,1])
                r[i] = d_bound
                continue
            
            # Get the vertices of this region
            verts = vor.vertices[region]
            if len(verts) == 0:
                d_bound = min(xy[i,0], 1-xy[i,0], xy[i,1], 1-xy[i,1])
                r[i] = d_bound
                continue
            
            # The inradius is the minimum distance from the point to any edge of the cell
            # Each edge is defined by two consecutive vertices
            min_dist = np.inf
            for k in range(len(verts)):
                v1 = verts[k]
                v2 = verts[(k+1) % len(verts)]
                # Distance from point to line segment v1-v2
                # First check if the perpendicular foot is on the segment
                d1 = xy[i,0] - v1[0]
                d2 = xy[i,1] - v1[1]
                d3 = v2[0] - v1[0]
                d4 = v2[1] - v1[1]
                
                denom = d3**2 + d4**2
                if denom < 1e-15:
                    dist = np.sqrt(d1**2 + d2**2)
                else:
                    t = max(0, min(1, (d1*d3 + d2*d4) / denom))
                    px = v1[0] + t*d3
                    py = v1[1] + t*d4
                    dist = np.sqrt((xy[i,0]-px)**2 + (xy[i,1]-py)**2)
                min_dist = min(min_dist, dist)
            
            # Also consider distance to boundary if cell is unbounded
            d_bound = min(xy[i,0], 1-xy[i,0], xy[i,1], 1-xy[i,1])
            r[i] = min(min_dist, d_bound)
        
        return r

    def optimize_with_voronoi(xy_init, r_init, max_iter=300):
        """Optimize positions and radii using Voronoi-based constraints."""
        n = N
        
        def objective(params):
            xy = params[:2*n].reshape(n, 2)
            r = params[2*n:]
            
            # Compute Voronoi inradii
            r_voronoi = voronoi_inradius(xy, n)
            
            # Objective: maximize sum of radii, but constrain r_i <= r_voronoi_i
            # and r_i <= distance to boundary
            d_bound = np.minimum(np.minimum(xy[:,0], 1-xy[:,0]), 
                                 np.minimum(xy[:,1], 1-xy[:,1]))
            r_max = np.minimum(r_voronoi, d_bound)
            
            # Penalty for exceeding max radius
            excess = np.maximum(0, r - r_max)
            penalty = np.sum(excess**2)
            
            return -np.sum(r) + 1000.0 * penalty
        
        # Initial radii: use Voronoi inradii
        r_v = voronoi_inradius(xy_init, n)
        d_bound = np.minimum(np.minimum(xy_init[:,0], 1-xy_init[:,0]), 
                             np.minimum(xy_init[:,1], 1-xy_init[:,1]))
        r_init_opt = np.minimum(r_v, d_bound) * 0.99
        
        params_init = np.concatenate([xy_init.flatten(), r_init_opt])
        
        bounds = [(0, 1)] * (2*n) + [(0, 0.5)] * n
        
        result = minimize(objective, params_init, method='L-BFGS-B', 
                         bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-14})
        
        return result.x

    def get_initial_config(idx, rng):
        n = N
        mod = idx % 10

        if mod == 0:
            # 6x6 grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row, col = i // 6, i % 6
                xy[i] = [(col + 0.5)/6, (row + 0.5)/6]
                r[i] = 0.5/6 - 0.003
        elif mod == 1:
            # Perturbed 6x6
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row, col = i // 6, i % 6
                xy[i] = [(col + 0.5)/6, (row + 0.5)/6]
                r[i] = 0.5/6 - 0.003
            xy += rng.uniform(-0.015, 0.015, size=(n, 2))
        elif mod == 2:
            # Hexagonal packing
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row, col = i // 6, i % 6
                x = (col + 0.5)/6 + (0.25/6 if row % 2 == 1 else 0)
                y = (row + 0.5)/6
                xy[i] = [x, y]
                r[i] = 0.073
        elif mod == 3:
            # 8x4 grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row, col = i // 8, i % 8
                xy[i] = [(col + 0.5)/8, (row + 0.5)/4]
                r[i] = 0.5/8 - 0.002
            xy += rng.uniform(-0.008, 0.008, size=(n, 2))
        elif mod == 4:
            # 4x8 grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row, col = i // 4, i % 4
                xy[i] = [(col + 0.5)/4, (row + 0.5)/8]
                r[i] = 0.5/4 - 0.002
            xy += rng.uniform(-0.008, 0.008, size=(n, 2))
        elif mod == 5:
            # Random uniform
            xy = rng.uniform(0.08, 0.92, size=(n, 2))
            r = rng.uniform(0.055, 0.085, size=n)
        elif mod == 6:
            # 7x5 grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row, col = i // 7, i % 7
                xy[i] = [(col + 0.5)/7, (row + 0.5)/5]
                r[i] = 0.5/7 - 0.002
            xy += rng.uniform(-0.008, 0.008, size=(n, 2))
        elif mod == 7:
            # 5x7 grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row, col = i // 5, i % 5
                xy[i] = [(col + 0.5)/5, (row + 0.5)/7]
                r[i] = 0.5/5 - 0.002
            xy += rng.uniform(-0.008, 0.008, size=(n, 2))
        elif mod == 8:
            # Hexagonal variant
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row, col = i // 6, i % 6
                x = (col + 0.5)/6 + (0.2/6 if row % 2 == 0 else 0)
                y = (row + 0.5)/6
                xy[i] = [x, y]
                r[i] = 0.07
        else:
            # Random medium
            xy = rng.uniform(0.1, 0.9, size=(n, 2))
            r = rng.uniform(0.06, 0.08, size=n)

        return np.concatenate([xy.flatten(), r])

    rng = np.random.RandomState(42)
    best_result = None
    best_score = -np.inf

    # Phase 1: Diverse restarts with Voronoi-based optimization
    n_restarts = 50
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break

        params = get_initial_config(restart, rng)
        params = push_apart(params, n_iters=50)

        xy = params[:2*N].reshape(N, 2).copy()
        r = params[2*N:].copy()

        # Use Voronoi-based optimization
        try:
            result = optimize_with_voronoi(xy, r, max_iter=200)
            if is_valid(result):
                score = get_score(result)
                if score > best_score:
                    best_score = score
                    best_result = result.copy()
        except Exception:
            pass

        # Also try standard penalty-based refinement
        def smooth_penalty(params, beta=50.0):
            n = N
            xy = params[:2*n].reshape(n, 2)
            r = params[2*n:]

            def smooth_max_neg(arr, beta=50.0):
                return np.logaddexp(0, -beta * arr) / beta

            d_left = xy[:, 0] - r
            d_right = 1.0 - xy[:, 0] - r
            d_bottom = xy[:, 1] - r
            d_top = 1.0 - xy[:, 1] - r

            boundary_pen = np.sum(smooth_max_neg(d_left, beta)) + \
                           np.sum(smooth_max_neg(d_right, beta)) + \
                           np.sum(smooth_max_neg(d_bottom, beta)) + \
                           np.sum(smooth_max_neg(d_top, beta))

            dx = xy[:, 0][:, None] - xy[:, 0][None, :]
            dy = xy[:, 1][:, None] - xy[:, 1][None, :]
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = r[:, None] + r[None, :]
            iu = np.triu_indices(n, k=1)
            overlap_viol = min_dist[iu] - dist[iu]
            overlap_pen = np.sum(smooth_max_neg(-overlap_viol, beta))

            return -np.sum(r) + 100.0 * (boundary_pen + overlap_pen)

        try:
            refined = minimize(lambda p: smooth_penalty(p, 50.0), params,
                            method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-14})
            if is_valid(refined.x):
                score = get_score(refined.x)
                if score > best_score:
                    best_score = score
                    best_result = refined.x.copy()
        except Exception:
            pass

    # Phase 2: Grow radii iteratively on best solution
    if best_result is not None and time.time() < deadline - 10:
        for grow_iter in range(20):
            if time.time() > deadline - 10:
                break
            xy = best_result[:2*N].reshape(N, 2).copy()
            r = best_result[2*N:].copy()
            r_new = r * 1.003
            params_trial = np.concatenate([xy.flatten(), r_new])
            
            def smooth_penalty2(params, beta=50.0):
                n = N
                xy = params[:2*n].reshape(n, 2)
                r = params[2*n:]
                def smooth_max_neg(arr, beta=50.0):
                    return np.logaddexp(0, -beta * arr) / beta
                d_left = xy[:, 0] - r
                d_right = 1.0 - xy[:, 0] - r
                d_bottom = xy[:, 1] - r
                d_top = 1.0 - xy[:, 1] - r
                boundary_pen = np.sum(smooth_max_neg(d_left, beta)) + \
                               np.sum(smooth_max_neg(d_right, beta)) + \
                               np.sum(smooth_max_neg(d_bottom, beta)) + \
                               np.sum(smooth_max_neg(d_top, beta))
                dx = xy[:, 0][:, None] - xy[:, 0][None, :]
                dy = xy[:, 1][:, None] - xy[:, 1][None, :]
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = r[:, None] + r[None, :]
                iu = np.triu_indices(n, k=1)
                overlap_viol = min_dist[iu] - dist[iu]
                overlap_pen = np.sum(smooth_max_neg(-overlap_viol, beta))
                return -np.sum(r) + 100.0 * (boundary_pen + overlap_pen)

            try:
                refined = minimize(lambda p: smooth_penalty2(p, 50.0), params_trial,
                                method='L-BFGS-B', options={'maxiter': 150, 'ftol': 1e-14})
                if is_valid(refined.x):
                    score = get_score(refined.x)
                    if score > best_score:
                        best_score = score
                        best_result = refined.x.copy()
                else:
                    break
            except Exception:
                break

    # Final cleanup
    if best_result is not None:
        xy = best_result[:2*N].reshape(N, 2).copy()
        r = best_result[2*N:].copy()
        r = np.clip(r, 0, 0.5)
        xy[:, 0] = np.clip(xy[:, 0], r, 1.0 - r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1.0 - r)
        final_params = np.concatenate([xy.flatten(), r])
        if is_valid(final_params):
            return np.column_stack([xy, r])

    # Fallback
    side = 6
    cells = []
    for i in range(N):
        row, col = i // 6, i % 6
        cells.append([(col + 0.5)/6, (row + 0.5)/6, 0.5/6 - 1e-3])
    return np.array(cells)
