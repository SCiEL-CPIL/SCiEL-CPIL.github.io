"""
Circle packing in perimeter-4 rectangle using Voronoi-cell inradii.
Strategy:
1. Represent packing by centers only.
2. Compute Voronoi diagram of centers within the rectangle.
3. Each circle's radius is the inradius of its Voronoi cell (distance to nearest edge).
4. This guarantees disjoint interiors by construction (circles inscribed in Voronoi cells).
5. Optimize center positions and rectangle aspect ratio to maximize sum of inradii.
6. Use SLSQP on a smooth, differentiable objective (no overlap constraints needed).
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    def _voronoi_inradii(x, y, w, h):
        """Compute inradius of each point's Voronoi cell within rectangle [0,w]x[0,h]."""
        pts = np.column_stack((x, y))
        # Add rectangle corners to Voronoi to clip cells at boundary
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h],
                            w / 2, -0.5 * h, w / 2, h + 0.5 * h,
                            -0.5 * w, h / 2, w + 0.5 * w, h / 2])
        # Actually, better approach: compute Voronoi of points, then for each cell,
        # find min distance from point to cell edges (including rectangle boundary).
        # Simpler: use scipy Voronoi on points, then for each finite vertex, check.
        # But we need distance to rectangle walls too.
        
        # Robust approach: for each point, radius = min(
        #   min over other points j of (dist(i,j) / 2) -- but this is only for interior
        #   min(x[i], w-x[i], y[i], h-y[i]) -- wall distance
        # )
        # However, the true Voronoi inradius is the distance to the nearest Voronoi edge,
        # which is max over all j of min(dist(i,j)/2, ...) -- no.
        # The inradius of the Voronoi cell of point i is the maximum r such that
        # a disk of radius r centered at i is contained in the cell.
        # This equals min( min_j (dist(i,j)/2), x[i], w-x[i], y[i], h-y[i] )
        # Wait, that's not quite right either. The Voronoi cell edges are perpendicular
        # bisectors between i and its neighbors. The distance from i to the edge between
        # i and j is dist(i,j)/2. The inradius is the minimum over all Voronoi edges
        # of the distance from i to that edge. For interior neighbors, this is dist/2.
        # For boundary, it's the distance to the wall.
        # So: r_i = min( min_{j != i} (dist(i,j)/2), x[i], w-x[i], y[i], h-y[i] )
        
        r = np.zeros(N)
        wall_dist = np.minimum(np.minimum(x, w - x), np.minimum(y, h - y))
        r[:] = wall_dist
        
        # For each pair, the Voronoi edge is at dist/2 from each center
        # The inradius is limited by the nearest such edge
        for i in range(N):
            min_edge_dist = np.inf
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                # Distance from i to the perpendicular bisector of i-j is d/2
                # But only if the bisector actually forms a Voronoi edge
                # For inradius, we take min over all j of d/2
                if d / 2 < min_edge_dist:
                    min_edge_dist = d / 2
            if min_edge_dist < r[i]:
                r[i] = min_edge_dist
        
        return r

    def _objective(z):
        """Objective: negative sum of inradii."""
        x = z[0::2]
        y = z[1::2]
        w = z[-1]
        h = 2.0 - w
        
        if w < 0.3 or h < 0.3:
            return 1e6
        
        # Ensure points are within bounds (with small margin)
        margin = 0.001
        if np.any(x < margin) or np.any(x > w - margin) or \
           np.any(y < margin) or np.any(y > h - margin):
            # Soft penalty for going out of bounds
            pen = 0.0
            pen += np.sum(np.maximum(margin - x, 0) ** 2)
            pen += np.sum(np.maximum(x - (w - margin), 0) ** 2)
            pen += np.sum(np.maximum(margin - y, 0) ** 2)
            pen += np.sum(np.maximum(y - (h - margin), 0) ** 2)
            return 1e4 + pen
        
        r = _voronoi_inradii(x, y, w, h)
        return -np.sum(r)

    def _get_initial_guesses():
        """Generate diverse initial configurations."""
        guesses = []
        rng = np.random.default_rng(42)
        
        # Grid-based initializations with different aspect ratios
        aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.55]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            for i in range(N):
                row, col = divmod(i, side)
                x[i] = (col + 0.5) / side * w0
                y[i] = (row + 0.5) / side * h0
            guesses.append(np.concatenate([x, y, [w0]]))
        
        # Random initializations
        for _ in range(12):
            w0 = rng.uniform(0.5, 1.5)
            h0 = 2.0 - w0
            x = rng.uniform(0.05, w0 - 0.05, N)
            y = rng.uniform(0.05, h0 - 0.05, N)
            guesses.append(np.concatenate([x, y, [w0]]))
        
        return guesses

    guesses = _get_initial_guesses()
    
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 25:
            break
        
        x0 = z0[0::2]
        y0 = z0[1::2]
        w0 = z0[-1]
        h0 = 2.0 - w0
        
        # Bounds: x in [0.01, w-0.01], y in [0.01, h-0.01], w in [0.3, 1.7]
        # Since w is variable, we use loose bounds and rely on objective penalty
        bounds = []
        for _ in range(N):
            bounds.append((0.005, 1.995))  # x
        for _ in range(N):
            bounds.append((0.005, 1.995))  # y
        bounds.append((0.3, 1.7))  # w
        
        remaining = deadline - (time.time() - start_time) - 15
        if remaining < 5:
            break
        
        maxiter = int(min(400, max(80, remaining * 30)))
        
        try:
            res = minimize(_objective, z0, method='SLSQP', bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-10, 'maxls': 50})
            
            z_opt = res.x
            x = z_opt[0::2]
            y = z_opt[1::2]
            w = z_opt[-1]
            h = 2.0 - w
            
            # Clip to valid range
            x = np.clip(x, 0.001, w - 0.001)
            y = np.clip(y, 0.001, h - 0.001)
            
            r = _voronoi_inradii(x, y, w, h)
            
            # Verify no overlaps (should be guaranteed, but check)
            valid = True
            for i in range(N):
                if r[i] <= 0:
                    valid = False
                    break
                if x[i] - r[i] < -1e-7 or x[i] + r[i] > w + 1e-7:
                    valid = False
                    break
                if y[i] - r[i] < -1e-7 or y[i] + r[i] > h + 1e-7:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        d2 = dx * dx + dy * dy
                        if d2 < (r[i] + r[j]) ** 2 - 1e-7:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                val = float(np.sum(r))
                if val > best_value:
                    best_value = val
                    best_result = (x.copy(), y.copy(), r.copy(), w, h)
                    
        except Exception:
            pass

    # Polish best result with more iterations
    if best_result is not None and time.time() < deadline - 20:
        x, y, r, w, h = best_result
        z0 = np.concatenate([x, y, [w]])
        
        bounds = []
        for _ in range(N):
            bounds.append((0.005, 1.995))
        for _ in range(N):
            bounds.append((0.005, 1.995))
        bounds.append((0.3, 1.7))
        
        remaining = deadline - (time.time() - start_time) - 10
        if remaining > 5:
            maxiter = int(min(600, max(100, remaining * 40)))
            try:
                res = minimize(_objective, z0, method='SLSQP', bounds=bounds,
                               options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})
                z_opt = res.x
                x = z_opt[0::2]
                y = z_opt[1::2]
                w = z_opt[-1]
                h = 2.0 - w
                
                x = np.clip(x, 0.001, w - 0.001)
                y = np.clip(y, 0.001, h - 0.001)
                
                r = _voronoi_inradii(x, y, w, h)
                
                valid = True
                for i in range(N):
                    if r[i] <= 0:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i + 1, N):
                            dx = x[i] - x[j]
                            dy = y[i] - y[j]
                            d2 = dx * dx + dy * dy
                            if d2 < (r[i] + r[j]) ** 2 - 1e-7:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    val = float(np.sum(r))
                    if val > best_value:
                        best_value = val
                        best_result = (x.copy(), y.copy(), r.copy(), w, h)
            except Exception:
                pass

    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        w = 1.0
        h = 1.0
        for i in range(N):
            row, col = divmod(i, side)
            x[i] = (col + 0.5) / side * w
            y[i] = (row + 0.5) / side * h
            r[i] = min(w / side, h / side) * 0.45
        return np.column_stack((x, y, r))

    x, y, r, w, h = best_result
    return np.column_stack((x, y, r))
