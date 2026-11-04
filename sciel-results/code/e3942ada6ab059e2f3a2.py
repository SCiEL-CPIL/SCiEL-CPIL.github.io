"""
Circle packing in unit square using hierarchical decomposition:
partition into cells, optimize each cell locally, then coordinate globally.

Strategy:
1. Partition unit square into coarse grid cells (e.g., 2x2 or 3x3)
2. Assign circles to cells based on initial positions
3. Optimize each cell independently (smaller, more tractable sub-problems)
4. Handle inter-cell constraints via iterative penalty on shared boundaries
5. Final global polish with subgradient descent
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay, ConvexHull
import time

N = 26
DEADLINE = 115.0  # Safety margin from 120s limit


def construct():
    start_time = time.time()

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
        r = np.maximum(r, 1e-6)
        return r

    def compute_radii_vectorized(x, y):
        """Vectorized computation of radii - same as compute_radii but with explicit vectorization."""
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
        r = np.maximum(r, 1e-6)
        return r

    def project_feasible(x, y, r, tol=1e-9):
        """Project circles to satisfy all constraints by scaling down radii if needed."""
        # Check boundary constraints
        r = np.minimum(r, x)
        r = np.minimum(r, 1.0 - x)
        r = np.minimum(r, y)
        r = np.minimum(r, 1.0 - y)
        
        # Check pairwise constraints iteratively
        for _ in range(50):
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dist, np.inf)
            
            # For each pair, if dist < r_i + r_j, scale both down
            violated = (dist < r[:, None] + r[None, :] - tol) & (dist > 0)
            if not np.any(violated):
                break
            
            # Find maximum violation
            violations = (r[:, None] + r[None, :] - dist) * violated
            violations[~violated] = 0
            max_viol_idx = np.unravel_index(np.argmax(violations), violations.shape)
            i, j = max_viol_idx
            
            if i == j:
                continue
            
            # Scale down both circles to fix this violation
            needed = dist[i, j]
            current = r[i] + r[j]
            scale = needed / current if current > 0 else 0.5
            r[i] *= scale
            r[j] *= scale
        
        r = np.maximum(r, 1e-8)
        return r

    def compute_radii_smooth(x, y, beta):
        """Smooth approximation of radii using log-sum-exp soft-min."""
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        half_dist = 0.5 * dist
        full_constraints = half_dist.copy()
        np.fill_diagonal(full_constraints, r_bound)

        neg_beta = -beta * full_constraints
        max_neg_row = np.max(neg_beta, axis=1, keepdims=True)
        exp_vals = np.exp(neg_beta - max_neg_row)
        exp_vals = np.where(np.isinf(full_constraints), 0.0, exp_vals)
        log_sum_row = np.log(np.sum(exp_vals, axis=1, keepdims=True))
        soft_min_all = -(max_neg_row + log_sum_row) / beta
        soft_min_all = soft_min_all[:, 0]

        r = np.maximum(soft_min_all, 1e-6)
        return r

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

    def partition_into_cells(x, y, n_cells_x=2, n_cells_y=2):
        """
        Partition circles into cells based on their positions.
        Returns list of (cell_idx, circle_indices) and cell boundaries.
        """
        cell_w = 1.0 / n_cells_x
        cell_h = 1.0 / n_cells_y
        
        # Assign each circle to a cell
        cell_assignments = np.zeros(N, dtype=int)
        for i in range(N):
            cx = int(min(x[i] / cell_w, n_cells_x - 1))
            cy = int(min(y[i] / cell_h, n_cells_y - 1))
            cell_assignments[i] = cy * n_cells_x + cx
        
        # Group circles by cell
        cell_groups = {}
        for i in range(N):
            cell_id = cell_assignments[i]
            if cell_id not in cell_groups:
                cell_groups[cell_id] = []
            cell_groups[cell_id].append(i)
        
        return cell_groups, cell_w, cell_h, n_cells_x, n_cells_y

    def optimize_cell_local(cell_indices, x_all, y_all, time_limit=0.5, seed=0):
        """
        Optimize circles within a single cell, treating other circles as fixed obstacles.
        Uses local simulated annealing + gradient polish.
        """
        if len(cell_indices) == 0:
            return None
        
        idx = np.array(cell_indices)
        x = x_all[idx].copy()
        y = y_all[idx].copy()
        n_local = len(idx)
        
        # Compute radii considering all circles (global context)
        def local_compute_radii(x_loc, y_loc):
            r_bound = np.minimum(
                np.minimum(x_loc, 1.0 - x_loc),
                np.minimum(y_loc, 1.0 - y_loc)
            )
            
            # Distance to all circles (local + global)
            all_x = np.concatenate([x_loc, x_all[np.setdiff1d(np.arange(N), idx)]])
            all_y = np.concatenate([y_loc, y_all[np.setdiff1d(np.arange(N), idx)]])
            
            dx = x_loc[:, None] - all_x[None, :]
            dy = y_loc[:, None] - all_y[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            
            # Exclude self-distances
            for i in range(n_local):
                dist[i, i] = np.inf
            
            min_dist = np.min(dist, axis=1)
            r = np.minimum(r_bound, 0.5 * min_dist)
            r = np.maximum(r, 1e-6)
            return r
        
        r = local_compute_radii(x, y)
        current_val = np.sum(r)
        best_x = x.copy()
        best_y = y.copy()
        best_val = current_val
        
        # Local SA
        rng = np.random.RandomState(seed)
        T_start = 0.01
        T_end = 0.0005
        max_iters = min(500, int(time_limit * 200))
        cool_rate = (T_end / T_start) ** (1.0 / max_iters) if max_iters > 0 else 1.0
        
        sa_start = time.time()
        it = 0
        while time.time() - sa_start < time_limit:
            if time.time() - start_time > DEADLINE - 5:
                break
            
            T = T_start * (cool_rate ** it)
            scale = 0.005 * (T / T_start) + 0.001
            dx_prop = rng.normal(0, scale, n_local)
            dy_prop = rng.normal(0, scale, n_local)
            
            x_new = x + dx_prop
            y_new = y + dy_prop
            x_new = np.clip(x_new, 0.02, 0.98)
            y_new = np.clip(y_new, 0.02, 0.98)
            
            new_r = local_compute_radii(x_new, y_new)
            new_val = np.sum(new_r)
            
            delta_val = new_val - current_val
            if delta_val > 0 or (T > 1e-6 and rng.random() < np.exp(delta_val / T)):
                x = x_new
                y = y_new
                r = new_r
                current_val = new_val
                
                if current_val > best_val:
                    best_val = current_val
                    best_x = x.copy()
                    best_y = y.copy()
            
            it += 1
            if T < 1e-5:
                break
        
        # Local gradient polish
        x = best_x.copy()
        y = best_y.copy()
        r = local_compute_radii(x, y)
        current_val = np.sum(r)
        best_val = current_val
        best_x = x.copy()
        best_y = y.copy()
        
        step_size = 0.03
        decay = 0.99
        
        polish_start = time.time()
        while time.time() - polish_start < time_limit * 0.5:
            if time.time() - start_time > DEADLINE - 3:
                break
            
            # Compute subgradient
            r_bound = np.minimum(
                np.minimum(x, 1.0 - x),
                np.minimum(y, 1.0 - y)
            )
            
            all_x = np.concatenate([x, x_all[np.setdiff1d(np.arange(N), idx)]])
            all_y = np.concatenate([y, y_all[np.setdiff1d(np.arange(N), idx)]])
            
            dx_mat = x[:, None] - all_x[None, :]
            dy_mat = y[:, None] - all_y[None, :]
            dist = np.sqrt(dx_mat**2 + dy_mat**2)
            for i in range(n_local):
                dist[i, i] = np.inf
            
            min_dist = np.min(dist, axis=1)
            nn_idx_local = np.argmin(dist, axis=1)
            
            # Determine active constraints
            c_x = x
            c_1x = 1.0 - x
            c_y = y
            c_1y = 1.0 - y
            c_nn = 0.5 * min_dist
            
            grad_x = np.zeros(n_local)
            grad_y = np.zeros(n_local)
            
            for i in range(n_local):
                vals = [c_x[i], c_1x[i], c_y[i], c_1y[i], c_nn[i]]
                min_val = np.min(vals)
                tol = 1e-6
                
                if abs(c_x[i] - min_val) < tol:
                    grad_x[i] += 1.0
                if abs(c_1x[i] - min_val) < tol:
                    grad_x[i] -= 1.0
                if abs(c_y[i] - min_val) < tol:
                    grad_y[i] += 1.0
                if abs(c_1y[i] - min_val) < tol:
                    grad_y[i] -= 1.0
                if abs(c_nn[i] - min_val) < tol:
                    j_global = nn_idx_local[i]
                    d_ij = dist[i, j_global]
                    if d_ij > 1e-9:
                        grad_x[i] += 0.5 * (x[i] - all_x[j_global]) / d_ij
                        grad_y[i] += 0.5 * (y[i] - all_y[j_global]) / d_ij
            
            grad_norm = np.linalg.norm(np.concatenate([grad_x, grad_y]))
            if grad_norm > 1e-9:
                grad_x /= grad_norm
                grad_y /= grad_norm
            
            x_new = x + step_size * grad_x
            y_new = y + step_size * grad_y
            x_new = np.clip(x_new, 0.0, 1.0)
            y_new = np.clip(y_new, 0.0, 1.0)
            
            new_r = local_compute_radii(x_new, y_new)
            new_val = np.sum(new_r)
            
            if new_val > current_val:
                x = x_new
                y = y_new
                r = new_r
                current_val = new_val
                if current_val > best_val:
                    best_val = current_val
                    best_x = x.copy()
                    best_y = y.copy()
                step_size *= decay
            else:
                step_size *= 0.95
            
            if step_size < 0.001:
                break
        
        # Update global arrays
        x_all[idx] = best_x
        y_all[idx] = best_y
        
        r_global = compute_radii(x_all, y_all)
        return r_global[idx], np.sum(r_global[idx])

    def make_initial_guess(seed, mode='grid'):
        rng = np.random.RandomState(seed)
        if mode == 'grid':
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    x[idx] = (i + 0.5) / side
                    y[idx] = (j + 0.5) / side
                    idx += 1
            x += rng.uniform(-0.08, 0.08, N)
            y += rng.uniform(-0.08, 0.08, N)
        elif mode == 'hex':
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    x[idx] = (i + 0.5) / side + (0.5 / side) * (j % 2)
                    y[idx] = (j + 0.5) / side
                    idx += 1
            x += rng.uniform(-0.05, 0.05, N)
            y += rng.uniform(-0.05, 0.05, N)
        else:
            x = rng.uniform(0.1, 0.9, N)
            y = rng.uniform(0.1, 0.9, N)

        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        return x, y

    def hierarchical_optimize(x0, y0, seed, time_limit=8.0):
        """
        Hierarchical decomposition optimization:
        1. Partition into cells
        2. Optimize each cell locally
        3. Global coordination with penalty method
        4. Final global polish
        """
        x = x0.copy()
        y = y0.copy()
        rng = np.random.RandomState(seed)
        
        # Phase 1: Local cell optimization (multiple passes)
        n_passes = 3
        cell_configs = [(2, 2), (3, 3), (2, 3)]
        
        for pass_idx in range(n_passes):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            nx, ny = cell_configs[pass_idx % len(cell_configs)]
            cell_groups, cell_w, cell_h, n_cx, n_cy = partition_into_cells(x, y, nx, ny)
            
            # Optimize each cell
            for cell_id, indices in sorted(cell_groups.items()):
                if time.time() - start_time > DEADLINE - 5:
                    break
                if len(indices) < 2:
                    continue
                
                cell_time = min(1.0, (DEADLINE - 5 - (time.time() - start_time)) / (len(cell_groups) - pass_idx))
                cell_time = max(cell_time, 0.2)
                
                optimize_cell_local(indices, x, y, time_limit=cell_time, seed=seed + pass_idx * 100 + cell_id)
            
            # Re-partition and repeat
        
        # Phase 2: Global coordination with penalty method
        # Penalize violations of inter-cell constraints
        penalty_weight = 0.1
        coord_start = time.time()
        
        while time.time() - coord_start < time_limit * 0.3:
            if time.time() - start_time > DEADLINE - 3:
                break
            
            r = compute_radii(x, y)
            current_val = np.sum(r)
            
            # Check for inter-cell violations (circles in adjacent cells overlapping)
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dist, np.inf)
            
            # Compute penalty for overlaps
            overlap_penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    if dist[i, j] < r[i] + r[j] - 1e-9:
                        overlap_penalty += (r[i] + r[j] - dist[i, j])**2
            
            # If no overlaps, we're in a good state
            if overlap_penalty < 1e-12:
                break
            
            # Apply penalty-based adjustment
            # Move circles apart to reduce overlaps
            for i in range(N):
                for j in range(i+1, N):
                    if dist[i, j] < r[i] + r[j] - 1e-9 and dist[i, j] > 1e-9:
                        # Push apart
                        direction_x = (x[i] - x[j]) / dist[i, j]
                        direction_y = (y[i] - y[j]) / dist[i, j]
                        push = penalty_weight * (r[i] + r[j] - dist[i, j])
                        x[i] += push * direction_x
                        y[i] += push * direction_y
                        x[j] -= push * direction_x
                        y[j] -= push * direction_y
            
            x = np.clip(x, 0.01, 0.99)
            y = np.clip(y, 0.01, 0.99)
            
            # Decay penalty
            penalty_weight *= 0.9
        
        # Phase 3: Final global polish
        x_final, y_final, r_final, val_final = run_subgradient_polish(x, y, time_limit=time_limit * 0.4)
        
        if x_final is not None and check_validity(x_final, y_final, r_final):
            return x_final, y_final, r_final, val_final
        
        # Fallback: use current best
        r = compute_radii(x, y)
        if check_validity(x, y, r):
            return x, y, r, np.sum(r)
        
        return None

    def get_delaunay_voids(x, y):
        """
        Identify interstitial voids using Delaunay triangulation.
        Returns list of (cx, cy, max_radius) for each void where a circle could fit.
        """
        try:
            # Add boundary points to ensure we capture edge/corner voids
            boundary_pts = []
            # Corners
            boundary_pts.append([0.0, 0.0])
            boundary_pts.append([1.0, 0.0])
            boundary_pts.append([0.0, 1.0])
            boundary_pts.append([1.0, 1.0])
            # Edge midpoints
            for t in np.linspace(0.1, 0.9, 4):
                boundary_pts.append([t, 0.0])
                boundary_pts.append([t, 1.0])
                boundary_pts.append([0.0, t])
                boundary_pts.append([1.0, t])
            
            all_pts = np.vstack([np.column_stack([x, y]), np.array(boundary_pts)])
            
            try:
                tri = Delaunay(all_pts)
            except Exception:
                return []
            
            voids = []
            n_circles = len(x)
            
            for simplex in tri.simplices:
                # Get the 3 vertices of this triangle
                verts = all_pts[simplex]
                
                # Check if any vertex is a circle center (not boundary)
                # Circle centers are indices 0..n_circles-1
                circle_indices = [i for i in simplex if i < n_circles]
                
                if len(circle_indices) == 0:
                    # All boundary - this is a corner/edge region, skip for now
                    continue
                
                # Compute circumcenter and circumradius of this triangle
                v1, v2, v3 = verts
                ax, ay = v1
                bx, by = v2
                cx, cy = v3
                
                # Circumcenter calculation
                d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
                if abs(d) < 1e-12:
                    continue
                
                ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + 
                      (cx**2 + cy**2) * (ay - by)) / d
                uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + 
                      (cx**2 + cy**2) * (bx - ax)) / d
                
                # Circumradius
                r_circ = np.sqrt((ux - ax)**2 + (uy - ay)**2)
                
                # Check if circumcenter is inside the triangle
                # Using barycentric coordinates
                def sign(p1, p2, p3):
                    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
                
                d1 = sign([ux, uy], v1, v2)
                d2 = sign([ux, uy], v2, v3)
                d3 = sign([ux, uy], v3, v1)
                
                has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
                has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
                
                if has_neg and has_pos:
                    # Circumcenter outside triangle - skip
                    continue
                
                # The max radius for a circle centered at circumcenter is r_circ
                # But we need to check against all other circles and boundaries
                max_r = r_circ
                
                # Check against boundaries
                max_r = min(max_r, ux, 1.0 - ux, uy, 1.0 - uy)
                
                # Check against all circle centers
                for i in range(n_circles):
                    dist_to_center = np.sqrt((ux - x[i])**2 + (uy - y[i])**2)
                    # The circle at (ux, uy) with radius r must not overlap with circle i
                    # If circle i has radius r_i, we need dist >= r + r_i
                    # But we don't know r_i yet, so use conservative estimate
                    # For now, just use dist as upper bound (assuming other circles have ~0 radius)
                    # This is a lower bound on the true max radius
                    max_r = min(max_r, 0.5 * dist_to_center)
                
                if max_r > 0.005:  # Only keep meaningful voids
                    voids.append((ux, uy, max_r))
            
            return voids
            
        except Exception:
            return []

    def sa_optimize(x0, y0, seed, time_limit=2.0):
        """
        Fast Coarse Simulated Annealing.
        """
        rng = np.random.RandomState(seed)
        x = x0.copy()
        y = y0.copy()
        r = compute_radii(x, y)
        current_val = np.sum(r)
        best_x = x.copy()
        best_y = y.copy()
        best_val = current_val

        T_start = 0.02
        T_end = 0.001
        sa_start = time.time()
        elapsed = 0.0

        # Reduced iterations for speed, relying on gradient step later
        max_iters = 1500 
        cool_rate = (T_end / T_start) ** (1.0 / max_iters)

        it = 0
        while elapsed < time_limit:
            if time.time() - start_time > DEADLINE - 5:
                break

            T = T_start * (cool_rate ** it)

            # Simple random walk with slight bias towards center to avoid boundary sticking
            scale = 0.01 * (T / T_start) + 0.002
            dx_prop = rng.normal(0, scale, N)
            dy_prop = rng.normal(0, scale, N)

            x_new = x + dx_prop
            y_new = y + dy_prop

            x_new = np.clip(x_new, 0.02, 0.98)
            y_new = np.clip(y_new, 0.02, 0.98)

            new_r = compute_radii(x_new, y_new)
            new_val = np.sum(new_r)

            delta_val = new_val - current_val

            if delta_val > 0 or (T > 1e-6 and rng.random() < np.exp(delta_val / T)):
                x = x_new
                y = y_new
                r = new_r
                current_val = new_val

                if current_val > best_val:
                    best_val = current_val
                    best_x = x.copy()
                    best_y = y.copy()

            it += 1
            elapsed = time.time() - sa_start

            if T < 1e-5:
                break

        return best_x, best_y, best_val

    def run_subgradient_polish(x0, y0, time_limit=3.0):
        """
        Direct Subgradient Descent on the nonsmooth objective.
        """
        x = x0.copy()
        y = y0.copy()
        rng = np.random.RandomState(42)  # For degenerate cases
        
        # Compute initial radii and identify active constraints
        r = compute_radii(x, y)
        current_val = np.sum(r)
        best_x = x.copy()
        best_y = y.copy()
        best_val = current_val

        # Subgradient step size parameters
        step_size = 0.05
        decay = 0.995
        min_step = 0.001

        polish_start = time.time()
        it = 0
        
        # Precompute distance matrices for efficiency
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        # Find nearest neighbor indices
        # For each i, find j that minimizes dist[i, j]
        nn_idx = np.argmin(dist, axis=1)
        nn_dist = dist[np.arange(N), nn_idx]

        # Boundary constraints
        # x < r, x > 1-r, y < r, y > 1-r
        # r = min(x, 1-x, y, 1-y, 0.5*nn_dist)
        
        # Gradients of active constraints w.r.t x, y
        # If r_i = x_i, grad_x = 1, grad_y = 0
        # If r_i = 1-x_i, grad_x = -1, grad_y = 0
        # If r_i = y_i, grad_x = 0, grad_y = 1
        # If r_i = 1-y_i, grad_x = 0, grad_y = -1
        # If r_i = 0.5 * dist(i, j), grad_x_i = 0.5 * (x_i - x_j) / dist, grad_y_i = 0.5 * (y_i - y_j) / dist
        
        while time.time() - polish_start < time_limit:
            if time.time() - start_time > DEADLINE - 3:
                break

            # Update distances and nearest neighbors
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dist, np.inf)
            
            nn_idx = np.argmin(dist, axis=1)
            nn_dist = dist[np.arange(N), nn_idx]

            # Determine active constraint for each circle
            # Candidates: x, 1-x, y, 1-y, 0.5*nn_dist
            c_x = x
            c_1x = 1.0 - x
            c_y = y
            c_1y = 1.0 - y
            c_nn = 0.5 * nn_dist

            # We want to maximize sum(r). 
            # r_i = min(c_x, c_1x, c_y, c_1y, c_nn)
            # Subgradient of max(sum(r)) is sum of gradients of active mins.
            
            # Initialize gradients
            grad_x = np.zeros(N)
            grad_y = np.zeros(N)

            for i in range(N):
                # Find the minimum constraint value
                vals = [c_x[i], c_1x[i], c_y[i], c_1y[i], c_nn[i]]
                min_val = np.min(vals)
                
                # Tolerance for considering a constraint "active"
                tol = 1e-6
                
                # If multiple constraints are active (within tol), sum their gradients
                # This handles the case where a circle is touching a wall and another circle simultaneously
                
                # Constraint 1: x
                if abs(c_x[i] - min_val) < tol:
                    grad_x[i] += 1.0
                # Constraint 2: 1-x
                if abs(c_1x[i] - min_val) < tol:
                    grad_x[i] -= 1.0
                # Constraint 3: y
                if abs(c_y[i] - min_val) < tol:
                    grad_y[i] += 1.0
                # Constraint 4: 1-y
                if abs(c_1y[i] - min_val) < tol:
                    grad_y[i] -= 1.0
                # Constraint 5: 0.5 * dist(i, j)
                if abs(c_nn[i] - min_val) < tol:
                    j = nn_idx[i]
                    d_ij = dist[i, j]
                    if d_ij > 1e-9:
                        grad_x[i] += 0.5 * (x[i] - x[j]) / d_ij
                        grad_y[i] += 0.5 * (y[i] - y[j]) / d_ij
                    else:
                        # Degenerate case: overlapping centers, pick random direction
                        grad_x[i] += rng.normal(0, 1)
                        grad_y[i] += rng.normal(0, 1)

            # Normalize gradient to avoid huge steps
            grad_norm = np.linalg.norm(np.concatenate([grad_x, grad_y]))
            if grad_norm > 1e-9:
                grad_x /= grad_norm
                grad_y /= grad_norm

            # Update positions
            # We want to increase r, so we move in the direction of the gradient of r.
            # Since we are maximizing sum(r), we move x += step * grad_x
            x_new = x + step_size * grad_x
            y_new = y + step_size * grad_y
            
            # Clip to valid bounds
            x_new = np.clip(x_new, 0.0, 1.0)
            y_new = np.clip(y_new, 0.0, 1.0)

            new_r = compute_radii(x_new, y_new)
            new_val = np.sum(new_r)

            if new_val > current_val:
                x = x_new
                y = y_new
                r = new_r
                current_val = new_val
                
                if current_val > best_val:
                    best_val = current_val
                    best_x = x.copy()
                    best_y = y.copy()
                    
                # If we improved, keep step size or increase slightly if stuck?
                # Standard subgradient: decay step size
                step_size *= decay
            else:
                # If no improvement, reduce step size more aggressively
                step_size *= 0.9

            if step_size < min_step:
                break
                
            it += 1

        x_final = best_x
        y_final = best_y
        r_final = compute_radii(x_final, y_final)
        val_final = np.sum(r_final)

        if check_validity(x_final, y_final, r_final):
            return x_final, y_final, r_final, val_final
        else:
            for scale in [0.99, 0.98, 0.95, 0.90, 0.85]:
                r_test = r_final * scale
                if check_validity(x_final, y_final, r_test):
                    return x_final, y_final, r_test, np.sum(r_test)

        return None

    def run_optimization(seed, mode, use_hierarchical=True):
        x0, y0 = make_initial_guess(seed, mode)

        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 8:
            return None

        if use_hierarchical and remaining > 15:
            # Use hierarchical decomposition
            hier_time = min(10.0, remaining * 0.7)
            result = hierarchical_optimize(x0, y0, seed, hier_time)
            if result is not None:
                return result
        
        # Fallback to standard SA + polish
        sa_time = min(2.0, remaining * 0.5)
        x_sa, y_sa, val_sa = sa_optimize(x0, y0, seed, sa_time)

        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 5:
            r_sa = compute_radii(x_sa, y_sa)
            if check_validity(x_sa, y_sa, r_sa):
                return x_sa, y_sa, r_sa, val_sa
            return None

        polish_time = min(3.0, remaining * 0.5)
        result = run_subgradient_polish(x_sa, y_sa, polish_time)
        if result is not None:
            return result

        r_sa = compute_radii(x_sa, y_sa)
        if check_validity(x_sa, y_sa, r_sa):
            return x_sa, y_sa, r_sa, val_sa

        return None

    # Initial best: grid configuration
    side = int(np.ceil(np.sqrt(N)))
    best_x = np.zeros(N)
    best_y = np.zeros(N)
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            best_x[idx] = (i + 0.5) / side
            best_y[idx] = (j + 0.5) / side
            idx += 1
    best_r = compute_radii(best_x, best_y)
    best_val = np.sum(best_r)

    # Multi-start loop with different initialization modes
    modes = ['grid', 'hex', 'random']
    num_starts = 8  # Reduced to allow more time per start with hierarchical

    for i in range(num_starts):
        if time.time() > DEADLINE - 10:
            break

        mode = modes[i % len(modes)]
        seed = i * 7 + 3

        # Use hierarchical for first few starts, fallback for later
        use_hier = i < 5 and (DEADLINE - (time.time() - start_time)) > 20
        res = run_optimization(seed, mode, use_hierarchical=use_hier)
        if res is not None:
            x_opt, y_opt, r_opt, val = res
            if val > best_val:
                best_val = val
                best_x = x_opt
                best_y = y_opt
                best_r = r_opt

    # Final validity check with projection
    if not check_validity(best_x, best_y, best_r):
        # Try projection first
        r_projected = project_feasible(best_x, best_y, best_r)
        if check_validity(best_x, best_y, r_projected):
            best_r = r_projected
        else:
            for scale in np.linspace(0.99, 0.85, 15):
                r_test = best_r * scale
                if check_validity(best_x, best_y, r_test):
                    best_r = r_test
                    break
            else:
                best_r = compute_radii(best_x, best_y)

    # Final polish if time allows
    if time.time() < DEADLINE - 5:
        final_time = min(3.0, DEADLINE - 5 - (time.time() - start_time))
        if final_time > 0.5:
            result = run_subgradient_polish(best_x, best_y, final_time)
            if result is not None:
                x_opt, y_opt, r_opt, val = result
                if val > best_val:
                    best_val = val
                    best_x = x_opt
                    best_y = y_opt
                    best_r = r_opt

    return np.column_stack((best_x, best_y, best_r))
