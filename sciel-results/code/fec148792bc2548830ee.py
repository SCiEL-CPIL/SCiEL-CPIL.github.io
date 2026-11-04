"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    if n == 0:
        return 0.0, True
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True

def _generate_random_topology(n, seed):
    """
    Generate a random contact graph.
    Returns a list of edges (i, j) representing contacts.
    """
    rng = random.Random(seed)
    # Randomly shuffle nodes
    nodes = list(range(n))
    rng.shuffle(nodes)
    
    edges = set()
    # Each node connects to a few random others
    # Heuristic: ~1.5 * n edges for a stable 2D packing (triangular mesh ~3n/2)
    target_edges = int(1.5 * n) 
    
    # Simple random graph generation avoiding self-loops and duplicates
    attempts = 0
    while len(edges) < target_edges and attempts < target_edges * 10:
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u != v:
            edges.add((min(u, v), max(u, v)))
        attempts += 1
        
    return list(edges)

def _solve_topology(xy_init, r_init, edges, deadline):
    """
    Solve for circle positions and radii given a set of contact constraints.
    We want to maximize sum(r) subject to:
    1. All circles inside [0,1]^2
    2. For each edge (i,j): dist(i,j) == r_i + r_j
    3. For non-edges: dist(i,j) >= r_i + r_j
    
    Strategy:
    Use a smooth objective function that penalizes violations of the contact constraints
    and non-contact overlaps, while rewarding large radii.
    We use L-BFGS-B for speed.
    """
    n = len(xy_init)
    # Variables: x[0..n-1], y[0..n-1], r[0..n-1]
    v0 = np.concatenate([xy_init[:, 0], xy_init[:, 1], r_init])
    
    # Precompute edge indices for fast access
    edge_list = []
    for u, v in edges:
        edge_list.append((u, v))
        
    # Non-edges for penalty
    all_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            all_pairs.append((i, j))
            
    edge_set = set((min(u,v), max(u,v)) for u,v in edges)
    non_edge_pairs = [p for p in all_pairs if p not in edge_set]

    def objective(v):
        x = v[0:n]
        y = v[n:2*n]
        r = v[2*n:3*n]
        
        # Penalty for going out of bounds (soft)
        # We want r <= min(x, 1-x, y, 1-y)
        # Let's use a smooth penalty for r > min_dist_to_wall
        min_wall = np.minimum(np.minimum(x, 1.0-x), np.minimum(y, 1.0-y))
        wall_viol = np.maximum(0, r - min_wall)
        pen_wall = np.sum(wall_viol**2)
        
        # Contact constraints: dist == r_i + r_j
        # We want to minimize (dist - (r_i+r_j))^2
        contact_pen = 0.0
        for u, v in edge_list:
            dx = x[u] - x[v]
            dy = y[u] - y[v]
            dist = np.sqrt(dx*dx + dy*dy)
            target = r[u] + r[v]
            diff = dist - target
            contact_pen += diff*diff
            
        # Non-contact constraints: dist >= r_i + r_j
        # Penalty for overlap
        overlap_pen = 0.0
        for u, v in non_edge_pairs:
            dx = x[u] - x[v]
            dy = y[u] - y[v]
            dist = np.sqrt(dx*dx + dy*dy)
            target = r[u] + r[v]
            viol = target - dist
            if viol > 0:
                overlap_pen += viol*viol
                
        # Objective: Maximize sum(r) -> Minimize -sum(r)
        # Weights: 
        # -sum(r) is the main goal.
        # Penalties must be strong enough to keep constraints satisfied.
        # However, if we make penalties too strong, we might shrink circles to satisfy them easily.
        # Better: Use a barrier or large multiplier.
        
        # Let's try: -sum(r) + 1000*contact_pen + 1000*overlap_pen + 1000*pen_wall
        # This might be stiff. Let's use a smooth approximation for the objective.
        
        return -np.sum(r) + 500.0 * (contact_pen + overlap_pen + pen_wall)

    def gradient(v):
        x = v[0:n].copy()
        y = v[n:2*n].copy()
        r = v[2*n:3*n].copy()
        
        grad = np.zeros(3*n)
        
        # Gradient of -sum(r)
        grad[2*n:3*n] -= 1.0
        
        # Gradient of wall penalty: 2 * (r - min_wall) * d(r - min_wall)/dv
        min_wall = np.minimum(np.minimum(x, 1.0-x), np.minimum(y, 1.0-y))
        wall_viol = np.maximum(0, r - min_wall)
        
        for i in range(n):
            if wall_viol[i] > 0:
                factor = 2.0 * 500.0 * wall_viol[i]
                # d(r_i - min_wall_i)/dr_i = 1
                grad[2*n + i] += factor
                # d(r_i - min_wall_i)/dx_i
                # min_wall_i = min(x_i, 1-x_i, y_i, 1-y_i)
                # If x_i is the min, d/dx_i (r_i - x_i) = -1
                # If 1-x_i is min, d/dx_i (r_i - (1-x_i)) = +1
                # etc.
                if min_wall[i] == x[i]:
                    grad[i] -= factor
                elif min_wall[i] == 1.0 - x[i]:
                    grad[i] += factor
                elif min_wall[i] == y[i]:
                    grad[n + i] -= factor
                else:
                    grad[n + i] += factor

        # Gradient of contact penalty
        for u, v in edge_list:
            dx = x[u] - x[v]
            dy = y[u] - y[v]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1e-12: continue
            target = r[u] + r[v]
            diff = dist - target
            # d/dv (diff^2) = 2 * diff * d(diff)/dv
            # d(diff)/dx_u = dx/dist, d(diff)/dx_v = -dx/dist
            # d(diff)/dr_u = 1, d(diff)/dr_v = 1
            factor = 2.0 * 500.0 * diff
            if dist > 1e-12:
                grad[u] += factor * dx / dist
                grad[n + u] += factor * dy / dist
                grad[v] -= factor * dx / dist
                grad[n + v] -= factor * dy / dist
            grad[2*n + u] += factor
            grad[2*n + v] += factor

        # Gradient of overlap penalty
        for u, v in non_edge_pairs:
            dx = x[u] - x[v]
            dy = y[u] - y[v]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1e-12: continue
            target = r[u] + r[v]
            viol = target - dist
            if viol > 0:
                factor = 2.0 * 500.0 * viol
                # d(viol)/dx_u = -dx/dist, etc
                # d(viol)/dr_u = 1
                if dist > 1e-12:
                    grad[u] -= factor * dx / dist
                    grad[n + u] -= factor * dy / dist
                    grad[v] += factor * dx / dist
                    grad[n + v] += factor * dy / dist
                grad[2*n + u] += factor
                grad[2*n + v] += factor

        return grad

    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.001, 0.5))

    try:
        result = minimize(objective, v0, jac=gradient, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8,
                                   'maxfun': 1000})
    except Exception:
        return xy_init, r_init

    v_opt = result.x
    new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
    new_r = v_opt[2*n:3*n]
    
    return new_xy, new_r

def _refine_with_local_search(xy, r, deadline, grid_size=50):
    """
    Quick local search to fix any small overlaps or improve radii.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    
    if not best_valid:
        return xy, r

    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    max_sweeps = 5
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]

            if len(others_xy) > 0:
                dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)
                r_candidates = np.minimum(wall_clear, circle_clear)
            else:
                r_candidates = wall_clear.copy()

            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]

            old_r = r[i]
            if new_r > old_r + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True

        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Initial grid packing as a baseline
    side = int(np.ceil(np.sqrt(N)))
    xy_init = np.zeros((N, 2))
    r_init = np.zeros(N)
    for i in range(N):
        gx = i % side
        gy = i // side
        xy_init[i, 0] = (gx + 0.5) / side
        xy_init[i, 1] = (gy + 0.5) / side
        r_init[i] = 0.5 / side - 0.002
    
    total, valid = _eval_packing(xy_init, r_init)
    if valid:
        best_total = total
        best_valid = True
        best_xy = xy_init.copy()
        best_r = r_init.copy()

    # Topology Search
    # Try multiple random topologies
    num_topologies = 50
    for t in range(num_topologies):
        if time.time() > deadline - 5:
            break
            
        seed = t * 12345 + 6789
        edges = _generate_random_topology(N, seed)
        
        # Start from the current best or a random perturbation
        # Using current best helps if the topology is similar, random helps escape local minima
        if t % 2 == 0 and best_valid:
            xy_start = best_xy + np.random.normal(0, 0.05, (N, 2))
            r_start = best_r + np.random.normal(0, 0.02, N)
            # Clip to valid range
            for i in range(N):
                xy_start[i, 0] = np.clip(xy_start[i, 0], r_start[i], 1.0-r_start[i])
                xy_start[i, 1] = np.clip(xy_start[i, 1], r_start[i], 1.0-r_start[i])
                r_start[i] = np.clip(r_start[i], 0.01, 0.4)
        else:
            xy_start = xy_init + np.random.normal(0, 0.1, (N, 2))
            r_start = np.full(N, 0.1)
            for i in range(N):
                xy_start[i, 0] = np.clip(xy_start[i, 0], r_start[i], 1.0-r_start[i])
                xy_start[i, 1] = np.clip(xy_start[i, 1], r_start[i], 1.0-r_start[i])

        sol_xy, sol_r = _solve_topology(xy_start, r_start, edges, deadline)
        
        # Refine with local search
        ref_xy, ref_r = _refine_with_local_search(sol_xy, sol_r, deadline)
        
        total, valid = _eval_packing(ref_xy, ref_r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = ref_xy
            best_r = ref_r

    # Final polish with gradient on the best found
    if best_valid and time.time() < deadline - 10:
        # Use the smooth objective from the current program's style for a final push
        # But since we are changing strategy, let's just do a high-res local search
        final_xy, final_r = _refine_with_local_search(best_xy, best_r, deadline, grid_size=100)
        total, valid = _eval_packing(final_xy, final_r)
        if valid and total > best_total:
            best_total = total
            best_xy = final_xy
            best_r = final_r

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
        return result

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
            result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
