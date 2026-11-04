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


def _grid_init(side):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.002
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])


def _random_init(n, seed=0):
    """Random initialization with small radii."""
    rng = np.random.RandomState(seed)
    xy = rng.uniform(0.1, 0.9, size=(n, 2))
    r = rng.uniform(0.02, 0.05, size=n)
    return xy, r


def _best_fit_local_search(xy, r, deadline, grid_size=80):
    """
    Iteratively update each circle to the best position (maximizing its radius)
    given all other circles fixed. Vectorized over a grid of candidate positions.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0

    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    max_sweeps = 30
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

    return best_xy, best_r, best_total, best_valid


def _smooth_objective(v, n):
    """
    Smooth objective: minimize -sum(r) + penalty for violations.
    Uses softplus for smooth max of constraint violations.
    """
    x = v[0:n]
    y = v[n:2*n]
    r = v[2*n:3*n]

    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min

    beta = 20.0
    wall_pen = np.sum(np.log1p(np.exp(beta * wall_viol)) / beta)

    total_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            viol = r[i] + r[j] - dist
            total_viol += np.log1p(np.exp(beta * viol)) / beta

    return -np.sum(r) + 100.0 * (wall_pen + total_viol)


def _smooth_grad(v, n):
    """
    Gradient of the smooth objective using softplus surrogate.
    Vectorized over pairs for speed.
    """
    x = v[0:n].copy()
    y = v[n:2*n].copy()
    r = v[2*n:3*n].copy()

    grad = np.zeros(3 * n)
    beta = 20.0

    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    sigmoid_w = 1.0 / (1.0 + np.exp(-beta * wall_viol))

    for i in range(n):
        sw = sigmoid_w[i]
        if sw > 1e-10:
            dists = [x[i], 1.0 - x[i], y[i], 1.0 - y[i]]
            min_idx = int(np.argmin(dists))
            grad[2*n + i] += 100.0 * sw
            if min_idx == 0:
                grad[i] -= 100.0 * sw
            elif min_idx == 1:
                grad[i] += 100.0 * sw
            elif min_idx == 2:
                grad[n + i] -= 100.0 * sw
            else:
                grad[n + i] += 100.0 * sw

    # Vectorized pairwise gradient
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            viol = r[i] + r[j] - dist
            s = 1.0 / (1.0 + np.exp(-beta * viol))
            if s > 1e-10:
                factor = 100.0 * s
                if dist > 1e-12:
                    grad[i] += factor * dx / dist
                    grad[n + i] += factor * dy / dist
                    grad[j] -= factor * dx / dist
                    grad[n + j] -= factor * dy / dist
                grad[2*n + i] += factor
                grad[2*n + j] += factor

    grad[2*n:3*n] -= 1.0

    return grad


def _gradient_polish(xy, r, deadline, tol=1e-8, max_iter=500):
    """
    Gradient-based polish using L-BFGS-B on the smooth objective.
    """
    n = len(xy)
    v0 = np.concatenate([xy[:, 0], xy[:, 1], r])

    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
    for i in range(n):
        bounds.append((0.001, 0.999))
    for i in range(n):
        bounds.append((0.001, 0.5))

    def obj(v):
        if time.time() > deadline:
            return float('inf')
        return _smooth_objective(v, n)

    def grad(v):
        if time.time() > deadline:
            return np.zeros(3 * n)
        return _smooth_grad(v, n)

    try:
        result = minimize(obj, v0, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6,
                                   'maxfun': 10000})
    except Exception:
        return xy, r

    v_opt = result.x
    new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
    new_r = v_opt[2*n:3*n]

    total, valid = _eval_packing(new_xy, new_r)
    if valid:
        return new_xy, new_r

    return xy, r


def _high_precision_polish(xy, r, deadline):
    """
    Final high-precision polish: tight tolerance, more iterations.
    """
    n = len(xy)
    v0 = np.concatenate([xy[:, 0], xy[:, 1], r])

    bounds = []
    for i in range(n):
        bounds.append((0.0001, 0.9999))
    for i in range(n):
        bounds.append((0.0001, 0.9999))
    for i in range(n):
        bounds.append((0.0001, 0.4999))

    def obj(v):
        if time.time() > deadline:
            return float('inf')
        return _smooth_objective(v, n)

    def grad(v):
        if time.time() > deadline:
            return np.zeros(3 * n)
        return _smooth_grad(v, n)

    try:
        result = minimize(obj, v0, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8,
                                   'maxfun': 50000})
    except Exception:
        return xy, r

    v_opt = result.x
    new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
    new_r = v_opt[2*n:3*n]

    total, valid = _eval_packing(new_xy, new_r)
    if valid:
        return new_xy, new_r

    return xy, r


# --- NEW: Contact Graph Topology Search ---

def _build_contact_graph_from_packing(xy, r, tol=1e-3):
    """
    Build a contact graph from a valid packing. Two circles are 'in contact'
    if their distance is within tol of the sum of their radii.
    Returns adjacency list.
    """
    n = len(xy)
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if abs(dist - (r[i] + r[j])) < tol:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def _generate_random_topology(n, seed=0, max_degree=5):
    """
    Generate a random contact graph topology with n nodes.
    Ensures connectivity and bounded degree.
    """
    rng = random.Random(seed)
    adj = {i: set() for i in range(n)}
    
    # Start with a random spanning tree to ensure connectivity
    nodes = list(range(n))
    rng.shuffle(nodes)
    for i in range(1, n):
        parent = rng.choice(nodes[:i])
        child = nodes[i]
        adj[parent].add(child)
        adj[child].add(parent)
    
    # Add random extra edges up to max_degree
    for _ in range(n * 2):  # Try to add some extra edges
        i, j = rng.sample(range(n), 2)
        if j not in adj[i] and len(adj[i]) < max_degree and len(adj[j]) < max_degree:
            adj[i].add(j)
            adj[j].add(i)
    
    return adj


def _solve_topology(adj, deadline, seed=0):
    """
    Given a contact graph topology, solve for positions and radii using
    constraint propagation / optimization.
    
    We use a simplified approach: assign random initial positions, then
    use gradient descent to satisfy the tangency constraints while
    maximizing sum of radii.
    """
    n = len(adj)
    rng = np.random.RandomState(seed)
    
    # Initialize positions randomly
    x = rng.uniform(0.2, 0.8, size=n)
    y = rng.uniform(0.2, 0.8, size=n)
    r = rng.uniform(0.03, 0.05, size=n)
    
    # Constraint: for each edge (i,j), dist(i,j) = r_i + r_j
    # We want to maximize sum(r) subject to these constraints and wall constraints.
    
    def objective(v):
        x_v = v[0:n]
        y_v = v[n:2*n]
        r_v = v[2*n:3*n]
        
        # Wall violations
        wall_min = np.minimum(np.minimum(x_v, 1.0 - x_v), np.minimum(y_v, 1.0 - y_v))
        wall_viol = np.sum(np.maximum(r_v - wall_min, 0.0) ** 2)
        
        # Tangency violations
        tang_viol = 0.0
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    dx = x_v[i] - x_v[j]
                    dy = y_v[i] - y_v[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    viol = dist - (r_v[i] + r_v[j])
                    tang_viol += viol ** 2
        
        return -np.sum(r_v) + 10.0 * wall_viol + 10.0 * tang_viol
    
    def gradient(v):
        x_v = v[0:n]
        y_v = v[n:2*n]
        r_v = v[2*n:3*n]
        
        grad = np.zeros(3 * n)
        
        # Wall gradient
        wall_min = np.minimum(np.minimum(x_v, 1.0 - x_v), np.minimum(y_v, 1.0 - y_v))
        for i in range(n):
            if r_v[i] > wall_min[i]:
                dists = [x_v[i], 1.0 - x_v[i], y_v[i], 1.0 - y_v[i]]
                min_idx = int(np.argmin(dists))
                grad[2*n + i] += 20.0 * (r_v[i] - wall_min[i])
                if min_idx == 0:
                    grad[i] -= 20.0 * (r_v[i] - wall_min[i])
                elif min_idx == 1:
                    grad[i] += 20.0 * (r_v[i] - wall_min[i])
                elif min_idx == 2:
                    grad[n + i] -= 20.0 * (r_v[i] - wall_min[i])
                else:
                    grad[n + i] += 20.0 * (r_v[i] - wall_min[i])
        
        # Tangency gradient
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    dx = x_v[i] - x_v[j]
                    dy = y_v[i] - y_v[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    viol = dist - (r_v[i] + r_v[j])
                    if dist > 1e-12:
                        grad[i] += 20.0 * viol * dx / dist
                        grad[n + i] += 20.0 * viol * dy / dist
                        grad[j] -= 20.0 * viol * dx / dist
                        grad[n + j] -= 20.0 * viol * dy / dist
                        grad[2*n + i] += 20.0 * viol
                        grad[2*n + j] += 20.0 * viol
        
        grad[2*n:3*n] -= 1.0
        
        return grad
    
    v0 = np.concatenate([x, y, r])
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
    for i in range(n):
        bounds.append((0.001, 0.999))
    for i in range(n):
        bounds.append((0.001, 0.5))
    
    try:
        result = minimize(objective, v0, jac=gradient, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-6,
                                   'maxfun': 5000})
    except Exception:
        return None
    
    v_opt = result.x
    new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
    new_r = v_opt[2*n:3*n]
    
    total, valid = _eval_packing(new_xy, new_r)
    if valid:
        return new_xy, new_r, total
    
    return None


def _topology_search(deadline, num_topologies=50):
    """
    Search over random contact graph topologies.
    """
    best_xy = None
    best_r = None
    best_total = 0.0
    
    for t in range(num_topologies):
        if time.time() > deadline:
            break
        
        seed = t
        adj = _generate_random_topology(N, seed=seed, max_degree=5)
        
        result = _solve_topology(adj, deadline, seed=seed)
        if result is not None:
            xy, r, total = result
            if total > best_total:
                best_total = total
                best_xy = xy
                best_r = r
    
    return best_xy, best_r, best_total


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Phase 1: Grid initialization + local search
    for side in [5, 6, 7]:
        if time.time() > deadline:
            break
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()

        out_xy, out_r, total, valid = _best_fit_local_search(xy, r, deadline)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 2: Random initializations + local search
    for seed in range(8):
        if time.time() > deadline:
            break
        xy, r = _random_init(N, seed=seed)
        out_xy, out_r, total, valid = _best_fit_local_search(xy, r, deadline, grid_size=60)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 3: NEW - Contact Graph Topology Search
    if time.time() < deadline - 20:
        topo_xy, topo_r, topo_total = _topology_search(deadline - 10, num_topologies=30)
        if topo_xy is not None and topo_total > best_total:
            best_total = topo_total
            best_valid = True
            best_xy = topo_xy
            best_r = topo_r

    # Phase 4: Gradient polish
    if best_valid and best_xy is not None:
        if time.time() < deadline - 15:
            polished_xy, polished_r = _gradient_polish(best_xy, best_r, deadline)
            total, valid = _eval_packing(polished_xy, polished_r)
            if valid and total > best_total:
                best_total = total
                best_xy = polished_xy
                best_r = polished_r

        # Phase 5: High-precision polish
        if time.time() < deadline - 8:
            hp_xy, hp_r = _high_precision_polish(best_xy, best_r, deadline)
            total, valid = _eval_packing(hp_xy, hp_r)
            if valid and total > best_total:
                best_total = total
                best_xy = hp_xy
                best_r = hp_r

        # Phase 6: Final local search from polished result
        if time.time() < deadline - 3:
            final_xy, final_r, final_total, final_valid = _best_fit_local_search(
                best_xy, best_r, deadline, grid_size=100)
            if final_valid and final_total > best_total:
                best_total = final_total
                best_xy = final_xy
                best_r = final_r

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        total, valid = _eval_packing(xy, r)
        if valid:
            best_total = total
            best_valid = True
            best_xy = xy
            best_r = r
        else:
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
