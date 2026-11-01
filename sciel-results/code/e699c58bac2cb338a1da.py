import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay, ConvexHull

N = 50

def _score(centers):
    """Compute the maximum common radius for given centers (exact, for evaluation)."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        return -np.inf
    wall_min = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_min)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    return min(r_wall, r_pair)

def _soft_min(values, beta=50.0):
    """Smooth approximation to min via negative log-sum-exp."""
    values = np.asarray(values)
    m = np.min(values)
    exponent = -beta * (values - m)
    exponent = np.clip(exponent, -700, 700)
    log_sum = np.log(np.sum(np.exp(exponent)))
    return m - log_sum / beta

def _epigraph_objective(x, beta=100.0):
    """
    Epigraph formulation: minimize -t + penalty
    where t is bounded above by all constraints (softly).
    x = [centers (150), t (1)]
    """
    c = x[:3*N].reshape(N, 3)
    t = x[3*N]
    
    # Wall constraints: 0.5 - |c_ik|
    wall_constraints = 0.5 - np.abs(c).ravel()
    
    # Pairwise half-distances using Delaunay neighbors for efficiency
    # But for correctness in optimization, use all pairs (N=50 is small)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_constraints = dists / 2.0
    pair_constraints = pair_constraints[np.triu_indices_from(pair_constraints, k=1)]
    
    all_constraints = np.concatenate([wall_constraints, pair_constraints])
    
    # Soft constraint: t should be <= all constraints
    # Penalty: sum of max(0, t - constraint)^2 for each constraint
    violations = np.maximum(0, t - all_constraints)
    penalty = np.sum(violations ** 2)
    
    # Also penalize t going too high
    # Objective: -t + penalty
    return -t + penalty

def _epigraph_objective_grad(x, beta=100.0):
    """Gradient of epigraph objective."""
    c = x[:3*N].reshape(N, 3)
    t = x[3*N]
    
    # Wall constraints: 0.5 - |c_ik|
    wall_constraints = 0.5 - np.abs(c).ravel()
    
    # Pairwise half-distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_constraints = dists / 2.0
    pair_constraints = pair_constraints[np.triu_indices_from(pair_constraints, k=1)]
    
    all_constraints = np.concatenate([wall_constraints, pair_constraints])
    
    # Gradient of penalty
    violations = np.maximum(0, t - all_constraints)
    active = violations > 0
    
    grad = np.zeros_like(x)
    
    # dt = -1 + 2 * sum(violations)
    grad[3*N] = -1.0 + 2.0 * np.sum(violations)
    
    # For wall constraints: constraint_i = 0.5 - |c[k,i]|
    # d(constraint)/d(c[k,i]) = -sign(c[k,i])
    # d(penalty)/d(c[k,i]) = 2 * violation * d(constraint)/d(c[k,i])
    wall_violations = violations[:3*N]
    for i in range(3*N):
        if wall_violations[i] > 0:
            sphere_idx = i // 3
            coord_idx = i % 3
            sign = np.sign(c[sphere_idx, coord_idx])
            if sign == 0:
                sign = 1.0
            grad[i] += 2.0 * wall_violations[i] * (-sign)
    
    # For pair constraints: constraint = |c_i - c_j| / 2
    # d(constraint)/d(c_i) = (c_i - c_j) / |c_i - c_j|
    # d(constraint)/d(c_j) = -(c_i - c_j) / |c_i - c_j|
    pair_violations = violations[3*N:]
    idx_i, idx_j = np.triu_indices(N, k=1)
    for p in range(len(pair_violations)):
        if pair_violations[p] > 0:
            i, j = idx_i[p], idx_j[p]
            d = c[i] - c[j]
            norm_d = np.linalg.norm(d)
            if norm_d > 1e-12:
                grad_dir = d / norm_d
                grad[3*i:3*i+3] += 2.0 * pair_violations[p] * grad_dir
                grad[3*j:3*j+3] += 2.0 * pair_violations[p] * (-grad_dir)
    
    return grad

def _delaunay_refine(centers, n_iters=30, step_size=0.01):
    """
    Use Delaunay triangulation to identify contact pairs and perform
    targeted optimization on those pairs.
    """
    centers = centers.copy()
    best_score = _score(centers)
    
    for it in range(n_iters):
        # Build Delaunay triangulation
        try:
            tri = Delaunay(centers)
        except Exception:
            break
        
        # Get unique edges from Delaunay
        edges = set()
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i+1, len(simplex)):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a,b), max(a,b)))
        
        # Also add wall contacts: spheres near walls
        wall_contacts = []
        for i in range(N):
            for coord in range(3):
                if abs(centers[i, coord]) > 0.45:
                    wall_contacts.append((i, coord))
        
        # Compute current radius
        cur_score = _score(centers)
        
        # For each contact pair, try to increase the distance
        improved = False
        for (i, j) in edges:
            d = centers[i] - centers[j]
            norm_d = np.linalg.norm(d)
            if norm_d < 1e-10:
                continue
            direction = d / norm_d
            
            # Try moving i away from j and vice versa
            for delta in [step_size, step_size*0.5, -step_size, -step_size*0.5]:
                new_c = centers.copy()
                new_c[i] += delta * direction * 0.5
                new_c[j] -= delta * direction * 0.5
                # Clip to cube
                new_c = np.clip(new_c, -0.5, 0.5)
                s = _score(new_c)
                if s > cur_score + 1e-12:
                    centers = new_c
                    cur_score = s
                    improved = True
                    break
        
        # Try moving wall-contact spheres away from walls
        for (i, coord) in wall_contacts:
            sign = np.sign(centers[i, coord])
            if sign == 0:
                continue
            for delta in [step_size, step_size*0.5]:
                new_c = centers.copy()
                new_c[i, coord] -= delta * sign
                new_c = np.clip(new_c, -0.5, 0.5)
                s = _score(new_c)
                if s > cur_score + 1e-12:
                    centers = new_c
                    cur_score = s
                    improved = True
                    break
        
        if cur_score > best_score:
            best_score = cur_score
        
        if not improved:
            step_size *= 0.7
            if step_size < 1e-6:
                break
    
    return centers, best_score

def _initial_grid():
    """4x4x4 grid initialization, pick 50."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(grid))
    return grid[perm[:N]].copy()

def _initial_fcc_like():
    """FCC-like layered hexagonal initialization."""
    centers = np.zeros((N, 3))
    n_layers = 5
    per_layer = N // n_layers
    remaining = N - per_layer * n_layers
    
    for layer in range(n_layers):
        z = (layer + 0.5) / n_layers - 0.5
        n_in_layer = per_layer + (1 if layer < remaining else 0)
        
        for i in range(n_in_layer):
            row = i // 4
            col = i % 4
            x = (col + 0.5) / 4.0 - 0.5
            y = (row + 0.5) / 3.0 - 0.5
            if row % 2 == 1:
                x += 0.25 / 4.0
            idx = layer * per_layer + i
            if idx < N:
                centers[idx] = [x, y, z]
    
    # Fill any remaining
    idx = per_layer * n_layers
    while idx < N:
        centers[idx] = [0, 0, 0]
        idx += 1
    
    return centers

def _lloyd_step(centers, n_iters=20, alpha=0.3):
    """
    Lloyd's algorithm: move each center toward the centroid of its Voronoi cell,
    clipped to the cube. This equalizes the influence regions.
    """
    centers = centers.copy()
    for _ in range(n_iters):
        # Compute Voronoi centroids using a simple nearest-neighbor partition
        try:
            from scipy.spatial import Voronoi
            vor = Voronoi(centers)
            
            new_centers = centers.copy()
            for i in range(N):
                region_idx = vor.point_region[i]
                region = vor.regions[region_idx]
                if -1 in region or len(region) == 0:
                    continue
                verts = vor.vertices[region]
                mask = (np.all(verts >= -0.5, axis=1) & np.all(verts <= 0.5, axis=1))
                if np.sum(mask) > 0:
                    centroid = np.mean(verts[mask], axis=0)
                    new_centers[i] = (1 - alpha) * centers[i] + alpha * centroid
            centers = new_centers
        except Exception:
            for i in range(N):
                dists = np.linalg.norm(centers[i] - centers, axis=1)
                dists[i] = np.inf
                j = np.argmin(dists)
                if dists[j] > 1e-10:
                    direction = centers[i] - centers[j]
                    direction /= np.linalg.norm(direction)
                    new_pos = centers[i] + alpha * 0.01 * direction
                    if np.all(np.abs(new_pos) <= 0.5):
                        centers[i] = new_pos
            for i in range(N):
                for k in range(3):
                    if centers[i, k] > 0.4:
                        centers[i, k] -= alpha * 0.01
                    elif centers[i, k] < -0.4:
                        centers[i, k] += alpha * 0.01
    
    return centers

def _polish_coordinate_descent(centers, deadline, max_rounds=200):
    """
    Final polish via coordinate descent on the exact objective.
    """
    best_score = _score(centers)
    step = 0.001
    
    for _ in range(max_rounds):
        if time.time() > deadline - 3:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline - 3:
                break
            for coord in range(3):
                for delta in [step, -step, step*0.5, -step*0.5, step*2, -step*2]:
                    new_c = centers.copy()
                    new_c[i, coord] += delta
                    if np.all(np.abs(new_c) <= 0.5):
                        s = _score(new_c)
                        if s > best_score + 1e-12:
                            centers = new_c
                            best_score = s
                            improved = True
        if not improved:
            step *= 0.7
            if step < 1e-7:
                break
    
    return centers, best_score

def _delaunay_graph_optimize(centers, deadline):
    """
    Delaunay-based graph optimization: use the contact graph from Delaunay
    to guide local optimization, focusing on the critical constraints.
    """
    centers = centers.copy()
    best_score = _score(centers)
    
    # Build initial Delaunay graph
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i+1, len(simplex)):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a,b), max(a,b)))
    except Exception:
        edges = set()
        # Fallback: use k-nearest neighbors
        for i in range(N):
            dists = np.linalg.norm(centers[i] - centers, axis=1)
            dists[i] = np.inf
            nn = np.argsort(dists)[:6]
            for j in nn:
                edges.add((min(i,j), max(i,j)))
    
    # Iteratively refine using the contact graph
    step_size = 0.005
    for it in range(50):
        if time.time() > deadline - 5:
            break
        
        cur_score = _score(centers)
        improved = False
        
        # Update Delaunay periodically
        if it % 10 == 0:
            try:
                tri = Delaunay(centers)
                edges = set()
                for simplex in tri.simplices:
                    for i in range(len(simplex)):
                        for j in range(i+1, len(simplex)):
                            a, b = simplex[i], simplex[j]
                            edges.add((min(a,b), max(a,b)))
            except Exception:
                pass
        
        # Optimize along each edge direction
        for (i, j) in sorted(edges):
            if time.time() > deadline - 5:
                break
            d = centers[i] - centers[j]
            norm_d = np.linalg.norm(d)
            if norm_d < 1e-10:
                continue
            direction = d / norm_d
            
            # Try symmetric expansion
            for delta in [step_size, -step_size, step_size*0.5, -step_size*0.5]:
                new_c = centers.copy()
                new_c[i] += delta * direction * 0.5
                new_c[j] -= delta * direction * 0.5
                new_c = np.clip(new_c, -0.5, 0.5)
                s = _score(new_c)
                if s > cur_score + 1e-12:
                    centers = new_c
                    cur_score = s
                    improved = True
                    break
        
        # Wall optimization for boundary spheres
        for i in range(N):
            if time.time() > deadline - 5:
                break
            for coord in range(3):
                if abs(centers[i, coord]) > 0.4:
                    sign = np.sign(centers[i, coord])
                    if sign == 0:
                        continue
                    for delta in [step_size, step_size*0.5]:
                        new_c = centers.copy()
                        new_c[i, coord] -= delta * sign
                        new_c = np.clip(new_c, -0.5, 0.5)
                        s = _score(new_c)
                        if s > cur_score + 1e-12:
                            centers = new_c
                            cur_score = s
                            improved = True
                            break
        
        if cur_score > best_score:
            best_score = cur_score
        
        if not improved:
            step_size *= 0.7
            if step_size < 1e-7:
                break
    
    return centers, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    # Multiple initializations
    inits = [
        ("grid", _initial_grid()),
        ("fcc", _initial_fcc_like()),
    ]
    
    # Add a perturbed grid
    rng = np.random.RandomState(123)
    perturbed = _initial_grid() + rng.normal(0, 0.02, (N, 3))
    perturbed = np.clip(perturbed, -0.5, 0.5)
    inits.append(("perturbed", perturbed))
    
    best_centers = None
    best_score = -np.inf
    
    for name, init in inits:
        if time.time() > deadline - 25:
            break
        
        cur = init.copy()
        cur_score = _score(cur)
        
        if cur_score > best_score:
            best_score = cur_score
            best_centers = cur.copy()
        
        # Phase 1: Lloyd's algorithm for structural improvement
        if time.time() < deadline - 20:
            lloyd_centers = _lloyd_step(cur, n_iters=15, alpha=0.25)
            lloyd_score = _score(lloyd_centers)
            if lloyd_score > cur_score:
                cur = lloyd_centers
                cur_score = lloyd_score
                if cur_score > best_score:
                    best_score = cur_score
                    best_centers = cur.copy()
        
        # Phase 2: Delaunay-based graph optimization (NEW)
        if time.time() < deadline - 15:
            delaunay_centers, delaunay_score = _delaunay_graph_optimize(cur, deadline)
            if delaunay_score > best_score:
                best_score = delaunay_score
                best_centers = delaunay_centers.copy()
                cur = delaunay_centers.copy()
        
        # Phase 3: Epigraph optimization with SLSQP
        if time.time() < deadline - 10:
            try:
                # Initialize x = [centers, t]
                x0 = np.concatenate([cur.ravel(), [_score(cur)]])
                
                res = minimize(
                    _epigraph_objective,
                    x0,
                    jac=_epigraph_objective_grad,
                    method='SLSQP',
                    bounds=[(-0.5, 0.5)] * 3*N + [(0.0, 0.5)],
                    options={'maxiter': 100, 'ftol': 1e-10, 'disp': False}
                )
                
                new_centers = res.x[:3*N].reshape(N, 3)
                new_score = _score(new_centers)
                
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_centers.copy()
                    cur = new_centers.copy()
            except Exception:
                pass
        
        # Phase 4: Final coordinate descent polish
        if time.time() < deadline - 5:
            cur = best_centers.copy()
            polished, polished_score = _polish_coordinate_descent(cur, deadline)
            if polished_score > best_score:
                best_score = polished_score
                best_centers = polished
    
    if best_centers is None:
        return _initial_grid()
    return best_centers
