"""Packing 200 equal circles in a unit square using Delaunay-based optimization."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay, cKDTree
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import minimize


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _get_delaunay_info(c):
    """
    Get Delaunay triangulation and identify critical triangles.
    Returns:
        tri: Delaunay triangulation
        min_circumradius: minimum circumradius among triangles
        critical_indices: indices of triangles with small circumradius
    """
    try:
        tri = Delaunay(c)
    except Exception:
        return None, 0.0, []
    
    simplices = tri.simplices
    n_simp = len(simplices)
    if n_simp == 0:
        return tri, 0.0, []
    
    # Compute circumradius for each triangle
    circumradii = np.zeros(n_simp)
    for i, (a, b, c_idx) in enumerate(simplices):
        pa, pb, pc = c[a], c[b], c[c_idx]
        # Side lengths
        ab = np.linalg.norm(pa - pb)
        bc = np.linalg.norm(pb - pc)
        ca = np.linalg.norm(pc - pa)
        # Area
        area = 0.5 * np.abs(np.cross(pb - pa, pc - pa))
        if area < 1e-15:
            circumradii[i] = 0.0
        else:
            circumradii[i] = ab * bc * ca / (4.0 * area)
    
    min_circ = np.min(circumradii)
    
    # Find critical triangles (those with circumradius close to minimum)
    threshold = min_circ * 1.05 + 1e-6
    critical_indices = np.where(circumradii <= threshold)[0]
    
    return tri, min_circ, critical_indices


def _delaunay_gradient(c, step_size=1e-5):
    """
    Compute gradient of the minimum circumradius w.r.t. circle centers.
    Uses finite differences on the Delaunay structure.
    """
    n = len(c)
    grad = np.zeros_like(c)
    
    tri, min_circ, critical_indices = _get_delaunay_info(c)
    if tri is None or len(critical_indices) == 0:
        return grad, min_circ
    
    # For each critical triangle, compute how moving each vertex affects its circumradius
    simplices = tri.simplices
    
    for tri_idx in critical_indices:
        a, b, c_idx = simplices[tri_idx]
        pa, pb, pc = c[a].copy(), c[b].copy(), c[c_idx].copy()
        
        # Compute circumradius
        ab = np.linalg.norm(pa - pb)
        bc = np.linalg.norm(pb - pc)
        ca = np.linalg.norm(pc - pa)
        area = 0.5 * np.abs(np.cross(pb - pa, pc - pa))
        if area < 1e-15:
            continue
        circ_r = ab * bc * ca / (4.0 * area)
        
        if circ_r > min_circ * 1.1:
            continue
        
        # Gradient of circumradius w.r.t. each vertex
        # R = abc/(4A) where A is area
        # dR/dpa involves derivatives of ab, ca, and A
        
        # Use finite differences for simplicity
        for vertex_idx, point in [(a, pa), (b, pb), (c_idx, pc)]:
            for dim in range(2):
                c_plus = c.copy()
                c_plus[vertex_idx, dim] += step_size
                _, _, crit_plus = _get_delaunay_info(c_plus)
                if len(crit_plus) > 0:
                    # Recompute circumradius for this specific triangle
                    a2, b2, c2 = simplices[tri_idx]
                    pa2, pb2, pc2 = c_plus[a2], c_plus[b2], c_plus[c2]
                    ab2 = np.linalg.norm(pa2 - pb2)
                    bc2 = np.linalg.norm(pb2 - pc2)
                    ca2 = np.linalg.norm(pc2 - pa2)
                    area2 = 0.5 * np.abs(np.cross(pb2 - pa2, pc2 - pa2))
                    if area2 > 1e-15:
                        circ_r_plus = ab2 * bc2 * ca2 / (4.0 * area2)
                    else:
                        circ_r_plus = 0.0
                else:
                    circ_r_plus = circ_r
                
                c_minus = c.copy()
                c_minus[vertex_idx, dim] -= step_size
                a2, b2, c2 = simplices[tri_idx]
                pa2, pb2, pc2 = c_minus[a2], c_minus[b2], c_minus[c2]
                ab2 = np.linalg.norm(pa2 - pb2)
                bc2 = np.linalg.norm(pb2 - pc2)
                ca2 = np.linalg.norm(pc2 - pa2)
                area2 = 0.5 * np.abs(np.cross(pb2 - pa2, pc2 - pa2))
                if area2 > 1e-15:
                    circ_r_minus = ab2 * bc2 * ca2 / (4.0 * area2)
                else:
                    circ_r_minus = 0.0
                
                grad[vertex_idx, dim] += (circ_r_plus - circ_r_minus) / (2 * step_size)
    
    return grad, min_circ


def _soft_min_objective(c, p=30):
    """
    Soft minimum of pairwise distances/2 and boundary distances.
    Returns negative value (to be minimized).
    """
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    dists = pdist(c, metric='euclidean')
    
    # Soft min using power sum
    # Focus on close pairs
    mask = dists < 0.15
    if np.any(mask):
        soft_min_pair = np.sum(dists[mask]**p)**(-1.0/p)
    else:
        soft_min_pair = np.min(dists)
    
    soft_min_boundary = np.sum(boundary_dists**p)**(-1.0/p)
    
    r_pair = soft_min_pair / 2.0
    r_bound = soft_min_boundary
    
    # Soft minimum of the two
    T = 0.01
    soft_min_combined = -T * np.logaddexp(-r_pair/T, -r_bound/T)
    
    return -soft_min_combined


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.055, 0.075, 25):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2 + 1e-9:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2 + 1e-9:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        if len(points) >= N:
            c = np.array(points[:N])
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
        elif len(points) > 0 and best_c is None:
            c = np.array(points)
            if len(c) < N:
                extra = np.random.uniform(-0.45, 0.45, size=(N - len(c), 2))
                c = np.vstack([c, extra])
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N].copy()
    
    return best_c


def _local_optimize(c0, max_time=30.0):
    """Local optimization using L-BFGS-B with soft-min energy."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    temperatures = [0.008, 0.012, 0.018, 0.025]
    
    for T in temperatures:
        if time.time() > start_time + max_time - 2:
            break
        
        def obj(params):
            c = params.reshape(-1, 2)
            c = np.clip(c, -0.4999, 0.4999)
            return _soft_min_objective(c, p=30)
        
        try:
            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 80}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            continue
    
    return best_c, best_r


def _delaunay_refine(c0, max_time=40.0):
    """
    Refine using Delaunay-based gradient ascent on minimum circumradius.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    lr = 0.003
    momentum = 0.9
    vel = np.zeros_like(c0)
    
    while time.time() < start_time + max_time - 2:
        # Compute gradient of soft min
        c_clipped = np.clip(best_c, -0.4999, 0.4999)
        
        # Use soft min gradient via finite differences
        eps = 1e-5
        grad = np.zeros_like(c_clipped)
        base_val = _soft_min_objective(c_clipped, p=30)
        
        for i in range(0, 2*N, 2):
            for dim in range(2):
                c_plus = c_clipped.flatten().copy()
                c_plus[i+dim] += eps
                val_plus = _soft_min_objective(c_plus.reshape(-1, 2), p=30)
                
                c_minus = c_clipped.flatten().copy()
                c_minus[i+dim] -= eps
                val_minus = _soft_min_objective(c_minus.reshape(-1, 2), p=30)
                
                grad[i+dim] = (val_plus - val_minus) / (2 * eps)
        
        # Gradient ascent (we're minimizing, so go opposite)
        vel = momentum * vel - lr * grad
        c_new = c_clipped + vel.reshape(-1, 2)
        c_new = np.clip(c_new, -0.4999, 0.4999)
        
        r_new = _compute_radius(c_new)
        if r_new > best_r:
            best_c = c_new
            best_r = r_new
        else:
            # Reduce learning rate
            lr *= 0.95
            vel *= 0.5
        
        if lr < 1e-7:
            break
    
    return best_c, best_r


def _anneal_optimize(c0, max_time=35.0):
    """Simulated-annealing-like optimization with cluster moves."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    T = 0.04
    T_min = 0.003
    cooling = 0.96
    
    c_current = c0.copy()
    r_current = best_r
    
    step_size = 0.015
    iteration = 0
    
    # Precompute clusters
    tree = cKDTree(c_current)
    clusters = []
    visited = np.zeros(N, dtype=bool)
    for i in range(N):
        if not visited[i]:
            cluster = [i]
            visited[i] = True
            queue = [i]
            while queue:
                idx = queue.pop(0)
                neighbors = tree.query_ball_point(c_current[idx], 0.12)
                for nb in neighbors:
                    if not visited[nb]:
                        visited[nb] = True
                        cluster.append(nb)
                        queue.append(nb)
            if len(cluster) > 1:
                clusters.append(cluster)
    
    while time.time() < start_time + max_time - 2:
        iteration += 1
        
        if iteration % 100 == 1:
            tree = cKDTree(c_current)
            clusters = []
            visited = np.zeros(N, dtype=bool)
            for i in range(N):
                if not visited[i]:
                    cluster = [i]
                    visited[i] = True
                    queue = [i]
                    while queue:
                        idx = queue.pop(0)
                        neighbors = tree.query_ball_point(c_current[idx], 0.12)
                        for nb in neighbors:
                            if not visited[nb]:
                                visited[nb] = True
                                cluster.append(nb)
                                queue.append(nb)
                    if len(cluster) > 1:
                        clusters.append(cluster)
        
        if np.random.random() < 0.7 or len(clusters) == 0:
            c_trial = c_current.copy()
            idx = np.random.randint(N)
            c_trial[idx] += np.random.normal(0, step_size, size=2)
        else:
            c_trial = c_current.copy()
            cluster = clusters[np.random.randint(len(clusters))]
            displacement = np.random.normal(0, step_size * 0.6, size=2)
            for idx in cluster:
                c_trial[idx] += displacement
        
        c_trial = np.clip(c_trial, -0.4999, 0.4999)
        r_trial = _compute_radius(c_trial)
        
        if r_trial >= r_current:
            accept = True
        else:
            delta = r_trial - r_current
            prob = np.exp(delta / T)
            accept = np.random.random() < prob
        
        if accept:
            c_current = c_trial
            r_current = r_trial
            if r_current > best_r:
                best_r = r_current
                best_c = c_current.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
            step_size *= 0.75
    
    return best_c, best_r


def _slsqp_polish(c0, max_time=25.0):
    """SLSQP-based polish with penalty-based objective."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    r_target = best_r
    
    for polish_iter in range(8):
        if time.time() > start_time + max_time - 2:
            break
        
        def objective_fixed_r(params):
            c = params.reshape(-1, 2)
            boundary_viol = np.maximum(0, np.abs(c) - (0.5 - r_target))
            boundary_penalty = np.sum(boundary_viol**2)
            
            dists = pdist(c, metric='euclidean')
            pair_viol = np.maximum(0, 2 * r_target - dists)
            pair_penalty = np.sum(pair_viol**2)
            
            return boundary_penalty + pair_penalty
        
        try:
            result = minimize(
                objective_fixed_r,
                best_c.flatten(),
                method='SLSQP',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 150, 'ftol': 1e-13, 'disp': False}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                r_target = r_new
            else:
                r_target = best_r * 0.9995
        except Exception:
            break
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_radius = 0.0
    best_c = None
    
    # Step 1: Hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    best_radius = r_hex
    best_c = c_hex.copy()
    
    # Step 2: Local optimization from hex
    if time.time() < deadline - 50:
        max_opt_time = min(35.0, deadline - time.time() - 10)
        c_opt, r_opt = _local_optimize(c_hex, max_time=max_opt_time)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Step 3: Delaunay-based refinement
    if time.time() < deadline - 40:
        remaining = deadline - time.time() - 8
        if remaining > 10:
            c_del, r_del = _delaunay_refine(best_c, max_time=min(25.0, remaining))
            if r_del > best_radius:
                best_radius = r_del
                best_c = c_del.copy()
    
    # Step 4: Multiple random restarts
    np.random.seed(42)
    n_restarts = 4
    
    for trial in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        m = int(np.ceil(np.sqrt(N)))
        grid_x, grid_y = np.meshgrid(
            np.linspace(-0.45, 0.45, m),
            np.linspace(-0.45, 0.45, m)
        )
        grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        
        np.random.shuffle(grid)
        c_init = grid[:N].copy()
        c_init += np.random.normal(0, 0.01, size=(N, 2))
        c_init = np.clip(c_init, -0.4999, 0.4999)
        
        r_init = _compute_radius(c_init)
        if r_init > best_radius:
            best_radius = r_init
            best_c = c_init.copy()
        
        remaining = deadline - time.time() - 10
        if remaining > 8:
            max_opt_time = min(18.0, remaining)
            c_opt, r_opt = _local_optimize(c_init, max_time=max_opt_time)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Step 5: Annealing
    if time.time() < deadline - 15:
        remaining = deadline - time.time() - 8
        if remaining > 10:
            c_ann, r_ann = _anneal_optimize(best_c, max_time=min(20.0, remaining))
            if r_ann > best_radius:
                best_radius = r_ann
                best_c = c_ann.copy()
    
    # Step 6: Final SLSQP polish
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        if remaining > 5:
            c_pol, r_pol = _slsqp_polish(best_c, max_time=min(15.0, remaining))
            if r_pol > best_radius:
                best_radius = r_pol
                best_c = c_pol.copy()
    
    # Final clip to ensure validity
    best_c = np.clip(best_c, -0.5, 0.5)
    
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        best_c = _initialize_hexagonal()
        best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
