"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree


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


def _soft_min(values, beta):
    """Smooth approximation of min using log-sum-exp."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    v_min = np.min(values)
    shifted = values - v_min
    shifted = np.clip(shifted, 0, 100.0 / beta)
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return v_min - log_sum / beta


def _smooth_objective(params, beta):
    """Smooth objective: minimize -soft_min(radii_components)."""
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    boundary_dists = 0.5 - np.abs(c)
    if n < 2:
        return -np.min(boundary_dists)
    dists = pdist(c, metric='euclidean')
    pair_dists_half = dists / 2.0
    all_dists = np.concatenate([pair_dists_half, boundary_dists.ravel()])
    r_soft = _soft_min(all_dists, beta)
    if r_soft <= 0:
        return 1.0
    return -r_soft


def _energy_focused(params, temperature=0.02):
    """Focused energy: only considers pairs within a threshold of the current minimum."""
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    boundary_dists = 0.5 - np.abs(c)
    b_min = np.min(boundary_dists)
    if N < 2:
        return -b_min
    dists = pdist(c, metric='euclidean')
    d_min = np.min(dists)
    T = max(temperature, 1e-8)
    threshold = d_min * 1.15 + 1e-6
    mask = dists < threshold
    if np.any(mask):
        d_sub = dists[mask]
        d_min_sub = np.min(d_sub)
        soft_min_pair = d_min_sub - T * np.log(np.sum(np.exp(-(d_sub - d_min_sub) / T)))
    else:
        soft_min_pair = d_min
    b_threshold = b_min * 1.15 + 1e-6
    b_mask = boundary_dists < b_threshold
    if np.any(b_mask):
        b_sub = boundary_dists[b_mask]
        b_min_sub = np.min(b_sub)
        soft_min_bound = b_min_sub - T * np.log(np.sum(np.exp(-(b_sub - b_min_sub) / T)))
    else:
        soft_min_bound = b_min
    r_pair = soft_min_pair / 2.0
    r_bound = soft_min_bound
    r_vals = np.array([r_pair, r_bound])
    m = np.min(r_vals)
    soft_min_combined = m - T * np.log(np.sum(np.exp(-(r_vals - m) / T)))
    return -soft_min_combined


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    for d in np.linspace(0.055, 0.075, 30):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    if best_c is not None:
        return best_c
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _local_optimize_smooth(c0, max_time=30.0):
    """Local optimization using smooth objective with annealing."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    betas = [15.0, 30.0, 60.0, 120.0, 250.0, 500.0, 1000.0, 2000.0]
    for beta in betas:
        if time.time() > start_time + max_time - 3:
            break
        try:
            def obj(params):
                return _smooth_objective(params, beta)
            maxiter = 80 if beta < 100 else 150
            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    return best_c, best_r


def _local_optimize_focused(c0, max_time=20.0, temps=None):
    """Local optimization using the focused energy function."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    if temps is None:
        temps = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    current_c = c0.copy()
    for T in temps:
        if time.time() > start_time + max_time - 1:
            break
        def obj(params, _T=T):
            return _energy_focused(params, temperature=_T)
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 80}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
            current_c = c_new
        except Exception:
            continue
    return best_c, best_r


def _simulated_annealing_rigid(c0, r0, max_time=40.0):
    """Simulated annealing with rigid-body cluster moves to escape local minima."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    current_c = c0.copy()
    current_r = r0
    T_start = 0.001
    T_end = 1e-8
    n_steps = 5000

    def _build_clusters(c, r_target):
        n = c.shape[0]
        tree = cKDTree(c)
        dist_limit = 2.5 * r_target
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            neighbors = tree.query_ball_point(c[i], dist_limit)
            for j in neighbors:
                if i < j:
                    union(i, j)
        clusters = {}
        for i in range(n):
            root = find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)
        return list(clusters.values())

    clusters = _build_clusters(current_c, current_r)
    cluster_rebuild_interval = 500

    for step in range(n_steps):
        if time.time() > start_time + max_time:
            break
        T = T_start * (T_end / T_start) ** (step / n_steps)
        if step % cluster_rebuild_interval == 0 and step > 0:
            clusters = _build_clusters(current_c, current_r)
        if np.random.random() < 0.6 and len(clusters) > 1:
            cluster = clusters[np.random.randint(len(clusters))]
            angle = np.random.random() * 2 * np.pi
            magnitude = np.random.exponential(0.003)
            dx = magnitude * np.cos(angle)
            dy = magnitude * np.sin(angle)
            new_c = current_c.copy()
            for idx in cluster:
                new_c[idx, 0] += dx
                new_c[idx, 1] += dy
        else:
            idx = np.random.randint(N)
            angle = np.random.random() * 2 * np.pi
            magnitude = np.random.exponential(0.002)
            dx = magnitude * np.cos(angle)
            dy = magnitude * np.sin(angle)
            new_c = current_c.copy()
            new_c[idx, 0] += dx
            new_c[idx, 1] += dy
        new_c = np.clip(new_c, -0.5 + 1e-12, 0.5 - 1e-12)
        new_r = _compute_radius(new_c)
        delta_r = new_r - current_r
        if delta_r > 0 or np.random.random() < np.exp(delta_r / max(T, 1e-12)):
            current_c = new_c
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_c = current_c.copy()
    return best_c, best_r


def _slsqp_polish(c0, r0, max_time=30.0):
    """High-precision polish using SLSQP with active constraints."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    n = N

    def _get_pairs_vectorized(c, r_target, margin=0.005):
        tree = cKDTree(c)
        dist_limit = 2 * r_target + margin
        pairs = []
        for i in range(n):
            neighbors = tree.query_ball_point(c[i], dist_limit)
            for j in neighbors:
                if i < j:
                    pairs.append((i, j))
        return np.array(pairs, dtype=np.int32) if pairs else np.empty((0, 2), dtype=np.int32)

    step = 5e-5
    max_iter = 80
    for _ in range(max_iter):
        if time.time() > start_time + max_time:
            break
        r_target = best_r + step
        pairs = _get_pairs_vectorized(best_c, r_target, margin=0.008)
        npairs = len(pairs)

        def constraints_func(x):
            c = x.reshape(-1, 2)
            bd = 0.5 - np.abs(c) - r_target
            bd_flat = bd.ravel()
            if npairs == 0:
                return bd_flat
            ci = c[pairs[:, 0]]
            cj = c[pairs[:, 1]]
            diffs = ci - cj
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            pd_vals = dists - 2 * r_target
            return np.concatenate([bd_flat, pd_vals])

        def jac_func(x):
            c = x.reshape(-1, 2)
            total_constraints = 2 * n + npairs
            jac = np.zeros((total_constraints, 2 * n))
            signs = -np.sign(c)
            for i in range(n):
                jac[2*i, 2*i] = signs[i, 0]
                jac[2*i+1, 2*i+1] = signs[i, 1]
            if npairs == 0:
                return jac
            ci = c[pairs[:, 0]]
            cj = c[pairs[:, 1]]
            diffs = ci - cj
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            dists_safe = np.where(dists < 1e-12, 1e-12, dists)
            unit_vecs = diffs / dists_safe[:, np.newaxis]
            row_indices = 2 * n + np.arange(npairs)
            col_i = pairs[:, 0] * 2
            col_j = pairs[:, 1] * 2
            np.add.at(jac, (row_indices, col_i), unit_vecs[:, 0])
            np.add.at(jac, (row_indices, col_i + 1), unit_vecs[:, 1])
            np.add.at(jac, (row_indices, col_j), -unit_vecs[:, 0])
            np.add.at(jac, (row_indices, col_j + 1), -unit_vecs[:, 1])
            return jac

        try:
            result = minimize(
                lambda x: 0.0,
                best_c.flatten(),
                method='SLSQP',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * n),
                constraints=[
                    {'type': 'ineq', 'fun': constraints_func, 'jac': jac_func}
                ],
                options={'maxiter': 150, 'ftol': 1e-14, 'disp': False}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r + 1e-12:
                best_c = c_new
                best_r = r_new
                step = min(step * 1.05, 2e-4)
            else:
                step *= 0.5
                if step < 1e-7:
                    break
        except Exception:
            break
    return best_c, best_r


def _cutting_plane_polish(c0, r0, max_time=15.0):
    """
    Cutting-plane method: approximate max radius from below using affine
    valid inequalities from active pairwise and boundary distance constraints.
    At each iteration, solve a convex QP to maximize a piecewise-linear lower
    bound on r, then update the constraint set with newly active constraints.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    
    # Active constraint set: list of (type, indices, coeff_vector, rhs)
    # type 'pair': r <= (d_ij - 2*r) / 1 + ... actually r <= d_ij/2
    # type 'bound': r <= 0.5 - |c_ik|
    # We store constraints as: r <= a^T c + b
    # For pair (i,j): r <= ||c_i - c_j||/2 (nonlinear, linearize at current point)
    # For bound i,k: r <= 0.5 - |c_ik| (linearize)
    
    def _get_active_constraints(c, r_target, margin=0.003):
        """Get active constraints near current radius."""
        n = c.shape[0]
        constraints = []
        
        # Boundary constraints
        for i in range(n):
            for k in range(2):
                d = 0.5 - np.abs(c[i, k])
                if d <= r_target + margin:
                    # r <= 0.5 - |c_ik|
                    # Linearize: if c_ik >= 0, r <= 0.5 - c_ik; if c_ik < 0, r <= 0.5 + c_ik
                    sign = np.sign(c[i, k])
                    if sign == 0:
                        sign = 1.0
                    # a^T c + b: a has -sign at position (i*2+k), b = 0.5
                    a = np.zeros(2 * n)
                    a[i * 2 + k] = -sign
                    b = 0.5
                    constraints.append(('bound', (i, k), a, b))
        
        # Pair constraints
        tree = cKDTree(c)
        dist_limit = 2 * (r_target + margin)
        for i in range(n):
            neighbors = tree.query_ball_point(c[i], dist_limit)
            for j in neighbors:
                if i < j:
                    d = np.linalg.norm(c[i] - c[j])
                    if d <= 2 * (r_target + margin):
                        # r <= d/2, linearize: r <= (c_i - c_j)·(c_i - c_j)/d + ... 
                        # Actually: r <= ||c_i - c_j||/2
                        # Gradient w.r.t. c_i: (c_i - c_j)/d, w.r.t. c_j: -(c_i - c_j)/d
                        diff = c[i] - c[j]
                        d_safe = max(d, 1e-12)
                        unit = diff / d_safe
                        a = np.zeros(2 * n)
                        a[i*2] = unit[0]
                        a[i*2+1] = unit[1]
                        a[j*2] = -unit[0]
                        a[j*2+1] = -unit[1]
                        b = d_safe / 2.0
                        constraints.append(('pair', (i, j), a, b))
        
        return constraints
    
    def _solve_qp(c_current, r_target, constraints, max_iter=50):
        """
        Solve: maximize r
        subject to: r <= a_m^T c + b_m for all m
                    -0.5 + eps <= c_i <= 0.5 - eps
        Using projected gradient on the dual or direct QP via scipy.
        """
        n = c_current.shape[0]
        eps = 1e-12
        bounds = [(-0.5 + eps, 0.5 - eps)] * (2 * n)
        
        # Objective: minimize -r (r is variable index 2n)
        # Variables: x = [c_1, ..., c_n, r], size 2n+1
        # Constraints: r - a_m^T c <= b_m  =>  -a_m^T c + r <= b_m
        #              -0.5+eps <= c_i <= 0.5-eps
        #              r >= 0
        
        def objective(x):
            return -x[-1]  # minimize -r
        
        # Inequality constraints: g(x) >= 0
        # For each constraint m: a_m^T c + b_m - r >= 0
        cons = []
        for ctype, cidx, a, b in constraints:
            def con_func(x, _a=a, _b=b):
                c_part = np.dot(_a[:2*n], x[:2*n])
                return c_part + _b - x[-1]
            def con_jac(x, _a=a):
                jac = np.zeros(2*n+1)
                jac[:2*n] = _a[:2*n]
                jac[-1] = -1.0
                return jac
            cons.append({'type': 'ineq', 'fun': con_func, 'jac': con_jac})
        
        # r >= 0
        cons.append({'type': 'ineq', 'fun': lambda x: x[-1], 'jac': lambda x: np.array([0.0]*(2*n) + [1.0])})
        
        # Bounds for r
        bounds.append((0.0, 0.5))
        
        x0 = np.concatenate([c_current.flatten(), [r_target]])
        
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False}
            )
            if result.success or result.fun < 0:
                c_new = np.clip(result.x[:2*n].reshape(-1, 2), -0.5+eps, 0.5-eps)
                r_new = result.x[-1]
                return c_new, r_new
        except Exception:
            pass
        
        return c_current, r_target
    
    # Main loop
    max_rounds = 20
    for round_idx in range(max_rounds):
        if time.time() > start_time + max_time:
            break
        
        # Get active constraints at current best
        constraints = _get_active_constraints(best_c, best_r, margin=0.005)
        
        if len(constraints) == 0:
            break
        
        # Solve QP
        c_qp, r_qp = _solve_qp(best_c, best_r, constraints, max_iter=80)
        
        if r_qp > best_r + 1e-10:
            best_c = c_qp.copy()
            best_r = r_qp
        
        # Check convergence
        if r_qp - best_r < 1e-8:
            # Try with tighter margin
            constraints_tight = _get_active_constraints(best_c, best_r, margin=0.002)
            c_qp2, r_qp2 = _solve_qp(best_c, best_r, constraints_tight, max_iter=50)
            if r_qp2 > best_r + 1e-10:
                best_c = c_qp2.copy()
                best_r = r_qp2
        
        if time.time() > start_time + max_time - 2:
            break
    
    return best_c, best_r


def _spectral_objective(params, alpha=50.0):
    """
    Spectral objective: maximize the smallest eigenvalue of a Laplacian-like
    matrix constructed from pairwise distance constraints.
    
    The idea: build a graph where edge weights are functions of distances.
    Minimize the negative of the smallest eigenvalue of the graph Laplacian,
    which encourages uniform spreading of points.
    """
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    
    # Compute pairwise distances
    dists = pdist(c, metric='euclidean')
    # Also include boundary distances as virtual "nodes"
    boundary_dists = 0.5 - np.abs(c)
    b_min = np.min(boundary_dists)
    
    # Build a weight matrix based on inverse distances
    # W_ij = 1 / (d_ij + eps) for i != j, 0 for i == j
    eps = 1e-8
    dists_matrix = squareform(dists)
    np.fill_diagonal(dists_matrix, 1.0)  # Avoid division by zero
    W = 1.0 / (dists_matrix + eps)
    np.fill_diagonal(W, 0.0)
    
    # Also add boundary constraint as a penalty
    # Penalize points too close to boundary
    boundary_penalty = np.sum(np.maximum(0, 0.001 - boundary_dists)) ** 2
    
    # Graph Laplacian L = D - W where D is degree matrix
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    
    # Add a small regularization to ensure positive semi-definiteness
    L += eps * np.eye(n)
    
    # Compute smallest eigenvalue (excluding the trivial zero eigenvalue)
    # For a connected graph, the smallest non-zero eigenvalue is the algebraic connectivity
    try:
        eigvals = np.linalg.eigvalsh(L)
        # The smallest eigenvalue should be ~0 for a connected graph
        # We want to maximize the smallest non-zero eigenvalue
        # Filter out eigenvalues very close to 0
        non_zero_eigs = eigvals[eigvals > 1e-6]
        if len(non_zero_eigs) > 0:
            lambda_min = non_zero_eigs[0]
        else:
            lambda_min = 0.0
    except np.linalg.LinAlgError:
        lambda_min = 0.0
    
    # Combine with a term that encourages large minimum distance
    min_dist = np.min(dists)
    
    # Objective: minimize -(alpha * lambda_min + min_dist/2 + b_min)
    # This encourages both uniform spreading and large minimum distance
    obj = -(alpha * lambda_min + min_dist / 2.0 + b_min)
    
    return obj + boundary_penalty


def _spectral_optimize(c0, max_time=25.0, alphas=None):
    """Local optimization using spectral objective."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    if alphas is None:
        alphas = [10.0, 30.0, 50.0, 100.0]
    current_c = c0.copy()
    for alpha in alphas:
        if time.time() > start_time + max_time - 2:
            break
        def obj(params, _alpha=alpha):
            return _spectral_objective(params, alpha=_alpha)
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 60}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
            current_c = c_new
        except Exception:
            continue
    return best_c, best_r


def _randomize_clusters(c0, magnitude):
    """Randomize some clusters to create diversity."""
    n = c0.shape[0]
    r_est = _compute_radius(c0)
    tree = cKDTree(c0)
    dist_limit = 2.5 * r_est
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        neighbors = tree.query_ball_point(c0[i], dist_limit)
        for j in neighbors:
            if i < j:
                union(i, j)

    clusters = {}
    for i in range(n):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)
    cluster_list = list(clusters.values())
    new_c = c0.copy()
    n_randomize = max(1, len(cluster_list) // 4)
    random_clusters = np.random.choice(len(cluster_list), n_randomize, replace=False)
    for ci in random_clusters:
        cluster = cluster_list[ci]
        angle = np.random.random() * 2 * np.pi
        mag = np.random.exponential(magnitude)
        dx = mag * np.cos(angle)
        dy = mag * np.sin(angle)
        for idx in cluster:
            new_c[idx, 0] += dx
            new_c[idx, 1] += dy
    return np.clip(new_c, -0.5 + 1e-12, 0.5 + 1e-12)


def construct():
    start_time = time.time()
    deadline = start_time + 145
    best_radius = 0.0
    best_c = None
    np.random.seed(42)

    # 1. Start with hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()

    # 2. Primary optimization with smooth objective
    if time.time() < deadline - 55:
        remaining = deadline - time.time() - 55
        c_opt, r_opt = _local_optimize_smooth(c_hex, max_time=min(35.0, remaining))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()

    # 3. Focused energy optimization
    if best_c is not None and time.time() < deadline - 45:
        remaining = max(8.0, deadline - time.time() - 45)
        c_foc, r_foc = _local_optimize_focused(best_c, max_time=min(20.0, remaining))
        if r_foc > best_radius:
            best_radius = r_foc
            best_c = c_foc.copy()

    # 4. Spectral optimization to encourage uniform spreading
    if best_c is not None and time.time() < deadline - 40:
        remaining = max(10.0, deadline - time.time() - 40)
        c_spec, r_spec = _spectral_optimize(best_c, max_time=min(20.0, remaining))
        if r_spec > best_radius:
            best_radius = r_spec
            best_c = c_spec.copy()

    # 5. Simulated annealing with rigid-body moves
    if best_c is not None and time.time() < deadline - 30:
        remaining = max(10.0, deadline - time.time() - 30)
        c_sa, r_sa = _simulated_annealing_rigid(best_c, best_radius, max_time=min(20.0, remaining))
        if r_sa > best_radius:
            best_radius = r_sa
            best_c = c_sa.copy()

    # 6. Another smooth optimization pass
    if best_c is not None and time.time() < deadline - 20:
        remaining = max(5.0, deadline - time.time() - 20)
        c_opt2, r_opt2 = _local_optimize_smooth(best_c, max_time=min(15.0, remaining))
        if r_opt2 > best_radius:
            best_radius = r_opt2
            best_c = c_opt2.copy()

    # 7. Focused energy refinement after SA
    if best_c is not None and time.time() < deadline - 12:
        remaining = max(5.0, deadline - time.time() - 12)
        c_foc2, r_foc2 = _local_optimize_focused(best_c, max_time=min(10.0, remaining))
        if r_foc2 > best_radius:
            best_radius = r_foc2
            best_c = c_foc2.copy()

    # 8. Cutting-plane polish for tighter radius bound
    if best_c is not None and time.time() < deadline - 4:
        remaining = max(5.0, deadline - time.time() - 4)
        c_cp, r_cp = _cutting_plane_polish(best_c, best_radius, max_time=min(12.0, remaining))
        if r_cp > best_radius:
            best_radius = r_cp
            best_c = c_cp

    # 9. Final SLSQP polish
    if best_c is not None and time.time() < deadline - 1:
        remaining = max(2.0, deadline - time.time() - 0.5)
        c_final, r_final = _slsqp_polish(best_c, best_radius, max_time=min(5.0, remaining))
        if r_final > best_radius:
            best_radius = r_final
            best_c = c_final

    if best_c is None:
        best_c = _initialize_hexagonal()
    best_c = np.clip(best_c, -0.5, 0.5)
    return best_c
# EVOLVE-BLOCK-END
