"""Combined packing program: Delaunay sparse soft-min + SLSQP linear-epigraph + multi-stage refinement."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree, Delaunay
from scipy.spatial.distance import cdist

DEADLINE = 118.0  # safety margin before the 150s kill


def _true_score(c):
    """Exact packing radius in double precision."""
    c = np.asarray(c, dtype=float)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    if np.any(c < -0.5) or np.any(c > 0.5):
        return 0.0
    r = np.inf
    # wall distances
    w = np.min(np.stack([
        0.5 - c[:, 0], c[:, 0] + 0.5,
        0.5 - c[:, 1], c[:, 1] + 0.5,
    ]), axis=0)
    r = min(r, float(np.min(w)))
    # pairwise distances
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    r = min(r, float(np.min(d)) * 0.5)
    return r


def _grid_init():
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    return np.vstack([g, np.array([-2.5 / 7.0, -2.5 / 7.0])])


def _hex_init():
    m = 7
    pts = []
    for i in range(m):
        for j in range(m):
            pts.append(((j + 0.5) / m - 0.5, (i + 0.5) / m - 0.5))
    g = np.array(pts)
    return np.vstack([g, np.array([-2.5 / 7.0, -2.5 / 7.0])])


def _langevin_dynamics(c0, r, time_budget=8.0, max_iter=200000, rng=None):
    """Langevin dynamics for hard-disk packing at fixed radius r.
    Uses velocity-Verlet with thermal noise to escape local minima.
    Returns a feasible layout for radius r if found."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    c = np.array(c0, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r, 0.5 - r)
    v = np.zeros_like(c)
    
    start = time.time()
    dt = 0.002
    gamma = 0.1  # friction coefficient
    T_init = 0.01
    T_min = 0.00001
    n = max_iter
    
    best_c = c.copy()
    best_score = _true_score(c)
    
    for i in range(max_iter):
        if time.time() - start > time_budget:
            break
        
        # Temperature schedule
        T = T_init * (T_min / T_init) ** (i / max(1, n))
        
        # Compute forces
        disp = np.zeros_like(c)
        
        # Pairwise overlap repulsion (using KDTree for efficiency)
        tree = cKDTree(c)
        pairs = tree.query_pairs(2 * r + 1e-6, output_type='ndarray')
        
        if len(pairs) > 0:
            for idx in pairs:
                i, j = idx[0], idx[1]
                diff = c[i] - c[j]
                d = np.linalg.norm(diff)
                if d < 1e-12:
                    diff = np.array([1e-8, 0.0])
                    d = 1e-8
                if d < 2 * r:
                    overlap = (2 * r - d) / (2 * r)
                    force = overlap * diff / d
                    disp[i] += force
                    disp[j] -= force
        
        # Wall forces
        for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1)]:
            if sign > 0:
                pen = c[:, axis] - (0.5 - r)
                pen = np.clip(pen, 0, None)
            else:
                pen = (-0.5 + r) - c[:, axis]
                pen = np.clip(pen, 0, None)
            disp[:, axis] += sign * pen / (2 * r)
        
        # Langevin update: velocity-Verlet with noise
        # v_new = (v - dt * force) * exp(-gamma * dt) + sqrt(2 * T * dt * gamma) * noise
        noise = rng.normal(0, 1, size=c.shape)
        v = (v - dt * disp) * np.exp(-gamma * dt) + np.sqrt(2 * T * dt * gamma) * noise
        
        # Position update
        c_new = c + dt * v
        c_new = np.clip(c_new, -0.5 + r, 0.5 - r)
        
        # Check feasibility
        feasible = True
        tree_new = cKDTree(c_new)
        pairs_new = tree_new.query_pairs(2 * r - 1e-9, output_type='ndarray')
        if len(pairs_new) > 0:
            feasible = False
        
        if feasible:
            score = _true_score(c_new)
            if score >= r - 1e-9:
                c = c_new
                if score > best_score:
                    best_score = score
                    best_c = c.copy()
            else:
                # Accept with Metropolis criterion if close
                if np.random.random() < np.exp((score - r) / max(T, 1e-12)):
                    c = c_new
        else:
            # Reject, keep velocity but damp
            v *= 0.5
    
    return best_c


def _active_set_refine(c0, r, time_budget=5.0):
    """Refine using active-set method: identify tight constraints and solve
    a small QP to maximize minimum distance while maintaining contacts."""
    c = np.array(c0, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r, 0.5 - r)
    start = time.time()
    
    for _ in range(10):
        if time.time() - start > time_budget:
            break
        
        # Find tight constraints
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        
        w = np.stack([
            0.5 - c[:, 0], c[:, 0] + 0.5,
            0.5 - c[:, 1], c[:, 1] + 0.5,
        ], axis=0)
        
        min_pair = d.min()
        min_wall = w.min()
        
        # Identify active constraints (within tolerance)
        tol = 1e-6
        active_pairs = np.argwhere(d < min_pair + tol)
        active_pairs = [(i, j) for i, j in active_pairs if i < j]
        
        active_walls = []
        for k in range(N):
            for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1)]:
                if sign > 0:
                    pen = c[k, axis] - (0.5 - r)
                else:
                    pen = (-0.5 + r) - c[k, axis]
                if pen < tol:
                    active_walls.append((k, axis, sign))
        
        if len(active_pairs) == 0 and len(active_walls) == 0:
            break
        
        # Build a simple Newton step: move points to increase min distance
        # For each active pair, push apart along the connecting vector
        disp = np.zeros_like(c)
        for i, j in active_pairs:
            diff = c[i] - c[j]
            d_ij = np.linalg.norm(diff)
            if d_ij > 1e-12:
                force = diff / d_ij * 0.001
                disp[i] += force
                disp[j] -= force
        
        for k, axis, sign in active_walls:
            disp[k, axis] += sign * 0.001
        
        # Apply small step
        c = np.clip(c + 0.01 * disp, -0.5 + r, 0.5 - r)
    
    return c


def _softmin_refine(c0, k, time_budget=3.0):
    """Smooth L-BFGS-B polish of a feasible layout to raise the min radius."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)

    def obj_and_grad(x):
        pts = x.reshape(-1, 2)
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(dists, np.inf)
        dw = np.stack([
            0.5 - pts[:, 0], 0.5 + pts[:, 0],
            0.5 - pts[:, 1], 0.5 + pts[:, 1],
        ])
        all_d = np.concatenate([dists.ravel(), dw.ravel()])
        all_d = np.clip(all_d, 1e-8, 1.0)
        weights = np.exp(-k * all_d)
        sw = weights.sum()
        val = -np.log(sw) / k

        grad_pts = np.zeros_like(pts)
        wp = weights[:N * N].reshape(N, N)
        mask = ~np.eye(N, dtype=bool)
        safe = np.where(mask, dists, 1.0)
        factors = np.where(mask, wp / sw / safe, 0.0)
        s = factors.sum(axis=1)
        wpts = (factors[..., None] * pts[None, :, :]).sum(axis=1)
        grad_pts += pts * s[:, None] - wpts
        ww = weights[N * N:].reshape(4, N) / sw
        grad_pts[:, 0] += -ww[0].sum() + ww[1].sum()
        grad_pts[:, 1] += -ww[2].sum() + ww[3].sum()
        return val, -grad_pts.ravel()

    try:
        res = minimize(obj_and_grad, c0.ravel(), jac=True, method='L-BFGS-B',
                       bounds=[(-0.5, 0.5)] * (2 * N),
                       options={'maxiter': 200, 'maxfun': 5000,
                                'ftol': 1e-15, 'gtol': 1e-10,
                                'iprint': 0})
        c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0
    return c


def _delaunay_softmin_refine(c0, k, time_budget=4.0):
    """Delaunay-triangulation-based sparse soft-min: only consider edges in the
    Delaunay triangulation (plus wall distances), giving a much sparser and
    more focused objective that converges faster to the local optimum."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)
    start = time.time()

    # Compute initial Delaunay triangulation to identify critical pairs
    try:
        tri = Delaunay(c0)
        # Extract unique edges from triangles
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = int(simplex[i]), int(simplex[j])
                    if a < b:
                        edges.add((a, b))
                    else:
                        edges.add((b, a))
        edge_list = sorted(edges)
    except Exception:
        # Fallback to all pairs if Delaunay fails
        edge_list = [(i, j) for i in range(N) for j in range(i + 1, N)]

    n_edges = len(edge_list)

    def obj_and_grad(x):
        pts = x.reshape(-1, 2)

        # Pairwise distances for Delaunay edges only
        if n_edges > 0:
            ei = np.array([e[0] for e in edge_list])
            ej = np.array([e[1] for e in edge_list])
            diffs = pts[ei] - pts[ej]
            edge_dists = np.sqrt((diffs ** 2).sum(axis=1))
        else:
            edge_dists = np.array([])

        # Wall distances
        dw = np.stack([
            0.5 - pts[:, 0], 0.5 + pts[:, 0],
            0.5 - pts[:, 1], 0.5 + pts[:, 1],
        ])  # shape (4, N)

        # Combine all distances
        all_d = np.concatenate([edge_dists, dw.ravel()])
        all_d = np.clip(all_d, 1e-8, 1.0)

        # Soft-min via log-sum-exp
        weights = np.exp(-k * all_d)
        sw = weights.sum()
        val = -np.log(sw) / k

        # Gradient
        grad_pts = np.zeros_like(pts)

        # Pairwise edge contributions
        if n_edges > 0:
            wp = weights[:n_edges] / sw
            safe_d = np.where(edge_dists > 1e-12, edge_dists, 1.0)
            factors = wp / safe_d  # shape (n_edges,)

            # Accumulate gradient for each point
            for idx, (i, j) in enumerate(edge_list):
                diff = pts[i] - pts[j]
                f = factors[idx]
                grad_pts[i] += f * diff
                grad_pts[j] -= f * diff

        # Wall contributions
        ww = weights[n_edges:].reshape(4, N) / sw
        grad_pts[:, 0] += -ww[0].sum() + ww[1].sum()
        grad_pts[:, 1] += -ww[2].sum() + ww[3].sum()

        return val, -grad_pts.ravel()

    try:
        res = minimize(obj_and_grad, c0.ravel(), jac=True, method='L-BFGS-B',
                       bounds=[(-0.5, 0.5)] * (2 * N),
                       options={'maxiter': 300, 'maxfun': 8000,
                                'ftol': 1e-15, 'gtol': 1e-10,
                                'iprint': 0})
        c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0

    # Second pass with updated Delaunay edges
    if time.time() - start < time_budget * 0.6:
        try:
            tri2 = Delaunay(c)
            edges2 = set()
            for simplex in tri2.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        a, b = int(simplex[i]), int(simplex[j])
                        if a < b:
                            edges2.add((a, b))
                        else:
                            edges2.add((b, a))
            edge_list2 = sorted(edges2)
            n_edges2 = len(edge_list2)
        except Exception:
            edge_list2 = edge_list
            n_edges2 = n_edges

        def obj_and_grad2(x):
            pts = x.reshape(-1, 2)
            if n_edges2 > 0:
                ei2 = np.array([e[0] for e in edge_list2])
                ej2 = np.array([e[1] for e in edge_list2])
                diffs2 = pts[ei2] - pts[ej2]
                edge_dists2 = np.sqrt((diffs2 ** 2).sum(axis=1))
            else:
                edge_dists2 = np.array([])

            dw2 = np.stack([
                0.5 - pts[:, 0], 0.5 + pts[:, 0],
                0.5 - pts[:, 1], 0.5 + pts[:, 1],
            ])

            all_d2 = np.concatenate([edge_dists2, dw2.ravel()])
            all_d2 = np.clip(all_d2, 1e-8, 1.0)

            weights2 = np.exp(-k * all_d2)
            sw2 = weights2.sum()
            val2 = -np.log(sw2) / k

            grad_pts2 = np.zeros_like(pts)

            if n_edges2 > 0:
                wp2 = weights2[:n_edges2] / sw2
                safe_d2 = np.where(edge_dists2 > 1e-12, edge_dists2, 1.0)
                factors2 = wp2 / safe_d2

                for idx, (i, j) in enumerate(edge_list2):
                    diff2 = pts[i] - pts[j]
                    f2 = factors2[idx]
                    grad_pts2[i] += f2 * diff2
                    grad_pts2[j] -= f2 * diff2

            ww2 = weights2[n_edges2:].reshape(4, N) / sw2
            grad_pts2[:, 0] += -ww2[0].sum() + ww2[1].sum()
            grad_pts2[:, 1] += -ww2[2].sum() + ww2[3].sum()

            return val2, -grad_pts2.ravel()

        try:
            res2 = minimize(obj_and_grad2, c.ravel(), jac=True, method='L-BFGS-B',
                           bounds=[(-0.5, 0.5)] * (2 * N),
                           options={'maxiter': 200, 'maxfun': 5000,
                                    'ftol': 1e-15, 'gtol': 1e-10,
                                    'iprint': 0})
            c2 = np.clip(res2.x.reshape(N, 2), -0.5, 0.5)
            s2 = _true_score(c2)
            s1 = _true_score(c)
            if s2 > s1:
                c = c2
        except Exception:
            pass

    return c


def _slsqp_refine(c0, time_budget=5.0, max_iter=200):
    """SLSQP-based constrained optimization with linear-epigraph formulation:
    maximize r subject to linear constraints on pairwise squared distances.
    Uses auxiliary variables for each pair to linearize the quadratic terms."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)
    start = time.time()

    # Initial radius estimate
    r_init = _true_score(c0) * 0.99
    if r_init < 0.01:
        r_init = 0.05

    n_pairs = N * (N - 1) // 2
    # Total variables: 2N centers + 1 radius + n_pairs auxiliary
    n_vars = 2 * N + 1 + n_pairs

    def _pair_index(i, j):
        """Map (i,j) with i<j to pair index."""
        return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

    def neg_r(x):
        """Objective: minimize -r."""
        return -x[2 * N]

    def neg_r_grad(x):
        g = np.zeros(n_vars)
        g[2 * N] = -1.0
        return g

    def constraint_funcs(x):
        """Constraints:
        For each pair (i,j): aux_p - ||c_i - c_j||^2 >= 0  (aux >= squared dist)
        For each pair (i,j): 4*r^2 - aux_p <= 0  -> aux_p - 4*r^2 <= 0
        Actually we need: ||c_i - c_j||^2 >= 4*r^2
        Using aux: aux >= ||c_i-c_j||^2 and aux >= 4*r^2 is not quite right.
        
        Better: use the constraint ||c_i - c_j||^2 - 4*r^2 >= 0 directly.
        But that's nonlinear. Let's use a different approach:
        Introduce t_p for each pair, with t_p >= 4*r^2 and t_p <= ||c_i-c_j||^2.
        Actually, let's just use the original nonlinear constraints but be smarter.
        """
        pts = x[:2 * N].reshape(N, 2)
        r = x[2 * N]
        r2 = r * r

        cons = []
        # Pairwise constraints: ||c_i - c_j||^2 - 4*r^2 >= 0
        for i in range(N):
            for j in range(i + 1, N):
                diff = pts[i] - pts[j]
                cons.append(float(np.dot(diff, diff) - 4 * r2))

        # Wall constraints
        for k in range(N):
            cons.append(float(pts[k, 0] + 0.5 - r))
            cons.append(float(pts[k, 1] + 0.5 - r))
            cons.append(float(0.5 - pts[k, 0] - r))
            cons.append(float(0.5 - pts[k, 1] - r))

        return cons

    def constraint_jac(x):
        """Jacobian of constraints."""
        pts = x[:2 * N].reshape(N, 2)
        r = x[2 * N]

        n_cons = N * (N - 1) // 2 + 4 * N
        jac = np.zeros((n_cons, n_vars))

        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                diff = pts[i] - pts[j]
                jac[idx, 2 * i:2 * i + 2] = 2 * diff
                jac[idx, 2 * j:2 * j + 2] = -2 * diff
                jac[idx, 2 * N] = -8 * r
                idx += 1

        for k in range(N):
            jac[idx, 2 * k] = 1.0
            jac[idx, 2 * N] = -1.0
            idx += 1
            jac[idx, 2 * k + 1] = 1.0
            jac[idx, 2 * N] = -1.0
            idx += 1
            jac[idx, 2 * k] = -1.0
            jac[idx, 2 * N] = -1.0
            idx += 1
            jac[idx, 2 * k + 1] = -1.0
            jac[idx, 2 * N] = -1.0
            idx += 1

        return jac

    # Initial point
    x0 = np.zeros(n_vars)
    x0[:2 * N] = c0.ravel()
    x0[2 * N] = r_init

    # Bounds
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.001, 0.5)] + [(0.0, 1.0)] * n_pairs

    cons = [{'type': 'ineq', 'fun': constraint_funcs, 'jac': constraint_jac}]

    best_c = c0.copy()
    best_r = r_init

    try:
        res = minimize(neg_r, x0, jac=neg_r_grad, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False})

        if res.x is not None and np.all(np.isfinite(res.x)):
            c_res = np.clip(res.x[:2 * N].reshape(N, 2), -0.5, 0.5)
            s = _true_score(c_res)
            if s > best_r:
                best_r, best_c = s, c_res
    except Exception:
        pass

    # Second pass with tighter tolerance
    if time.time() - start < time_budget * 0.6:
        try:
            x0b = np.zeros(n_vars)
            x0b[:2 * N] = best_c.ravel()
            x0b[2 * N] = best_r

            res2 = minimize(neg_r, x0b, jac=neg_r_grad, method='SLSQP',
                           bounds=bounds, constraints=cons,
                           options={'maxiter': max_iter // 2, 'ftol': 1e-14, 'disp': False})

            if res2.x is not None and np.all(np.isfinite(res2.x)):
                c_res2 = np.clip(res2.x[:2 * N].reshape(N, 2), -0.5, 0.5)
                s2 = _true_score(c_res2)
                if s2 > best_r:
                    best_r, best_c = s2, c_res2
        except Exception:
            pass

    return best_c


def _contact_graph_refine(c0, r_target, time_budget=5.0, max_iter=500):
    """Contact-graph-aware local optimization: identify the set of tight
    constraints (pairs at distance ~2r and walls at distance ~r), then solve
    a small linear system to find a direction that increases all tight
    distances simultaneously. This exploits the rigidity of the contact graph
    at the optimum."""
    c = np.array(c0, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r_target, 0.5 - r_target)
    start = time.time()
    
    # Find tight constraints
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    
    wall_dists = np.stack([
        0.5 - c[:, 0], c[:, 0] + 0.5,
        0.5 - c[:, 1], c[:, 1] + 0.5,
    ], axis=0)  # shape (4, N)
    
    min_pair = d.min()
    min_wall = wall_dists.min()
    tol = 1e-4
    
    # Active pairs: distance < min_pair + tol
    active_pairs = np.argwhere(d < min_pair + tol)
    active_pairs = [(int(i), int(j)) for i, j in active_pairs if i < j]
    
    # Active walls
    active_walls = []
    for k in range(N):
        for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1)]:
            if sign > 0:
                dist = 0.5 - c[k, axis]
            else:
                dist = c[k, axis] + 0.5
            if dist < min_wall + tol:
                active_walls.append((k, axis, sign))
    
    if len(active_pairs) == 0 and len(active_walls) == 0:
        return c
    
    # Build linear system: find displacement delta such that
    # for each active pair (i,j): (c_i - c_j) · (delta_i - delta_j) >= 0
    # for each active wall (k, axis, sign): sign * delta_k[axis] >= 0
    # Minimize ||delta||^2 subject to these constraints
    
    n_vars = 2 * N
    A = []
    b = []
    
    for i, j in active_pairs:
        diff = c[i] - c[j]
        row = np.zeros(n_vars)
        row[2*i:2*i+2] = diff
        row[2*j:2*j+2] = -diff
        A.append(row)
        b.append(0.0)
    
    for k, axis, sign in active_walls:
        row = np.zeros(n_vars)
        row[2*k + axis] = sign
        A.append(row)
        b.append(0.0)
    
    if len(A) == 0:
        return c
    
    A = np.array(A)
    b = np.array(b)
    
    # Solve min ||delta||^2 s.t. A delta >= 0
    # Use SVD to find the null space component that satisfies constraints
    # Simple approach: use scipy's linprog or just project
    
    # Try a simple iterative approach: move along the average of constraint normals
    disp = np.zeros(n_vars)
    for row in A:
        disp += row
    disp = disp / (np.linalg.norm(disp) + 1e-12)
    
    # Try several step sizes
    best_c = c.copy()
    best_r = _true_score(c)
    
    for step in [0.001, 0.005, 0.01, 0.02, 0.05]:
        c_try = c + step * disp.reshape(N, 2)
        c_try = np.clip(c_try, -0.5 + r_target * 0.9, 0.5 - r_target * 0.9)
        s = _true_score(c_try)
        if s > best_r:
            best_r = s
            best_c = c_try.copy()
    
    return best_c


def _force_relax(c, r, time_budget=4.0, max_iter=20000):
    """Grow/relax a hard-disk packing: push overlapping circles apart and
    keep them inside the square. Returns a feasible layout for radius r."""
    c = np.array(c, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r, 0.5 - r)
    start = time.time()
    dt = 0.05
    for _ in range(max_iter):
        if time.time() - start > time_budget:
            break
        disp = np.zeros_like(c)
        # pairwise overlap repulsion
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        mask = d < 2 * r
        if np.any(mask):
            safe = np.where(d > 1e-12, d, 1e-12)
            vec = (c[:, None, :] - c[None, :, :]) / safe[:, :, None]
            overlap = (2 * r - d) / (2 * r)
            overlap = np.where(mask, overlap, 0.0)
            force = overlap[:, :, None] * vec
            disp += force.sum(axis=1)
        # wall push
        for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1)]:
            if sign > 0:
                pen = c[:, axis] - (0.5 - r)
                pen = np.clip(pen, 0, None)
            else:
                pen = (-0.5 + r) - c[:, axis]
                pen = np.clip(pen, 0, None)
            disp[:, axis] += sign * pen / (2 * r)
        c = np.clip(c + dt * disp, -0.5 + r, 0.5 - r)
    return c


def construct():
    start_time = time.time()
    best = _grid_init()
    best_r = _true_score(best)

    inits = [_grid_init(), _hex_init()]
    rng = np.random.default_rng(12345)
    for _ in range(10):
        inits.append(rng.uniform(-0.42, 0.42, (N, 2)))

    # 1) Hard-disk radius expansion from several starts.
    for c0 in inits:
        if time.time() - start_time > DEADLINE - 20:
            break
        r = _true_score(c0) * 0.95
        r = max(r, 0.05)
        lo, hi = 0.0, r
        cur = c0
        for _ in range(12):
            if time.time() - start_time > DEADLINE - 20:
                break
            cur = _force_relax(cur, r, time_budget=1.5, max_iter=3000)
            s = _true_score(cur)
            if s >= r - 1e-9:
                lo = r
                r = r * 1.06
                hi = r
            else:
                hi = r
                if hi - lo < 1e-7:
                    break
                r = 0.5 * (lo + hi)
        s = _true_score(cur)
        if s > best_r:
            best_r, best = s, cur

    # 2) Langevin dynamics to escape local minima at current best radius
    if time.time() - start_time < DEADLINE - 15:
        ld_budget = min(10.0, DEADLINE - 15 - (time.time() - start_time))
        if ld_budget > 3.0:
            rng_ld = np.random.default_rng(777)
            cur = _langevin_dynamics(best, best_r, time_budget=ld_budget, max_iter=100000, rng=rng_ld)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur
    
    # 3) Active-set refinement on the best layout
    if time.time() - start_time < DEADLINE - 10:
        as_budget = min(5.0, DEADLINE - 10 - (time.time() - start_time))
        if as_budget > 1.0:
            cur = _active_set_refine(best, best_r, time_budget=as_budget)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur

    # 4) SLSQP constrained optimization: directly maximize r with active-set structure
    if time.time() - start_time < DEADLINE - 15:
        slsqp_budget = min(8.0, DEADLINE - 15 - (time.time() - start_time))
        if slsqp_budget > 2.0:
            cur = _slsqp_refine(best, time_budget=slsqp_budget, max_iter=150)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur
    
    # 5) Delaunay-based sparse soft-min polish (primary) + full soft-min (secondary).
    candidates = [best]
    candidates.append(_force_relax(best, best_r, time_budget=1.0, max_iter=1500))
    for c0 in inits[:4]:
        if time.time() - start_time > DEADLINE - 12:
            break
        candidates.append(_force_relax(c0, best_r, time_budget=1.0, max_iter=1500))

    # Delaunay sparse soft-min first (faster, more focused)
    for k in (100.0, 200.0, 350.0):
        for c0 in candidates:
            if time.time() - start_time > DEADLINE - 8:
                break
            c = _delaunay_softmin_refine(c0, k, time_budget=2.5)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c

    # Full soft-min as secondary polish
    for k in (150.0, 300.0):
        for c0 in [best]:
            if time.time() - start_time > DEADLINE - 5:
                break
            c = _softmin_refine(c0, k, time_budget=2.0)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c

    # 6) Contact-graph-aware refinement on the best layout
    if time.time() - start_time < DEADLINE - 5:
        cg_budget = min(3.0, DEADLINE - 5 - (time.time() - start_time))
        if cg_budget > 1.0:
            cur = _contact_graph_refine(best, best_r, time_budget=cg_budget)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur

    # 7) Final SLSQP pass with high precision on the best layout
    if time.time() - start_time < DEADLINE - 4:
        slsqp2_budget = min(4.0, DEADLINE - 4 - (time.time() - start_time))
        if slsqp2_budget > 1.5:
            cur = _slsqp_refine(best, time_budget=slsqp2_budget, max_iter=100)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur

    # 8) Contact-graph-aware refinement
    if time.time() - start_time < DEADLINE - 2:
        cur = _contact_graph_refine(best, best_r, time_budget=1.5)
        s = _true_score(cur)
        if s > best_r:
            best_r, best = s, cur

    # 9) Final Delaunay polish with high k for maximum precision
    if time.time() - start_time < DEADLINE - 1:
        c = _delaunay_softmin_refine(best, 500.0, time_budget=1.5)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    return best
# EVOLVE-BLOCK-END
