"""Rewritten packing program: Langevin dynamics + active-set contact refinement."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import cKDTree

DEADLINE = 115.0  # safety margin before the 150s kill


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


def _slsqp_refine(c0, time_budget=5.0, max_iter=200):
    """SLSQP-based constrained optimization: maximize r subject to
    ||c_i - c_j||^2 >= 4*r^2 and wall constraints c_ik >= -0.5 + r, c_ik <= 0.5 - r.
    Uses the epigraph formulation with auxiliary variable r."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)
    start = time.time()
    
    # Initial radius estimate
    r_init = _true_score(c0) * 0.99
    if r_init < 0.01:
        r_init = 0.05
    
    def neg_r(x):
        """Objective: minimize -r (i.e., maximize r)."""
        return -x[2 * N]

    def neg_r_grad(x):
        g = np.zeros(2 * N + 1)
        g[2 * N] = -1.0
        return g

    def constraint_funcs(x):
        """All constraints in a single vector:
        For each pair (i,j): ||c_i - c_j||^2 - 4*r^2 >= 0
        For each point k, axis a: c_ka + 0.5 - r >= 0 (lower wall)
        For each point k, axis a: 0.5 - c_ka - r >= 0 (upper wall)
        """
        pts = x[:2 * N].reshape(N, 2)
        r = x[2 * N]
        r2 = r * r
        
        cons = []
        # Pairwise constraints: ||c_i - c_j||^2 - 4*r^2
        for i in range(N):
            for j in range(i + 1, N):
                diff = pts[i] - pts[j]
                cons.append(float(np.dot(diff, diff) - 4 * r2))
        
        # Wall constraints
        for k in range(N):
            cons.append(float(pts[k, 0] + 0.5 - r))   # left wall
            cons.append(float(pts[k, 1] + 0.5 - r))   # bottom wall
            cons.append(float(0.5 - pts[k, 0] - r))   # right wall
            cons.append(float(0.5 - pts[k, 1] - r))   # top wall
        
        return cons

    def constraint_jac(x):
        """Jacobian of constraints."""
        pts = x[:2 * N].reshape(N, 2)
        r = x[2 * N]
        
        n_cons = N * (N - 1) // 2 + 4 * N
        jac = np.zeros((n_cons, 2 * N + 1))
        
        idx = 0
        # Pairwise constraint gradients
        for i in range(N):
            for j in range(i + 1, N):
                diff = pts[i] - pts[j]
                jac[idx, 2 * i:2 * i + 2] = 2 * diff
                jac[idx, 2 * j:2 * j + 2] = -2 * diff
                jac[idx, 2 * N] = -8 * r
                idx += 1
        
        # Wall constraint gradients
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

    # Initial point: centers + radius
    x0 = np.zeros(2 * N + 1)
    x0[:2 * N] = c0.ravel()
    x0[2 * N] = r_init
    
    # Bounds: centers in [-0.5, 0.5], r in [0.001, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.001, 0.5)]
    
    # Inequality constraints (>= 0)
    cons = [{'type': 'ineq', 'fun': constraint_funcs, 'jac': constraint_jac}]
    
    best_c = c0.copy()
    best_r = r_init
    
    try:
        res = minimize(neg_r, x0, jac=neg_r_grad, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False})
        
        if res.x is not None and np.all(np.isfinite(res.x)):
            c_res = np.clip(res.x[:2 * N].reshape(N, 2), -0.5, 0.5)
            r_res = res.x[2 * N]
            s = _true_score(c_res)
            if s > best_r:
                best_r, best_c = s, c_res
    except Exception:
        pass
    
    # If SLSQP didn't improve, try a second pass with tighter tolerance
    if time.time() - start < time_budget * 0.7:
        try:
            x0b = np.zeros(2 * N + 1)
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
    
    # 5) Smooth soft-min polish from the best layout (and a couple others).
    candidates = [best]
    candidates.append(_force_relax(best, best_r, time_budget=1.0, max_iter=1500))
    for c0 in inits[:4]:
        if time.time() - start_time > DEADLINE - 10:
            break
        candidates.append(_force_relax(c0, best_r, time_budget=1.0, max_iter=1500))

    for k in (80.0, 150.0, 250.0):
        for c0 in candidates:
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
    if time.time() - start_time < DEADLINE - 3:
        slsqp2_budget = min(4.0, DEADLINE - 3 - (time.time() - start_time))
        if slsqp2_budget > 1.5:
            cur = _slsqp_refine(best, time_budget=slsqp2_budget, max_iter=100)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur
    
    # 8) One more contact-graph pass after final polish
    if time.time() - start_time < DEADLINE - 1:
        cur = _contact_graph_refine(best, best_r, time_budget=1.5)
        s = _true_score(cur)
        if s > best_r:
            best_r, best = s, cur

    return best
# EVOLVE-BLOCK-END
