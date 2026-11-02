"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles.
Strategy: Contact-graph constrained optimization using SLSQP.
We define constraints based on nearest-neighbor pairs (contact graph) and enforce
non-overlap directly via inequalities, rather than using a penalty method.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
EPS = 1e-9
DEADLINE = 140.0  # Safety margin for 150s limit


def _max_radius(pos, r):
    """Maximal radius for each circle given current positions and radii."""
    n = len(r)
    mr = np.empty(n)
    for i in range(n):
        dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        for j in range(n):
            if i != j:
                dmin = min(dmin, np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]) - r[j])
        mr[i] = max(0.0, dmin)
    return mr


def _get_contact_pairs(pos, k=3):
    """Get nearest neighbor pairs using Delaunay triangulation or k-NN."""
    # Using Delaunay gives the natural contact graph for circle packing
    try:
        tri = Delaunay(pos)
        edges = set()
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    edges.add((simplex[i], simplex[j]))
        # Also add nearest neighbors to ensure connectivity if Delaunay is sparse
        from scipy.spatial import cKDTree
        tree = cKDTree(pos)
        _, idx = tree.query(pos, k=k + 1)
        for i in range(N):
            for j in idx[i]:
                if i != j:
                    edges.add((min(i, j), max(i, j)))
        return list(edges)
    except Exception:
        # Fallback to all pairs if triangulation fails (unlikely)
        return [(i, j) for i in range(N) for j in range(i + 1, N)]


def _constraints(x, pairs):
    """Inequality constraints: ||p_i - p_j|| - (r_i + r_j) >= 0"""
    p = x[:2 * N].reshape(N, 2)
    r = x[2 * N:]
    cons = []
    for i, j in pairs:
        d = np.linalg.norm(p[i] - p[j])
        cons.append(d - r[i] - r[j])
    return cons


def _constraint_jac(x, pairs):
    """Jacobian for inequality constraints."""
    p = x[:2 * N].reshape(N, 2)
    r = x[2 * N:]
    J = np.zeros((len(pairs), 3 * N))
    for k, (i, j) in enumerate(pairs):
        diff = p[i] - p[j]
        d = np.linalg.norm(diff)
        if d < EPS:
            d = EPS
        unit = diff / d
        # d(||p_i - p_j||)/dp_i = unit
        J[k, 2 * i] = unit[0]
        J[k, 2 * i + 1] = unit[1]
        # d(||p_i - p_j||)/dp_j = -unit
        J[k, 2 * j] = -unit[0]
        J[k, 2 * j + 1] = -unit[1]
        # d(||p_i - p_j|| - r_i - r_j)/dr_i = -1
        J[k, 2 * N + i] = -1.0
        # d(||p_i - p_j|| - r_i - r_j)/dr_j = -1
        J[k, 2 * N + j] = -1.0
    return J


def _objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2 * N:])


def _objective_grad(x):
    g = np.zeros_like(x)
    g[2 * N:] = -1.0
    return g


def construct():
    t_start = time.time()
    rng = np.random.default_rng(1234)
    best = None
    best_score = -1.0

    def consider(cands):
        nonlocal best, best_score
        # keep only valid (non-overlapping, in-square) circles
        valid = []
        for x, y, r in cands:
            if r <= 0:
                continue
            if x < r + EPS or x > 1 - r - EPS or y < r + EPS or y > 1 - r - EPS:
                continue
            ok = True
            for x2, y2, r2 in valid:
                if (x - x2) ** 2 + (y - y2) ** 2 < (r + r2) ** 2 - 1e-9:
                    ok = False
                    break
            if ok:
                valid.append((x, y, r))
        if len(valid) < N:
            return
        arr = np.array(valid[:N])
        s = float(arr[:, 2].sum())
        if s > best_score:
            best_score = s
            best = arr

    # ---- Phase A: Generate diverse initializations ----
    all_starts = []

    # 1. Grid initializations
    for side in range(4, 8):
        if time.time() > DEADLINE:
            break
        cells = []
        for i in range(side):
            for j in range(side):
                cells.append(((i + 0.5) / side, (j + 0.5) / side))
        cells = cells[:N]
        pos = np.array(cells, dtype=float)
        r = np.full(N, 0.5 / side - 1e-3)
        for _ in range(10):
            r = _max_radius(pos, r)
        all_starts.append(np.concatenate([pos, r]))

    # 2. Hexagonal packing initialization
    # Hex grid is denser than square
    try:
        pos_hex = []
        r_hex = []
        # Generate hex lattice points
        # Spacing s = 2*r/sqrt(3) for vertical, s for horizontal?
        # Let's just place them in a grid and relax
        side = int(np.ceil(np.sqrt(N)))
        # Hex offset
        for i in range(side):
            for j in range(side):
                if len(pos_hex) >= N:
                    break
                x = (j + 0.5 * (i % 2)) / side
                y = (i + 0.5) / side
                # Scale to fit
                pos_hex.append((x, y))
        pos_hex = np.array(pos_hex[:N])
        # Ensure inside
        pos_hex = np.clip(pos_hex, 0.1, 0.9)
        r_hex = np.full(N, 0.05)
        for _ in range(20):
            r_hex = _max_radius(pos_hex, r_hex)
        all_starts.append(np.concatenate([pos_hex, r_hex]))
    except Exception:
        pass

    # 3. Random perturbations of grid
    for side in [5, 6, 7]:
        if time.time() > DEADLINE:
            break
        cells = []
        for i in range(side):
            for j in range(side):
                cells.append(((i + 0.5) / side, (j + 0.5) / side))
        cells = cells[:N]
        pos = np.array(cells, dtype=float)
        r = np.full(N, 0.5 / side - 1e-3)
        for _ in range(10):
            r = _max_radius(pos, r)
        
        for k in range(5):
            if time.time() > DEADLINE:
                break
            p0 = pos + 0.05 * (rng.random((N, 2)) * 2 - 1)
            p0 = np.clip(p0, 0.05, 0.95)
            r0 = r.copy()
            r0 = _max_radius(p0, r0)
            all_starts.append(np.concatenate([p0, r0]))

    # 4. Pure random
    for k in range(5):
        if time.time() > DEADLINE:
            break
        p0 = rng.random((N, 2)) * 0.8 + 0.1
        r0 = np.full(N, 0.03)
        r0 = _max_radius(p0, r0)
        all_starts.append(np.concatenate([p0, r0]))

    if best is not None:
        all_starts.append(best.copy())

    # ---- Phase B: Constrained Optimization (SLSQP) ----
    # We will iterate:
    # 1. Update contact pairs based on current positions
    # 2. Run SLSQP with these constraints
    # 3. Polish radii
    
    def _polish(cands):
        """Final polish: fix positions, maximize radii exactly."""
        if not cands or len(cands) < N:
            return cands
        arr = np.array(cands[:N])
        pos = arr[:, :2].copy()
        rad = arr[:, 2].copy()
        
        # Iterate: fix pos -> max rad
        for it in range(20):
            if time.time() > DEADLINE - 5:
                break
            rad = _max_radius(pos, rad)
            # Check if converged
            if np.allclose(rad, arr[:, 2]):
                break
        out = np.column_stack([pos, rad])
        return out.tolist()

    # Optimization loop
    # We do multiple rounds of SLSQP, updating the contact graph each time
    # This helps because the contact graph changes as circles move
    
    max_rounds = 10
    for x0 in all_starts:
        if time.time() > DEADLINE:
            break
        
        current_x = x0.copy()
        
        # Initial contact pairs
        pos_init = current_x[:2 * N].reshape(N, 2)
        pairs = _get_contact_pairs(pos_init)
        
        # Run SLSQP
        try:
            # Bounds: positions in [0,1], radii in [0, 0.5]
            bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N
            
            # Inequality constraints: g(x) >= 0
            # SLSQP expects g(x) >= 0 for 'ineq'
            
            # We might need to relax constraints slightly if initial point is infeasible
            # due to floating point or initial overlap.
            # _max_radius ensures initial feasibility, but perturbation might break it slightly.
            
            res = minimize(
                _objective,
                current_x,
                jac=_objective_grad,
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': _constraints, 'jac': _constraint_jac, 'args': (pairs,)}
                ],
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
            )
            
            if res.success or res.fun < -np.sum(current_x[2 * N:]):
                current_x = res.x
        except Exception:
            pass
            
        # After SLSQP, the contact graph might have changed.
        # Let's do a few more iterations with updated graph if time permits.
        for _ in range(3):
            if time.time() > DEADLINE - 10:
                break
            pos_cur = current_x[:2 * N].reshape(N, 2)
            new_pairs = _get_contact_pairs(pos_cur)
            # If pairs are the same, stop
            if set(new_pairs) == set(pairs):
                break
            pairs = new_pairs
            try:
                bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N
                res = minimize(
                    _objective,
                    current_x,
                    jac=_objective_grad,
                    bounds=bounds,
                    constraints=[
                        {'type': 'ineq', 'fun': _constraints, 'jac': _constraint_jac, 'args': (pairs,)}
                    ],
                    method='SLSQP',
                    options={'maxiter': 100, 'ftol': 1e-9, 'disp': False}
                )
                if res.success or res.fun < -np.sum(current_x[2 * N:]):
                    current_x = res.x
            except Exception:
                break

        # Convert to list of tuples and polish
        pos_final = current_x[:2 * N].reshape(N, 2)
        rad_final = current_x[2 * N:]
        cands = list(zip(pos_final[:, 0], pos_final[:, 1], rad_final))
        polished = _polish(cands)
        consider(polished)

    # Final polish on the best solution found
    if best is not None:
        cands = best.tolist()
        polished = _polish(cands)
        consider(polished)

    if best is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    return best


if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
