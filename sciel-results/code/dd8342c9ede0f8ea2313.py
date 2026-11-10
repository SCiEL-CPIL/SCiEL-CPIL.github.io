# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0) * np.arange(6)
    return np.column_stack((cx + s * np.cos(th), cy + s * np.sin(th)))

def _edge_normals(V):
    n = np.empty((6, 2))
    for i in range(6):
        e = V[(i + 1) % 6] - V[i]
        L = np.hypot(e[0], e[1])
        n[i] = np.array([e[1], -e[0]]) / L
    return n

def _gap_and_margin(A, B):
    """Return (best_margin, active_normal). best_margin>0 means separated."""
    nA = _edge_normals(A)
    nB = _edge_normals(B)
    best = -np.inf
    best_n = nA[0]
    for n in nA:
        loA, hiA = (A @ n).min(), (A @ n).max()
        loB, hiB = (B @ n).min(), (B @ n).max()
        sep = max(loB - hiA, loA - hiB)
        if sep > best:
            best = sep
            best_n = n
    for n in nB:
        loA, hiA = (A @ n).min(), (A @ n).max()
        loB, hiB = (B @ n).min(), (B @ n).max()
        sep = max(loB - hiA, loA - hiB)
        if sep > best:
            best = sep
            best_n = n
    return best, best_n

def _outer_side_for_points(pts):
    best_s = np.inf
    best_a = 0.0
    for a_deg in np.linspace(0, 60, 361):
        a = np.deg2rad(a_deg)
        th = a + np.deg2rad(60.0) * np.arange(6)
        normals = np.column_stack((np.cos(th), np.sin(th)))
        proj = pts @ normals.T
        maxproj = proj.max(axis=0)
        apothem = maxproj.max()
        s = 2.0 * apothem / np.sqrt(3.0)
        if s < best_s:
            best_s = s
            best_a = a_deg
    return best_s, best_a

def _all_valid(inner, outer, side):
    """Full validity check. inner (11,3), outer (3,), side float."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(side):
        return False
    if side < np.sqrt(11.0) - 1e-9:
        return False
    # separation
    Vs = [_hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2]) for i in range(11)]
    for i in range(11):
        for j in range(i + 1, 11):
            m, _ = _gap_and_margin(Vs[i], Vs[j])
            if m <= 1e-6:
                return False
    # containment
    cx, cy, ca = outer[0], outer[1], outer[2]
    OV = _hex_vertices(cx, cy, side, ca)
    for V in Vs:
        for i in range(6):
            p1 = OV[i]
            p2 = OV[(i + 1) % 6]
            cross = (p2[0] - p1[0]) * (V[:, 1] - p1[1]) - (p2[1] - p1[1]) * (V[:, 0] - p1[0])
            if np.any(cross < -1e-6):
                return False
    # edge length check after shifting by outer center
    for V in Vs:
        Vs2 = V - np.array([cx, cy])
        for i in range(6):
            L = np.hypot(*(Vs2[(i + 1) % 6] - Vs2[i]))
            if abs(L - 1.0) > 1e-8:
                return False
    return True

def hexagon_packing_11():
    t0 = time.time()
    deadline = t0 + 600.0

    def _outer_active(pts):
        """Return (s, active_normal) for pts (N,2)."""
        best_s = np.inf
        best_n = np.zeros(2)
        for a_deg in np.linspace(0, 60, 361):
            a = np.deg2rad(a_deg)
            th = a + np.deg2rad(60.0) * np.arange(6)
            normals = np.column_stack((np.cos(th), np.sin(th)))
            proj = pts @ normals.T
            maxproj = proj.max(axis=0)
            apothem = maxproj.max()
            s = 2.0 * apothem / np.sqrt(3.0)
            if s < best_s:
                best_s = s
                idx = int(np.argmax(maxproj))
                best_n = normals[idx]
        return best_s, best_n

    def _objective_grad(state):
        x = state[0::3].copy()
        y = state[1::3].copy()
        a = state[2::3].copy()
        # center shift
        cx = x.mean()
        cy = y.mean()
        pts = np.column_stack((x - cx, y - cy))
        s, n_act = _outer_active(pts)
        # gradient of s w.r.t. x,y: s = 2/sqrt(3) * max_k (pts @ n_k).max over k
        # active normal n_act; ds/dx_i = 2/sqrt(3) * (n_act if this i is the max projecter)
        # We approximate: gradient of max over points of (p_i . n_act) w.r.t. p_i.
        proj_act = pts @ n_act
        max_idx = int(np.argmax(proj_act))
        grad = np.zeros_like(state)
        # For the active point: ds/dx_i = 2/sqrt(3)*n_act.x, ds/dy_i = 2/sqrt(3)*n_act.y
        # But because of centering (subtract mean), d/dx_i = (1 - 1/11) factor for the active,
        # and -1/11 for others. We use a simpler, robust finite-difference-free approach:
        # analytic subgradient ignoring centering derivative (centering is a constant shift
        # that doesn't change s). Actually s depends on pts = x - mean(x), so d s/d x_i for
        # active = 2/sqrt3 * n_act, and for others 0, but mean shift: d(pts_j)/dx_i = -1/11 for
        # all j, so the max over j of (pts_j . n) has subgradient: for the active point j*,
        # d/dx_i = 2/sqrt3 * (delta_{i,j*} - 1/11) * n_act.
        factor = 2.0 / np.sqrt(3.0)
        for i in range(11):
            coef = factor * (1.0 if i == max_idx else 0.0) - factor * (1.0 / 11.0)
            grad[3 * i + 0] += coef * n_act[0]
            grad[3 * i + 1] += coef * n_act[1]
            # angle has no effect on s (only on separation), so 0 here.

        # separation penalty + gradient
        Vs = [_hex_vertices(x[i], y[i], 1.0, a[i]) for i in range(11)]
        pen = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                m, n_act2 = _gap_and_margin(Vs[i], Vs[j])
                if m <= 1e-6:
                    viol = 1e-6 - m
                    pen += viol * viol
                    # gradient of viol w.r.t. positions: viol = 1e-6 - sep, sep along n_act2.
                    # d sep/d x: for hexagon i, moving it by +dx decreases sep by n_act2 . dx
                    # if i is the "min" side... we approximate subgradient:
                    # For the active normal, sep = max(loB-hiA, loA-hiB). We just push them apart
                    # along n_act2: gradient on i = -n_act2 * 2*viol, on j = +n_act2 * 2*viol
                    # (sign chosen to increase separation). Use magnitude 2*viol for smoothness.
                    gmag = 2.0 * viol
                    grad[3 * i + 0] += -gmag * n_act2[0]
                    grad[3 * i + 1] += -gmag * n_act2[1]
                    grad[3 * j + 0] += +gmag * n_act2[0]
                    grad[3 * j + 1] += +gmag * n_act2[1]
        total = s + 1000.0 * pen
        return total, grad

    # ---- Multi-start search ----
    best_state = None
    best_val = -np.inf

    def _eval_state(state):
        x = state[0::3]
        y = state[1::3]
        a = state[2::3]
        cx = x.mean()
        cy = y.mean()
        pts = np.column_stack((x - cx, y - cy))
        s, a_out = _outer_side_for_points(pts)
        inner = np.column_stack((x, y, a))
        outer = np.array([cx, cy, a_out])
        if _all_valid(inner, outer, s):
            return 1.0 / s, inner, outer, s
        return None, inner, outer, s

    # Seed arrangements
    def _seed_hex():
        return np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [-2.0, 0.0],
            [1.0, 1.732],
            [-1.0, 1.732],
            [1.0, -1.732],
            [-1.0, -1.732],
            [3.0, 1.732],
            [-3.0, 1.732],
            [3.0, -1.732],
            [-3.0, -1.732],
        ])

    seeds = []
    base = _seed_hex()
    seeds.append(np.concatenate([base[:, 0], base[:, 1], np.zeros(11)]))
    # a couple of random perturbations as additional starts
    rng = np.random.default_rng(42)
    for _ in range(3):
        p = base + rng.normal(0, 0.3, size=(11, 2))
        ang = rng.uniform(-30, 30, size=11)
        seeds.append(np.concatenate([p[:, 0], p[:, 1], ang]))

    for seed_state in seeds:
        if time.time() > deadline - 60.0:
            break
        # L-BFGS-B optimization
        res = minimize(_objective_grad, seed_state, jac=True, method='L-BFGS-B',
                       options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-9,
                                'maxfun': 2000, 'maxls': 50})
        if res.success or res.fun < np.inf:
            st = res.x
            val, inner, outer, s = _eval_state(st)
            if val is not None and val > best_val:
                best_val = val
                best_state = (inner, outer, s)
        # also evaluate the seed itself
        val, inner, outer, s = _eval_state(seed_state)
        if val is not None and val > best_val:
            best_val = val
            best_state = (inner, outer, s)

    if best_state is None:
        # fallback loose packing
        inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ], dtype=float)
        outer = np.array([0, 0, 0], dtype=float)
        return inner, outer, 8.0

    inner, outer, s = best_state
    return inner, outer, float(s)

# EVOLVE-BLOCK-END
