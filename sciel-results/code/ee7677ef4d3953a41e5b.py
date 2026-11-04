import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11
S3 = np.sqrt(3.0)

# 6 unit-hexagon edge normals (angles 0,60,...300)
NORM = np.array([
    [1.0, 0.0],
    [0.5, S3 / 2.0],
    [-0.5, S3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -S3 / 2.0],
    [0.5, -S3 / 2.0],
])

# Minimum separation along each normal for two non-overlapping unit hexagons (same orientation)
MIN_SEP = S3  # = 2 * inradius = 2 * (sqrt(3)/2)


def outer_side_from_centers(C, outer_angle_deg=0.0):
    """Smallest outer hexagon side length enclosing all hexagons centered at C,
    with outer hexagon rotated by outer_angle_deg."""
    theta = np.radians(outer_angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    # Rotated normals
    n_rot = (NORM @ R.T).T  # shape (6,2)
    m = 0.0
    for k in range(6):
        n = n_rot[k]
        proj = C @ n + S3 / 2.0  # + inradius of unit hex
        m = max(m, float(proj.max()))
    # m = required inradius of outer hexagon
    # inradius = side * sqrt(3)/2 => side = m * 2/sqrt(3)
    return 2.0 * m / S3


def separation_violation(C):
    """Sum of squared violations of separation constraints."""
    viol = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = C[i] - C[j]
            sep = np.abs(d @ NORM.T).max()
            if sep < MIN_SEP:
                viol += (MIN_SEP - sep) ** 2
    return viol


def objective(params):
    """params: [x0,y0,...,x10,y10, outer_angle_deg]"""
    C = params[:2 * N].reshape(N, 2)
    outer_angle = params[2 * N]
    side = outer_side_from_centers(C, outer_angle)
    viol = separation_violation(C)
    return side + 50.0 * viol


def exact_nonoverlap(C, eps=1e-9):
    for i in range(N):
        for j in range(i + 1, N):
            d = C[i] - C[j]
            sep = np.abs(d @ NORM.T).max()
            if sep < MIN_SEP - eps:
                return False
    return True


def inside_outer(C, S, outer_angle_deg=0.0, eps=1e-9):
    theta = np.radians(outer_angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    n_rot = (NORM @ R.T).T
    apothem_outer = S * S3 / 2.0
    for k in range(6):
        n = n_rot[k]
        proj = C @ n + S3 / 2.0
        if proj.max() > apothem_outer + eps:
            return False
    return True


def valid(C, S, outer_angle_deg=0.0):
    return exact_nonoverlap(C) and inside_outer(C, S, outer_angle_deg)


def init_flower():
    C = [[0.0, 0.0]]
    for k in range(6):
        a = k * np.pi / 3
        C.append([S3 * np.cos(a), S3 * np.sin(a)])
    for k in range(4):
        a = k * np.pi / 2 + np.pi / 6
        C.append([2.0 * S3 * np.cos(a), 2.0 * S3 * np.sin(a)])
    return np.array(C[:N])


def init_ring():
    C = [[0.0, 0.0]]
    for k in range(10):
        a = k * 2 * np.pi / 10
        C.append([2.2 * S3 * np.cos(a), 2.2 * S3 * np.sin(a)])
    return np.array(C[:N])


def init_lattice():
    C = []
    rows = [(3, 0.0), (4, 1.5), (3, 3.0), (1, 4.5)]
    for count, y in rows:
        for i in range(count):
            x = (i - (count - 1) / 2.0) * S3
            C.append([x, y])
    return np.array(C[:N])


def init_staggered():
    C = []
    for i in range(6):
        C.append([(i - 2.5) * S3, 0.0])
    for i in range(5):
        C.append([(i - 2.0) * S3 + S3 / 2, 1.5])
    return np.array(C[:N])


def init_compact_hex():
    """Tighter hexagonal arrangement: center + 6 ring + 4 more in second ring."""
    C = [[0.0, 0.0]]
    # First ring: 6 hexagons at distance sqrt(3) in 6 directions
    for k in range(6):
        a = k * np.pi / 3
        C.append([S3 * np.cos(a), S3 * np.sin(a)])
    # Second ring: 4 hexagons at distance 2*sqrt(3) in directions 30, 90, 210, 270
    for k in range(4):
        a = np.radians(30 + k * 90)
        C.append([2.0 * S3 * np.cos(a), 2.0 * S3 * np.sin(a)])
    return np.array(C[:N])


def init_optimal_11():
    """Hand-crafted arrangement known to be near-optimal for 11 hexagons."""
    # Based on known packings: a hexagon of side 3 has 19 cells,
    # we need 11. Try a compact shape.
    C = []
    # Center
    C.append([0.0, 0.0])
    # 6 neighbors
    for k in range(6):
        a = k * np.pi / 3
        C.append([S3 * np.cos(a), S3 * np.sin(a)])
    # 4 more: at (±3*sqrt(3)/2, ±1.5) type positions
    C.append([1.5 * S3, 1.5])
    C.append([-1.5 * S3, 1.5])
    C.append([1.5 * S3, -1.5])
    C.append([-1.5 * S3, -1.5])
    return np.array(C[:N])


def polish_lbfgs(params0, deadline, maxiter=2000):
    """Polish with L-BFGS-B."""
    try:
        res = minimize(objective, params0, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8})
        if time.time() > deadline:
            return params0
        return res.x
    except Exception:
        return params0


def polish_nm(params0, deadline, maxiter=3000):
    """Polish with Nelder-Mead."""
    try:
        res = minimize(objective, params0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-7, 'fatol': 1e-7,
                                'adaptive': True})
        if time.time() > deadline:
            return params0
        return res.x
    except Exception:
        return params0


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    inits = [init_flower(), init_ring(), init_lattice(), init_staggered(),
             init_compact_hex(), init_optimal_11()]

    best_params = None
    best_side = float('inf')

    def eval_params(params):
        nonlocal best_params, best_side
        C = params[:2 * N].reshape(N, 2)
        outer_angle = params[2 * N]
        side = outer_side_from_centers(C, outer_angle)
        if valid(C, side, outer_angle):
            if side < best_side:
                best_side = side
                best_params = params.copy()

    # Phase 1: Polish each initialization
    for C in inits:
        if time.time() > deadline - 80:
            break
        # Try a few outer angles
        for oa in [0.0, 15.0, 30.0, 45.0]:
            if time.time() > deadline - 70:
                break
            params0 = np.concatenate([C.flatten(), [oa]])
            # L-BFGS-B polish
            p1 = polish_lbfgs(params0, deadline - 30, maxiter=1000)
            eval_params(p1)
            # Nelder-Mead polish
            p2 = polish_nm(p1, deadline - 20, maxiter=2000)
            eval_params(p2)

    # Phase 2: Differential evolution on a subset of promising regions
    if time.time() < deadline - 60:
        # Use DE on the best few initializations
        best_C = None
        best_s = float('inf')
        for C in inits:
            s = outer_side_from_centers(C, 0.0)
            if valid(C, s, 0.0) and s < best_s:
                best_s = s
                best_C = C

        if best_C is not None:
            # DE bounds: centers in [-4, 4], outer angle in [-30, 30]
            bounds = [(-4.0, 4.0)] * (2 * N) + [(-30.0, 30.0)]
            # Use the best initialization as a seed for DE population
            try:
                res = differential_evolution(
                    objective, bounds,
                    seed=42, maxiter=50, popsize=15,
                    tol=1e-6, polish=True,
                    init='sobol'
                )
                if time.time() < deadline - 10:
                    eval_params(res.x)
            except Exception:
                pass

    # Phase 3: Final high-precision polish on the best
    if best_params is not None and time.time() < deadline - 15:
        p_final = polish_lbfgs(best_params, deadline - 5, maxiter=5000)
        eval_params(p_final)
        p_final2 = polish_nm(p_final, deadline - 3, maxiter=5000)
        eval_params(p_final2)

    # Fallback
    if best_params is None:
        C = init_flower()
        params = np.concatenate([C.flatten(), [0.0]])
        p = polish_lbfgs(params, deadline - 5, maxiter=5000)
        C = p[:2 * N].reshape(N, 2)
        oa = p[2 * N]
        side = outer_side_from_centers(C, oa)
        if not valid(C, side, oa):
            # Last resort: spread out
            C = np.zeros((N, 2))
            for i in range(N):
                C[i] = [6.0 * i, 0.0]
            oa = 0.0
            side = outer_side_from_centers(C, oa)
        best_params = np.concatenate([C.flatten(), [oa]])
        best_side = side

    C = best_params[:2 * N].reshape(N, 2)
    outer_angle = float(best_params[2 * N])
    inner = [[float(C[i, 0]), float(C[i, 1]), 0.0] for i in range(N)]
    return inner, (0.0, 0.0), float(best_side), outer_angle
