import numpy as np
import time
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3.0)

# 6 unit-hexagon edge normals
NORM = np.array([
    [1.0, 0.0],
    [0.5, S3 / 2.0],
    [-0.5, S3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -S3 / 2.0],
    [0.5, -S3 / 2.0],
])


def outer_side(C):
    m = 0.0
    for k in range(6):
        n = NORM[k]
        proj = C @ n + 1.0
        m = max(m, float(proj.max()))
    return 2.0 * m / S3


def separation_violation(C):
    viol = 0.0
    for i in range(len(C)):
        for j in range(i + 1, len(C)):
            d = C[i] - C[j]
            sep = np.abs(d @ NORM.T).max()
            if sep < 2.0:
                viol += (2.0 - sep) ** 2
    return viol


def exact_nonoverlap(C, eps=1e-9):
    for i in range(len(C)):
        for j in range(i + 1, len(C)):
            d = C[i] - C[j]
            sep = np.abs(d @ NORM.T).max()
            if sep < 2.0 - eps:
                return False
    return True


def inside_outer(C, S, eps=1e-9):
    apothem = S * S3 / 2.0
    for k in range(6):
        n = NORM[k]
        proj = (C @ n) + 1.0
        if proj.max() > apothem + eps:
            return False
    return True


def valid(C, S):
    return exact_nonoverlap(C) and inside_outer(C, S)


def build_rows(row_counts, gaps, row_offsets):
    """Build hexagon centers from row structure.
    row_counts: list of ints, count per row
    gaps: list of vertical gaps between consecutive rows (len = len(row_counts)-1)
    row_offsets: horizontal shift per row (len = len(row_counts))
    Returns (C, total_height) where C is (N,2) array.
    """
    C = []
    y = 0.0
    for i, count in enumerate(row_counts):
        offset = row_offsets[i] if i < len(row_offsets) else 0.0
        for j in range(count):
            x = (j - (count - 1) / 2.0) * S3 + offset
            C.append([x, y])
        if i < len(gaps):
            y += gaps[i]
    C = np.array(C)
    total_height = y
    return C, total_height


def row_objective(params, row_counts, gaps_default):
    """params: [row_offsets..., gap_scales...]
    gap_scales multiply default gaps (1.5) to allow compression.
    """
    n_rows = len(row_counts)
    n_offsets = n_rows
    n_gaps = n_rows - 1
    offsets = params[:n_offsets]
    gap_scales = params[n_offsets:n_offsets + n_gaps]
    
    gaps = [1.5 * max(1.0, gs) for gs in gap_scales]
    C, total_height = build_rows(row_counts, gaps, offsets)
    
    side = outer_side(C)
    viol = separation_violation(C)
    return side + 100.0 * viol


def optimize_rows(row_counts, deadline, maxiter=5000):
    """Optimize a given row structure."""
    n_rows = len(row_counts)
    n_gaps = max(0, n_rows - 1)
    n_params = n_rows + n_gaps
    x0 = np.zeros(n_params)
    
    def obj(p):
        return row_objective(p, row_counts, 1.5)
    
    try:
        res = minimize(obj, x0, method='Nelder-Mead',
                      options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8,
                               'adaptive': True})
    except Exception:
        return None
    
    offsets = res.x[:n_rows]
    gap_scales = res.x[n_rows:n_rows + n_gaps]
    gaps = [1.5 * max(1.0, gs) for gs in gap_scales]
    C, _ = build_rows(row_counts, gaps, offsets)
    return C


def polish_free(C, deadline, maxiter=3000):
    """Final free-coordinate polish."""
    if len(C) != N:
        return C
    flat0 = C.flatten()
    
    def obj(flat):
        Cx = flat.reshape(N, 2)
        side = outer_side(Cx)
        viol = separation_violation(Cx)
        return side + 100.0 * viol
    
    try:
        res = minimize(obj, flat0, method='Nelder-Mead',
                      options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-9,
                               'adaptive': True})
    except Exception:
        return C
    return res.x.reshape(N, 2)


def try_rotation(C, deadline):
    """Try rotating the whole cluster to reduce bounding hexagon side."""
    if len(C) != N:
        return C
    centroid = C.mean(axis=0)
    Cc = C - centroid
    
    best_C = C
    best_side = outer_side(C)
    
    def rigid_obj(params):
        theta, tx, ty = params
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        Cx = Cc @ R.T + np.array([tx, ty])
        side = outer_side(Cx)
        viol = separation_violation(Cx)
        return side + 100.0 * viol
    
    for theta0 in [0, np.pi/12, np.pi/6, np.pi/4, np.pi/3, np.pi/2]:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(rigid_obj, [theta0, 0, 0], method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9,
                                   'adaptive': True})
            if res.fun < best_side:
                theta, tx, ty = res.x
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s], [s, c]])
                best_C = Cc @ R.T + np.array([tx, ty])
                best_side = res.fun
        except Exception:
            pass
    return best_C


def construct():
    start = time.time()
    deadline = start + 140.0
    
    best = None
    best_side = float('inf')
    
    def update(C):
        nonlocal best, best_side
        if C is None or len(C) != N:
            return
        S = outer_side(C)
        if valid(C, S) and S < best_side:
            best_side = S
            best = C
    
    # Row structures to try (most promising first)
    row_structures = [
        [2, 4, 2, 2, 1],   # known optimal ~4.23
        [2, 4, 4, 1],
        [3, 4, 3, 1],
        [2, 3, 4, 2],
        [1, 4, 4, 2],
        [4, 3, 4],
        [3, 3, 3, 2],
        [2, 5, 4],
        [5, 3, 3],
        [4, 4, 3],
        [2, 4, 3, 2],
        [3, 4, 4],
    ]
    
    for rc in row_structures:
        if sum(rc) != N:
            continue
        if time.time() > deadline - 30:
            break
        C = optimize_rows(rc, deadline, maxiter=3000)
        update(C)
    
    # Also try the classic flower + ring initializations
    inits = []
    # Flower: 1 center + 6 ring + 4 outer
    C_flower = [[0.0, 0.0]]
    for k in range(6):
        a = k * np.pi / 3
        C_flower.append([S3 * np.cos(a), S3 * np.sin(a)])
    for k in range(4):
        a = k * np.pi / 2 + np.pi / 6
        C_flower.append([2.0 * S3 * np.cos(a), 2.0 * S3 * np.sin(a)])
    inits.append(np.array(C_flower[:N]))
    
    # Staggered 6+5
    C_stag = []
    for i in range(6):
        C_stag.append([(i - 2.5) * S3, 0.0])
    for i in range(5):
        C_stag.append([(i - 2.0) * S3 + S3 / 2, 1.5])
    inits.append(np.array(C_stag[:N]))
    
    # Lattice 3-4-3-1
    C_lat = []
    rows = [(3, 0.0), (4, 1.5), (3, 3.0), (1, 4.5)]
    for count, y in rows:
        for i in range(count):
            x = (i - (count - 1) / 2.0) * S3
            C_lat.append([x, y])
    inits.append(np.array(C_lat[:N]))
    
    for C in inits:
        if time.time() > deadline - 25:
            break
        Cn = polish_free(C, deadline, maxiter=2000)
        update(Cn)
    
    # Rotation optimization on best
    if best is not None and time.time() < deadline - 15:
        Cr = try_rotation(best, deadline)
        update(Cr)
    
    # Final high-precision polish
    if best is not None and time.time() < deadline - 5:
        Cf = polish_free(best, deadline, maxiter=10000)
        update(Cf)
    
    # Fallback
    if best is None:
        C = np.zeros((N, 2))
        for i in range(N):
            C[i] = [6.0 * i, 0.0]
        best = C
        best_side = outer_side(C)
    
    inner = [[float(best[i, 0]), float(best[i, 1]), 0.0] for i in range(N)]
    return inner, (0.0, 0.0), float(best_side), 0.0
