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
# 12 separation directions (6 normals + 6 rotated by 30 deg)
N12 = np.vstack([NORM, NORM @ np.array([[0.5, -S3 / 2.0], [S3 / 2.0, 0.5]])])

# unit hexagon vertices
UNIT = np.array([
    [1.0, 0.0],
    [0.5, S3 / 2.0],
    [-0.5, S3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -S3 / 2.0],
    [0.5, -S3 / 2.0],
])


def outer_side(C, A):
    """Smallest outer hexagon (centered at origin, rot 0) side length enclosing all hexagons."""
    m = 0.0
    for k in range(6):
        n = NORM[k]
        proj = C @ n + 1.0
        m = max(m, float(proj.max()))
    return 2.0 * m / S3


def separation_violation(C, A):
    """For each pair, the max separation along the 12 directions minus 2. Negative => overlap."""
    viol = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = C[i] - C[j]
            sep = np.abs(d @ N12.T).max()
            if sep < 2.0:
                viol += (2.0 - sep) ** 2
    return viol


def objective_flat(flat):
    C = flat[0::3].reshape(N, 2)
    A = flat[1::3]
    side = outer_side(C, A)
    viol = separation_violation(C, A)
    return side + 100.0 * viol


def exact_nonoverlap(C, A, eps=1e-9):
    """Exact check: two convex polygons non-overlapping iff a separating axis exists."""
    for i in range(N):
        for j in range(i + 1, N):
            rel = (A[i] - A[j]) * np.pi / 180.0
            c, s = np.cos(rel), np.sin(rel)
            R = np.array([[c, -s], [s, c]])
            n_j = NORM @ R
            axes = np.vstack([NORM, n_j])
            d = C[i] - C[j]
            sep = np.abs(d @ axes.T).max()
            if sep < 2.0 - eps:
                return False
    return True


def inside_outer(C, A, S, eps=1e-9):
    """All hexagons inside outer hexagon of side S centered at origin rot 0."""
    apothem = S * S3 / 2.0
    for k in range(6):
        n = NORM[k]
        proj = (C @ n) + 1.0
        if proj.max() > apothem + eps:
            return False
    return True


def valid(C, A, S):
    return exact_nonoverlap(C, A) and inside_outer(C, A, S)


def init_flower():
    C = [[0.0, 0.0]]
    for k in range(6):
        a = k * np.pi / 3
        C.append([S3 * np.cos(a), S3 * np.sin(a)])
    for k in range(4):
        a = k * np.pi / 2 + np.pi / 6
        C.append([2.0 * S3 * np.cos(a), 2.0 * S3 * np.sin(a)])
    C = np.array(C[:N])
    A = np.zeros(N)
    return C, A


def init_ring():
    C = [[0.0, 0.0]]
    for k in range(10):
        a = k * 2 * np.pi / 10
        C.append([2.2 * S3 * np.cos(a), 2.2 * S3 * np.sin(a)])
    C = np.array(C[:N])
    A = np.zeros(N)
    return C, A


def init_lattice():
    C = []
    rows = [
        (3, 0.0),
        (4, 1.5),
        (3, 3.0),
        (1, 4.5),
    ]
    for count, y in rows:
        for i in range(count):
            x = (i - (count - 1) / 2.0) * S3
            C.append([x, y])
    C = np.array(C[:N])
    A = np.zeros(N)
    return C, A


def init_staggered():
    C = []
    for i in range(6):
        C.append([(i - 2.5) * S3, 0.0])
    for i in range(5):
        C.append([(i - 2.0) * S3 + S3 / 2, 1.5])
    C = np.array(C[:N])
    A = np.zeros(N)
    return C, A


def init_random_cluster():
    rng = np.random.RandomState(0)
    C = rng.uniform(-1.5 * S3, 1.5 * S3, (N, 2))
    A = rng.uniform(-15, 15, N)
    return C, A


def flat_from_CA(C, A):
    flat = np.empty(3 * N)
    flat[0::3] = C[:, 0]
    flat[1::3] = A
    flat[2::3] = C[:, 1]
    return flat


def ca_from_flat(flat):
    C = np.column_stack([flat[0::3], flat[2::3]])
    A = flat[1::3]
    return C, A


def optimize_nm(C, A, deadline, maxiter=5000):
    """Local polish with Nelder-Mead."""
    flat0 = flat_from_CA(C, A)
    try:
        res = minimize(objective_flat, flat0, method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-6,
                                "adaptive": True})
    except Exception:
        return C, A
    f = res.x
    Cn, An = ca_from_flat(f)
    return Cn, An


def construct():
    start = time.time()
    deadline = start + 140.0

    # Build diverse initializations
    inits = [init_flower(), init_ring(), init_lattice(), init_staggered(), init_random_cluster()]

    best = None
    best_side = float('inf')

    # Check all initializations first
    for C, A in inits:
        Si = outer_side(C, A)
        if valid(C, A, Si) and Si < best_side:
            best_side = Si
            best = (C.copy(), A.copy(), Si)

    # Single DE run with large population
    bound = 12.0
    bounds = []
    for i in range(N):
        bounds.append((-bound, bound))    # x
        bounds.append((-180, 180))        # angle
        bounds.append((-bound, bound))    # y

    try:
        # Use a larger population to cover diverse regions
        res = differential_evolution(objective_flat, bounds, maxiter=300,
                                      popsize=30, tol=1e-5, seed=42,
                                      polish=True, workers=1)
        if time.time() > deadline - 30:
            raise TimeoutError()

        f = res.x
        Cn, An = ca_from_flat(f)
        Sn = outer_side(Cn, An)
        if valid(Cn, An, Sn) and Sn < best_side:
            best_side = Sn
            best = (Cn, An, Sn)

        # Also try to extract good individuals from DE's population
        # by checking a few random points near the best
    except Exception:
        pass

    # Polish top candidates with NM
    if best is not None:
        Cn, An, Sn = best
        if time.time() < deadline - 20:
            Cn2, An2 = optimize_nm(Cn, An, deadline, maxiter=5000)
            Sn2 = outer_side(Cn2, An2)
            if valid(Cn2, An2, Sn2) and Sn2 < best_side:
                best_side = Sn2
                best = (Cn2, An2, Sn2)

    # If we have time, try additional local searches from different starting points
    if time.time() < deadline - 30:
        # Try NM from each initialization
        for C, A in inits:
            if time.time() > deadline - 20:
                break
            Cn, An = optimize_nm(C, A, deadline, maxiter=3000)
            Sn = outer_side(Cn, An)
            if valid(Cn, An, Sn) and Sn < best_side:
                best_side = Sn
                best = (Cn, An, Sn)

    # If nothing valid, fallback
    if best is None:
        C, A = init_flower()
        Cn, An = optimize_nm(C, A, deadline, maxiter=30000)
        Sn = outer_side(Cn, An)
        if not valid(Cn, An, Sn):
            Cn = np.zeros((N, 2))
            for i in range(N):
                Cn[i] = [6.0 * i, 0.0]
            An = np.zeros(N)
            Sn = outer_side(Cn, An)
        best = (Cn, An, Sn)

    Cn, An, Sn = best
    inner = [[float(Cn[i, 0]), float(Cn[i, 1]), float(An[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(Sn), 0.0
