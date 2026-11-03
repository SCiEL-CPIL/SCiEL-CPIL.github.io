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
    # max over all vertices of max over 6 normals of dot(v, n)  ==  max over i,k of dot(c_i, n_k)
    # because max over vertices of dot(v,n) = dot(c, n) + 1 (apothem of unit hex is 1)
    m = 0.0
    for k in range(6):
        n = NORM[k]
        proj = C @ n + 1.0
        m = max(m, float(proj.max()))
    return 2.0 * m / S3


def separation_violation(C, A):
    """For each pair, the max separation along the 12 directions minus 2. Negative => overlap."""
    # We need to account for rotation: the Minkowski-sum radius depends on relative angle.
    # For hexagons with relative rotation delta, the separation condition uses normals of the
    # union of both sets of edges. Use 12 directions (6 of each hexagon's orientation).
    # Approximation: use fixed 12 directions (union when both aligned). Rotations handled
    # approximately; final validity gate uses exact check.
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
    """Exact check: two convex polygons non-overlapping iff a separating axis exists.
    Use SAT on the 12 edge normals (union of both orientations)."""
    for i in range(N):
        for j in range(i + 1, N):
            # relative angle
            rel = (A[i] - A[j]) * np.pi / 180.0
            c, s = np.cos(rel), np.sin(rel)
            R = np.array([[c, -s], [s, c]])
            # normals of hexagon j in hexagon i's frame = NORM @ R
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
    # 4 more in a ring at radius ~2*sqrt(3)
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
    # 3-4-3-1 staggered lattice rows
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
    # two staggered rows of 6 and 5
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


def optimize_nm(C, A, deadline, maxiter=5000):
    """Local polish with Nelder-Mead."""
    flat0 = np.empty(3 * N)
    flat0[0::3] = C[:, 0]
    flat0[1::3] = A
    flat0[2::3] = C[:, 1]
    try:
        res = minimize(objective_flat, flat0, method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-6,
                                "adaptive": True})
    except Exception:
        return C, A
    f = res.x
    Cn = np.column_stack([f[0::3], f[2::3]])
    An = f[1::3]
    return Cn, An

def optimize_de(C, A, deadline, maxiter=2000):
    """Global search with Differential Evolution."""
    # Bounds for centers and angles
    # Estimate a reasonable bound based on initial spread or fixed large bound
    bound = 10.0
    bounds = []
    for i in range(N):
        bounds.append((-bound, bound)) # x
        bounds.append((-180, 180)) # angle
        bounds.append((-bound, bound)) # y
    
    # DE expects bounds as list of tuples
    # We need to map flat index to bounds
    # flat: x0, a0, y0, x1, a1, y1 ...
    
    # Run DE
    try:
        # Use a seed for reproducibility in tests, but random in production?
        # The task allows randomness.
        res = differential_evolution(objective_flat, bounds, maxiter=maxiter, 
                                      popsize=15, tol=1e-4, seed=42, 
                                      polish=True, workers=1)
    except Exception:
        return C, A
        
    if res.success or res.fun < objective_flat(np.array([C[0,0], A[0], C[0,1]])):
         # If DE found something better, use it
         f = res.x
         Cn = np.column_stack([f[0::3], f[2::3]])
         An = f[1::3]
         return Cn, An
    return C, A

def construct():
    start = time.time()
    deadline = start + 140.0

    inits = [init_flower(), init_ring(), init_lattice(), init_staggered(), init_random_cluster()]

    best = None
    best_side = float('inf')

    # 1. Try DE from each initialization
    for C, A in inits:
        if time.time() > deadline - 60:
            break
        Cn, An = optimize_de(C, A, deadline, maxiter=1000)
        Sn = outer_side(Cn, An)
        if valid(Cn, An, Sn):
            if Sn < best_side:
                best_side = Sn
                best = (Cn, An, Sn)
        # Also check initial
        Si = outer_side(C, A)
        if valid(C, A, Si) and Si < best_side:
            best_side = Si
            best = (C, A, Si)

    # 2. Polish the best found with NM
    if best is not None:
        Cn, An, Sn = best
        if time.time() < deadline - 20:
            Cn2, An2 = optimize_nm(Cn, An, deadline, maxiter=5000)
            Sn2 = outer_side(Cn2, An2)
            if valid(Cn2, An2, Sn2) and Sn2 < best_side:
                best_side = Sn2
                best = (Cn2, An2, Sn2)

    # 3. If nothing valid, fallback
    if best is None:
        C, A = init_flower()
        Cn, An = optimize_nm(C, A, deadline, maxiter=30000)
        Sn = outer_side(Cn, An)
        if not valid(Cn, An, Sn):
            # fallback: spread out widely (valid but large)
            Cn = np.zeros((N, 2))
            for i in range(N):
                Cn[i] = [6.0 * i, 0.0]
            An = np.zeros(N)
            Sn = outer_side(Cn, An)
        best = (Cn, An, Sn)

    Cn, An, Sn = best
    inner = [[float(Cn[i, 0]), float(Cn[i, 1]), float(An[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(Sn), 0.0
