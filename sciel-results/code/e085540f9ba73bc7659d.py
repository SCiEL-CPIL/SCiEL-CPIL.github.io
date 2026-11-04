import numpy as np
import time

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


def init_ring():
    C = [[0.0, 0.0]]
    for k in range(10):
        a = k * 2 * np.pi / 10
        C.append([2.2 * S3 * np.cos(a), 2.2 * S3 * np.sin(a)])
    C = np.array(C[:N])
    A = np.zeros(N)
    return C, A


def init_random_cluster():
    rng = np.random.RandomState(0)
    C = rng.uniform(-1.5 * S3, 1.5 * S3, (N, 2))
    A = rng.uniform(-15, 15, N)
    return C, A


def separation_violation(C, A):
    """For each pair, the max separation along the 12 directions minus 2. Negative => overlap."""
    viol = 0.0
    N12 = np.vstack([NORM, NORM @ np.array([[0.5, -S3 / 2.0], [S3 / 2.0, 0.5]])])
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


def optimize_nm(C, A, maxiter=5000):
    """Local polish with Nelder-Mead."""
    from scipy.optimize import minimize
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


def simulated_annealing(C, A, deadline, seed):
    """
    Run Simulated Annealing on a single cohort.
    Returns the best valid configuration found.
    """
    rng = np.random.RandomState(seed)
    current_C = C.copy()
    current_A = A.copy()

    # Initial side length
    current_S = outer_side(current_C, current_A)
    current_valid = valid(current_C, current_A, current_S)
    current_cost = current_S if current_valid else current_S + 1000 * separation_violation(current_C, current_A)

    best_C = current_C.copy()
    best_A = current_A.copy()
    best_S = current_S if current_valid else float('inf')

    T = 0.5
    T_min = 1e-4
    cooling = 0.9995

    while time.time() < deadline and T > T_min:
        # Generate move
        move_type = rng.random()
        if move_type < 0.6:
            # Translate
            idx = rng.randint(0, N - 1)
            delta = rng.normal(0, 0.2, 2)
            new_C = current_C.copy()
            new_C[idx] += delta
            new_A = current_A.copy()
        elif move_type < 0.8:
            # Rotate
            idx = rng.randint(0, N - 1)
            new_C = current_C.copy()
            new_A = current_A.copy()
            new_A[idx] += rng.normal(0, 5.0)
        else:
            # Push apart overlapping pair
            # Find a random pair
            i, j = rng.randint(0, N - 1), rng.randint(0, N - 1)
            if i == j:
                continue
            new_C = current_C.copy()
            new_A = current_A.copy()
            d = new_C[i] - new_C[j]
            if np.linalg.norm(d) > 1e-6:
                d = d / np.linalg.norm(d)
                push = 0.1
                new_C[i] += d * push
                new_C[j] -= d * push
            else:
                new_C[i] += [0.1, 0]

        new_S = outer_side(new_C, new_A)
        new_valid = valid(new_C, new_A, new_S)

        # Calculate cost
        if new_valid:
            new_cost = new_S
        else:
            new_cost = new_S + 1000 * separation_violation(new_C, new_A)

        delta_cost = new_cost - current_cost

        # Acceptance
        if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
            current_C = new_C
            current_A = new_A
            current_S = new_S
            current_valid = new_valid
            current_cost = new_cost

            if current_valid and current_S < best_S:
                best_S = current_S
                best_C = current_C.copy()
                best_A = current_A.copy()

        T *= cooling

    return best_C, best_A, best_S


def build_rigid_cluster(scale_factor=1.002):
    """
    Build a rigid 11-hexagon cluster on the triangular lattice.
    Returns (cluster_centers, flat_verts) where flat_verts are all 66 vertices
    in the cluster frame (before global rotation).
    """
    d = S3
    # Lattice vectors for unit hexagons (side 1, vertex at 0 deg)
    u = np.array([1.5, d / 2])
    v = np.array([0.0, d])

    # 11 closest points to origin on the triangular lattice
    lattice_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, -1), (2, -1), (-2, 1)
    ]

    cluster_centers = np.array([i * u + j * v for i, j in lattice_coords])
    cluster_centers *= scale_factor

    # All vertices for each hexagon (relative to its center)
    all_base_verts = np.tile(UNIT, (N, 1, 1))
    cluster_verts = all_base_verts + cluster_centers[:, np.newaxis, :]
    flat_verts = cluster_verts.reshape(-1, 2)

    return cluster_centers, flat_verts


def compute_side_for_angles(cluster_angle_deg, phi_deg, flat_verts):
    """
    Compute the minimum outer hexagon side length for given cluster rotation
    and outer hexagon rotation angle.
    """
    angle_rad = np.deg2rad(cluster_angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    rotated = flat_verts @ R.T

    phi_rad = np.deg2rad(phi_deg)
    normals = np.array([
        [np.cos(phi_rad + k * np.pi / 3), np.sin(phi_rad + k * np.pi / 3)]
        for k in range(6)
    ])
    projs = rotated @ normals.T
    max_proj = np.max(projs)
    return 2.0 * max_proj / S3


def golden_section_search(f, a, b, tol=1e-8, max_iter=200):
    """
    Golden section search to find the minimum of f in [a, b].
    """
    gr = (np.sqrt(5) - 1) / 2  # golden ratio conjugate
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)

    for _ in range(max_iter):
        if b - a < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = f(d)

    return (a + b) / 2, f((a + b) / 2)


def optimize_rigid_cluster(flat_verts, deadline):
    """
    Optimize the rigid cluster by searching over global rotation angle
    and outer hexagon angle. Uses a combination of grid search, golden section
    search, and Nelder-Mead for high-precision optimization.
    Returns (best_cluster_angle, best_phi, best_side).
    """
    start = time.time()

    # Phase 1: Coarse grid search over cluster angle (0 to 60 degrees)
    # and outer hexagon angle (0 to 60 degrees)
    angles_coarse = np.linspace(0, 60, 61, endpoint=False)
    phis_coarse = np.linspace(0, 60, 61, endpoint=False)

    best_side = float('inf')
    best_angle = 0.0
    best_phi = 0.0

    for a in angles_coarse:
        for p in phis_coarse:
            s = compute_side_for_angles(a, p, flat_verts)
            if s < best_side:
                best_side = s
                best_angle = a
                best_phi = p

    if time.time() > deadline - 15:
        return best_angle, best_phi, best_side

    # Phase 2: Fine grid search around the best coarse result
    fine_range = 3.0  # degrees
    angles_fine = np.linspace(best_angle - fine_range, best_angle + fine_range, 121)
    phis_fine = np.linspace(best_phi - fine_range, best_phi + fine_range, 121)

    for a in angles_fine:
        for p in phis_fine:
            s = compute_side_for_angles(a, p, flat_verts)
            if s < best_side:
                best_side = s
                best_angle = a
                best_phi = p

    if time.time() > deadline - 10:
        return best_angle, best_phi, best_side

    # Phase 3: Golden section search on cluster angle (fixing phi)
    def obj_angle(a_arr):
        return compute_side_for_angles(a_arr[0], best_phi, flat_verts)

    a_center = best_angle
    a_range = 1.5
    a_lower = max(0, a_center - a_range)
    a_upper = min(60, a_center + a_range)

    best_a, best_s_a = golden_section_search(
        lambda x: obj_angle(np.array([x])), a_lower, a_upper, tol=1e-8
    )
    if best_s_a < best_side:
        best_angle = best_a
        best_side = best_s_a

    if time.time() > deadline - 5:
        return best_angle, best_phi, best_side

    # Phase 4: Golden section search on phi (fixing cluster angle)
    def obj_phi(p_arr):
        return compute_side_for_angles(best_angle, p_arr[0], flat_verts)

    p_center = best_phi
    p_range = 1.5
    p_lower = max(0, p_center - p_range)
    p_upper = min(60, p_center + p_range)

    best_p, best_s_p = golden_section_search(
        lambda x: obj_phi(np.array([x])), p_lower, p_upper, tol=1e-8
    )
    if best_s_p < best_side:
        best_phi = best_p
        best_side = best_s_p

    if time.time() > deadline - 2:
        return best_angle, best_phi, best_side

    # Phase 5: Nelder-Mead polish on both angles simultaneously
    from scipy.optimize import minimize

    def polish_obj(theta):
        return compute_side_for_angles(theta[0], theta[1], flat_verts)

    try:
        res = minimize(
            polish_obj,
            np.array([best_angle, best_phi]),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        if res.x[0] > 0 and res.x[0] < 60 and res.x[1] > 0 and res.x[1] < 60:
            side_res = compute_side_for_angles(res.x[0], res.x[1], flat_verts)
            if side_res < best_side:
                best_angle = float(res.x[0])
                best_phi = float(res.x[1])
                best_side = side_res
    except Exception:
        pass

    return best_angle, best_phi, best_side


def construct():
    start = time.time()
    deadline = start + 145.0  # Safety margin (evaluator kills at 150s)

    # Build diverse initializations
    inits = [init_flower(), init_lattice(), init_staggered(), init_ring(), init_random_cluster()]

    best = None
    best_side = float('inf')

    # Check all initializations first
    for C, A in inits:
        Si = outer_side(C, A)
        if valid(C, A, Si) and Si < best_side:
            best_side = Si
            best = (C.copy(), A.copy(), Si)

    # Run multiple SA cohorts
    sa_time_budget = 60.0
    num_cohorts = 3
    time_per_cohort = sa_time_budget / num_cohorts

    for i in range(num_cohorts):
        if time.time() > start + sa_time_budget:
            break

        # Pick an init, slightly perturb it for diversity
        base_C, base_A = inits[i % len(inits)]
        rng = np.random.RandomState(100 + i)
        C0 = base_C + rng.normal(0, 0.1, (N, 2))
        A0 = base_A + rng.normal(0, 5.0, N)

        cohort_deadline = time.time() + time_per_cohort
        C_best, A_best, S_best = simulated_annealing(C0, A0, cohort_deadline, seed=1000 + i)

        if S_best < best_side:
            best_side = S_best
            best = (C_best, A_best, S_best)

    # Polish top candidate with NM
    if best is not None:
        Cn, An, Sn = best
        if time.time() < deadline - 30:
            Cn2, An2 = optimize_nm(Cn, An, maxiter=5000)
            Sn2 = outer_side(Cn2, An2)
            if valid(Cn2, An2, Sn2) and Sn2 < best_side:
                best_side = Sn2
                best = (Cn2, An2, Sn2)

    # If we have time, try additional local searches from different starting points
    if time.time() < deadline - 40:
        for C, A in inits:
            if time.time() > deadline - 30:
                break
            Cn, An = optimize_nm(C, A, maxiter=3000)
            Sn = outer_side(Cn, An)
            if valid(Cn, An, Sn) and Sn < best_side:
                best_side = Sn
                best = (Cn, An, Sn)

    # Rigid cluster optimization (the key structural change)
    # This exploits the fixed internal lattice geometry and optimizes only
    # the global rotation and outer hexagon angle
    if time.time() < deadline - 20:
        cluster_centers, flat_verts = build_rigid_cluster(scale_factor=1.002)
        best_angle, best_phi, best_side_rigid = optimize_rigid_cluster(flat_verts, deadline - 5)

        if best_side_rigid < best_side:
            # Apply the optimal rotation to the cluster centers
            angle_rad = np.deg2rad(best_angle)
            c, s = np.cos(angle_rad), np.sin(angle_rad)
            R = np.array([[c, -s], [s, c]])
            C_rotated = cluster_centers @ R.T

            # The inner hexagons all have the same rotation as the cluster
            A_rotated = np.full(N, best_angle)

            # Verify validity
            if valid(C_rotated, A_rotated, best_side_rigid):
                best_side = best_side_rigid
                best = (C_rotated, A_rotated, best_side_rigid)

    # If nothing valid, fallback
    if best is None:
        C, A = init_flower()
        Cn, An = optimize_nm(C, A, maxiter=30000)
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
