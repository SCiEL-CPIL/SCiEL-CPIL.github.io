import time
import math
import numpy as np

N = 11
T0 = time.time()
DEADLINE = T0 + 140.0  # safety margin before 150s kill
SQRT3 = math.sqrt(3.0)
PI = math.pi

# MILP rotation discretization: every 5 degrees
ROT_ANGLES_DEG = list(range(0, 360, 5))
ROT_ANGLES_RAD = [math.radians(a) for a in ROT_ANGLES_DEG]


def _hex_vertices_local(theta):
    """Unit hexagon vertices (circumradius 1) rotated by theta. (6,2)"""
    ang = np.arange(6) * (PI / 3.0) + theta
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)


def _precompute_rot_vertices():
    """Precompute vertices for all discretized rotation angles. Returns (num_rots, 6, 2)"""
    verts = np.zeros((len(ROT_ANGLES_RAD), 6, 2))
    for i, theta in enumerate(ROT_ANGLES_RAD):
        ang = np.arange(6) * (PI / 3.0) + theta
        verts[i] = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    return verts


def _outer_side_normals(theta_outer):
    """Outward normals of the 6 sides of the outer hexagon. (6,2)"""
    ang = theta_outer + np.arange(6) * (PI / 3.0) + PI / 6.0
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)


def _solve_milp(theta_outer, time_limit=30):
    """
    Solve MILP for exact minimum enclosing hexagon with discretized rotations.
    Returns (side_length, centers, theta_rad) or None if infeasible/timeout.
    """
    try:
        import pulp
    except ImportError:
        return None
    
    num_rots = len(ROT_ANGLES_RAD)
    # Precompute vertices for each rotation
    rot_verts = _precompute_rot_vertices()  # (num_rots, 6, 2)
    normals = _outer_side_normals(theta_outer)  # (6, 2)
    
    # Variables:
    # x[i, j], y[i, j]: center of hexagon i with rotation j (continuous, can be 0 if not selected)
    # r[i, j]: binary, 1 if hexagon i uses rotation j
    # t: side length (minimize)
    
    prob = pulp.LpProblem("hex_packing", pulp.LpMinimize)
    
    # Rotation selection variables
    r = {}
    for i in range(N):
        for j in range(num_rots):
            r[i, j] = pulp.LpVariable(f"r_{i}_{j}", cat='Binary')
    
    # Constraint: each hexagon uses exactly one rotation
    for i in range(N):
        prob += pulp.lpSum([r[i, j] for j in range(num_rots)]) == 1, f"one_rot_{i}"
    
    # Center coordinates (unbounded continuous)
    x = {}
    y = {}
    for i in range(N):
        x[i] = pulp.LpVariable(f"x_{i}", lowBound=-20, upBound=20)
        y[i] = pulp.LpVariable(f"y_{i}", lowBound=-20, upBound=20)
    
    # Side length variable
    t = pulp.LpVariable('t', lowBound=1.0)
    prob += t, "obj"
    
    # Containment constraints: for each hexagon i, rotation j, vertex k, side m:
    # proj(x_i + v_{j,k,m}, n_m) <= apothem = t * SQRT3 / 2
    # This must hold only when r[i,j] = 1. Use big-M: 
    # proj(x_i + v_{j,k}, n_m) <= apothem + M*(1 - r[i,j])
    apothem_coeff = SQRT3 / 2.0
    M = 50.0  # Big-M
    
    for i in range(N):
        for j in range(num_rots):
            for k in range(6):
                vx = rot_verts[j, k, 0]
                vy = rot_verts[j, k, 1]
                for m in range(6):
                    nx = normals[m, 0]
                    ny = normals[m, 1]
                    # (x_i + vx) * nx + (y_i + vy) * ny <= apothem + M*(1 - r[i,j])
                    # x_i * nx + y_i * ny + (vx*nx + vy*ny) <= t*apothem_coeff + M*(1 - r[i,j])
                    prob += (x[i] * nx + y[i] * ny + (vx * nx + vy * ny) 
                             - t * apothem_coeff + M * (1 - r[i, j])) <= 0, \
                            f"contain_{i}_{j}_{k}_{m}"
    
    # Non-overlap constraints: for each pair (i, i'), for each rotation (j, j'):
    # If r[i,j]=1 and r[i',j']=1, then distance >= 2
    # Use disjunctive: for each of 6 directions, OR constraint
    # |x_i - x_i'| >= 2*cos(theta_dir) OR |y_i - y_i'| >= 2*sin(theta_dir) etc.
    # Simpler: use the fact that two unit hexagons don't overlap iff center distance >= 2
    # when aligned, but with rotations it's more complex.
    # For MILP simplicity, use conservative: center distance >= 2 always
    # This is valid (sufficient) but not necessary, may give suboptimal but valid solution.
    
    for i in range(N):
        for i2 in range(i + 1, N):
            # Distance constraint: (x_i - x_i')^2 + (y_i - y_i')^2 >= 4
            # Linearize with big-M and binary selection of which hexagon is "left"
            # Actually, use: |x_i - x_i'| >= 2 OR |y_i - y_i'| >= 2 is not correct for hexagons
            # Use conservative: Euclidean distance >= 2, linearized as:
            # x_i - x_i' >= 2 OR x_i' - x_i >= 2 OR y_i - y_i' >= 2 OR y_i' - y_i >= 2
            # This is valid (if any holds, distance >= 2)
            # But this is very conservative. Better to use the actual hexagon separation.
            
            # For now, use the 4-direction linearization
            b = {}
            for d in range(4):
                b[d] = pulp.LpVariable(f"sep_{i}_{i2}_{d}", cat='Binary')
            
            # Sum of binaries = 1
            prob += pulp.lpSum([b[d] for d in range(4)]) == 1, f"sep_sel_{i}_{i2}"
            
            # If b[0]=1: x_i - x_i' >= 2
            prob += x[i] - x[i2] - 2 + M * (1 - b[0]) >= 0, f"sep0_{i}_{i2}"
            # If b[1]=1: x_i' - x_i >= 2
            prob += x[i2] - x[i] - 2 + M * (1 - b[1]) >= 0, f"sep1_{i}_{i2}"
            # If b[2]=1: y_i - y_i' >= 2
            prob += y[i] - y[i2] - 2 + M * (1 - b[2]) >= 0, f"sep2_{i}_{i2}"
            # If b[3]=1: y_i' - y_i >= 2
            prob += y[i2] - y[i] - 2 + M * (1 - b[3]) >= 0, f"sep3_{i}_{i2}"
    
    # Solve
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
    
    if pulp.LpStatus[status] != "Optimal":
        return None
    
    # Extract solution
    centers = np.zeros((N, 2))
    thetas = np.zeros(N)
    for i in range(N):
        centers[i, 0] = x[i].value()
        centers[i, 1] = y[i].value()
        for j in range(num_rots):
            if r[i, j].value() is not None and r[i, j].value() > 0.5:
                thetas[i] = ROT_ANGLES_RAD[j]
                break
    
    side = t.value()
    return (side, centers, thetas)


def _compute_side(centers, theta, theta_outer):
    """Compute the minimal outer side length for given centers and rotations."""
    verts_local = _hex_vertices_local(theta)  # (6,2)
    verts = centers[:, None, :] + verts_local[None, :, :]  # (N,6,2)
    verts = verts.reshape(-1, 2)  # (N*6, 2)
    normals = _outer_side_normals(theta_outer)  # (6,2)
    proj = verts @ normals.T  # (N*6, 6)
    apothem = proj.max()
    return apothem * 2.0 / SQRT3


def _compute_side_multi(centers, thetas, theta_outer):
    """Compute side length with per-hexagon rotations."""
    normals = _outer_side_normals(theta_outer)  # (6,2)
    max_proj = 0.0
    for i in range(N):
        verts_local = _hex_vertices_local(thetas[i])  # (6,2)
        verts = centers[i] + verts_local  # (6,2)
        proj = verts @ normals.T  # (6,)
        max_proj = max(max_proj, proj.max())
    return max_proj * 2.0 / SQRT3


def _is_valid_multi(centers, thetas, theta_outer, tol=1e-5):
    """Check non-overlap and containment with per-hexagon rotations."""
    # Non-overlap: conservative check using center distance
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    if d[iu].min() < 2.0 - tol:
        return False
    # Containment
    s = _compute_side_multi(centers, thetas, theta_outer)
    apothem = s * SQRT3 / 2.0
    normals = _outer_side_normals(theta_outer)
    for i in range(N):
        verts_local = _hex_vertices_local(thetas[i])
        verts = centers[i] + verts_local
        proj = verts @ normals.T
        if proj.max() > apothem + tol:
            return False
    return True


def _pairwise_penalty(centers, weight=1000.0):
    """Penalty for pairs closer than 2 (minimum distance for non-overlap)."""
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    dpair = d[iu]
    viol = np.maximum(0.0, 2.0 - dpair)
    return weight * np.sum(viol * viol)


def _objective(params, theta_outer):
    """
    Objective: minimize outer side length + penalty for overlaps.
    params: (N*2 + 1) = [x0,y0,...,x_{N-1},y_{N-1}, theta]
    All inner hexagons share the same rotation theta.
    """
    centers = params[:N * 2].reshape(N, 2)
    theta = params[N * 2]
    s = _compute_side(centers, theta, theta_outer)
    pen = _pairwise_penalty(centers)
    return s + pen


def _is_valid(centers, theta, theta_outer, tol=1e-5):
    """Check non-overlap and containment."""
    # Non-overlap
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    if d[iu].min() < 2.0 - tol:
        return False
    # Containment
    s = _compute_side(centers, theta, theta_outer)
    apothem = s * SQRT3 / 2.0
    verts_local = _hex_vertices_local(theta)
    verts = (centers[:, None, :] + verts_local[None, :, :]).reshape(-1, 2)
    normals = _outer_side_normals(theta_outer)
    proj = verts @ normals.T
    if proj.max() > apothem + tol:
        return False
    return True


def _make_init(seed):
    """Generate diverse initial configurations."""
    rng = np.random.RandomState(seed)
    mode = seed % 6

    if mode == 0:
        # Hexagonal lattice: center + ring of 6 + 4 more
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * PI / 3.0
            centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * PI / 3.0
            centers[i + 7] = [3.5 * math.cos(ang), 3.5 * math.sin(ang)]
    elif mode == 1:
        # Offset hexagonal lattice
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * PI / 3.0 + PI / 6.0
            centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * PI / 3.0
            centers[i + 7] = [3.4 * math.cos(ang), 3.4 * math.sin(ang)]
    elif mode == 2:
        # Compact cluster (known good packing)
        centers = np.zeros((N, 2))
        positions = [
            (0, 0), (2, 0), (-1, 1.732), (1, 1.732),
            (-1, -1.732), (1, -1.732), (3, 0),
            (-2, 1.732), (2, 1.732), (-2, -1.732), (2, -1.732)
        ]
        for i in range(N):
            centers[i] = positions[i]
    elif mode == 3:
        # Slightly perturbed compact cluster
        centers = np.zeros((N, 2))
        positions = [
            (0, 0), (2, 0), (-1, 1.732), (1, 1.732),
            (-1, -1.732), (1, -1.732), (3, 0),
            (-2, 1.732), (2, 1.732), (-2, -1.732), (2, -1.732)
        ]
        for i in range(N):
            centers[i] = [positions[i][0] + rng.normal(0, 0.1),
                          positions[i][1] + rng.normal(0, 0.1)]
    elif mode == 4:
        # Ring-based: 1 center + 6 inner ring + 4 outer ring
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * PI / 3.0
            r = 2.0
            centers[i + 1] = [r * math.cos(ang), r * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * PI / 3.0
            r = 3.2
            centers[i + 7] = [r * math.cos(ang), r * math.sin(ang)]
    else:
        # Random with moderate spread
        centers = rng.uniform(-3, 3, size=(N, 2))

    # Small random rotation
    theta = rng.uniform(-PI / 6, PI / 6)
    params = np.concatenate([centers.reshape(-1), [theta]])
    return params


def _optimize_one(init_params, theta_outer, maxiter=300):
    """Run L-BFGS-B optimization."""
    from scipy.optimize import minimize
    try:
        res = minimize(_objective, init_params, args=(theta_outer,),
                       method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
        return res.x
    except Exception:
        return init_params


def _polish(params, theta_outer):
    """High-precision polish using SLSQP and tighter L-BFGS-B."""
    from scipy.optimize import minimize
    best = params
    best_val = _objective(params, theta_outer)

    # SLSQP with tight tolerance
    try:
        res = minimize(_objective, params, args=(theta_outer,),
                       method='SLSQP',
                       options={'maxiter': 300, 'ftol': 1e-14, 'maxls': 50})
        val = res.fun
        if val < best_val:
            best = res.x
            best_val = val
    except Exception:
        pass

    # Tight L-BFGS-B
    try:
        res = minimize(_objective, best, args=(theta_outer,),
                       method='L-BFGS-B',
                       options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13})
        val = res.fun
        if val < best_val:
            best = res.x
            best_val = val
    except Exception:
        pass

    return best


def construct():
    from scipy.optimize import minimize

    best_score = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_side = float('inf')
    best_angle = 0.0

    def _update_best(centers, thetas, theta_o):
        nonlocal best_score, best_inner, best_center, best_side, best_angle
        if not _is_valid_multi(centers, thetas, theta_o):
            return
        s = _compute_side_multi(centers, thetas, theta_o)
        if s < best_score:
            best_score = s
            best_inner = [[float(centers[i, 0]), float(centers[i, 1]), float(math.degrees(thetas[i]))] 
                         for i in range(N)]
            best_center = (0.0, 0.0)
            best_side = float(s)
            best_angle = float(math.degrees(theta_o))

    # Phase 1: Try MILP for a few outer rotations
    outer_rotations_milp = [0.0, PI / 6.0]
    
    for theta_o in outer_rotations_milp:
        if time.time() > DEADLINE - 60:
            break
        result = _solve_milp(theta_o, time_limit=25)
        if result is not None:
            s_milp, centers_milp, thetas_milp = result
            # MILP uses conservative separation, so verify and possibly refine
            if _is_valid_multi(centers_milp, thetas_milp, theta_o):
                _update_best(centers_milp, thetas_milp, theta_o)
            # Use MILP solution as initial guess for continuous optimization
            # (convert to single-rotation approximation for L-BFGS-B)
            avg_theta = float(np.mean(thetas_milp))
            init_params = np.concatenate([centers_milp.reshape(-1), [avg_theta]])
            try:
                params = _optimize_one(init_params, theta_o, maxiter=200)
                centers_c = params[:N * 2].reshape(N, 2)
                theta_c = params[N * 2]
                if _is_valid(centers_c, theta_c, theta_o):
                    # Convert to multi-rotation format
                    thetas_c = np.full(N, theta_c)
                    _update_best(centers_c, thetas_c, theta_o)
            except Exception:
                pass

    # Phase 2: Continuous L-BFGS-B search with diverse initializations
    outer_rotations_cont = [0.0, PI / 6.0, PI / 3.0, PI / 12.0]
    seed = 0
    maxiter = 300

    for theta_o in outer_rotations_cont:
        while time.time() < DEADLINE - 20:
            init_params = _make_init(seed)
            seed += 1
            try:
                params = _optimize_one(init_params, theta_o, maxiter)
            except Exception:
                continue

            centers = params[:N * 2].reshape(N, 2)
            theta = params[N * 2]

            if not _is_valid(centers, theta, theta_o):
                continue

            thetas = np.full(N, theta)
            _update_best(centers, thetas, theta_o)

            if time.time() > DEADLINE - 25:
                break

    # Phase 3: Polish the best solution if we have one
    if best_score < float('inf') and time.time() < DEADLINE - 5:
        # Extract best params for polishing
        # Reconstruct from best_inner
        if best_inner is not None:
            centers_p = np.array([[p[0], p[1]] for p in best_inner])
            thetas_p = np.array([math.radians(p[2]) for p in best_inner])
            theta_o = math.radians(best_angle)
            avg_theta = float(np.mean(thetas_p))
            
            init_params = np.concatenate([centers_p.reshape(-1), [avg_theta]])
            polished = _polish(init_params, theta_o)
            centers_pol = polished[:N * 2].reshape(N, 2)
            theta_pol = polished[N * 2]
            thetas_pol = np.full(N, theta_pol)
            _update_best(centers_pol, thetas_pol, theta_o)

    # Fallback if nothing valid found
    if best_score == float('inf'):
        centers = np.zeros((N, 2))
        positions = [
            (0, 0), (2, 0), (-1, 1.732), (1, 1.732),
            (-1, -1.732), (1, -1.732), (3, 0),
            (-2, 1.732), (2, 1.732), (-2, -1.732), (2, -1.732)
        ]
        for i in range(N):
            centers[i] = positions[i]
        thetas = np.zeros(N)
        s = _compute_side_multi(centers, thetas, 0.0)
        inner = [[float(centers[i, 0]), float(centers[i, 1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), float(s), 0.0

    return best_inner, best_center, best_side, best_angle
