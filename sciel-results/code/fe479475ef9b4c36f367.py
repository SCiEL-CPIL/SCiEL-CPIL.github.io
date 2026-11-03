import numpy as np
import time
import scipy.optimize
from scipy.spatial import ConvexHull

N = 11
SQRT3 = np.sqrt(3.0)

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    # Unit hexagon vertices (flat-top orientation)
    unit_verts = np.array([
        [1.0, 0.0],
        [0.5, SQRT3/2],
        [-0.5, SQRT3/2],
        [-1.0, 0.0],
        [-0.5, -SQRT3/2],
        [0.5, -SQRT3/2]
    ])

    # Face normals of outer hexagon (pointing outward from center)
    normals = np.array([
        [1.0, 0.0],
        [0.5, SQRT3/2],
        [-0.5, SQRT3/2],
        [-1.0, 0.0],
        [-0.5, -SQRT3/2],
        [0.5, -SQRT3/2]
    ])

    def get_all_verts(params):
        """Get vertices for all N hexagons. Returns (N, 6, 2) array."""
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            x, y, ang = params[i*3], params[i*3+1], params[i*3+2]
            a = np.deg2rad(ang)
            c, s = np.cos(a), np.sin(a)
            R = np.array([[c, -s], [s, c]])
            verts[i] = unit_verts @ R.T + np.array([x, y])
        return verts

    def compute_side(params):
        """Compute minimum outer hexagon side length to contain all hexagons."""
        verts = get_all_verts(params)
        # For each vertex, compute max projection onto each normal
        # Shape: (N, 6, 6) -> max over vertices for each normal
        # proj[i, v, n] = dot(verts[i,v], normals[n])
        # We need max over all i,v for each n
        projs = np.einsum('ivk,k->ivn', verts, normals)  # (N, 6, 6)
        max_apothem = projs.max()
        return 2.0 * max_apothem / SQRT3

    def compute_overlap_penalty(params, eps=1e-9):
        """Compute overlap penalty between pairs of hexagons."""
        penalty = 0.0
        min_center_dist = SQRT3 - eps  # ~1.732
        for i in range(N):
            xi, yi = params[i*3], params[i*3+1]
            for j in range(i+1, N):
                xj, yj = params[j*3], params[j*3+1]
                dx, dy = xi - xj, yi - yj
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < min_center_dist:
                    penalty += (min_center_dist - dist)**2 * 50.0
        return penalty

    def objective(params):
        side = compute_side(params)
        penalty = compute_overlap_penalty(params)
        return side + penalty

    def sat_feasible(S, max_iters=200):
        """Check if N hexagons can fit in outer hexagon of side S using MILP."""
        # For a regular hexagon with side S, the apothem is S*sqrt(3)/2
        # A point (x,y) is inside if for all normals n: n·(x,y) <= apothem
        apothem = S * SQRT3 / 2

        # Variables: x_i, y_i for each hexagon center (2N variables)
        # Constraints: for each hexagon i, each vertex v, each normal n:
        #   n · (center_i + R(theta_i) * v) <= apothem
        # This is nonlinear due to theta_i.
        # Simplification: assume all hexagons have same orientation (theta=0)
        # This is a common approach and often optimal for hexagonal packings.

        # For theta=0, vertex positions relative to center are fixed: unit_verts
        # Constraint: for each i, v, n: n · (center_i + unit_verts[v]) <= apothem
        # i.e., n · center_i <= apothem - n · unit_verts[v]

        # Variables: x_0..x_{N-1}, y_0..y_{N-1}
        n_vars = 2 * N

        # Objective: minimize 0 (feasibility)
        c = np.zeros(n_vars)

        # Inequality constraints: A_ub @ x <= b_ub
        # For each hexagon i, each vertex v, each normal n:
        #   n[0]*x_i + n[1]*y_i <= apothem - n·unit_verts[v]
        A_rows = []
        b_rows = []
        for i in range(N):
            for v in range(6):
                for n_idx in range(6):
                    row = np.zeros(n_vars)
                    row[2*i] = normals[n_idx, 0]
                    row[2*i+1] = normals[n_idx, 1]
                    rhs = apothem - np.dot(normals[n_idx], unit_verts[v])
                    A_rows.append(row)
                    b_rows.append(rhs)

        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)

        # Bounds: centers should be within reasonable range
        bounds = [(-S, S)] * (2 * N)

        try:
            res = scipy.optimize.milp(
                c,
                constraints=[scipy.optimize.LinearConstraint(A_ub, -np.inf, b_ub)],
                integrality=np.zeros(n_vars),
                bounds=bounds,
                options={'time_limit': 30, 'mip_gap': 1e-6}
            )
            if res.status == 0:
                return True, res.x
            else:
                return False, None
        except Exception:
            return False, None

    def local_search(params, max_iters=500, step=0.1):
        """Local search with adaptive step size."""
        current = params.copy()
        current_obj = objective(current)
        best_local = current.copy()
        best_obj = current_obj
        step_size = step
        no_improve = 0

        for it in range(max_iters):
            if time.time() > deadline - 3:
                break

            perturbation = np.random.normal(0, step_size, len(current))
            perturbation[2::3] *= 0.3  # Reduce angle perturbations
            candidate = current + perturbation

            cand_obj = objective(candidate)

            if cand_obj < current_obj:
                current = candidate
                current_obj = cand_obj
                no_improve = 0
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_local = current.copy()
            else:
                no_improve += 1
                if no_improve > 40:
                    step_size *= 0.9
                    if step_size < 0.001:
                        break

        return best_local, best_obj

    # Generate initial configurations
    d = SQRT3

    def make_config_hex_cluster():
        """7-hex cluster + 4 more arranged around"""
        centers = [
            (0, 0),
            (d, 0), (d/2, 1.5), (-d/2, 1.5),
            (-d, 0), (-d/2, -1.5), (d/2, -1.5),
            (0, 3.0), (d*1.5, 1.5), (d*1.5, -1.5), (0, -3.0)
        ]
        params = []
        for x, y in centers:
            params.extend([x, y, 0.0])
        return np.array(params)

    def make_config_two_rows():
        """Two staggered rows"""
        params = []
        for i in range(6):
            params.extend([i * d, 0.0, 0.0])
        for i in range(5):
            params.extend([(i + 0.5) * d, 1.5, 0.0])
        return np.array(params)

    def make_config_three_rows():
        """Three rows: 4-3-4"""
        params = []
        for i in range(4):
            params.extend([(i - 1.5) * d, 0.0, 0.0])
        for i in range(3):
            params.extend([(i - 1.0) * d + d/2, 1.5, 0.0])
        for i in range(4):
            params.extend([(i - 1.5) * d, 3.0, 0.0])
        return np.array(params)

    def make_config_linear():
        """Linear arrangement"""
        params = []
        for i in range(N):
            params.extend([(i - N//2) * d, 0.0, 0.0])
        return np.array(params)

    initial_configs = [
        make_config_hex_cluster(),
        make_config_two_rows(),
        make_config_three_rows(),
        make_config_linear()
    ]

    np.random.seed(42)

    best_side = float('inf')
    best_params = None
    best_penalty = float('inf')

    # Phase 1: Local search from multiple initial configurations
    for config in initial_configs:
        if time.time() > deadline - 10:
            break
        try:
            result_params, result_obj = local_search(config, max_iters=400, step=0.15)
            side = compute_side(result_params)
            penalty = compute_overlap_penalty(result_params)

            if penalty < 1e-6 and side < best_side:
                best_side = side
                best_params = result_params.copy()
                best_penalty = penalty
        except Exception:
            pass

    # Phase 2: Binary search with SAT feasibility
    if time.time() < deadline - 15:
        # Get a good upper bound from local search or estimate
        if best_side < float('inf'):
            high = best_side + 0.1
        else:
            high = 5.0

        low = 3.0  # Theoretical minimum for 11 hexagons

        # First, find a feasible upper bound using SAT
        for test_S in [4.0, 3.8, 3.6, 3.4, 3.2]:
            if time.time() > deadline - 15:
                break
            feasible, sat_solution = sat_feasible(test_S)
            if feasible:
                high = test_S
                if sat_solution is not None:
                    # Convert MILP solution to params
                    params = np.zeros(3 * N)
                    for i in range(N):
                        params[i*3] = sat_solution[2*i]
                        params[i*3+1] = sat_solution[2*i+1]
                        params[i*3+2] = 0.0
                    side = compute_side(params)
                    penalty = compute_overlap_penalty(params)
                    if penalty < 1e-6 and side < best_side:
                        best_side = side
                        best_params = params.copy()
                        best_penalty = penalty
                break

        # Binary search for minimal feasible S
        for _ in range(8):
            if time.time() > deadline - 10:
                break
            mid = (low + high) / 2
            feasible, sat_solution = sat_feasible(mid)
            if feasible:
                high = mid
                if sat_solution is not None:
                    params = np.zeros(3 * N)
                    for i in range(N):
                        params[i*3] = sat_solution[2*i]
                        params[i*3+1] = sat_solution[2*i+1]
                        params[i*3+2] = 0.0
                    side = compute_side(params)
                    penalty = compute_overlap_penalty(params)
                    if penalty < 1e-6 and side < best_side:
                        best_side = side
                        best_params = params.copy()
                        best_penalty = penalty
            else:
                low = mid

    # Phase 3: Refine best solution with local search
    if best_params is not None and time.time() < deadline - 5:
        refined_params, refined_obj = local_search(best_params, max_iters=300, step=0.05)
        side = compute_side(refined_params)
        penalty = compute_overlap_penalty(refined_params)
        if penalty < 1e-6 and side < best_side:
            best_side = side
            best_params = refined_params.copy()
            best_penalty = penalty

    # Fallback
    if best_side == float('inf') or best_params is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Construct output
    inner = []
    for i in range(N):
        x, y, ang = best_params[i*3], best_params[i*3+1], best_params[i*3+2]
        inner.append([float(x), float(y), float(ang)])

    return inner, (0.0, 0.0), float(best_side), 0.0
