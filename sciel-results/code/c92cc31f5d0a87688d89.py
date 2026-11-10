import numpy as np
import time
import scipy.optimize as so

def moving_sofa_2d() -> np.ndarray:
    """
    Optimizes the path of the moving sofa to maximize the area of the intersection.
    """
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 30.0

    def time_guard():
        return time.time() - start_time > (time_limit - safety_margin)

    # Sofa points definition
    i_vals = np.arange(-3.0, 1.5, 0.01)
    j_vals = np.arange(-1.5, 1.5, 0.01)
    P_x, P_y = np.meshgrid(i_vals, j_vals, indexing='xy')
    P_x = P_x.flatten()
    P_y = P_y.flatten()
    N_pts = len(P_x)

    # Pre-compute dense path structure for a given set of angles
    # Returns:
    # - dense_a: array of angles for all dense points
    # - coeffs_x: matrix where each row corresponds to a dense point and columns are weights for anchors
    #   Anchors: 0=Start, 1..N=Waypoints, N+1=End
    # - coeffs_y: same as coeffs_x
    # Actually, since x and y are linearly interpolated, the coefficients are the same.
    # Let's return a matrix W of shape (n_dense, n_anchors) such that:
    # dense_pos = W @ anchor_pos
    # anchor_pos = [(0,0), (x1,y1), ..., (xN,yN), (-2,0)]

    def build_dense_structure(angles):
        N = len(angles)
        n_anchors = N + 2
        dense_anchors_idx = []
        dense_t = []

        # Segment 0: Start (0) to Waypoint 0 (1)
        a1, a2 = 0.0, angles[0]
        m = max(1, int(np.ceil(abs(a2 - a1))))
        dense_anchors_idx.append(0) # Start
        dense_t.append(0.0)
        if m > 1:
            for k in range(1, m):
                t = k / m
                dense_anchors_idx.append(0)
                dense_t.append(t)
        dense_anchors_idx.append(1) # Waypoint 0
        dense_t.append(1.0)

        # Segments between waypoints
        for i in range(N - 1):
            a1, a2 = angles[i], angles[i+1]
            m = max(1, int(np.ceil(abs(a2 - a1))))
            if m > 1:
                for k in range(1, m):
                    t = k / m
                    dense_anchors_idx.append(i + 1)
                    dense_t.append(t)
            dense_anchors_idx.append(i + 2) # Waypoint i+1
            dense_t.append(1.0)

        # Segment N: Waypoint N-1 (N) to End (N+1)
        a1, a2 = angles[N-1], 90.0
        m = max(1, int(np.ceil(abs(a2 - a1))))
        if m > 1:
            for k in range(1, m):
                t = k / m
                dense_anchors_idx.append(N)
                dense_t.append(t)
        dense_anchors_idx.append(N + 1) # End
        dense_t.append(1.0)

        n_dense = len(dense_anchors_idx)
        # Build weight matrix
        # W[i, j] is the weight of anchor j in dense point i
        W = np.zeros((n_dense, n_anchors))
        for i in range(n_dense):
            a_idx = dense_anchors_idx[i]
            t = dense_t[i]
            W[i, a_idx] = 1.0 - t
            W[i, a_idx + 1] = t

        # Compute angles for dense points
        # Anchor angles: 0, angles[0], ..., angles[N-1], 90
        anchor_angles = np.concatenate([[0.0], angles, [90.0]])
        dense_a = W @ anchor_angles

        return W, dense_a

    # Pre-compute for a given set of angles
    # We will use this structure in the objective function

    def make_evaluator(angles):
        W, dense_a = build_dense_structure(angles)
        n_dense, n_anchors = W.shape
        cos_a = np.cos(np.deg2rad(dense_a))
        sin_a = np.sin(np.deg2rad(dense_a))

        def evaluate(x_vals, y_vals):
            # Anchor positions
            # 0: (0,0)
            # 1..N: (x_vals, y_vals)
            # N+1: (-2, 0)
            anchor_x = np.zeros(n_anchors)
            anchor_y = np.zeros(n_anchors)
            if n_anchors > 1:
                anchor_x[1:-1] = x_vals
                anchor_y[1:-1] = y_vals
            if n_anchors > 0:
                anchor_x[-1] = -2.0
                anchor_y[-1] = 0.0

            dense_x = W @ anchor_x
            dense_y = W @ anchor_y

            in_sofa = np.ones(N_pts, dtype=bool)

            for k in range(n_dense):
                dx = P_x - dense_x[k]
                dy = P_y - dense_y[k]

                u = dx * cos_a[k] + dy * sin_a[k]
                v = -dx * sin_a[k] + dy * cos_a[k]

                cond1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
                cond2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
                mask = cond1 | cond2

                in_sofa &= mask

                if not np.any(in_sofa):
                    return 0.0

            return np.sum(in_sofa) * 0.0001

        return evaluate

    # Initial configuration
    # Let's start with 8 waypoints
    angles = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    n_angles = len(angles)

    # Initial guess
    # Simple linear interpolation in x, y?
    # Start (0,0), End (-2,0)
    # x: 0 -> -2
    # y: 0 -> 0
    init_x = np.linspace(0.0, -2.0, n_angles + 2)[1:-1]
    init_y = np.zeros(n_angles)

    p0 = np.concatenate([init_x, init_y])

    evaluator = make_evaluator(angles)

    def objective(params):
        x_vals = params[:n_angles]
        y_vals = params[n_angles:]
        return -evaluator(x_vals, y_vals)

    best_params = p0.copy()
    best_area = -objective(p0)

    # Optimization
    # Powell is good for this
    try:
        res = so.minimize(objective, p0, method='Powell',
                          options={'maxiter': 1000, 'maxfev': 2000, 'disp': False})
        if not time_guard():
            if res.fun < -best_area:
                best_area = -res.fun
                best_params = res.x
    except Exception:
        pass

    if not time_guard():
        # Try Nelder-Mead for local refinement
        try:
            res2 = so.minimize(objective, best_params, method='Nelder-Mead',
                               options={'maxiter': 1000, 'maxfev': 2000, 'disp': False})
            if res2.fun < -best_area:
                best_area = -res2.fun
                best_params = res2.x
        except Exception:
            pass

    # If time remains, try with different number of waypoints
    # e.g. 10 waypoints
    if not time_guard():
        angles2 = np.linspace(10.0, 80.0, 10)
        n_angles2 = len(angles2)
        init_x2 = np.linspace(0.0, -2.0, n_angles2 + 2)[1:-1]
        init_y2 = np.zeros(n_angles2)
        p0_2 = np.concatenate([init_x2, init_y2])

        evaluator2 = make_evaluator(angles2)

        def objective2(params):
            x_vals = params[:n_angles2]
            y_vals = params[n_angles2:]
            return -evaluator2(x_vals, y_vals)

        try:
            res3 = so.minimize(objective2, p0_2, method='Powell',
                               options={'maxiter': 500, 'maxfev': 1000, 'disp': False})
            if not time_guard():
                if res3.fun < -best_area:
                    best_area = -res3.fun
                    best_params = res3.x
                    angles = angles2
                    n_angles = n_angles2
        except Exception:
            pass

    # Construct final path
    x_vals = best_params[:n_angles]
    y_vals = best_params[n_angles:]

    final_path = []
    for i in range(n_angles):
        final_path.append([x_vals[i], y_vals[i], angles[i]])

    path_arr = np.array(final_path, dtype=np.float64)

    return path_arr
