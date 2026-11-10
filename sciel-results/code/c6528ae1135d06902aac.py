# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    """
    Search-based approach using coarser subgrid for fast optimization,
    then verification on full grid. Uses Nelder-Mead and coordinate descent.
    """
    start_time = time.time()
    deadline = start_time + 620  # 10s safety margin before 630s limit

    # Build the full evaluation grid
    xs_full = np.arange(-3.0, 1.5, 0.01)
    ys_full = np.arange(-1.5, 1.5, 0.01)
    Xf, Yf = np.meshgrid(xs_full, ys_full, indexing='ij')
    gx_full, gy_full = Xf.ravel(), Yf.ravel()
    n_points_full = len(gx_full)

    # Build a coarser subgrid for fast evaluation during optimization
    step = 3
    xs_sub = np.arange(-3.0, 1.5, 0.01 * step)
    ys_sub = np.arange(-1.5, 1.5, 0.01 * step)
    Xs, Ys = np.meshgrid(xs_sub, ys_sub, indexing='ij')
    gx_sub, gy_sub = Xs.ravel(), Ys.ravel()
    n_points_sub = len(gx_sub)

    def eval_single_placement(gx, gy, x, y, a_deg):
        """Returns boolean mask of grid points inside this placement."""
        a_rad = np.deg2rad(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        dx = gx - x
        dy = gy - y
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        arm1 = (u >= -0.001) & (u <= 1.001) & (v >= -3.001) & (v <= 1.001)
        arm2 = (u >= -3.001) & (u <= 1.001) & (v >= -0.001) & (v <= 1.001)
        return arm1 | arm2

    def densify_trajectory(trajectory):
        """
        Densify the trajectory as the evaluator does.
        trajectory: array of shape (n, 3) with (x, y, angle_deg)
        Returns list of dense placements.
        """
        full = np.vstack([[0.0, 0.0, 0.0], trajectory, [-2.0, 0.0, 90.0]])
        dense = [full[0]]
        for i in range(len(full) - 1):
            p1 = full[i]
            p2 = full[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            for k in range(1, m + 1):
                t = k / m
                interp = p1 + t * (p2 - p1)
                dense.append(interp)
        return dense

    def evaluate_trajectory_sub(trajectory):
        """Evaluate on subgrid - fast."""
        dense = densify_trajectory(trajectory)
        mask = np.ones(n_points_sub, dtype=bool)
        for x, y, a in dense:
            in_place = eval_single_placement(gx_sub, gy_sub, x, y, a)
            mask &= in_place
            if not mask.any():
                return 0.0
        return mask.sum() / n_points_sub

    def evaluate_trajectory_full(trajectory):
        """Evaluate on full grid - accurate but slow."""
        dense = densify_trajectory(trajectory)
        mask = np.ones(n_points_full, dtype=bool)
        for x, y, a in dense:
            in_place = eval_single_placement(gx_full, gy_full, x, y, a)
            mask &= in_place
            if not mask.any():
                return 0.0
        return mask.sum() / n_points_full

    def params_to_traj(params, n_ctrl):
        """Convert parameter vector to trajectory with fixed endpoints."""
        # params has 3*(n_ctrl-2) free parameters (interior points)
        t = np.linspace(0, 1, n_ctrl)
        x = np.zeros(n_ctrl)
        y = np.zeros(n_ctrl)
        angle = np.zeros(n_ctrl)
        x[0], y[0], angle[0] = 0.0, 0.0, 0.0
        x[-1], y[-1], angle[-1] = -2.0, 0.0, 90.0
        for i in range(1, n_ctrl - 1):
            x[i] = params[3*(i-1)]
            y[i] = params[3*(i-1)+1]
            angle[i] = params[3*(i-1)+2]
        return np.column_stack([x, y, angle])

    def traj_to_params(traj, n_ctrl):
        """Convert trajectory to parameter vector (interior points only)."""
        params = []
        for i in range(1, n_ctrl - 1):
            params.extend([traj[i, 0], traj[i, 1], traj[i, 2]])
        return np.array(params)

    # Initial best
    best_score_full = 0.0
    best_traj = None

    # --- SEARCH APPROACH ---
    n_ctrl = 8  # Number of control points
    n_restarts = 20

    for restart in range(n_restarts):
        if time.time() > deadline - 30:  # Leave time for final verification
            break
        
        # Generate initial trajectory
        t = np.linspace(0, 1, n_ctrl)
        
        # Different parameterizations for diversity
        if restart % 4 == 0:
            # Smooth sine curve
            x = -0.25 * t
            y = 0.3 * np.sin(np.pi * t)
            angle = 12.5 * t
        elif restart % 4 == 1:
            # Higher arc
            x = -0.25 * t
            y = 0.5 * np.sin(np.pi * t)
            angle = 12.5 * t
        elif restart % 4 == 2:
            # Curved x
            x = -0.25 * t + 0.1 * np.sin(np.pi * t)
            y = 0.4 * np.sin(np.pi * t)
            angle = 12.5 * t
        else:
            # Random perturbation
            x = -0.25 * t + np.random.normal(0, 0.05, n_ctrl)
            y = 0.35 * np.sin(np.pi * t) + np.random.normal(0, 0.05, n_ctrl)
            angle = 12.5 * t + np.random.normal(0, 2, n_ctrl)
        
        # Ensure proper endpoints
        x[0], y[0], angle[0] = 0.0, 0.0, 0.0
        x[-1], y[-1], angle[-1] = -2.0, 0.0, 90.0
        
        traj = np.column_stack([x, y, angle])
        params = traj_to_params(traj, n_ctrl)
        
        # Nelder-Mead optimization on subgrid
        def neg_score_sub(p):
            tr = params_to_traj(p, n_ctrl)
            return -evaluate_trajectory_sub(tr)
        
        try:
            result = minimize(neg_score_sub, params, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 0.01, 'fatol': 1e-6})
            if result.fun < 0:  # Valid result
                best_params = result.x
                best_sub_score = -result.fun
            else:
                best_params = params
                best_sub_score = evaluate_trajectory_sub(traj)
        except Exception:
            best_params = params
            best_sub_score = evaluate_trajectory_sub(traj)
        
        # Coordinate descent refinement on subgrid
        improved = True
        max_cd_iters = 50
        cd_iter = 0
        while improved and cd_iter < max_cd_iters:
            improved = False
            cd_iter += 1
            if time.time() > deadline - 30:
                break
            for i in range(len(best_params)):
                for delta in [0.1, 0.05, 0.02, 0.01, -0.1, -0.05, -0.02, -0.01]:
                    trial_params = best_params.copy()
                    trial_params[i] += delta
                    # Clip angle values to reasonable range
                    idx = i % 3
                    if idx == 2:
                        trial_params[i] = np.clip(trial_params[i], -5, 95)
                    trial_traj = params_to_traj(trial_params, n_ctrl)
                    trial_score = evaluate_trajectory_sub(trial_traj)
                    if trial_score > best_sub_score + 1e-9:
                        best_params = trial_params
                        best_sub_score = trial_score
                        improved = True
                        break
        
        # Evaluate best on full grid
        best_traj_candidate = params_to_traj(best_params, n_ctrl)
        full_score = evaluate_trajectory_full(best_traj_candidate)
        
        if full_score > best_score_full:
            best_score_full = full_score
            best_traj = best_traj_candidate.copy()
    
    # Final local refinement on full grid for the best trajectory
    if best_traj is not None and time.time() < deadline - 10:
        params = traj_to_params(best_traj, n_ctrl)
        # A few coordinate descent steps on full grid
        for i in range(len(params)):
            for delta in [0.05, 0.02, 0.01, -0.05, -0.02, -0.01]:
                trial_params = params.copy()
                trial_params[i] += delta
                idx = i % 3
                if idx == 2:
                    trial_params[i] = np.clip(trial_params[i], -5, 95)
                trial_traj = params_to_traj(trial_params, n_ctrl)
                trial_score = evaluate_trajectory_full(trial_traj)
                if trial_score > best_score_full + 1e-9:
                    params = trial_params
                    best_score_full = trial_score
                    best_traj = trial_traj.copy()
    
    # If no trajectory was found, return a minimal valid one
    if best_traj is None:
        return np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    
    return best_traj

# EVOLVE-BLOCK-END
