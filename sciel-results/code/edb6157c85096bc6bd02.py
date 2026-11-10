# EVOLVE-BLOCK-START
import numpy as np
import time


def moving_sofa_2d() -> np.ndarray:
    """
    The 2D moving-sofa problem: find the largest-area rigid shape that can be moved
    through an L-shaped corridor of width 1 that makes a single 90-degree turn.

    The shape is specified IMPLICITLY by a motion "path". Each path point is a triple
    (x, y, angle_degrees) giving the position of the corridor's inner corner and the
    corridor's rotation at one instant of the move. In the reference frame of the sofa
    the corridor sweeps over the plane; the sofa is then the intersection of every
    corridor placement along the path, so by construction it always clears the corridor.
    The objective is to maximize the area of that intersection.

    The evaluator prepends the start state (0, 0, 0) and appends the end state
    (-2, 0, 90), then densifies the path to <=1-degree rotation steps, exactly as in the
    AlphaEvolve "single turn" corridor notebook.

    Returns:
        path: np.ndarray of shape (n, 3); each row is (x, y, angle_degrees) for the
            corridor's inner corner. Here n = 8. Endpoints are added by the evaluator.
    """
    # Set up the sofa grid
    x_grid = np.arange(-3, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_grid, y_grid)
    P = np.column_stack((X.ravel(), Y.ravel()))
    N_points = P.shape[0]

    # Start and End states (added by evaluator)
    start_state = np.array([0.0, 0.0, 0.0])
    end_state = np.array([-2.0, 0.0, 90.0])

    def densify_path(path):
        """
        Densify the path by adding linear interpolations between consecutive points.
        The evaluator does: between consecutive rows, add linear interpolations in
        m = max(1, ceil(|a2 - a1|)) equal steps.
        
        Returns the full densified path including start and end.
        """
        # Prepend start, append end
        full_path = np.vstack([start_state, path, end_state])
        
        densified = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            for k in range(1, m + 1):
                t = k / m
                interp = p1 + t * (p2 - p1)
                densified.append(interp)
        
        return np.array(densified)

    def evaluate_path(path):
        """
        Evaluate a path by computing the number of sofa points inside all corridor placements.
        Returns the area (count * 0.0001).
        """
        densified = densify_path(path)
        
        # For each placement, check which points are inside
        # A point w is inside placement (x, y, a) if R(-a)(w - (x, y)) lies in the base corridor
        # Base corridor: union of {0 <= u <= 1, -3 <= v <= 1} and {-3 <= u <= 1, 0 <= v <= 1}
        
        # Start with all points as potentially inside
        inside_all = np.ones(N_points, dtype=bool)
        
        for placement in densified:
            x, y, a_deg = placement
            a_rad = np.radians(a_deg)
            
            # Rotate points by -a around (x, y)
            dx = P[:, 0] - x
            dy = P[:, 1] - y
            
            cos_a = np.cos(-a_rad)
            sin_a = np.sin(-a_rad)
            
            u = cos_a * dx - sin_a * dy
            v = sin_a * dx + cos_a * dy
            
            # Check if (u, v) is in base corridor
            # Union of two rectangles:
            # Rect 1: 0 <= u <= 1, -3 <= v <= 1
            # Rect 2: -3 <= u <= 1, 0 <= v <= 1
            in_rect1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_rect2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            in_base = in_rect1 | in_rect2
            
            inside_all &= in_base
        
        return np.sum(inside_all) * 0.0001

    # Initial path (same as baseline)
    best_path = np.array(
        [
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ],
        dtype=np.float64,
    )
    best_score = evaluate_path(best_path)

    # Deadline guard
    deadline = time.time() + 600  # 600 seconds max

    # Simple local search: perturb each point and keep improvements
    np.random.seed(42)
    n_points_path = len(best_path)
    
    # Try random perturbations
    max_iter = 10000
    for iter in range(max_iter):
        if time.time() > deadline - 60:  # Safety margin of 60s
            break
        
        # Pick a random point to perturb
        idx = np.random.randint(n_points_path)
        new_path = best_path.copy()
        
        # Perturb x, y, angle
        new_path[idx, 0] += np.random.normal(0, 0.1)
        new_path[idx, 1] += np.random.normal(0, 0.1)
        new_path[idx, 2] += np.random.normal(0, 5)
        
        # Ensure angles are in reasonable range
        new_path[idx, 2] = np.clip(new_path[idx, 2], 0, 90)
        
        score = evaluate_path(new_path)
        if score > best_score:
            best_score = score
            best_path = new_path

    return best_path


# EVOLVE-BLOCK-END
