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
    (-2, 0, 90), then densifies the path to <=1-degree rotation steps.

    Returns:
        path: np.ndarray of shape (n, 3); each row is (x, y, angle_degrees) for the
            corridor's inner corner.
    """
    start_time = time.time()
    deadline = 620.0  # evaluator kills at 630s, keep safety margin

    # Sofa grid
    x_grid = np.arange(-3, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_grid, y_grid)
    P = np.column_stack((X.ravel(), Y.ravel()))  # (N, 2)
    N = P.shape[0]

    # Start and end states (added by evaluator)
    start_state = np.array([0.0, 0.0, 0.0])
    end_state = np.array([-2.0, 0.0, 90.0])

    def densify_path(path):
        """
        Densify the path by adding linear interpolations between consecutive rows.
        Between consecutive rows (x1,y1,a1) and (x2,y2,a2), add m = max(1, ceil(|a2-a1|))
        equal steps of interpolation.
        """
        if len(path) == 0:
            return np.array([start_state, end_state])
        
        full_path = [start_state] + list(path) + [end_state]
        
        densified = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            for k in range(1, m + 1):
                t = k / m
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                a = p1[2] + t * (p2[2] - p1[2])
                densified.append([x, y, a])
        
        return np.array(densified)

    def check_point_in_corridor_vectorized(P, x, y, a_deg):
        """
        Vectorized check: which points in P are inside the corridor placement at (x, y, a_deg).
        Returns a boolean array of shape (N,).
        """
        a_rad = np.radians(a_deg)
        dx = P[:, 0] - x
        dy = P[:, 1] - y
        cos_a = np.cos(-a_rad)
        sin_a = np.sin(-a_rad)
        u = cos_a * dx - sin_a * dy
        v = sin_a * dx + cos_a * dy
        
        eps = 1e-9
        in_r1 = (u >= -eps) & (u <= 1 + eps) & (v >= -3 - eps) & (v <= 1 + eps)
        in_r2 = (u >= -3 - eps) & (u <= 1 + eps) & (v >= -eps) & (v <= 1 + eps)
        
        return in_r1 | in_r2

    def evaluate_path(path):
        """
        Evaluate the area of the sofa for a given path.
        The sofa is the set of grid points that are inside every placement.
        """
        densified = densify_path(path)
        
        # Initialize all points as potentially in the sofa
        in_sofa = np.ones(N, dtype=bool)
        
        for x, y, a in densified:
            # Check all points against this placement
            mask = check_point_in_corridor_vectorized(P, x, y, a)
            in_sofa &= mask
            if not in_sofa.any():
                break  # Early exit if no points remain
        
        return in_sofa.sum() * 0.0001

    # Initial path - hand-tuned starting point
    path = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)

    best_path = path.copy()
    best_score = evaluate_path(path)

    def coordinate_descent_optimize(path, max_iterations=500, base_step=0.1):
        """
        Coordinate descent optimization with adaptive step sizes.
        For each point in the path, try moving in each coordinate direction.
        """
        current_path = path.copy()
        current_score = evaluate_path(current_path)
        step = base_step
        
        for iteration in range(max_iterations):
            if (time.time() - start_time) > deadline - 30:
                break
            
            improved = False
            for i in range(len(current_path)):
                if (time.time() - start_time) > deadline - 30:
                    break
                
                # Try each coordinate: x, y, angle
                for dim in range(3):
                    # Try positive and negative directions
                    for direction in [1, -1]:
                        new_path = current_path.copy()
                        new_path[i, dim] += direction * step
                        
                        # Clip angle to valid range
                        if dim == 2:
                            new_path[i, 2] = np.clip(new_path[i, 2], 5, 85)
                        
                        new_score = evaluate_path(new_path)
                        
                        if new_score > current_score:
                            current_path = new_path
                            current_score = new_score
                            improved = True
                        else:
                            # If no improvement, reduce step size for this coordinate
                            step = step * 0.7
                    
                    # Reset step size after trying both directions
                    step = base_step * (0.7 ** iteration)
            
            if not improved:
                break
        
        return current_path, current_score

    # Run coordinate descent with decreasing base step sizes
    base_steps = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    
    for base_step in base_steps:
        if (time.time() - start_time) > deadline - 30:
            break
        
        path, score = coordinate_descent_optimize(path, max_iterations=200, base_step=base_step)
        
        if score > best_score:
            best_score = score
            best_path = path.copy()

    # Final polish with very small steps
    if (time.time() - start_time) < deadline - 10:
        path, score = coordinate_descent_optimize(path, max_iterations=100, base_step=0.001)
        if score > best_score:
            best_score = score
            best_path = path.copy()

    # Ensure we return the best path found
    if best_score > evaluate_path(path):
        path = best_path.copy()

    return path

# EVOLVE-BLOCK-END
