"""Heilbronn triangles: maximize the smallest triangle area for 11 points in a unit equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from itertools import combinations

N = 11
SQRT3 = 3 ** 0.5
DEADLINE_FRACTION = 0.85  # safety margin


def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area


def _objective_and_constraints(x, n):
    """
    x: flattened array of 2n coordinates.
    Returns (objective_value, constraints_list).
    We maximize min triangle area, so we minimize -min_area.
    Constraints: points must be inside the equilateral triangle.
    """
    points = x.reshape(n, 2)
    
    # Check if all points are inside the triangle (with small margin)
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Interior conditions:
    #   y >= 0
    #   y <= sqrt(3) * x  (below line from (0,0) to (0.5, sqrt(3)/2) extended... wait)
    # Actually, the three half-planes for triangle with vertices A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2):
    #   Above AB: y >= 0
    #   Left of BC: The line BC goes from (1,0) to (0.5, sqrt(3)/2). 
    #     Direction: (-0.5, sqrt(3)/2). Normal pointing inward: (sqrt(3)/2, 0.5)
    #     Equation: sqrt(3)/2 * (x - 1) + 0.5 * (y - 0) <= 0 => sqrt(3)*x + y <= sqrt(3)
    #   Right of AC: The line AC goes from (0,0) to (0.5, sqrt(3)/2).
    #     Direction: (0.5, sqrt(3)/2). Normal pointing inward: (sqrt(3)/2, -0.5)
    #     Equation: sqrt(3)/2 * x - 0.5 * y >= 0 => sqrt(3)*x - y >= 0 => y <= sqrt(3)*x
    
    margin = 1e-6
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    # Constraint 1: y >= margin
    c1 = y_coords - margin
    # Constraint 2: sqrt(3)*x - y >= margin  => y <= sqrt(3)*x - margin
    c2 = SQRT3 * x_coords - y_coords - margin
    # Constraint 3: sqrt(3)*x + y <= sqrt(3) - margin
    c3 = SQRT3 - margin - SQRT3 * x_coords - y_coords
    
    # All must be >= 0
    min_constraint = min(c1.min(), c2.min(), c3.min())
    
    if min_constraint < 0:
        # Points are outside; return a large penalty
        return -1e10, min_constraint
    
    min_area = _min_triangle_area(points)
    return -min_area, min_constraint


def _build_initial_points(rng):
    """Generate N random points inside the triangle."""
    points = []
    while len(points) < N:
        u = rng.random()
        v = rng.random()
        if u + v < 1.0:
            # Barycentric coordinates
            x = u + 0.5 * v
            y = (SQRT3 / 2) * v
            # Add small margin to stay inside
            # Check if inside
            if y > 1e-4 and SQRT3 * x - y > 1e-4 and SQRT3 * x + y < SQRT3 - 1e-4:
                points.append([x, y])
    return np.array(points)


def _simulated_annealing_step(points, temperature, rng):
    """
    Perform one simulated annealing step.
    Returns new points and whether it was accepted.
    """
    n = len(points)
    idx = rng.randint(0, n - 1)
    
    # Generate a random perturbation
    step_size = temperature * 0.1
    dx = rng.gauss(0, step_size)
    dy = rng.gauss(0, step_size)
    
    new_points = points.copy()
    new_points[idx, 0] += dx
    new_points[idx, 1] += dy
    
    # Check if new point is inside the triangle
    x, y = new_points[idx]
    margin = 1e-6
    if y < margin or SQRT3 * x - y < margin or SQRT3 * x + y > SQRT3 - margin:
        return points, False
    
    old_min_area = _min_triangle_area(points)
    new_min_area = _min_triangle_area(new_points)
    
    delta = new_min_area - old_min_area
    if delta >= 0:
        return new_points, True
    else:
        if temperature > 0:
            prob = np.exp(delta / temperature)
            if rng.random() < prob:
                return new_points, True
    return points, False


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_FRACTION * 150  # 127.5 seconds
    
    rng = np.random.RandomState(42)
    
    # Start with a few random initializations and pick the best
    best_points = None
    best_min_area = 0
    
    # Try several random starts
    for init in range(20):
        if time.time() > deadline:
            break
        
        points = _build_initial_points(rng)
        min_area = _min_triangle_area(points)
        
        if min_area > best_min_area:
            best_min_area = min_area
            best_points = points.copy()
    
    if best_points is None:
        # Fallback: deterministic points
        best_points = _build_initial_points(rng)
        best_min_area = _min_triangle_area(best_points)
    
    # Simulated annealing refinement
    current_points = best_points.copy()
    current_min_area = best_min_area
    
    # Cooling schedule
    T0 = 0.1
    T_end = 1e-6
    num_iterations = 5000
    cooling_factor = (T_end / T0) ** (1.0 / num_iterations)
    
    temperature = T0
    for it in range(num_iterations):
        if time.time() > deadline:
            break
        
        new_points, accepted = _simulated_annealing_step(current_points, temperature, rng)
        if accepted:
            current_points = new_points
            new_min_area = _min_triangle_area(current_points)
            if new_min_area > best_min_area:
                best_min_area = new_min_area
                best_points = current_points.copy()
        
        temperature *= cooling_factor
    
    # Final local optimization using Nelder-Mead on the best point
    if time.time() < deadline - 10:  # Leave 10 seconds margin
        # Flatten points for optimization
        x0 = best_points.flatten()
        
        def objective(x):
            points = x.reshape(N, 2)
            # Check bounds
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            margin = 1e-8
            if (y_coords < margin).any() or (SQRT3 * x_coords - y_coords < margin).any() or (SQRT3 * x_coords + y_coords > SQRT3 - margin).any():
                return 1e10
            return -_min_triangle_area(points)
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.success:
                opt_points = result.x.reshape(N, 2)
                # Verify points are inside
                x_coords = opt_points[:, 0]
                y_coords = opt_points[:, 1]
                margin = 1e-6
                if (y_coords >= margin).all() and (SQRT3 * x_coords - y_coords >= margin).all() and (SQRT3 * x_coords + y_coords <= SQRT3 - margin).all():
                    opt_min_area = _min_triangle_area(opt_points)
                    if opt_min_area > best_min_area:
                        best_min_area = opt_min_area
                        best_points = opt_points
        except Exception:
            pass
    
    # Ensure we return a valid array
    if best_points is None or len(best_points) != N:
        best_points = _build_initial_points(rng)
    
    return best_points
