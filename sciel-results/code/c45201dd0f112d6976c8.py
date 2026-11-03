"""Rewritten program for Heilbronn triangles - 11 points in equilateral triangle."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 11
SQRT3 = np.sqrt(3)
TRI_AREA = 0.5 * SQRT3  # area of equilateral triangle with side 1


def _min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by the points (vectorized)."""
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                # Vectorized cross product for area
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area


def _min_triangle_area_fast(points):
    """Faster min triangle area using vectorized operations over all triples."""
    n = len(points)
    # points shape (n, 2)
    # Compute all triangle areas at once
    # For each pair (i,j), compute cross with all k > j
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            v1 = points[j] - points[i]  # shape (2,)
            # For all k > j
            v2 = points[j + 1:] - points[i]  # shape (m, 2)
            areas = 0.5 * np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area


def _barycentric_to_cartesian(bary):
    """Convert barycentric coordinates to Cartesian in the equilateral triangle."""
    a, b, c = bary
    x = a + 0.5 * b
    y = (SQRT3 / 2) * b
    return x, y


def _cartesian_to_barycentric(x, y):
    """Convert Cartesian to barycentric (a, b, c) where a+b+c=1."""
    b = 2 * y / SQRT3
    a = x - 0.5 * b
    c = 1 - a - b
    return a, b, c


def _in_triangle(x, y, eps=1e-9):
    """Check if point is inside the equilateral triangle."""
    if y < -eps or y > SQRT3 / 2 + eps:
        return False
    if x < -eps or x > 1 + eps:
        return False
    if x < 0.5:
        if y > SQRT3 * x + eps:
            return False
    else:
        if y > SQRT3 * (1 - x) + eps:
            return False
    return True


def _objective_flat(flat_params):
    """Objective function for differential evolution (minimize negative min area)."""
    points = np.empty((N, 2))
    for i in range(N):
        u = flat_params[2 * i]
        v = flat_params[2 * i + 1]
        x = u + 0.5 * v
        y = (SQRT3 / 2) * v
        points[i] = [x, y]
    
    # Check bounds
    for i in range(N):
        x, y = points[i]
        if y < -1e-4 or y > SQRT3 / 2 + 1e-4 or x < -1e-4 or x > 1 + 1e-4:
            return -1e-10
        if x < 0.5:
            if y > SQRT3 * x + 1e-4:
                return -1e-10
        else:
            if y > SQRT3 * (1 - x) + 1e-4:
                return -1e-10
    
    min_area = _min_triangle_area_fast(points)
    return -min_area


def _local_refine(points, max_iters=500, step_size=0.01, time_limit=None):
    """Local refinement using gradient-free coordinate descent with adaptive step."""
    best_points = points.copy()
    best_area = _min_triangle_area_fast(best_points)
    
    if time_limit is not None:
        deadline = time.time() + time_limit
    else:
        deadline = float('inf')
    
    current_step = step_size
    no_improve = 0
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            for coord in range(2):
                # Try moving in 4 directions
                for direction in [-1, 1]:
                    new_points = best_points.copy()
                    new_points[i, coord] += direction * current_step
                    
                    # Check bounds
                    valid = True
                    for j in range(N):
                        x, y = new_points[j]
                        if y < -1e-6 or y > SQRT3 / 2 + 1e-6 or x < -1e-6 or x > 1 + 1e-6:
                            valid = False
                            break
                        if x < 0.5:
                            if y > SQRT3 * x + 1e-6:
                                valid = False
                                break
                        else:
                            if y > SQRT3 * (1 - x) + 1e-6:
                                valid = False
                                break
                    if not valid:
                        continue
                    
                    new_area = _min_triangle_area_fast(new_points)
                    if new_area > best_area + 1e-12:
                        best_points = new_points
                        best_area = new_area
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            no_improve += 1
            if no_improve >= 3:
                current_step *= 0.7
                if current_step < 1e-6:
                    break
                no_improve = 0
    
    return best_points, best_area


def _multi_start_local_refine(start_points, num_starts=5, time_budget=60):
    """Run local refinement from multiple perturbations of start_points."""
    best_points = start_points.copy()
    best_area = _min_triangle_area_fast(best_points)
    
    start_time = time.time()
    
    # First refine from the original
    refined, area = _local_refine(start_points, max_iters=300, step_size=0.02, 
                                   time_limit=max(0, time_budget - 5))
    if area > best_area:
        best_points = refined
        best_area = area
    
    # Try random perturbations
    for s in range(num_starts):
        if time.time() - start_time > time_budget - 5:
            break
        
        # Random perturbation
        perturbed = start_points.copy()
        for i in range(N):
            perturbed[i, 0] += np.random.uniform(-0.05, 0.05)
            perturbed[i, 1] += np.random.uniform(-0.05, 0.05)
        
        # Clip to triangle (approximate)
        for i in range(N):
            x, y = perturbed[i]
            if y < 0:
                y = 0.01
            if y > SQRT3 / 2:
                y = SQRT3 / 2 - 0.01
            if x < 0:
                x = 0.01
            if x > 1:
                x = 0.99
            if x < 0.5:
                max_y = SQRT3 * x
                if y > max_y - 0.01:
                    y = max_y - 0.02
            else:
                max_y = SQRT3 * (1 - x)
                if y > max_y - 0.01:
                    y = max_y - 0.02
            perturbed[i] = [x, y]
        
        remaining = max(0, time_budget - (time.time() - start_time) - 5)
        if remaining < 2:
            break
        
        refined, area = _local_refine(perturbed, max_iters=200, step_size=0.015,
                                       time_limit=remaining)
        if area > best_area:
            best_points = refined
            best_area = area
    
    return best_points, best_area


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    np.random.seed(42)
    
    # Initial good starting point: distribute points roughly evenly
    # Use a grid-like initial distribution in barycentric coordinates
    barycentric_init = [
        [0.1, 0.1, 0.8],
        [0.1, 0.8, 0.1],
        [0.8, 0.1, 0.1],
        [0.15, 0.15, 0.7],
        [0.15, 0.7, 0.15],
        [0.7, 0.15, 0.15],
        [0.7, 0.15, 0.15],
        [0.333, 0.333, 0.334],
        [0.2, 0.4, 0.4],
        [0.4, 0.2, 0.4],
        [0.4, 0.4, 0.2],
    ]
    
    initial_points = []
    for b in barycentric_init:
        x, y = _barycentric_to_cartesian(b)
        initial_points.append([x, y])
    initial_points = np.array(initial_points)
    
    initial_flat = []
    for p in initial_points:
        v = 2 * p[1] / SQRT3
        u = p[0] - 0.5 * v
        initial_flat.extend([u, v])
    initial_flat = np.array(initial_flat)
    
    # Define bounds
    bounds = [(0, 1), (0, 1)] * N
    
    # Phase 1: Differential evolution
    elapsed = time.time() - start_time
    de_time_budget = min(80, deadline - time.time() - 30)
    
    if de_time_budget > 10:
        # Use a callback to stop if we're running out of time
        de_result = differential_evolution(
            _objective_flat,
            bounds,
            x0=initial_flat,
            maxiter=150,
            popsize=15,
            tol=1e-5,
            mutation=(0.3, 1.0),
            recombination=0.8,
            polish=False,  # We'll do our own polishing
            seed=42,
            callback=lambda x, convergence: (time.time() - start_time) > deadline - 40
        )
        
        if de_result.fun < 0:
            best_flat = de_result.x
            points = np.empty((N, 2))
            for i in range(N):
                u = best_flat[2 * i]
                v = best_flat[2 * i + 1]
                x = u + 0.5 * v
                y = (SQRT3 / 2) * v
                points[i] = [x, y]
        else:
            points = initial_points
    else:
        points = initial_points
    
    # Phase 2: Local refinement
    elapsed = time.time() - start_time
    local_time_budget = max(5, deadline - time.time() - 5)
    
    if local_time_budget > 5:
        best_points, best_area = _multi_start_local_refine(
            points, num_starts=4, time_budget=local_time_budget
        )
    else:
        best_points = points
        best_area = _min_triangle_area_fast(points)
    
    return best_points
