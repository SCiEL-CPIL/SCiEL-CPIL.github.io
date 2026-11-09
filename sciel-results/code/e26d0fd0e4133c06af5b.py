"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize, dual_annealing

N = 11
H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])
TRIANGLE_AREA = 0.5 * 1.0 * H  # area of the equilateral triangle

# Precompute all C(11,3) = 165 triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i, dtype=np.int64)
_idx_j = np.array(_idx_j, dtype=np.int64)
_idx_k = np.array(_idx_k, dtype=np.int64)


def _min_triangle_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _is_inside_triangle(x, y):
    """Check if point (x,y) is inside or on the equilateral triangle."""
    if y < -1e-9 or y > H + 1e-9:
        return False
    # Left edge: from (0,0) to (0.5, H): x >= y / H
    # Right edge: from (1,0) to (0.5, H): x <= 1 - y / H
    if x < -1e-9 or x > 1 + 1e-9:
        return False
    if y > 0:
        if x < y / H - 1e-9:
            return False
        if x > 1 - y / H + 1e-9:
            return False
    return True


def _clamp_to_triangle(points):
    """Clamp points to be inside the triangle."""
    result = points.copy()
    for i in range(len(result)):
        x, y = result[i]
        if y < 0:
            y = 0.0
        if y > H:
            y = H
        if x < 0:
            x = 0.0
        if x > 1:
            x = 1.0
        # Clamp to triangle edges
        if y > 0:
            left_bound = y / H
            right_bound = 1 - y / H
            if x < left_bound:
                x = left_bound
            if x > right_bound:
                x = right_bound
        result[i] = [x, y]
    return result


def _params_to_points(params):
    """Convert 2N parameters to N points, ensuring they are inside the triangle."""
    params = np.array(params)
    points = params.reshape(N, 2)
    return _clamp_to_triangle(points)


def _objective(params):
    """Objective function to minimize: negative of min triangle area."""
    points = _params_to_points(params)
    min_area = _min_triangle_area(points)
    return -min_area


def _soft_min_objective(params, temperature=0.01):
    """Differentiable soft-minimum of triangle areas for gradient-based optimization."""
    points = _params_to_points(params)
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    # Soft minimum using log-sum-exp
    log_sum = np.log(np.sum(np.exp(-areas / temperature)))
    soft_min = -temperature * log_sum
    return -soft_min  # We want to maximize min area, so minimize negative


def _gradient_descent_refine(points, max_iter=1000, lr=0.01, temperature=0.01):
    """Refine points using gradient descent on soft-min objective."""
    params = points.flatten()
    best_params = params.copy()
    best_obj = _objective(best_params)
    
    for it in range(max_iter):
        # Compute gradient numerically
        grad = np.zeros_like(params)
        eps = 1e-6
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            obj_plus = _soft_min_objective(params_plus, temperature)
            grad[i] = (obj_plus - best_obj) / eps
        
        params -= lr * grad
        points = _clamp_to_triangle(params.reshape(N, 2))
        params = points.flatten()
        
        obj = _objective(params)
        if obj < best_obj:
            best_obj = obj
            best_params = params.copy()
    
    return _params_to_points(best_params), -best_obj


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    best_points = None
    best_score = -1.0
    
    # Initial diverse points
    rng = np.random.RandomState(42)
    
    # Generate several initial configurations
    initial_configs = []
    
    # 1. Random points
    for trial in range(20):
        points = np.zeros((N, 2))
        for i in range(N):
            u, v = rng.random(2)
            while u + v > 1:
                u, v = rng.random(2)
            points[i] = [u + 0.5 * v, H * v]
        initial_configs.append(points)
    
    # 2. Grid-based initialization
    for trial in range(10):
        points = np.zeros((N, 2))
        # Use a 3x4 grid with some randomization
        grid_positions = []
        for row in range(3):
            for col in range(4):
                y = (row + 0.5) / 3.0 * H
                x_left = y / H
                x_right = 1 - y / H
                x = x_left + (col + 0.5) / 4.0 * (x_right - x_left)
                grid_positions.append([x, y])
        # Add some random jitter
        for i in range(N):
            if i < len(grid_positions):
                points[i] = grid_positions[i] + rng.normal(0, 0.05, 2)
            else:
                u, v = rng.random(2)
                while u + v > 1:
                    u, v = rng.random(2)
                points[i] = [u + 0.5 * v, H * v]
        points = _clamp_to_triangle(points)
        initial_configs.append(points)
    
    # 3. Symmetric initialization
    for trial in range(5):
        points = np.zeros((N, 2))
        # Place points symmetrically
        # 3 on bottom edge, 4 on middle row, 3 on top area, 1 at top
        positions = [
            [0.1, 0.0], [0.5, 0.0], [0.9, 0.0],  # bottom
            [0.25, H/3], [0.45, H/3], [0.55, H/3], [0.75, H/3],  # middle
            [0.3, 2*H/3], [0.5, 2*H/3], [0.7, 2*H/3],  # upper
            [0.5, 0.9*H],  # top
        ]
        for i in range(N):
            points[i] = positions[i] + rng.normal(0, 0.03, 2)
        points = _clamp_to_triangle(points)
        initial_configs.append(points)
    
    # Evaluate initial configurations
    for points in initial_configs:
        score = _min_triangle_area(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Differential Evolution for global search
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        bounds = [(0, 1), (0, H)] * N
        result = differential_evolution(
            _objective,
            bounds=bounds,
            seed=42,
            maxiter=50,
            tol=1e-6,
            popsize=15,
            init='latinhypercube',
            polish=True
        )
        if result.success or result.fun < -best_score:
            candidate_points = _params_to_points(result.x)
            candidate_score = _min_triangle_area(candidate_points)
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points.copy()
    
    # L-BFGS-B refinement from multiple diverse initializations
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        # Use soft-min objective for smoother landscape
        for init_idx, points in enumerate(initial_configs[:10]):
            if time.time() > deadline - 30:
                break
            params = points.flatten()
            try:
                res = minimize(
                    _soft_min_objective,
                    x0=params,
                    method='L-BFGS-B',
                    bounds=[(0, 1), (0, H)] * N,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                candidate_points = _params_to_points(res.x)
                candidate_score = _min_triangle_area(candidate_points)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_points = candidate_points.copy()
            except Exception:
                pass
    
    # Dual annealing for additional global search
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        try:
            result_da = dual_annealing(
                _objective,
                bounds=[(0, 1), (0, H)] * N,
                seed=123,
                maxiter=100
            )
            candidate_points = _params_to_points(result_da.x)
            candidate_score = _min_triangle_area(candidate_points)
            if candidate_score > best_score:
                best_score = candidate_score
                best_points = candidate_points.copy()
        except Exception:
            pass
    
    # Final local refinement using gradient descent on worst triangles
    elapsed = time.time() - start_time
    if elapsed < deadline - 10 and best_points is not None:
        points = best_points.copy()
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            # Find the worst triangle
            p1 = points[_idx_i]
            p2 = points[_idx_j]
            p3 = points[_idx_k]
            cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
            areas = 0.5 * np.abs(cross)
            worst_idx = np.argmin(areas)
            
            # Try to move one of the worst triangle's vertices
            i, j, k = _idx_i[worst_idx], _idx_j[worst_idx], _idx_k[worst_idx]
            best_move = None
            best_move_score = _min_triangle_area(points)
            
            for vertex in [i, j, k]:
                for dx in [-0.02, -0.01, 0, 0.01, 0.02]:
                    for dy in [-0.02, -0.01, 0, 0.01, 0.02]:
                        new_points = points.copy()
                        new_points[vertex, 0] += dx
                        new_points[vertex, 1] += dy
                        new_points = _clamp_to_triangle(new_points)
                        score = _min_triangle_area(new_points)
                        if score > best_move_score:
                            best_move_score = score
                            best_move = new_points.copy()
            
            if best_move is not None:
                points = best_move
                if _min_triangle_area(points) > best_score:
                    best_score = _min_triangle_area(points)
                    best_points = points.copy()
            else:
                break
    
    if best_points is None:
        # Fallback: random points
        points = np.zeros((N, 2))
        for i in range(N):
            u, v = rng.random(2)
            while u + v > 1:
                u, v = rng.random(2)
            points[i] = [u + 0.5 * v, H * v]
        best_points = _clamp_to_triangle(points)
    
    return best_points
