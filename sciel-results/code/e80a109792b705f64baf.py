"""Heilbronn triangles: place 11 points in a unit-area equilateral triangle to maximize the minimum triangle area."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 11
# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
AREA_TRI = 0.5 * np.sqrt(3)  # Area of the equilateral triangle with side 1

def barycentric_to_cartesian(coords):
    """Convert barycentric coordinates (u, v, w) with u+v+w=1 to Cartesian.
    We parameterize with (u, v) where w = 1-u-v, requiring u>=0, v>=0, u+v<=1.
    Cartesian: p = u*V0 + v*V1 + w*V2 = v*V1 + (1-u-v)*V2
    """
    u = coords[0::2]
    v = coords[1::2]
    w = 1.0 - u - v
    x = v * V1[0] + w * V2[0]
    y = v * V1[1] + w * V2[1]
    return np.column_stack([x, y])

def cartesian_to_barycentric(points):
    """Convert Cartesian points to barycentric (u, v) parameters."""
    # p = v*V1 + w*V2 where w = 1-u-v
    # x = v + 0.5*w = v + 0.5*(1-u-v) = 0.5 + 0.5*v - 0.5*u
    # y = sqrt(3)/2 * w = sqrt(3)/2 * (1-u-v)
    # From y: w = 2*y/sqrt(3), so 1-u-v = 2*y/sqrt(3) => u+v = 1 - 2*y/sqrt(3)
    # From x: 0.5 + 0.5*v - 0.5*u = x => v - u = 2*x - 1
    # So: v = (1 - 2*y/sqrt(3) + 2*x - 1)/2 = (2*x - 2*y/sqrt(3))/2 = x - y/sqrt(3)
    #     u = 1 - 2*y/sqrt(3) - v = 1 - 2*y/sqrt(3) - x + y/sqrt(3) = 1 - x - y/sqrt(3)
    x = points[:, 0]
    y = points[:, 1]
    v = x - y / np.sqrt(3)
    u = 1.0 - x - y / np.sqrt(3)
    return np.column_stack([u, v])

def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the N points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def objective(params):
    """Negative minimum triangle area (to be minimized)."""
    points = barycentric_to_cartesian(params)
    # Check if all points are inside the triangle
    for i in range(N):
        u = params[2*i]
        v = params[2*i+1]
        if u < -1e-10 or v < -1e-10 or u + v > 1 + 1e-10:
            return 1.0  # Large penalty
    return -min_triangle_area(points)

def objective_smooth(params, temperature=0.01):
    """Smoothed objective using soft-min approximation for better gradient behavior."""
    points = barycentric_to_cartesian(params)
    # Check bounds
    for i in range(N):
        u = params[2*i]
        v = params[2*i+1]
        if u < -1e-10 or v < -1e-10 or u + v > 1 + 1e-10:
            return 1.0
    
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                areas.append(area)
    areas = np.array(areas)
    # Soft min: -log(sum(exp(-a/T))) * T
    min_area_smooth = -temperature * np.log(np.sum(np.exp(-areas / temperature)))
    return -min_area_smooth

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    best_points = None
    best_area = 0.0
    
    # Bounds for barycentric coordinates: u in [0,1], v in [0,1], u+v <= 1
    # For differential_evolution, we need simple bounds, so we'll use [0,1] for each
    # and handle the constraint in the objective (penalty)
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Strategy 1: Differential Evolution for global exploration
    try:
        result_de = differential_evolution(
            objective,
            bounds=bounds,
            seed=42,
            maxiter=100,
            tol=1e-8,
            polish=False,
            updating='immediate',
            workers=1
        )
        elapsed = time.time() - start_time
        if elapsed < deadline - 30:
            if result_de.fun < 0:  # Valid solution
                area_de = -result_de.fun
                if area_de > best_area:
                    best_area = area_de
                    best_points = barycentric_to_cartesian(result_de.x)
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        if best_points is not None:
            return best_points
        # Fallback: return a reasonable default
        return _default_points()
    
    # Strategy 2: Multi-start local optimization with L-BFGS-B
    # Use the DE result and several random starts
    starts = []
    if best_points is not None:
        starts.append(cartesian_to_barycentric(best_points).flatten())
    
    # Add some structured starts
    rng = np.random.RandomState(123)
    for _ in range(5):
        # Random points in barycentric space
        u = rng.random(N)
        v = rng.random(N)
        # Ensure u+v <= 1
        for i in range(N):
            if u[i] + v[i] > 1:
                scale = 1.0 / (u[i] + v[i])
                u[i] *= scale
                v[i] *= scale
        starts.append(np.column_stack([u, v]).flatten())
    
    # Also try a grid-like initial configuration
    grid_start = _grid_barycentric()
    starts.append(grid_start)
    
    for start in starts:
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        try:
            result_local = minimize(
                objective,
                x0=start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result_local.fun < 0:  # Valid solution
                area_local = -result_local.fun
                if area_local > best_area:
                    best_area = area_local
                    best_points = barycentric_to_cartesian(result_local.x)
        except Exception:
            continue
    
    # Strategy 3: Refine with smooth objective if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 20 and best_points is not None:
        try:
            bary = cartesian_to_barycentric(best_points).flatten()
            # Anneal temperature
            for temp in [0.05, 0.02, 0.01, 0.005]:
                result_smooth = minimize(
                    lambda p: objective_smooth(p, temperature=temp),
                    x0=bary,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if result_smooth.fun < 0:
                    area_smooth = -result_smooth.fun
                    if area_smooth > best_area:
                        best_area = area_smooth
                        best_points = barycentric_to_cartesian(result_smooth.x)
                        bary = result_smooth.x
        except Exception:
            pass
    
    if best_points is not None:
        # Final verification
        final_area = min_triangle_area(best_points)
        if final_area > best_area * 0.99:  # Sanity check
            return best_points
    
    return _default_points()

def _default_points():
    """Return a default set of 11 points (corners, edge midpoints, center)."""
    points = []
    # Corners
    points.append(V0)
    points.append(V1)
    points.append(V2)
    # Edge midpoints
    points.append((V0 + V1) / 2)
    points.append((V1 + V2) / 2)
    points.append((V2 + V0) / 2)
    # Center
    points.append((V0 + V1 + V2) / 3)
    # Additional points
    for t in [0.25, 0.75]:
        points.append(t * V0 + (1-t) * (V1 + V2) / 2)
    return np.array(points[:N])

def _grid_barycentric():
    """Create a grid-like initial configuration in barycentric coordinates."""
    # 3x3 grid in the triangle
    points = []
    for i in range(3):
        for j in range(3 - i):
            u = i / 4.0
            v = j / 4.0
            if u + v <= 1.0:
                points.append([u, v])
    # Add center if not already there
    if len(points) < N:
        points.append([1/3, 1/3])
    # Pad with center if needed
    while len(points) < N:
        points.append([1/3, 1/3])
    return np.array(points[:N]).flatten()
