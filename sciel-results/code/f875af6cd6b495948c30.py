"""Heilbronn triangles: place 11 points to maximize the smallest triangle area."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
SQRT3 = np.sqrt(3.0)

# Vertices of the equilateral triangle with area 1
# Area of equilateral triangle with side s is (sqrt(3)/4) * s^2 = 1
# So s^2 = 4/sqrt(3), s = 2 / (3^0.25)
SIDE = 2.0 / (3.0 ** 0.25)
V0 = np.array([0.0, 0.0])
V1 = np.array([SIDE, 0.0])
V2 = np.array([SIDE / 2.0, SIDE * SQRT3 / 2.0])


def bary_to_cart(bary):
    """Convert barycentric coordinates (u, v, w with u+v+w=1) to Cartesian.
    We use (u, v) as parameters, w = 1 - u - v."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = u * V0[0] + v * V1[0] + w * V2[0]
    y = u * V0[1] + v * V1[1] + w * V2[1]
    return np.column_stack([x, y])


def min_triangle_area_cartesian(points):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                if area < min_area:
                    min_area = area
    return min_area


def objective_bary(bary_flat):
    """Objective to minimize: negative of the soft-min of triangle areas.
    bary_flat is a 2N array of (u, v) pairs."""
    bary = bary_flat.reshape(N, 2)
    # Ensure valid barycentric coords: u >= 0, v >= 0, u+v <= 1
    # Small penalty if outside
    if np.any(bary[:, 0] < -1e-6) or np.any(bary[:, 1] < -1e-6) or np.any(bary[:, 0] + bary[:, 1] > 1 + 1e-6):
        return 1e6
    # Clip to valid range
    bary[:, 0] = np.clip(bary[:, 0], 0, 1)
    bary[:, 1] = np.clip(bary[:, 1], 0, 1 - bary[:, 0])
    
    points = bary_to_cart(bary)
    
    # Compute all triangle areas
    n = N
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                areas.append(area)
    areas = np.array(areas)
    
    # Use log-sum-exp trick for soft min
    beta = 100.0
    min_area = np.min(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - min_area))))
    soft_min = min_area - np.log(log_sum) / beta
    
    return -soft_min


def objective_bary_with_beta(bary_flat, beta):
    """Objective with a specific beta parameter."""
    bary = bary_flat.reshape(N, 2)
    if np.any(bary[:, 0] < -1e-6) or np.any(bary[:, 1] < -1e-6) or np.any(bary[:, 0] + bary[:, 1] > 1 + 1e-6):
        return 1e6
    bary[:, 0] = np.clip(bary[:, 0], 0, 1)
    bary[:, 1] = np.clip(bary[:, 1], 0, 1 - bary[:, 0])
    
    points = bary_to_cart(bary)
    
    n = N
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                areas.append(area)
    areas = np.array(areas)
    
    min_area = np.min(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - min_area))))
    soft_min = min_area - np.log(log_sum) / beta
    
    return -soft_min


def hexagonal_lattice_init():
    """Generate a hexagonal lattice pattern of 11 points in barycentric coords."""
    # For an equilateral triangle, a hexagonal packing can be approximated
    # by dividing the triangle into smaller triangles and placing points at centers
    
    # A simple structured init: use a grid in barycentric space
    # 11 points: try a 3-3-3-2 or similar arrangement
    
    # Let's use a more systematic approach: divide the triangle into regions
    # and place points at centroids of sub-triangles
    
    # For 11 points, we can use a pattern based on triangular numbers
    # T(4) = 10, so we need 1 more. Let's use a 4-row triangular grid plus 1 center point
    
    points_bary = []
    
    # Row 0 (top): 1 point at top vertex area
    # Row 1: 2 points
    # Row 2: 3 points
    # Row 3: 3 points
    # Row 4: 2 points
    # That's 11
    
    # Actually, let me use a simpler structured approach
    # Divide the triangle into 11 roughly equal regions
    
    # Use a 3x3 grid in barycentric coords with some adjustments
    # Barycentric coords (u, v) where u, v >= 0, u+v <= 1
    
    # Create a grid and filter valid points
    n_grid = 5
    us = np.linspace(0.1, 0.8, n_grid)
    vs = np.linspace(0.1, 0.8, n_grid)
    
    candidates = []
    for u in us:
        for v in vs:
            if u + v <= 0.9:
                candidates.append([u, v])
    
    # We need exactly 11 points, so take a subset or adjust
    # For now, let's use a fixed pattern
    # Pattern: approximate optimal positions for 11 points
    
    # Known good initial patterns for Heilbronn triangles
    # Let's use a symmetric arrangement
    
    # Center point
    points_bary.append([1/3, 1/3])
    
    # 3 points near each vertex (but not at vertices)
    eps = 0.15
    # Near V0 (u=1, v=0)
    points_bary.append([1 - eps, eps/2])
    # Near V1 (u=0, v=1)
    points_bary.append([eps/2, 1 - eps])
    # Near V2 (u=0, v=0)
    points_bary.append([eps/2, eps/2])
    
    # 6 points along the edges/midpoints
    # Midpoint of V0-V1 edge: (u=0.5, v=0.5)
    points_bary.append([0.5, 0.5])
    # Midpoint of V0-V2 edge: (u=0.5, v=0)
    points_bary.append([0.5, 0.05])
    # Midpoint of V1-V2 edge: (u=0, v=0.5)
    points_bary.append([0.05, 0.5])
    
    # 3 more interior points
    points_bary.append([0.3, 0.3])
    points_bary.append([0.3, 0.4])
    points_bary.append([0.4, 0.3])
    
    return np.array(points_bary)


def random_bary_init(rng):
    """Generate random barycentric coordinates."""
    bary = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = rng.uniform(0, 1)
            v = rng.uniform(0, 1)
            if u + v <= 1:
                bary[i] = [u, v]
                break
    return bary


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_score = -np.inf
    best_bary = None
    
    # Try multiple initializations
    inits = []
    
    # 1. Hexagonal lattice
    inits.append(hexagonal_lattice_init())
    
    # 2. Random inits with different seeds
    import random
    for seed in [1, 7, 42, 123, 999]:
        rng = random.Random(seed)
        inits.append(random_bary_init(rng))
    
    for init_bary in inits:
        if time.time() > deadline - 30:
            break
        
        # Multi-stage L-BFGS-B with increasing beta
        current_bary = init_bary.copy()
        best_local_score = -np.inf
        best_local_bary = current_bary.copy()
        
        for beta in [10.0, 50.0, 100.0, 200.0, 500.0]:
            if time.time() > deadline - 10:
                break
            
            # Define objective with current beta
            def obj(x):
                return objective_bary_with_beta(x, beta)
            
            # Bounds for barycentric coords
            bounds = [(0, 1), (0, 1)] * N
            # But we also need u+v <= 1, which is a linear constraint
            # L-BFGS-B doesn't handle linear constraints directly, so we use a penalty or clipping
            
            try:
                result = minimize(
                    obj,
                    current_bary.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < best_local_score:
                    new_bary = result.x.reshape(N, 2)
                    # Clip to valid region
                    new_bary[:, 0] = np.clip(new_bary[:, 0], 0, 1)
                    new_bary[:, 1] = np.clip(new_bary[:, 1], 0, 1 - new_bary[:, 0])
                    
                    # Evaluate actual min area
                    points = bary_to_cart(new_bary)
                    actual_min_area = min_triangle_area_cartesian(points)
                    
                    if actual_min_area > best_local_score:
                        best_local_score = actual_min_area
                        best_local_bary = new_bary.copy()
                    
                    current_bary = new_bary.copy()
            except Exception:
                pass
        
        if best_local_score > best_score:
            best_score = best_local_score
            best_bary = best_local_bary.copy()
    
    # If we still have time, try differential evolution for a global search
    if time.time() < deadline - 20 and best_bary is not None:
        def de_objective(x):
            bary = x.reshape(N, 2)
            if np.any(bary[:, 0] < -1e-6) or np.any(bary[:, 1] < -1e-6) or np.any(bary[:, 0] + bary[:, 1] > 1 + 1e-6):
                return 1e6
            bary[:, 0] = np.clip(bary[:, 0], 0, 1)
            bary[:, 1] = np.clip(bary[:, 1], 0, 1 - bary[:, 0])
            
            points = bary_to_cart(bary)
            min_area = min_triangle_area_cartesian(points)
            return -min_area
        
        de_bounds = [(0, 1), (0, 1)] * N
        
        try:
            de_result = differential_evolution(
                de_objective,
                de_bounds,
                seed=42,
                maxiter=50,
                popsize=10,
                tol=1e-6,
                polish=True
            )
            
            if de_result.fun < -best_score:
                de_bary = de_result.x.reshape(N, 2)
                de_bary[:, 0] = np.clip(de_bary[:, 0], 0, 1)
                de_bary[:, 1] = np.clip(de_bary[:, 1], 0, 1 - de_bary[:, 0])
                
                points = bary_to_cart(de_bary)
                actual_min_area = min_triangle_area_cartesian(points)
                
                if actual_min_area > best_score:
                    best_score = actual_min_area
                    best_bary = de_bary.copy()
        except Exception:
            pass
    
    # Final refinement with high beta
    if best_bary is not None and time.time() < deadline - 5:
        for beta in [500.0, 1000.0, 2000.0]:
            if time.time() > deadline - 2:
                break
            
            def obj(x):
                return objective_bary_with_beta(x, beta)
            
            bounds = [(0, 1), (0, 1)] * N
            
            try:
                result = minimize(
                    obj,
                    best_bary.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.fun < -best_score:
                    new_bary = result.x.reshape(N, 2)
                    new_bary[:, 0] = np.clip(new_bary[:, 0], 0, 1)
                    new_bary[:, 1] = np.clip(new_bary[:, 1], 0, 1 - new_bary[:, 0])
                    
                    points = bary_to_cart(new_bary)
                    actual_min_area = min_triangle_area_cartesian(points)
                    
                    if actual_min_area > best_score:
                        best_score = actual_min_area
                        best_bary = new_bary.copy()
            except Exception:
                pass
    
    # Convert to Cartesian coordinates
    if best_bary is not None:
        points = bary_to_cart(best_bary)
    else:
        # Fallback: return the original seed program's result
        import random
        rng = random.Random(7)
        points_list = []
        while len(points_list) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points_list.append([u + 0.5 * v, (3 ** 0.5) / 2 * v])
        points = np.array(points_list)
    
    return points
