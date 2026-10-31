import numpy as np
from scipy.optimize import minimize
import time
from itertools import combinations

N = 11
TIME_LIMIT = 115.0

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def cart_to_bary(cart):
    """Convert Cartesian (x, y) to barycentric (u, v)."""
    x = cart[:, 0]
    y = cart[:, 1]
    S = 1 - y / H
    u = 0.5 + 0.5 * S - x
    v = S - u
    return np.column_stack([u, v])

def calc_areas_bary(bary):
    """Calculate areas of all triangles formed by N points in barycentric coords."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def calc_min_area(bary):
    """Calculate minimum area efficiently."""
    return np.min(calc_areas_bary(bary))

def clamp_bary(bary, eps=1e-10):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def harmonic_function(bary, mode=0):
    """
    Define a harmonic function on the triangle using barycentric coordinates.
    Different modes give different critical point structures.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    if mode == 0:
        # Standard harmonic: u*v*w
        return u * v * w
    elif mode == 1:
        # Modified: (u - 1/3)^2 + (v - 1/3)^2 + (w - 1/3)^2
        return (u - 1.0/3)**2 + (v - 1.0/3)**2 + (w - 1.0/3)**2
    elif mode == 2:
        # Product form
        return u * v + v * w + w * u
    else:
        return u * v * w

def critical_points_grid(mode, n_per_edge=5):
    """
    Generate a grid of critical points of the harmonic function.
    These are points where the gradient is zero or where the function has local extrema.
    """
    points = []
    
    # For mode 0 (u*v*w), critical points are at:
    # - Center (1/3, 1/3)
    # - Midpoints of edges (1/2, 0), (0, 1/2), (1/2, 1/2) in barycentric
    # - We create a refined grid around these
    
    if mode == 0:
        # Center
        points.append([1.0/3, 1.0/3])
        
        # Edge midpoints and nearby
        for _ in range(n_per_edge - 1):
            # Along edge u=0 (from (0,0) to (0,1))
            t = (_ + 1) / n_per_edge
            points.append([0.0 + 0.02, t])  # Slightly inside
            
            # Along edge v=0 (from (0,0) to (1,0))
            points.append([t, 0.0 + 0.02])
            
            # Along edge w=0 (from (1,0) to (0,1))
            points.append([t * (1 - t), t])
        
        # Interior grid
        for i in range(1, n_per_edge):
            for j in range(1, n_per_edge - i + 1):
                u = i / n_per_edge
                v = j / n_per_edge
                if u + v < 1.0 - 0.05:
                    points.append([u, v])
    
    elif mode == 1:
        # Radial grid from center
        center = np.array([1.0/3, 1.0/3])
        for r in [0.1, 0.2, 0.25, 0.3]:
            for k in range(6):
                theta = np.pi/3 * k
                # In barycentric, move in directions
                du = r * np.cos(theta) * 0.5
                dv = r * np.sin(theta) * 0.5
                pt = center + np.array([du, dv])
                if pt[0] > 0.01 and pt[1] > 0.01 and pt[0] + pt[1] < 0.98:
                    points.append(pt)
    
    else:
        # General grid
        for i in range(1, n_per_edge):
            for j in range(1, n_per_edge - i + 1):
                u = i / n_per_edge
                v = j / n_per_edge
                if u + v < 1.0 - 0.05:
                    points.append([u, v])
    
    return np.array(points)

def topological_persistence_score(bary):
    """
    Compute a topological persistence-based score.
    We identify the longest-lived 2D homology features and 
    return the minimum area of triangles corresponding to these features.
    
    Simplified approach: Use the areas of all triangles, weight them by 
    their "persistence" (how long they survive in a filtration), and 
    maximize the minimum weighted area.
    """
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        return -1e10
    
    # Sort areas in ascending order
    sorted_areas = np.sort(areas)
    
    # Persistence: In a sublevel set filtration of the area function,
    # a triangle is "born" when the filtration level reaches its area.
    # The "death" is when it merges with another feature.
    # For 2D homology in a triangle, all features die at the same time (the max area).
    # So persistence = max_area - area for each triangle.
    
    max_area = np.max(areas)
    persistence = max_area - areas
    
    # The longest-lived features have the smallest areas (highest persistence)
    # We want to maximize the minimum area among the most persistent features
    
    # Top k most persistent (smallest areas)
    k = max(1, len(areas) // 10)
    top_persistent = np.sort(areas)[:k]
    
    # Score: minimum of the top persistent areas, weighted by persistence
    min_persistent_area = np.min(top_persistent)
    
    # Also consider the overall minimum
    overall_min = np.min(areas)
    
    # Combined score
    score = 0.7 * min_persistent_area + 0.3 * overall_min
    
    return score

def persistence_objective(bary_flat, weight=0.5):
    """
    Objective function based on topological persistence.
    Maximizes the minimum area of the most persistent triangles.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        return 1e10
    
    sorted_areas = np.sort(areas)
    
    # Top k most persistent (smallest areas)
    k = max(1, len(areas) // 10)
    top_persistent = sorted_areas[:k]
    
    min_persistent = np.min(top_persistent)
    overall_min = np.min(areas)
    
    # Soft minimum for smoothness
    beta = 50.0
    scaled = -beta * top_persistent
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_persistent = -log_sum / beta
    
    # Combined objective
    obj = -weight * soft_min_persistent - (1 - weight) * overall_min
    
    return obj

def generate_morse_init(mode, n_per_edge=6):
    """
    Generate initial points from critical points of the harmonic function.
    """
    critical_pts = critical_points_grid(mode, n_per_edge)
    
    if len(critical_pts) < N:
        # Add more points
        rng = np.random.default_rng(42 + mode)
        while len(critical_pts) < N:
            u = rng.random()
            v = rng.random() * (1 - u)
            critical_pts = np.vstack([critical_pts, [[u, v]]])
    
    # Select N points from critical points
    # Use a greedy approach: pick points that maximize minimum distance
    selected = []
    remaining = list(range(len(critical_pts)))
    
    # Start with center if available
    center_idx = np.argmin(np.sum((critical_pts - np.array([1/3, 1/3]))**2, axis=1))
    if center_idx in remaining:
        selected.append(center_idx)
        remaining.remove(center_idx)
    
    while len(selected) < N and remaining:
        best_idx = -1
        best_min_dist = -1
        
        for idx in remaining:
            pt = critical_pts[idx]
            min_dist = np.inf
            for sel_idx in selected:
                dist = np.linalg.norm(pt - critical_pts[sel_idx])
                min_dist = min(min_dist, dist)
            
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx
        
        if best_idx >= 0:
            selected.append(best_idx)
            remaining.remove(best_idx)
    
    if len(selected) < N:
        # Fill with random points
        rng = np.random.default_rng(123 + mode)
        while len(selected) < N:
            u = rng.random()
            v = rng.random() * (1 - u)
            selected.append(len(critical_pts))
            critical_pts = np.vstack([critical_pts, [[u, v]]])
    
    bary = critical_pts[selected[:N]]
    bary = clamp_bary(bary)
    
    return bary

def refine_persistence(bary_init, time_limit=5.0):
    """
    Refine a configuration using persistence-based optimization.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    curr_val = topological_persistence_score(bary_current)
    
    # Stage 1: Soft persistence optimization
    for weight in [0.3, 0.5, 0.7]:
        if time.time() - start_time > time_limit * 0.5:
            break
        try:
            result = minimize(
                persistence_objective,
                bary_current.flatten(),
                args=(weight,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            new_bary = clamp_bary(result.x.reshape(N, 2))
            new_val = topological_persistence_score(new_bary)
            if new_val > curr_val:
                bary_current = new_bary
                curr_val = new_val
        except Exception:
            pass
    
    # Stage 2: Hard min refinement
    if time.time() - start_time < time_limit * 0.8:
        try:
            def hard_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                return -np.min(calc_areas_bary(bary))
            
            result = minimize(
                hard_obj,
                bary_current.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-18}
            )
            new_bary = clamp_bary(result.x.reshape(N, 2))
            new_val = topological_persistence_score(new_bary)
            if new_val > curr_val:
                bary_current = new_bary
                curr_val = new_val
        except Exception:
            pass
    
    # Stage 3: Nelder-Mead polish
    if time.time() - start_time < time_limit:
        try:
            def hard_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                return -np.min(calc_areas_bary(bary))
            
            result = minimize(
                hard_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 15000, 'xatol': 1e-15, 'fatol': 1e-20, 'adaptive': True}
            )
            new_bary = clamp_bary(result.x.reshape(N, 2))
            new_val = topological_persistence_score(new_bary)
            if new_val > curr_val:
                bary_current = new_bary
                curr_val = new_val
        except Exception:
            pass
    
    # Stage 4: Adaptive perturbation
    if time.time() - start_time < time_limit:
        rng = np.random.default_rng(int(time.time() * 1000) % 100000)
        step_size = 0.04
        for _ in range(80):
            if time.time() - start_time > time_limit:
                break
            improved = False
            for _ in range(8):
                perturbation = rng.normal(0, step_size, (N, 2))
                new_bary = clamp_bary(bary_current + perturbation)
                new_val = topological_persistence_score(new_bary)
                if new_val > curr_val:
                    bary_current = new_bary
                    curr_val = new_val
                    improved = True
                    break
            if not improved:
                step_size *= 0.7
            if step_size < 1e-6:
                break
    
    return bary_current

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = topological_persistence_score(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # Generate initial points from different Morse function modes
    inits = []
    
    # Mode 0: u*v*w
    for n_edge in [5, 6, 7]:
        init = generate_morse_init(0, n_edge)
        inits.append(init)
    
    # Mode 1: Radial
    for n_edge in [5, 6]:
        init = generate_morse_init(1, n_edge)
        inits.append(init)
    
    # Mode 2: Product
    for n_edge in [5, 6]:
        init = generate_morse_init(2, n_edge)
        inits.append(init)
    
    # Add some random inits
    rng = np.random.default_rng(42)
    for _ in range(10):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        inits.append(np.column_stack([u, v]))
    
    # Sort by initial quality
    inits_with_vals = [(init, topological_persistence_score(init)) for init in inits]
    inits_with_vals.sort(key=lambda x: -x[1])
    inits = [x[0] for x in inits_with_vals]
    
    # Refine each init
    for i, init in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        remaining = len(inits) - i
        if i < 3:
            base_time = 6.0
        elif i < 8:
            base_time = 4.0
        else:
            base_time = 2.5
        
        time_per_init = min(base_time, (deadline - time.time() - 10) / max(1, remaining))
        if time_per_init < 0.5:
            time_per_init = 0.5
        
        refined = refine_persistence(init, time_limit=time_per_init)
        update_best(refined)
    
    # Final polish
    if best_bary is not None and time.time() < deadline - 5:
        try:
            def hard_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                return -np.min(calc_areas_bary(bary))
            
            bounds = [(0, 1)] * (N * 2)
            result = minimize(
                hard_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 10000, 'ftol': 1e-18}
            )
            new_bary = clamp_bary(result.x.reshape(N, 2))
            update_best(new_bary)
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 2:
            try:
                def hard_obj(x):
                    bary = x.reshape(N, 2)
                    bary = clamp_bary(bary)
                    return -np.min(calc_areas_bary(bary))
                
                result = minimize(
                    hard_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-21, 'adaptive': True}
                )
                new_bary = clamp_bary(result.x.reshape(N, 2))
                update_best(new_bary)
            except Exception:
                pass
    
    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
