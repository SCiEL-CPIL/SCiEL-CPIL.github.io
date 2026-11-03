"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools
import random

N = 13

# EVOLVE-BLOCK-START

def _hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 1e-10

def _all_triangle_areas(points):
    """Compute all triangle areas and return min area and the index of the min triangle."""
    n = len(points)
    min_area = float('inf')
    min_idx = None
    tri_count = 0
    for i, j, k in itertools.combinations(range(n), 3):
        p1, p2, p3 = points[i], points[j], points[k]
        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        if area < min_area:
            min_area = area
            min_idx = (i, j, k)
        tri_count += 1
    return min_area, min_idx

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_tri, _ = _all_triangle_areas(points)
    return min_tri / hull_area

def _soft_min_area(points, temperature=0.01):
    """Soft minimum of triangle areas using log-sum-exp for smoothness."""
    n = len(points)
    areas = []
    for i, j, k in itertools.combinations(range(n), 3):
        p1, p2, p3 = points[i], points[j], points[k]
        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        areas.append(area)
    areas = np.array(areas)
    # Soft min: -logsumexp(-areas/T) * T
    max_neg = np.max(-areas / temperature)
    soft_min = -temperature * (max_neg + np.log(np.sum(np.exp(-areas / temperature - max_neg))))
    return soft_min

def _neg_soft_score(points_flat, temperature=0.01):
    """Negative soft score for optimization."""
    points = points_flat.reshape(N, 2)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    soft_min = _soft_min_area(points, temperature)
    score = soft_min / hull_area
    return -score

def _neg_score(points_flat):
    """Negative hard score for optimization."""
    points = points_flat.reshape(N, 2)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    min_tri, _ = _all_triangle_areas(points)
    score = min_tri / hull_area
    return -score

def _concentric_init(n_rings, seed):
    """Initialize points on concentric circles."""
    rng = np.random.RandomState(seed)
    points = []
    
    # Distribute points among rings
    ring_sizes = [n_rings]  # innermost
    remaining = N - n_rings
    if remaining > 0:
        # Split remaining among outer rings
        outer_rings = 1
        per_ring = remaining // outer_rings
        ring_sizes.append(per_ring)
    
    radius = 0.0
    for ring_size in ring_sizes:
        if ring_size == 0:
            continue
        radius += 0.5
        angles = np.linspace(0, 2 * np.pi, ring_size, endpoint=False) + rng.uniform(0, 2 * np.pi, size=1)[0]
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        points.append(np.column_stack((x, y)))
    
    if len(points) == 0:
        # Fallback: single ring
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        points = [np.column_stack((np.cos(angles), np.sin(angles)))]
    
    result = np.vstack(points)
    # Center
    result -= np.mean(result, axis=0)
    return result

def _generate_concentric_initializations(seed):
    """Generate multiple concentric circle initializations with different ring structures."""
    inits = []
    rng = random.Random(seed)
    
    # Try different numbers of rings
    for n_rings in [1, 2, 3]:
        if n_rings == 1:
            # All on one circle
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(0, 0.2)
            r = 1.0 + rng.uniform(-0.1, 0.1)
            x = r * np.cos(angles)
            y = r * np.sin(angles)
            pts = np.column_stack((x, y))
            pts -= np.mean(pts, axis=0)
            inits.append(pts)
        elif n_rings == 2:
            # 4 inner, 9 outer (or similar split)
            inner_n = rng.randint(3, 5)
            outer_n = N - inner_n
            inner_angles = np.linspace(0, 2 * np.pi, inner_n, endpoint=False) + rng.uniform(0, 2 * np.pi)
            outer_angles = np.linspace(0, 2 * np.pi, outer_n, endpoint=False) + rng.uniform(0, 2 * np.pi)
            inner_r = 0.4 + rng.uniform(-0.05, 0.05)
            outer_r = 1.0 + rng.uniform(-0.05, 0.05)
            x = np.concatenate([inner_r * np.cos(inner_angles), outer_r * np.cos(outer_angles)])
            y = np.concatenate([inner_r * np.sin(inner_angles), outer_r * np.sin(outer_angles)])
            pts = np.column_stack((x, y))
            pts -= np.mean(pts, axis=0)
            inits.append(pts)
        elif n_rings == 3:
            # 3 inner, 4 middle, 6 outer
            sizes = [rng.randint(2, 4), rng.randint(3, 5), 0]
            sizes[2] = N - sizes[0] - sizes[1]
            if sizes[2] < 2:
                sizes = [3, 4, 6]
            radii = [0.35, 0.65, 1.0]
            x_all, y_all = [], []
            for s, r in zip(sizes, radii):
                if s <= 0:
                    continue
                angles = np.linspace(0, 2 * np.pi, s, endpoint=False) + rng.uniform(0, 2 * np.pi)
                r_val = r + rng.uniform(-0.05, 0.05)
                x_all.append(r_val * np.cos(angles))
                y_all.append(r_val * np.sin(angles))
            x = np.concatenate(x_all)
            y = np.concatenate(y_all)
            pts = np.column_stack((x, y))
            pts -= np.mean(pts, axis=0)
            inits.append(pts)
    
    # Also add Fermat spiral
    i = np.arange(N)
    r = np.sqrt(i / N)
    theta = i * 2.399963229728653 + rng.random() * 2 * np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.column_stack((x, y))
    pts -= np.mean(pts, axis=0)
    inits.append(pts)
    
    return inits

def _targeted_worst_triangle_opt(points, n_steps=50, step_size=0.05, seed=42):
    """Targeted optimization: perturb points in the worst (minimum area) triangle."""
    rng = np.random.RandomState(seed)
    points = points.copy()
    current_score = _score(points)
    
    for step in range(n_steps):
        min_area, min_idx = _all_triangle_areas(points)
        if min_idx is None:
            break
        i, j, k = min_idx
        p1, p2, p3 = points[i], points[j], points[k]
        
        # Try small random perturbations to each of the 3 points
        best_perturb = None
        best_new_score = current_score
        
        for _ in range(10):
            new_points = points.copy()
            # Perturb each of the 3 points
            for idx in [i, j, k]:
                delta = rng.normal(0, step_size, size=2)
                new_points[idx] += delta
            
            new_score = _score(new_points)
            if new_score > best_new_score:
                best_new_score = new_score
                best_perturb = new_points.copy()
        
        if best_perturb is not None:
            points = best_perturb
            current_score = best_new_score
        else:
            # Decrease step size
            step_size *= 0.7
            if step_size < 1e-4:
                break
    
    return points, current_score

def _refine_with_soft_min(points, seed=42):
    """Refine points using soft minimum objective with annealing."""
    rng = np.random.RandomState(seed)
    points = points.copy()
    current_score = _score(points)
    
    # Anneal temperature
    temperatures = [0.05, 0.02, 0.01, 0.005, 0.001]
    
    for temp in temperatures:
        def obj(flat):
            return _neg_soft_score(flat, temp)
        
        try:
            result = minimize(
                obj,
                points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if result.fun < 0:
                new_points = result.x.reshape(N, 2)
                new_score = _score(new_points)
                if new_score > current_score:
                    points = new_points
                    current_score = new_score
        except Exception:
            continue
    
    return points, current_score

def _refine_with_nelder_mead(points, seed=42):
    """Refine points using Nelder-Mead on hard score."""
    try:
        result = minimize(
            _neg_score,
            points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9}
        )
        if result.fun < 0:
            new_points = result.x.reshape(N, 2)
            new_score = _score(new_points)
            return new_points, new_score
    except Exception:
        pass
    
    return points, _score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 150s limit, 5s safety margin
    
    best_points = None
    best_score = 0.0
    
    # Generate diverse initializations
    inits = _generate_concentric_initializations(seed=12345)
    
    # Process each initialization
    for init_idx, init_points in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        # Phase 1: Soft min refinement
        if time.time() > deadline - 15:
            break
        refined, score = _refine_with_soft_min(init_points, seed=init_idx + 1)
        if score > best_score:
            best_score = score
            best_points = refined.copy()
        
        # Phase 2: Targeted worst triangle optimization
        if time.time() > deadline - 15:
            break
        targeted, score2 = _targeted_worst_triangle_opt(refined, n_steps=30, step_size=0.03, seed=init_idx + 100)
        if score2 > best_score:
            best_score = score2
            best_points = targeted.copy()
        
        # Phase 3: Final Nelder-Mead refinement
        if time.time() > deadline - 10:
            break
        final, score3 = _refine_with_nelder_mead(targeted, seed=init_idx + 200)
        if score3 > best_score:
            best_score = score3
            best_points = final.copy()
    
    # If we have time, do additional rounds with different seeds
    extra_rounds = 0
    while time.time() < deadline - 20 and extra_rounds < 5:
        extra_seed = 1000 + extra_rounds * 100
        extra_inits = _generate_concentric_initializations(seed=extra_seed)
        for init_idx, init_points in enumerate(extra_inits[:2]):  # Only first 2 to save time
            if time.time() > deadline - 15:
                break
            
            refined, score = _refine_with_soft_min(init_points, seed=extra_seed + init_idx)
            if score > best_score:
                best_score = score
                best_points = refined.copy()
            
            if time.time() > deadline - 10:
                break
            targeted, score2 = _targeted_worst_triangle_opt(refined, n_steps=20, step_size=0.02, seed=extra_seed + 100 + init_idx)
            if score2 > best_score:
                best_score = score2
                best_points = targeted.copy()
        
        extra_rounds += 1
    
    if best_points is None:
        # Fallback: Fermat spiral
        i = np.arange(N)
        r = np.sqrt(i / N)
        theta = i * 2.399963229728653
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        best_points = np.column_stack((x, y))
        best_points -= np.mean(best_points, axis=0)
    
    return best_points

# EVOLVE-BLOCK-END
