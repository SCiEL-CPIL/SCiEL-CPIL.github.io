# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize
from itertools import combinations

def calculate_score(points):
    """
    Calculate the metric R = (d_min / d_max)^2
    """
    if len(points) < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def objective_function(flat_points):
    """
    Objective function for minimization: minimize d_max / d_min
    which is equivalent to maximizing d_min / d_max
    """
    points = flat_points.reshape(-1, 2)
    dists = spd.pdist(points)
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_min == 0:
        return 1e10
    return d_max / d_min

def generate_hex_lattice(n_points=19):
    """
    Generate a hexagonal lattice of points.
    For 19 points, it's a hexagon of side 2 (radius 2).
    """
    points = []
    # Center
    points.append([0, 0])
    # Ring 1
    for i in range(6):
        angle = i * np.pi / 3
        points.append([np.cos(angle), np.sin(angle)])
    # Ring 2
    for i in range(12):
        # Hex lattice coordinates
        # Directions: 0, 60, 120, 180, 240, 300
        # Ring 2 points are at distance 2 from center in hex grid steps
        # We can generate them by moving 2 steps in each of the 6 directions and filling the gaps
        # Easier: iterate over hex coordinates (q, r) with |q|+|r|+|q+r| <= 2 * 2? No.
        # Hex distance is max(|q|, |r|, |q+r|) in axial coords?
        # Let's just use the known coordinates for a hexagon of radius 2 (distance 2 from center).
        pass
    
    # Simpler generation for hexagon of side k=2 (19 points)
    # Axial coordinates (q, r)
    # Point is in hex if max(|q|, |r|, |q+r|) <= k
    k = 2
    pts = []
    for q in range(-k, k + 1):
        for r in range(-k, k + 1):
            if abs(q + r) <= k:
                # Convert axial to cartesian
                x = 1.5 * q
                y = np.sqrt(3) / 2 * (q + 2 * r)
                pts.append([x, y])
    return np.array(pts)

def local_search_optimizer(points, time_limit):
    """
    Use Nelder-Mead to refine the points.
    """
    start_time = time.time()
    initial_flat = points.flatten()
    
    # Nelder-Mead parameters
    # We want to minimize d_max / d_min
    res = None
    try:
        res = minimize(
            objective_function, 
            initial_flat, 
            method='Nelder-Mead', 
            options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 10000, 'maxfev': 10000, 'adaptive': True}
        )
    except Exception:
        pass
        
    if res and res.success:
        best_points = res.x.reshape(-1, 2)
        return best_points
    
    return points

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    deadline = time.time() + 590
    
    # 1. Generate 19-point hexagonal lattice
    hex19 = generate_hex_lattice(19)
    
    best_score = -1
    best_points = None
    
    # 2. Try removing 3 points from the 19-point hexagon
    indices = list(range(19))
    # There are 969 combinations, this is fast
    for combo in combinations(indices, 3):
        remove_set = set(combo)
        keep_indices = [i for i in indices if i not in remove_set]
        candidate = hex19[keep_indices]
        score = calculate_score(candidate)
        if score > best_score:
            best_score = score
            best_points = candidate.copy()
            
    # 3. Refine the best candidate with Nelder-Mead
    if time.time() < deadline - 10:
        refined_points = local_search_optimizer(best_points, deadline - time.time() - 10)
        refined_score = calculate_score(refined_points)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points

    # 4. Fallback/Alternative: Try the current program's SA on a grid if time permits
    # But the hexagon approach is likely superior. 
    # Let's add a quick SA polish on the best found points just in case NM got stuck.
    if time.time() < deadline - 20:
        # Implement a quick SA polish
        current_points = best_points.copy()
        current_score = best_score
        temp = 0.01
        cooling = 0.99
        step = 0.1 * np.std(current_points)
        
        while time.time() < deadline - 20:
            idx = np.random.randint(16)
            delta = np.random.randn(2) * step
            new_points = current_points.copy()
            new_points[idx] += delta
            new_score = calculate_score(new_points)
            
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            else:
                if temp > 1e-6:
                    prob = np.exp((new_score - current_score) / temp)
                    if np.random.rand() < prob:
                        current_points = new_points
                        current_score = new_score
            
            temp *= cooling
            step *= 0.999

    return best_points

# EVOLVE-BLOCK-END
