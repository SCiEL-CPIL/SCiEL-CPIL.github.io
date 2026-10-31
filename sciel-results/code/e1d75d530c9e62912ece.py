"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

# EVOLVE-BLOCK-START

# The equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
SQRT3 = np.sqrt(3)

def _min_area_vectorized(pts):
    """Compute the minimum area of any triangle formed by three of the points, vectorized."""
    n = len(pts)
    # pts shape: (n, 2)
    # For each triple (i,j,k), area = 0.5 * |cross(pts[j]-pts[i], pts[k]-pts[i])|
    # Generate all index triples
    idx = np.array(list(combinations(range(n), 3)))
    # idx shape: (165, 3)
    a = pts[idx[:, 0]]  # (165, 2)
    b = pts[idx[:, 1]]  # (165, 2)
    c = pts[idx[:, 2]]  # (165, 2)
    # Cross product in 2D: (bx-ax)*(cy-ay) - (by-ay)*(cx-ax)
    ab = b - a
    ac = c - a
    cross = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle using barycentric coordinates."""
    x, y = p
    c = 2 * y / SQRT3
    b = x - 0.5 * c
    a = 1 - b - c
    
    # Project to simplex: a,b,c >= 0, a+b+c=1
    a = max(0.0, a)
    b = max(0.0, b)
    c = max(0.0, c)
    s = a + b + c
    if s < 1e-12:
        return (0.5, SQRT3 / 6)
    a /= s
    b /= s
    c /= s
    
    return (b + 0.5 * c, SQRT3 / 2 * c)

def _clamp_all(pts):
    """Clamp all points to triangle."""
    return np.array([_clamp_to_triangle(p) for p in pts])

def _barycentric_to_cartesian(bary):
    """Convert barycentric coords (a,b,c) with a+b+c=1 to Cartesian."""
    # p = a*V0 + b*V1 + c*V2 = (b + 0.5*c, SQRT3/2 * c)
    a, b, c = bary
    return np.array([b + 0.5 * c, SQRT3 / 2 * c])

def _cartesian_to_barycentric(pts):
    """Convert Cartesian points to barycentric coordinates (a,b,c)."""
    bary = np.zeros((len(pts), 3))
    for i, (x, y) in enumerate(pts):
        c = 2 * y / SQRT3
        b = x - 0.5 * c
        a = 1 - b - c
        bary[i] = [a, b, c]
    return bary

def _min_area_bary(bary_pts):
    """Compute min area from barycentric points, vectorized."""
    pts = np.array([_barycentric_to_cartesian(b) for b in bary_pts])
    return _min_area_vectorized(pts)

def _smooth_penalty(x, pts, margin=0.01):
    """
    Smooth penalty for points outside the triangle.
    x: flat array of 2*N coordinates
    pts: (N,2) array
    Returns penalty value (0 if all inside).
    """
    # Compute barycentric coordinates
    bary = _cartesian_to_barycentric(pts)
    # Penalty for negative barycentric coords
    neg = np.clip(-bary, 0, None)
    penalty = np.sum(neg**2)
    return penalty

def _objective_de(x):
    """Objective for differential evolution: minimize negative min area + penalty."""
    pts = x.reshape(N, 2)
    pts_c = _clamp_all(pts)
    min_a = _min_area_vectorized(pts_c)
    # Small penalty for being outside (after clamping this should be 0, but DE works in unconstrained space)
    bary = _cartesian_to_barycentric(pts)
    neg = np.clip(-bary, 0, None)
    penalty = 10.0 * np.sum(neg**2)
    return -min_a + penalty

def _initial_placement():
    """Create a structured initial placement."""
    pts = []
    # Row 1 (bottom): 4 points
    y1 = 0.1
    for i in range(4):
        x = 0.1 + i * 0.25
        pts.append([x, y1])
    # Row 2 (middle): 4 points  
    y2 = 0.35
    for i in range(4):
        x = 0.2 + i * 0.2
        pts.append([x, y2])
    # Row 3 (top): 3 points
    y3 = 0.6
    for i in range(3):
        x = 0.3 + i * 0.2
        pts.append([x, y3])
    return np.array(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -1
    
    # Strategy 1: Differential Evolution for global search
    # Bounds: x in [0, 1], y in [0, sqrt(3)/2]
    bounds = [(0, 1), (0, SQRT3 / 2)] * N
    
    try:
        de_result = differential_evolution(
            _objective_de,
            bounds=bounds,
            seed=42,
            maxiter=50,
            tol=1e-6,
            popsize=15,
            mutation=(0.3, 0.8),
            recombination=0.8,
            polish=False,  # We'll do our own polish
            callback=lambda xk, convergence: None
        )
        
        final_pts = de_result.x.reshape(N, 2)
        final_pts = _clamp_all(final_pts)
        score = _min_area_vectorized(final_pts)
        
        if score > best_score:
            best_score = score
            best_points = final_pts.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30:
        if best_points is not None:
            return best_points
        return _initial_placement()
    
    # Strategy 2: Local refinement using Nelder-Mead on the best DE result
    if best_points is not None:
        def objective_nm(x):
            pts = x.reshape(N, 2)
            pts_c = _clamp_all(pts)
            min_a = _min_area_vectorized(pts_c)
            return -min_a
        
        try:
            res = minimize(
                objective_nm,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            final_pts = res.x.reshape(N, 2)
            final_pts = _clamp_all(final_pts)
            score = _min_area_vectorized(final_pts)
            if score > best_score:
                best_score = score
                best_points = final_pts.copy()
        except Exception:
            pass
    
    # Check time
    if time.time() > deadline - 15:
        if best_points is not None:
            return best_points
        return _initial_placement()
    
    # Strategy 3: Multi-start Nelder-Mead with random restarts
    n_restarts = 5
    for i in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # Generate random points inside triangle
        pts = []
        for _ in range(N):
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            x = u + 0.5 * v
            y = v * SQRT3 / 2
            pts.append([x, y])
        pts = np.array(pts)
        
        def objective_nm(x):
            pts_x = x.reshape(N, 2)
            pts_c = _clamp_all(pts_x)
            min_a = _min_area_vectorized(pts_c)
            return -min_a
        
        try:
            res = minimize(
                objective_nm,
                pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            final_pts = res.x.reshape(N, 2)
            final_pts = _clamp_all(final_pts)
            score = _min_area_vectorized(final_pts)
            if score > best_score:
                best_score = score
                best_points = final_pts.copy()
        except Exception:
            pass
    
    if best_points is None:
        return _initial_placement()
    
    # Strategy 4: Final local polish with small steps
    if time.time() < deadline - 2:
        current_best = best_points.copy()
        current_score = best_score
        T = 0.005
        T_min = 1e-6
        cooling_rate = 0.998
        
        while time.time() < deadline - 1 and T > T_min:
            idx = rng.integers(0, N)
            delta = rng.normal(0, T, 2)
            new_pts = current_best.copy()
            new_pts[idx] = np.array(_clamp_to_triangle(current_best[idx] + delta))
            
            new_score = _min_area_vectorized(new_pts)
            
            if new_score > current_score:
                current_best = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_best.copy()
            else:
                try:
                    prob = np.exp((new_score - current_score) / max(T, 1e-10))
                except OverflowError:
                    prob = 0.0
                if rng.random() < prob:
                    current_best = new_pts
                    current_score = new_score
            
            T *= cooling_rate
    
    return best_points

# EVOLVE-BLOCK-END
