"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2  # area of unit equilateral triangle


def point_in_triangle(p, eps=1e-9):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, V0, V1)
    d2 = sign(p, V1, V2)
    d3 = sign(p, V2, V0)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)


def min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) -
                                 (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                if area < min_area:
                    min_area = area
    return min_area


def negative_min_area_flat(x):
    """Objective for Nelder-Mead: negative of minimum triangle area, with penalty for out-of-triangle points."""
    n = N
    points = x.reshape(n, 2)
    # Penalty for points outside triangle
    penalty = 0.0
    for i in range(n):
        if not point_in_triangle(points[i]):
            penalty += 100.0 * (1.0 + 1.0)  # large penalty
    min_area = min_triangle_area(points)
    return -(min_area / TRI_AREA) + penalty


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    # Generate diverse starting points using hexagonal/structured patterns
    def generate_starts():
        starts = []
        
        # Pattern 1: Hexagonal grid within triangle
        for scale in [0.3, 0.4, 0.5]:
            pts = []
            # Center
            cx, cy = 0.5, np.sqrt(3)/6
            pts.append([cx, cy])
            # Ring 1
            for ang in np.linspace(0, 2*np.pi, 6, endpoint=False):
                px = cx + scale * np.cos(ang)
                py = cy + scale * np.sin(ang)
                if point_in_triangle([px, py]):
                    pts.append([px, py])
            # Add more points if needed
            rng = np.random.RandomState(len(starts))
            while len(pts) < N:
                u, v = rng.random(), rng.random()
                if u + v < 1.0 and u > 0.02 and v > 0.02:
                    px = u + 0.5 * v
                    py = np.sqrt(3)/2 * v
                    pts.append([px, py])
            if len(pts) >= N:
                starts.append(np.array(pts[:N]))
        
        # Pattern 2: Vertices + midpoints + interior
        base_pts = [
            [0.1, 0.1], [0.9, 0.1], [0.5, 0.4],
            [0.2, 0.3], [0.8, 0.3], [0.5, 0.15],
            [0.3, 0.5], [0.7, 0.5], [0.5, 0.6],
            [0.4, 0.35], [0.6, 0.35]
        ]
        # Filter to those inside triangle
        filtered = [p for p in base_pts if point_in_triangle(p)]
        if len(filtered) >= N:
            starts.append(np.array(filtered[:N]))
        
        # Pattern 3: Random with fixed seed
        for seed in [42, 123, 456, 789]:
            rng = np.random.RandomState(seed)
            pts = []
            while len(pts) < N:
                u, v = rng.random(), rng.random()
                if u + v < 1.0 and u > 0.02 and v > 0.02:
                    px = u + 0.5 * v
                    py = np.sqrt(3)/2 * v
                    pts.append([px, py])
            starts.append(np.array(pts[:N]))
        
        return starts
    
    starts = generate_starts()
    
    best_x = None
    best_score = -float('inf')
    
    for s_idx, x0 in enumerate(starts):
        if time.time() > deadline:
            break
        
        # Flatten for Nelder-Mead
        x0_flat = x0.flatten()
        
        try:
            result = minimize(
                negative_min_area_flat,
                x0_flat,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
        except Exception:
            continue
        
        x_opt = result.x.reshape(N, 2)
        
        # Check all points are inside triangle
        valid = all(point_in_triangle(x_opt[i]) for i in range(N))
        
        if valid:
            score = min_triangle_area(x_opt) / TRI_AREA
            if score > best_score:
                best_score = score
                best_x = x_opt
        
        # Also check the initial point
        init_score = min_triangle_area(x0) / TRI_AREA
        if init_score > best_score:
            best_score = init_score
            best_x = x0
    
    if best_x is None:
        # Fallback: return a valid configuration
        best_x = starts[0]
    
    return best_x
# EVOLVE-BLOCK-END
