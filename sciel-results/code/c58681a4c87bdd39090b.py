"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
import math
from scipy.optimize import minimize_scalar, linprog

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def generate_polyhexes(n):
    """Generate all fixed polyhexes of size n."""
    polyhexes = []
    seen = set()
    
    def canonicalize(coords):
        # Normalize by translating so that min q is 0 and min r is 0
        min_q = min(q for q, r in coords)
        min_r = min(r for q, r in coords)
        norm_coords = sorted([(q - min_q, r - min_r) for q, r in coords])
        return tuple(norm_coords)
        
    # Axial directions
    dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    
    def backtrack(current, frontier):
        if len(current) == n:
            key = canonicalize(current)
            if key not in seen:
                seen.add(key)
                polyhexes.append(list(current))
            return
        
        # Sort frontier to ensure deterministic order and pruning
        # We only add hexes that are "new"
        for pos in frontier:
            if pos in current:
                continue
            
            new_current = current + [pos]
            new_frontier = []
            for d in dirs:
                npos = (pos[0] + d[0], pos[1] + d[1])
                if npos not in new_current:
                    new_frontier.append(npos)
            
            new_frontier = list(set(new_frontier))
            
            backtrack(new_current, new_frontier)
            
    backtrack([(0, 0)], [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)])
    return polyhexes

def axial_to_cartesian(q, r, scale=1.0):
    x = 1.5 * scale * q
    y = SQRT3 * scale * (r + q / 2.0)
    return (x, y)

def get_hex_vertices(cx, cy, side, angle_rad):
    angles = angle_rad + np.pi / 3.0 * np.arange(6)
    return [(cx + side * math.cos(a), cy + side * math.sin(a)) for a in angles]

def solve_lp_for_hexagon(vertices, theta):
    """
    Solve LP to find min side S of regular hexagon with angle theta containing vertices.
    Returns (S, cx, cy)
    """
    normals = []
    for i in range(6):
        angle = theta + math.pi/6 + i * math.pi/3
        normals.append((math.cos(angle), math.sin(angle)))
    
    A_ub = []
    b_ub = []
    
    for v in vertices:
        vx, vy = v
        for nx, ny in normals:
            A_ub.append([-nx, -ny, -1])
            b_ub.append(-vx * nx - vy * ny)
            
    A_ub.append([0, 0, -1])
    b_ub.append(0)
    
    res = linprog(c=[0, 0, 1], A_ub=A_ub, b_ub=b_ub, method='highs')
    
    if res.status == 0:
        R = res.fun
        S = 2.0 * R / SQRT3
        cx = res.x[0]
        cy = res.x[1]
        return S, cx, cy
    else:
        return float('inf'), 0, 0

def optimize_theta(vertices):
    """Find theta that minimizes the enclosing hexagon side."""
    def objective(theta):
        S, _, _ = solve_lp_for_hexagon(vertices, theta)
        return S
    
    res = minimize_scalar(objective, bounds=(0, math.pi/3), method='bounded', options={'xatol': 1e-6})
    return res.fun, res.x

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    polyhexes = generate_polyhexes(N)
    
    best_S = float('inf')
    best_config = None
    
    for polyhex in polyhexes:
        if time.time() > deadline - 10:
            break
            
        centers = [axial_to_cartesian(q, r, 1.0) for q, r in polyhex]
        
        vertices = []
        for cx, cy in centers:
            verts = get_hex_vertices(cx, cy, 1.0, 0.0)
            vertices.extend(verts)
        
        S, theta = optimize_theta(vertices)
        
        if S < best_S:
            # Get exact center
            S, cx, cy = solve_lp_for_hexagon(vertices, theta)
            
            inner = [(px, py, 0.0) for px, py in centers]
            best_config = (inner, (cx, cy), S, math.degrees(theta))
            best_S = S
            
    if best_config is None:
        # Fallback
        axial_coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, -1), (1, 1), (-2, 1), (-1, -1)
        ]
        centers = [axial_to_cartesian(q, r, 1.0) for q, r in axial_coords]
        inner = [(px, py, 0.0) for px, py in centers]
        S = 6.0 # Guess
        best_config = (inner, (0.0, 0.0), S, 0.0)
        
    return best_config

# EVOLVE-BLOCK-END
