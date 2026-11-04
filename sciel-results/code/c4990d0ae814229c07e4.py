"""Unit regular hexagons inside a regular hexagon n=12 - DE optimizer with SAT collision."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 12
START_TIME = time.time()
DEADLINE = 130.0  # Safety margin before 150s kill

def hex_vertices(cx, cy, side, angle_deg):
    """Get vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3.0
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def sat_overlap(v1, v2):
    """Check if two convex polygons overlap using Separating Axis Theorem.
    Returns True if they overlap (collision), False if separated."""
    axes = []
    for poly in [v1, v2]:
        for i in range(len(poly)):
            edge = poly[(i + 1) % len(poly)] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                axes.append(normal / norm)
    
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 or max2 < min1:
            return False  # Separating axis found, no overlap
    return True  # No separating axis, polygons overlap

def min_enclosing_hexagon_side(inner_hexes):
    """Compute minimum side length of outer hexagon that contains all inner hexagons.
    inner_hexes: list of (cx, cy, angle_deg) for unit hexagons.
    We find the smallest regular hexagon (any rotation) that encloses all vertices."""
    # Collect all vertices of all inner hexagons
    all_verts = []
    for cx, cy, ang in inner_hexes:
        verts = hex_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    
    all_verts = np.vstack(all_verts)
    
    # Try different outer hexagon rotations and centers
    best_side = float('inf')
    # The optimal enclosing hexagon center should be near the centroid
    centroid = all_verts.mean(axis=0)
    
    # Try a range of outer hexagon rotations
    for outer_angle_deg in np.arange(0, 60, 3.0):
        outer_angle = np.radians(outer_angle_deg)
        # Project all vertices onto the 6 normal directions of the outer hexagon
        # For a regular hexagon with side s centered at (ox, oy) rotated by theta,
        # the constraint is that for each of the 6 face normals, the max projection
        # of inner points minus center must be <= s * sqrt(3)/2 (the apothem)
        
        # Actually, for a regular hexagon with side length s:
        # - Apothem (inradius) = s * sqrt(3) / 2
        # - The 6 face normals point at angles: outer_angle + i*60deg for i=0..5
        # - For each normal direction, max_proj - min_proj <= s * sqrt(3)
        #   Wait, that's not quite right. Let me think more carefully.
        
        # A regular hexagon with side s has:
        # - 6 vertices at distance s from center
        # - 6 faces at distance s*sqrt(3)/2 from center (apothem)
        # - The hexagon can be described as intersection of 6 half-planes
        #   n_i . (p - c) <= s*sqrt(3)/2 for i=0..5
        #   where n_i are unit normals pointing outward
        
        # For the hexagon to contain all points:
        #   max_i (n_i . (p - c)) <= s*sqrt(3)/2 for all points p
        # So s >= max_i (n_i . (p - c)) / (sqrt(3)/2) for all p
        # Minimum s for a given center c and rotation = max over all p and i of (n_i . (p-c)) / (sqrt(3)/2)
        
        apothem_factor = np.sqrt(3) / 2.0
        
        # Try different center positions around the centroid
        for dx in np.linspace(-2, 2, 5):
            for dy in np.linspace(-2, 2, 5):
                center = centroid + np.array([dx, dy])
                # Compute required side length
                max_proj = 0.0
                for i in range(6):
                    theta = outer_angle + i * np.pi / 3.0
                    n = np.array([np.cos(theta), np.sin(theta)])
                    proj = np.dot(all_verts - center, n)
                    max_proj = max(max_proj, proj.max())
                
                if max_proj < 0:
                    continue
                side = max_proj / apothem_factor
                if side < best_side:
                    best_side = side
    
    return best_side

def objective(params):
    """Objective function for DE: minimize outer hexagon side length."""
    if time.time() > DEADLINE:
        return 1e10
    
    # Unpack parameters: 12 hexagons * 3 params (x, y, angle)
    inner_hexes = []
    for i in range(N):
        x = params[i * 3]
        y = params[i * 3 + 1]
        ang = params[i * 3 + 2]
        inner_hexes.append((x, y, ang))
    
    # Check for collisions between inner hexagons
    for i in range(N):
        for j in range(i + 1, N):
            v1 = hex_vertices(inner_hexes[i][0], inner_hexes[i][1], 1.0, inner_hexes[i][2])
            v2 = hex_vertices(inner_hexes[j][0], inner_hexes[j][1], 1.0, inner_hexes[j][2])
            if sat_overlap(v1, v2):
                # Penalize heavily for overlap
                return 100.0 + min_enclosing_hexagon_side(inner_hexes)
    
    return min_enclosing_hexagon_side(inner_hexes)

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Define bounds for each parameter
    # x, y: roughly in [-10, 10]
    # angle: [0, 60] degrees (hexagons have 60-degree symmetry)
    bounds = []
    for i in range(N):
        bounds.append((-10.0, 10.0))   # x
        bounds.append((-10.0, 10.0))   # y
        bounds.append((0.0, 60.0))     # angle
    
    # Initial best solution: linear chain (from seed)
    best_params = np.array([4.0 * k for k in range(-(N // 2), N - N // 2)] * 3)
    best_params[2::3] = 0.0  # angles
    best_params[1::3] = 0.0  # y positions
    best_score = 40.0
    
    try:
        # Run Differential Evolution
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.8,
            polish=False,
            init='sobol'
        )
        
        if result.fun < best_score:
            best_params = result.x
            best_score = result.fun
        
        # Polish with Nelder-Mead
        if time.time() < DEADLINE - 20:
            res2 = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res2.fun < best_score:
                best_params = res2.x
                best_score = res2.fun
        
    except Exception:
        pass
    
    # Extract final solution
    inner = []
    for i in range(N):
        x = best_params[i * 3]
        y = best_params[i * 3 + 1]
        ang = best_params[i * 3 + 2] % 60.0  # Normalize angle
        inner.append([x, y, ang])
    
    # Compute the actual outer hexagon parameters
    # Find optimal outer hexagon center and rotation
    inner_tuples = [(p[0], p[1], p[2]) for p in inner]
    
    all_verts = []
    for cx, cy, ang in inner_tuples:
        verts = hex_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    centroid = all_verts.mean(axis=0)
    
    best_side = float('inf')
    best_outer_angle = 0.0
    best_center = (0.0, 0.0)
    
    for outer_angle_deg in np.arange(0, 60, 1.0):
        outer_angle = np.radians(outer_angle_deg)
        apothem_factor = np.sqrt(3) / 2.0
        
        for dx in np.linspace(-3, 3, 7):
            for dy in np.linspace(-3, 3, 7):
                center = centroid + np.array([dx, dy])
                max_proj = 0.0
                for i in range(6):
                    theta = outer_angle + i * np.pi / 3.0
                    n = np.array([np.cos(theta), np.sin(theta)])
                    proj = np.dot(all_verts - center, n)
                    max_proj = max(max_proj, proj.max())
                
                if max_proj < 0:
                    continue
                side = max_proj / apothem_factor
                if side < best_side:
                    best_side = side
                    best_outer_angle = outer_angle_deg
                    best_center = (center[0], center[1])
    
    if best_side == float('inf'):
        best_side = 40.0
    
    return inner, best_center, best_side, best_outer_angle

# EVOLVE-BLOCK-START
# (construct function defined above)
# EVOLVE-BLOCK-END
