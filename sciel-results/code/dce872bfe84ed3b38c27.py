"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side=1.0, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = math.radians(angle_deg)
    # Regular hexagon: vertices at angles a + k*pi/3
    verts = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return np.array(verts)

def _support_function(verts, direction):
    """Max dot product of vertices with direction vector."""
    return np.max(verts @ direction)

def _outer_side_needed(inner_hexes, outer_center, outer_angle_deg):
    """
    Given inner hexagons (list of (x,y,angle_deg)) and outer hexagon center/angle,
    compute the minimum side length of the outer hexagon that contains all inner hexagons.
    
    The outer hexagon has 6 face normals. For a regular hexagon with side s and rotation phi,
    the face normals are at angles phi + k*pi/3 + pi/6 (i.e., perpendicular to edges).
    Actually, let's think: a regular hexagon with vertices at angles theta_k = phi + k*pi/3.
    The edges are between consecutive vertices. The outward normal to edge between vertex k and k+1
    is at angle phi + (k+0.5)*pi/3.
    
    The support function of the outer hexagon in direction n (unit) is s * (something).
    For a regular hexagon with side length s, the distance from center to each face (apothem) is s * sqrt(3)/2.
    The support function in the direction of a face normal is the apothem = s * sqrt(3)/2.
    In other directions, it's more complex.
    
    Actually, the outer hexagon contains a point p if for all 6 face normals n_i, 
    (p - center) . n_i <= apothem = s * sqrt(3)/2.
    
    So for each inner hexagon vertex v, we need (v - outer_center) . n_i <= s * sqrt(3)/2 for all i.
    Thus s >= max over all vertices v and all face normals n_i of (v - outer_center) . n_i / (sqrt(3)/2).
    """
    outer_a = math.radians(outer_angle_deg)
    # Face normals: perpendicular to edges. Edge between vertex k and k+1 has normal at angle outer_a + (k+0.5)*pi/3
    normals = []
    for k in range(6):
        theta_n = outer_a + (k + 0.5) * math.pi / 3.0
        normals.append(np.array([math.cos(theta_n), math.sin(theta_n)]))
    normals = np.array(normals)  # shape (6, 2)
    
    outer_c = np.array(outer_center)
    apothem_factor = math.sqrt(3) / 2.0
    
    max_proj = 0.0
    for (ix, iy, ia) in inner_hexes:
        verts = _hex_vertices(ix, iy, 1.0, ia)  # shape (6, 2)
        # (verts - outer_c) @ normals.T -> shape (6, 6)
        projs = (verts - outer_c) @ normals.T
        max_proj = max(max_proj, np.max(projs))
    
    s = max_proj / apothem_factor
    return s

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin
    
    # Initial guess: hexagonal arrangement of 11 hexagons
    # 1 in center, 6 in first ring, 4 more? Actually 1+6=7, need 11.
    # Let's do a simple grid-like arrangement first.
    
    # Better initial guess: place them in a honeycomb pattern
    # Hexagon side = 1, so distance between centers of adjacent hexagons = sqrt(3) (face-to-face)
    d = math.sqrt(3)
    
    # 11 hexagons: let's try a 4-3-4 arrangement or similar
    # Or just use a simple linear arrangement as seed, then optimize
    
    # Seed: linear arrangement (same as original but with proper spacing)
    # Original had spacing 4.0 which is way too large. Proper spacing is sqrt(3) ~ 1.732
    inner_seed = []
    for k in range(N):
        inner_seed.append([d * k, 0.0, 0.0])
    
    # Better seed: honeycomb cluster
    # Center at origin
    # Ring 0: (0,0)
    # Ring 1: 6 hexagons at distance d in 6 directions
    # Ring 2: partial ring to get to 11 total
    inner_seed = [(0.0, 0.0, 0.0)]
    for k in range(6):
        theta = k * math.pi / 3.0
        x = d * math.cos(theta)
        y = d * math.sin(theta)
        inner_seed.append((x, y, 0.0))
    # That's 7. Need 4 more. Place them in second ring at specific positions.
    # Second ring hexagons are at distance 2*d from center, at angles 0, 60, 120, 180, 240, 300
    # But some positions are already occupied by ring 1 neighbors.
    # Let's just add 4 more at the "corners" of the second ring
    for k in [0, 1, 2, 3]:
        theta = k * math.pi / 3.0
        x = 2 * d * math.cos(theta)
        y = 2 * d * math.sin(theta)
        # Check if this position is too close to existing ones - for seed it's okay
        inner_seed.append((x, y, 0.0))
    
    # Convert to list of lists
    inner_seed = [list(h) for h in inner_seed]
    
    # Variables: x1,y1,a1, x2,y2,a2, ..., x11,y11,a11, cx, cy, outer_angle
    # Total: 11*3 + 3 = 36 variables
    # But we can fix outer center to (0,0) by symmetry? No, let's optimize it too.
    # Actually, we can fix the center of mass of inner hexagons to be at origin to reduce symmetry.
    # Let's keep it simple: optimize all positions, angles, and outer params.
    
    def objective(params):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            return 1e10  # penalty
        
        # Unpack
        inner = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i + 1]
            a = params[3*i + 2]
            inner.append((x, y, a))
        cx = params[3*N]
        cy = params[3*N + 1]
        outer_angle = params[3*N + 2]
        
        # Compute required outer side
        s = _outer_side_needed(inner, (cx, cy), outer_angle)
        
        # Add penalty for overlapping inner hexagons
        # For each pair, check minimum distance between hexagons
        # Approximation: center distance minus 2*inradius
        inradius = math.sqrt(3) / 2.0  # for side=1
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, _ = inner[i]
                xj, yj, _ = inner[j]
                dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
                min_dist = 2 * inradius  # minimum center distance for no overlap (face-to-face)
                if dist < min_dist:
                    overlap_penalty += (min_dist - dist) ** 2
        
        return s + 100.0 * overlap_penalty
    
    # Initial parameters
    x0 = []
    for (x, y, a) in inner_seed:
        x0.append(x)
        x0.append(y)
        x0.append(math.radians(a))  # use radians internally
    x0.append(0.0)  # cx
    x0.append(0.0)  # cy
    x0.append(0.0)  # outer angle
    
    x0 = np.array(x0)
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))  # x
        bounds.append((-10, 10))  # y
        bounds.append((-math.pi, math.pi))  # angle
    bounds.append((-10, 10))  # cx
    bounds.append((-10, 10))  # cy
    bounds.append((-math.pi/6, math.pi/6))  # outer angle (symmetry)
    
    best_params = x0
    best_val = objective(x0)
    
    # Try differential evolution first (global search)
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        try:
            result = differential_evolution(
                objective, bounds, 
                maxiter=50, popsize=15, 
                tol=1e-6, seed=42,
                polish=False
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Polish with L-BFGS-B
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        try:
            result = minimize(
                objective, best_params, 
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Extract best solution
    inner_result = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i + 1]
        a_deg = math.degrees(best_params[3*i + 2])
        inner_result.append([x, y, a_deg])
    cx = best_params[3*N]
    cy = best_params[3*N + 1]
    outer_angle_deg = math.degrees(best_params[3*N + 2])
    
    # Compute the actual outer side for this configuration
    s = _outer_side_needed(inner_result, (cx, cy), outer_angle_deg)
    
    # If overlap penalty was present, the returned s might be off.
    # Let's recompute cleanly.
    # Actually, objective returns s + penalty. We want just s.
    # Recompute s without penalty.
    final_s = _outer_side_needed(inner_result, (cx, cy), outer_angle_deg)
    
    # Verify no overlaps (approximately)
    inradius = math.sqrt(3) / 2.0
    for i in range(N):
        for j in range(i+1, N):
            xi, yi, _ = inner_result[i]
            xj, yj, _ = inner_result[j]
            dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < 2 * inradius - 1e-6:
                # Overlap detected, this is bad. Return a safe fallback.
                # Fallback to seed with proper spacing
                inner_safe = [[d * k, 0.0, 0.0] for k in range(N)]
                return inner_safe, (0.0, 0.0), _outer_side_needed(inner_safe, (0.0, 0.0), 0.0), 0.0
    
    return inner_result, (cx, cy), final_s, outer_angle_deg
# EVOLVE-BLOCK-END
