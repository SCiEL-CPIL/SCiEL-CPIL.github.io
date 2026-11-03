"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s safety margin
    
    # Unit hexagon vertices (flat-top orientation, circumradius 1)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    hex_angles = np.radians(np.arange(0, 360, 60))
    hex_verts = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    # For a regular hexagon with side s and center at origin, flat-top:
    # The 6 outward normal directions (perpendicular to each edge)
    # Edge normals are at angles 30, 90, 150, 210, 270, 330 degrees
    normal_angles = np.radians(np.arange(30, 360, 60))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    def hexagon_support_function(center, angle_deg, normals):
        """Return the support function values (max dot product with normals) for a unit hexagon."""
        # Rotate hexagon vertices by angle
        theta = np.radians(angle_deg)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rotated_verts = (rot @ hex_verts.T).T
        # Support function: max dot product with each normal
        supports = np.max(rotated_verts @ normals.T, axis=0)
        return supports
    
    def compute_required_side(inner_positions, inner_angles, outer_center, outer_angle_deg):
        """Compute the minimum outer side length needed to contain all inner hexagons."""
        theta_outer = np.radians(outer_angle_deg)
        rot_outer = np.array([[np.cos(theta_outer), -np.sin(theta_outer)], [np.sin(theta_outer), np.cos(theta_outer)]])
        normals_outer = (rot_outer @ normals.T).T
        
        max_support = 0.0
        for pos, ang in zip(inner_positions, inner_angles):
            # Support function of inner hexagon relative to outer center
            theta_inner = np.radians(ang)
            rot_inner = np.array([[np.cos(theta_inner), -np.sin(theta_inner)], [np.sin(theta_inner), np.cos(theta_inner)]])
            rotated_verts = (rot_inner @ hex_verts.T).T
            # Translate to outer center frame
            translated_verts = rotated_verts + pos - outer_center
            # Project onto outer normals
            projections = translated_verts @ normals_outer.T
            supports = np.max(projections, axis=0)
            max_support = max(max_support, np.max(supports))
        
        return max_support
    
    def objective(params):
        if time.time() > deadline - 5.0:
            return 1e10
        # params: [x1,y1,ang1, x2,y2,ang2, ..., outer_cx, outer_cy, outer_ang]
        n_inner = N
        inner_positions = []
        inner_angles = []
        for i in range(n_inner):
            x = params[3*i]
            y = params[3*i+1]
            ang = params[3*i+2]
            inner_positions.append(np.array([x, y]))
            inner_angles.append(ang)
        outer_cx = params[3*n_inner]
        outer_cy = params[3*n_inner+1]
        outer_ang = params[3*n_inner+2]
        
        return compute_required_side(inner_positions, inner_angles, 
                                     np.array([outer_cx, outer_cy]), outer_ang)
    
    # Initial arrangement: hexagonal lattice pattern
    # For 12 hexagons, use a compact arrangement
    # Hexagonal lattice spacing for same-orientation non-overlapping: distance >= 2*inradius = sqrt(3)
    # Actually for two unit hexagons with same orientation, minimum center distance is 2*inradius = sqrt(3)
    # for face-to-face contact, or 2 for vertex-to-vertex... 
    # The minimum distance between centers of two non-overlapping unit hexagons with same orientation
    # is the width of the hexagon = 2 (distance between parallel sides) when aligned
    # Actually, the inradius is sqrt(3)/2, so face-to-face distance is 2*sqrt(3)/2 = sqrt(3)
    # Wait: inradius = sqrt(3)/2 * side = sqrt(3)/2 for side=1
    # Distance between opposite faces = 2 * inradius = sqrt(3)
    # So minimum center distance for same-orientation hexagons is sqrt(3) for face contact
    
    # Hexagonal lattice with spacing sqrt(3)
    spacing = np.sqrt(3)
    
    # Arrange in a hexagonal cluster: 
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 4 hexagons
    # That's 12, but let's do a more compact hexagonal shape
    
    # Better: 3-4-3-2 or use a proper hexagonal packing
    # Let's use: center at (0,0), then 6 around, then 5 more
    positions = []
    # Center
    positions.append(np.array([0.0, 0.0]))
    # 6 around at distance sqrt(3)
    for i in range(6):
        angle = np.radians(60 * i)
        positions.append(spacing * np.array([np.cos(angle), np.sin(angle)]))
    # 5 more: place them in the gaps
    # Next ring positions (distance 2*sqrt(3))
    for i in range(5):
        angle = np.radians(60 * i + 30)  # offset by 30 degrees
        positions.append(2 * spacing * np.array([np.cos(angle), np.sin(angle)]))
    
    # Verify we have 12
    assert len(positions) == 12
    
    # Initial parameters
    params = []
    for pos in positions:
        params.extend([pos[0], pos[1], 0.0])
    params.extend([0.0, 0.0, 0.0])  # outer center and angle
    
    # Optimize
    result = None
    best_side = 1e10
    
    try:
        result = minimize(objective, params, method='L-BFGS-B', 
                         options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        if result.success or result.fun < best_side:
            best_side = result.fun
    except Exception:
        pass
    
    # Try multiple restarts with different initial arrangements
    for restart in range(5):
        if time.time() > deadline - 10.0:
            break
        # Perturb initial positions
        np.random.seed(42 + restart)
        params_perturbed = []
        for i in range(12):
            x = positions[i][0] + np.random.uniform(-0.5, 0.5)
            y = positions[i][1] + np.random.uniform(-0.5, 0.5)
            ang = np.random.uniform(0, 60)
            params_perturbed.extend([x, y, ang])
        params_perturbed.extend([0.0, 0.0, 0.0])
        
        try:
            res = minimize(objective, params_perturbed, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_side:
                best_side = res.fun
                result = res
        except Exception:
            pass
    
    if result is None:
        # Fallback: return initial arrangement
        inner = [[pos[0], pos[1], 0.0] for pos in positions]
        return inner, (0.0, 0.0), best_side, 0.0
    
    # Extract result
    best_params = result.x
    inner = []
    for i in range(12):
        x = best_params[3*i]
        y = best_params[3*i+1]
        ang = best_params[3*i+2]
        inner.append([float(x), float(y), float(ang)])
    
    outer_cx = float(best_params[36])
    outer_cy = float(best_params[37])
    outer_ang = float(best_params[38])
    
    return inner, (outer_cx, outer_cy), float(best_side), outer_ang

# EVOLVE-BLOCK-END
