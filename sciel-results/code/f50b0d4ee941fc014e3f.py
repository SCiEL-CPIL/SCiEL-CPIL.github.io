import numpy as np
import time
from itertools import combinations

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _min_outer_side_from_vertices(vertices):
    """
    Compute the minimum side length s of a regular hexagon centered at origin
    (angle 0) that contains all given vertices.
    """
    # Outward normals of a regular hexagon with angle 0 (flat top/bottom? No, vertex at 0)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees.
    # Edges are between these.
    # Normal to edge between v_k and v_{k+1} is at angle (k * 60 + 30) degrees.
    outward_normals = np.array([
        [np.cos(np.radians(30 + 60*k)), np.sin(np.radians(30 + 60*k))]
        for k in range(6)
    ])
    dots = vertices @ outward_normals.T
    max_dot = np.max(dots, axis=1)
    apothem_min = np.max(max_dot)
    s_min = 2.0 * apothem_min / np.sqrt(3.0)
    return s_min

def _check_separation(pos1, ang1, pos2, ang2, side=1.0, tol=1e-8):
    """
    Check if two unit hexagons are separated using SAT.
    Returns True if separated.
    """
    v1 = _hex_vertices(pos1[0], pos1[1], side, ang1)
    v2 = _hex_vertices(pos2[0], pos2[1], side, ang2)
    
    # Get normals for both hexagons
    normals = []
    for v in [v1, v2]:
        for k in range(6):
            edge = v[(k+1)%6] - v[k]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                normals.append(n)
                
    for n in normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        # If there is a gap > tol, they are separated
        if max1 < min2 - tol or max2 < min1 - tol:
            return True
            
    return False

def _is_valid_config(positions, angles, s, tol=1e-6):
    """
    Full validity check.
    """
    N = len(positions)
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(angles)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
        
    # Check containment
    all_verts = []
    for i in range(N):
        all_verts.append(_hex_vertices(positions[i,0], positions[i,1], 1.0, angles[i]))
    all_verts = np.vstack(all_verts)
    
    outer_verts = _hex_vertices(0, 0, s, 0.0)
    for v in all_verts:
        inside = True
        for k in range(6):
            p1 = outer_verts[k]
            p2 = outer_verts[(k+1)%6]
            cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
            if cross < -tol:
                inside = False
                break
        if not inside:
            return False
            
    # Check separation
    for i in range(N):
        for j in range(i+1, N):
            if not _check_separation(positions[i], angles[i], positions[j], angles[j], tol=1e-8):
                return False
                
    return True

def _solve_topology(radii, angles_rad, s):
    """
    Given a topology defined by radii (12,) and angles_rad (12,) for the 12 hexagons,
    and an outer side s, check if it's valid.
    Returns (valid, actual_s_needed)
    """
    positions = np.column_stack([radii * np.cos(angles_rad), radii * np.sin(angles_rad)])
    angles_deg = np.degrees(angles_rad)
    
    # Compute actual s needed
    all_verts = []
    for i in range(12):
        all_verts.append(_hex_vertices(positions[i,0], positions[i,1], 1.0, angles_deg[i]))
    all_verts = np.vstack(all_verts)
    s_needed = _min_outer_side_from_vertices(all_verts)
    
    if s_needed > s + 1e-9:
        return False, s_needed
        
    # Check separation
    for i in range(12):
        for j in range(i+1, 12):
            if not _check_separation(positions[i], angles_deg[i], positions[j], angles_deg[j], tol=1e-8):
                return False, s_needed
                
    return True, s_needed

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580.0
    
    best_score = -1.0
    best_result = None
    
    # Define the "Star" topology:
    # 0: Center
    # 1-6: Inner ring
    # 7-11: Outer ring
    
    # Standard honeycomb angles for inner ring
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    
    # Outer ring angles: typically between inner ring vertices
    # If inner ring is at 0, 60..., outer ring can be at 30, 90...
    outer_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
    # But we only have 5 outer hexagons in a 12-hex cluster?
    # Wait, 1 center + 6 inner + 5 outer = 12.
    # A perfect hexagon of side 2 has 19 hexagons (1+6+12).
    # A hexagon of side 1 has 7.
    # 12 is 1 + 6 + 5.
    # The "missing" 1 from the full outer ring allows for tighter packing?
    # Or is it 1 center, 5 inner, 6 outer?
    # Let's try the 1-6-5 configuration.
    
    # We will optimize the radii of the 6 inner and 5 outer hexagons.
    # The center is fixed at (0,0).
    
    # Initial guess
    # Inner ring radius: sqrt(3) (touching center)
    # Outer ring radius: ~ 1.5 * sqrt(3) + something?
    
    # Let's use a simple local search on the radii.
    # Variables: r_1..r_6 (inner), r_7..r_11 (outer)
    # Angles are fixed for symmetry to reduce search space, but we might allow small perturbations later.
    
    # Fixed angles for symmetry:
    # Center: 0
    # Inner: 0, 60, 120, 180, 240, 300
    # Outer: 30, 90, 150, 210, 270 (5 hexagons, skipping 330 to break symmetry? Or just 5 of the 6?)
    # Actually, to be symmetric, we should probably pick 5 out of 6 outer positions.
    # Let's pick outer angles: 30, 90, 150, 210, 270. (Missing 330).
    
    inner_angs = base_angles
    outer_angs = np.array([30, 90, 150, 210, 270]) * np.pi / 180.0
    
    all_angs = np.concatenate([[0.0], inner_angs, outer_angs])
    
    # Optimization variables: radii for indices 1..11
    # r_inner (6), r_outer (5)
    
    def evaluate(radii):
        # radii: (11,)
        r_inner = radii[:6]
        r_outer = radii[6:]
        
        # Construct full radii array (12,)
        full_radii = np.zeros(12)
        full_radii[0] = 0.0
        full_radii[1:7] = r_inner
        full_radii[7:12] = r_outer
        
        # We want to minimize s.
        # Compute s_needed
        positions = np.column_stack([full_radii * np.cos(all_angs), full_radii * np.sin(all_angs)])
        angles_deg = np.degrees(all_angs)
        
        all_verts = []
        for i in range(12):
            all_verts.append(_hex_vertices(positions[i,0], positions[i,1], 1.0, angles_deg[i]))
        all_verts = np.vstack(all_verts)
        s_needed = _min_outer_side_from_vertices(all_verts)
        
        # Check constraints
        valid = True
        # Containment is handled by s_needed calculation (it's the min s to contain)
        # But we need to ensure that for this s_needed, the hexagons don't overlap.
        
        for i in range(12):
            for j in range(i+1, 12):
                if not _check_separation(positions[i], angles_deg[i], positions[j], angles_deg[j], tol=1e-8):
                    valid = False
                    break
            if not valid:
                break
                
        if not valid:
            return np.inf
        return s_needed

    # Local search: Simulated Annealing or Gradient Descent on radii
    # Let's use a simple gradient descent with numerical gradients? No, too slow.
    # Use scipy.optimize.minimize on the 11D space.
    
    from scipy.optimize import minimize
    
    # Initial guess
    # Inner ring touching center: sqrt(3)
    # Outer ring touching inner ring: sqrt(3) + sqrt(3) = 2*sqrt(3)? No, geometry is complex.
    # Distance between center and inner center: r1
    # Distance between inner center and outer center: d
    # Angle difference: 30 deg.
    # Law of cosines: r_outer^2 = r_inner^2 + d^2 - 2*r_inner*d*cos(30)
    # If they touch, distance between centers is sqrt(3) (for aligned faces) or 2 (for vertices).
    # Let's guess r_inner = sqrt(3), r_outer = 2.5 * sqrt(3)
    
    r0 = np.array([np.sqrt(3)] * 6 + [2.5 * np.sqrt(3)] * 5)
    
    # Run optimization
    try:
        res = minimize(evaluate, r0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.success and res.fun < np.inf:
            r_opt = res.x
            s_opt = res.fun
            # Reconstruct
            full_radii = np.zeros(12)
            full_radii[0] = 0.0
            full_radii[1:7] = r_opt[:6]
            full_radii[7:12] = r_opt[6:]
            
            positions = np.column_stack([full_radii * np.cos(all_angs), full_radii * np.sin(all_angs)])
            angles_deg = np.degrees(all_angs)
            
            # Final validation
            if _is_valid_config(positions, angles_deg, s_opt):
                score = 1.0 / s_opt
                if score > best_score:
                    best_score = score
                    best_result = (positions, angles_deg, s_opt)
    except Exception as e:
        pass
    
    # If Nelder-Mead failed or didn't find anything, try a few random perturbations of the initial guess
    if best_result is None:
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            r_rand = r0 * np.random.uniform(0.9, 1.1, 11)
            try:
                res = minimize(evaluate, r_rand, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.success and res.fun < np.inf:
                    r_opt = res.x
                    s_opt = res.fun
                    full_radii = np.zeros(12)
                    full_radii[0] = 0.0
                    full_radii[1:7] = r_opt[:6]
                    full_radii[7:12] = r_opt[6:]
                    
                    positions = np.column_stack([full_radii * np.cos(all_angs), full_radii * np.sin(all_angs)])
                    angles_deg = np.degrees(all_angs)
                    
                    if _is_valid_config(positions, angles_deg, s_opt):
                        score = 1.0 / s_opt
                        if score > best_score:
                            best_score = score
                            best_result = (positions, angles_deg, s_opt)
            except Exception:
                continue

    # Fallback: Honeycomb
    if best_result is None:
        pos = []
        ang = []
        # Center
        pos.append([0,0]); ang.append(0)
        # Inner
        for a in base_angles:
            pos.append([np.sqrt(3)*np.cos(a), np.sqrt(3)*np.sin(a)]); ang.append(np.degrees(a))
        # Outer
        for a in np.array([30, 90, 150, 210, 270]) * np.pi / 180.0:
             # Approx position
             r = 2.5 * np.sqrt(3)
             pos.append([r*np.cos(a), r*np.sin(a)]); ang.append(np.degrees(a))
        
        pos = np.array(pos)
        ang = np.array(ang)
        
        # Compute s
        all_verts = []
        for i in range(12):
            all_verts.append(_hex_vertices(pos[i,0], pos[i,1], 1.0, ang[i]))
        all_verts = np.vstack(all_verts)
        s = _min_outer_side_from_vertices(all_verts)
        
        # Adjust if invalid
        if _is_valid_config(pos, ang, s):
            best_result = (pos, ang, s)
            best_score = 1.0/s
        else:
            # Expand until valid
            scale = 1.0
            while not _is_valid_config(pos * scale, ang, s * scale) and scale < 10:
                scale += 0.1
            if _is_valid_config(pos * scale, ang, s * scale):
                best_result = (pos * scale, ang, s * scale)
                best_score = 1.0 / (s * scale)
            else:
                # Emergency fallback
                pos = np.random.uniform(-5, 5, (12,2))
                ang = np.zeros(12)
                s = 10.0
                while not _is_valid_config(pos, ang, s):
                    s += 0.5
                best_result = (pos, ang, s)
                best_score = 1.0/s

    pos_final, ang_final, s_final = best_result
    
    inner_hex_data = np.column_stack([pos_final, ang_final])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s_final
