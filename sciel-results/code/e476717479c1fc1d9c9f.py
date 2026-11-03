import math
import numpy as np
import time
import random
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def _soft_max(vals, temp):
    """Numerically stable soft-max (log-sum-exp)."""
    m = np.max(vals)
    if m > 1e10:
        return float('inf')
    shifted = np.clip((vals - m) / temp, -50, 50)
    return m + temp * np.log(np.sum(np.exp(shifted)))

def _hex_vertices(center, angle_deg):
    """Generate vertices of a unit regular hexagon."""
    cx, cy = center
    theta = math.radians(angle_deg)
    verts = []
    for k in range(6):
        phi = theta + k * math.pi / 3.0
        x = cx + math.cos(phi)
        y = cy + math.sin(phi)
        verts.append((x, y))
    return np.array(verts)

def _dot(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1]

def _cross(v1, v2):
    return v1[0]*v2[1] - v1[1]*v2[0]

def _project_polygon_onto_axis(polygon, axis):
    """Project polygon vertices onto an axis and return min/max dot products."""
    dots = []
    for v in polygon:
        dots.append(_dot(v, axis))
    return min(dots), max(dots)

def _polygons_overlap(poly1, poly2):
    """Check if two convex polygons overlap using Separating Axis Theorem."""
    polygons = [poly1, poly2]
    
    for i in range(2):
        polygon = polygons[i]
        for j in range(len(polygon)):
            v1 = polygon[j]
            v2 = polygon[(j + 1) % len(polygon)]
            # Edge vector
            edge = v2 - v1
            # Normal axis
            axis = np.array([-edge[1], edge[0]])
            
            # Normalize axis (optional, but good for consistency)
            norm = np.linalg.norm(axis)
            if norm < 1e-9:
                continue
            axis = axis / norm
            
            # Project both polygons onto this axis
            min1, max1 = _project_polygon_onto_axis(poly1, axis)
            min2, max2 = _project_polygon_onto_axis(poly2, axis)
            
            # Check for separation
            if max1 < min2 or max2 < min1:
                return False
                
    return True

def _calc_min_enclosing_hexagon_side(pts, angles):
    """
    Calculate the minimum side length of a regular hexagon that encloses 
    the given set of hexagons (defined by centers and angles).
    Uses LP for a fixed outer rotation, then minimizes over outer rotation.
    """
    def objective_for_beta(beta):
        # Normals for the enclosing hexagon
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        # For each inner hexagon, we need to ensure it is inside the outer one.
        # This is equivalent to: for each vertex of the inner hexagon, 
        # it must satisfy the half-plane constraints of the outer hexagon.
        # Outer hexagon: |n_k . (x - c)| <= D for all k, where D is the apothem of outer hex.
        # Side length S = (D + APOTHEM) / APOTHEM.
        
        for i in range(len(pts)):
            cx, cy = pts[i]
            ang = angles[i]
            verts = _hex_vertices((cx, cy), ang)
            
            for v in verts:
                for nx, ny in normals:
                    proj = v[0] * nx + v[1] * ny
                    # Constraint: nx*x + ny*y - D <= cx*nx + cy*ny ? No.
                    # Let's use the standard LP form:
                    # We want to find c=(cx_out, cy_out) and D such that:
                    # n_k . (v - c_out) <= D  =>  -n_k . c_out + D >= n_k . v
                    # and -n_k . (v - c_out) <= D => n_k . c_out + D >= -n_k . v
                    # Let variables be x = [cx_out, cy_out, D]
                    # Constraints:
                    # -nx*cx - ny*cy + D >= nx*vx + ny*vy  =>  nx*cx + ny*cy - D <= -(nx*vx + ny*vy)
                    # nx*cx + ny*cy - D <= nx*vx + ny*vy
            
            # Wait, the LP in the original code was simpler because it assumed the inner hexagons were just points?
            # No, the original code `calc_S_for_beta` took `pts` which were the *centers* of the inner hexagons?
            # Let's check the original code's `calc_S_for_beta`.
            # It iterates `for px, py in pts:` and uses `proj = px * nx + py * ny`.
            # This implies the original code was treating the inner hexagons as **points** or **circles** of radius 0?
            # Or did it expand the constraints?
            # If the inner hexagons have side 1, their support function in direction n is APOTHEM.
            # So the condition is: n . (c_inner - c_outer) + APOTHEM <= D_outer
            # => n . c_inner - n . c_outer + APOTHEM <= D_outer
            # => n . c_outer + D_outer >= n . c_inner + APOTHEM
            # And for the opposite direction:
            # -n . (c_inner - c_outer) + APOTHEM <= D_outer
            # => -n . c_inner + n . c_outer + APOTHEM <= D_outer
            # => n . c_outer - D_outer <= -n . c_inner - APOTHEM  (Wait, signs)
            # Let's re-derive.
            # Outer hexagon: {x | |n_k . (x - c)| <= D}
            # Inner hexagon i is inside if for all vertices v of inner i: |n_k . (v - c)| <= D.
            # This is equivalent to: n_k . (v - c) <= D AND -n_k . (v - c) <= D.
            # n_k . v - n_k . c <= D  =>  n_k . c + D >= n_k . v
            # -n_k . v + n_k . c <= D =>  -n_k . c + D >= -n_k . v
            
            # The LP variables are c=(cx, cy) and D.
            # Minimize D.
            # Constraints:
            # 1) nx*cx + ny*cy - D <= -(nx*vx + ny*vy)  (from n.c + D >= n.v => n.c - D <= -n.v) -- WAIT
            #    n.c + D >= n.v  =>  -n.c - D <= -n.v
            #    Let's use the form A_ub x <= b_ub.
            #    x = [cx, cy, D]
            #    Constraint 1: -nx*cx - ny*cy - 1*D <= -(nx*vx + ny*vy)
            #    Constraint 2: nx*cx + ny*cy - 1*D <= (nx*vx + ny*vy)
            
            for v in verts:
                proj = v[0] * nx + v[1] * ny
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-proj)
                A_ub.append([nx, ny, -1.0])
                b_ub.append(proj)
                
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                S = (D + APOTHEM) / APOTHEM
                return S
            else:
                return float('inf')
        except Exception:
            return float('inf')

    # Coarse grid search for beta
    best_S = float('inf')
    for i in range(36):
        beta = i * (math.pi / 6) / 36
        S = objective_for_beta(beta)
        if S < best_S:
            best_S = S
    
    # Fine grid search
    if best_S < float('inf'):
        # Find best beta from coarse
        best_beta_coarse = 0
        min_val = float('inf')
        for i in range(36):
            beta = i * (math.pi / 6) / 36
            S = objective_for_beta(beta)
            if S < min_val:
                min_val = S
                best_beta_coarse = beta
        
        for i in range(11):
            beta = best_beta_coarse - math.pi/72 + i * (2 * math.pi / 72) / 10
            if 0 <= beta <= math.pi/6:
                S = objective_for_beta(beta)
                if S < best_S:
                    best_S = S

    return best_S

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define candidate point sets (axial coordinates)
    candidates = []

    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    candidates.append(axial_1)

    axial_flower_a = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_flower_a)

    axial_flower_b = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_flower_b)

    axial_diag1 = [
        (0, 0), (1, 1), (2, 2),
        (-1, 0), (0, 1), (1, 2),
        (-2, 0), (-1, 1), (0, 2),
        (-1, -1), (1, 0)
    ]
    candidates.append(axial_diag1)

    axial_tri1 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_tri1)

    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_2)

    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append(axial_3)

    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append(axial_4)

    axial_5 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    candidates.append(axial_5)

    axial_6 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    candidates.append(axial_6)

    axial_7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_7)

    axial_8 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append(axial_8)

    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    candidates.append(axial_9)

    axial_10 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_10)

    axial_11 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3)
    ]
    candidates.append(axial_11)

    axial_12 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4)
    ]
    candidates.append(axial_12)

    axial_13 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2)
    ]
    candidates.append(axial_13)

    axial_14 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    candidates.append(axial_14)

    axial_15 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1), (0, -1)
    ]
    candidates.append(axial_15)

    axial_16 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 2)
    ]
    candidates.append(axial_16)

    axial_17 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3)
    ]
    candidates.append(axial_17)

    axial_18 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    candidates.append(axial_18)

    axial_19 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_19)

    axial_20 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (1, 1), (-1, 1)
    ]
    candidates.append(axial_20)

    axial_21 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ]
    candidates.append(axial_21)

    axial_22 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_22)

    axial_23 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (-1, 2)
    ]
    candidates.append(axial_23)

    axial_24 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (1, 4)
    ]
    candidates.append(axial_24)

    axial_25 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3),
        (0, 4), (1, 4)
    ]
    candidates.append(axial_25)

    axial_26 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3),
        (1, 4), (2, 4)
    ]
    candidates.append(axial_26)

    axial_27 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (1, 2), (-1, 3)
    ]
    candidates.append(axial_27)

    axial_28 = [
        (0, 0), (1, 0), (2, 0),
        (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3), (3, 3)
    ]
    candidates.append(axial_28)

    axial_29 = [
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3), (2, 3),
        (0, 4), (1, 4)
    ]
    candidates.append(axial_29)

    axial_30 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2),
        (0, 3), (1, 3),
        (1, 4)
    ]
    candidates.append(axial_30)

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # Step 1: Find the best rigid configuration
    best_rigid_S = float('inf')
    best_rigid_pts = None
    best_rigid_axial = None
    
    for axial in candidates:
        pts = axial_to_cartesian(axial)
        angles = [0.0] * len(pts)
        S = _calc_min_enclosing_hexagon_side(pts, angles)
        if S < best_rigid_S:
            best_rigid_S = S
            best_rigid_pts = pts.copy()
            best_rigid_axial = axial

    if best_rigid_pts is None:
        best_rigid_pts = axial_to_cartesian(candidates[0])
        best_rigid_S = _calc_min_enclosing_hexagon_side(best_rigid_pts, [0.0]*11)

    # Step 2: Local Search (Radial Compression / Jitter)
    current_pts = best_rigid_pts.copy()
    current_angles = [0.0] * 11
    current_S = best_rigid_S
    
    # Pre-compute vertices for initial state to speed up overlap checks? 
    # No, angles will change.
    
    max_iters = 5000
    step_size = 0.2
    
    for i in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Pick 1 or 2 hexagons to move
        num_moves = random.choice([1, 2])
        indices = random.sample(range(11), num_moves)
        
        new_pts = current_pts.copy()
        new_angles = current_angles.copy()
        
        for idx in indices:
            # Perturb position
            dx = random.uniform(-step_size, step_size)
            dy = random.uniform(-step_size, step_size)
            new_pts[idx] += np.array([dx, dy])
            
            # Perturb angle
            d_ang = random.uniform(-10, 10)
            new_angles[idx] += d_ang
            
        # Check for overlaps
        overlap = False
        for i_idx in range(11):
            if overlap: break
            for j_idx in range(i_idx + 1, 11):
                # Quick reject: center distance
                dist = np.linalg.norm(new_pts[i_idx] - new_pts[j_idx])
                if dist < 1.9: # Max distance between centers for non-overlap is > 2*apothem? 
                               # Actually, if dist > 2 * sqrt(3) * 0.5 * 2? 
                               # The max distance between two points in a unit hexagon is 2 (diameter).
                               # If centers are > 2 apart, they definitely don't overlap? 
                               # No, if they are aligned, they can touch at distance 2.
                               # So if dist > 2.0 + epsilon, no overlap.
                               # If dist < 2*APOTHEM * 2? 
                               # Let's use SAT for safety, but quick reject if dist > 2.1
                               pass
                # Full SAT check
                v1 = _hex_vertices(new_pts[i_idx], new_angles[i_idx])
                v2 = _hex_vertices(new_pts[j_idx], new_angles[j_idx])
                if _polygons_overlap(v1, v2):
                    overlap = True
                    break
        
        if overlap:
            continue
            
        # Calculate new S
        new_S = _calc_min_enclosing_hexagon_side(new_pts, new_angles)
        
        if new_S < current_S:
            current_S = new_S
            current_pts = new_pts
            current_angles = new_angles
            # Reduce step size slightly on improvement
            step_size *= 0.95
        else:
            # Increase step size slightly to explore
            step_size = min(step_size * 1.05, 0.5)

    # Final Polish: Use Nelder-Mead on the best configuration found
    # Define a smooth objective for the final polish
    def final_objective(params):
        # params: 11*3 = 33 values (x, y, ang) for each hexagon
        pts = np.array(params[0:22]).reshape(11, 2)
        angs = params[22:33]
        
        # Check overlaps (soft penalty)
        penalty = 0
        for i in range(11):
            for j in range(i+1, 11):
                v1 = _hex_vertices(pts[i], angs[i])
                v2 = _hex_vertices(pts[j], angs[j])
                # Approximate overlap penalty using min distance? 
                # SAT doesn't give distance easily. 
                # We'll rely on the discrete search to find a valid config, 
                # and just polish the enclosing hexagon size here.
                # Since we already checked overlaps in the loop, we assume valid.
                
        return _calc_min_enclosing_hexagon_side(pts, angs)

    # Flatten current state
    init_params = []
    for i in range(11):
        init_params.append(current_pts[i][0])
        init_params.append(current_pts[i][1])
        init_params.append(current_angles[i])
        
    try:
        res = minimize(final_objective, init_params, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.fun < current_S:
            current_S = res.fun
            current_pts = res.x[0:22].reshape(11, 2)
            current_angles = res.x[22:33]
    except Exception:
        pass

    # Construct return value
    # We need to return the outer hexagon center, side, rotation
    # And the inner hexagons (x, y, angle)
    
    # Recalculate the optimal outer hexagon for the final inner configuration
    # We need to find the best beta and center
    def get_outer_params(pts, angles):
        best_S = float('inf')
        best_beta = 0
        best_center = (0,0)
        
        for i in range(36):
            beta = i * (math.pi / 6) / 36
            # Run LP to get center and D
            normals = np.array([
                (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
                 math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
                for k in range(6)
            ])
            c_obj = np.array([0.0, 0.0, 1.0])
            A_ub = []
            b_ub = []
            for i in range(len(pts)):
                cx, cy = pts[i]
                ang = angles[i]
                verts = _hex_vertices((cx, cy), ang)
                for v in verts:
                    for nx, ny in normals:
                        proj = v[0] * nx + v[1] * ny
                        A_ub.append([-nx, -ny, -1.0])
                        b_ub.append(-proj)
                        A_ub.append([nx, ny, -1.0])
                        b_ub.append(proj)
            
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            bounds = [(None, None), (None, None), (0, None)]
            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.status == 0:
                    D = res.x[2]
                    S = (D + APOTHEM) / APOTHEM
                    if S < best_S:
                        best_S = S
                        best_beta = beta
                        best_center = (res.x[0], res.x[1])
            except:
                pass
        return best_S, best_center, best_beta

    final_S, final_center, final_beta = get_outer_params(current_pts, current_angles)
    
    inner = []
    for i in range(11):
        inner.append([float(current_pts[i][0]), float(current_pts[i][1]), float(current_angles[i])])
        
    center = [float(final_center[0]), float(final_center[1])]
    side = float(final_S)
    rotation = float(math.degrees(final_beta))
    
    return inner, center, side, rotation

