import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def polygons_disjoint_sat(p1, p2, eps=1e-5):
        """
        Returns True if two convex polygons are disjoint (allowing touching with eps tolerance).
        Uses Separating Axis Theorem (SAT).
        """
        for i in range(len(p1)):
            j = (i + 1) % len(p1)
            edge = p1[j] - p1[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal /= norm
            
            proj1 = p1.dot(normal)
            proj2 = p2.dot(normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - eps or max2 < min1 - eps:
                return True
        
        for i in range(len(p2)):
            j = (i + 1) % len(p2)
            edge = p2[j] - p2[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal /= norm
            
            proj1 = p1.dot(normal)
            proj2 = p2.dot(normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - eps or max2 < min1 - eps:
                return True
        
        return False

    def check_validity(vertices_list):
        """
        vertices_list: list of (6, 2) arrays for each hexagon.
        Returns True if all are pairwise disjoint.
        """
        for i in range(len(vertices_list)):
            for j in range(i + 1, len(vertices_list)):
                if not polygons_disjoint_sat(vertices_list[i], vertices_list[j]):
                    return False
        return True

    def objective(params):
        """
        params: (3,) array [shear, rotation_deg, scale_factor]
        Returns outer hexagon side length + penalty if invalid.
        """
        if time.time() > deadline:
            return 1e10
            
        shear = params[0]
        rot_deg = params[1]
        scale = params[2]
        
        # Base lattice vectors for hexagonal tiling (side 1)
        # Distance between centers of touching hexagons is sqrt(3)
        # e1 = (sqrt(3), 0)
        # e2 = (sqrt(3)/2, 1.5)
        
        e1 = np.array([np.sqrt(3), 0])
        e2 = np.array([np.sqrt(3)/2, 1.5])
        
        # Apply shear to e1
        # Shear in y direction proportional to x
        # e1' = (e1[0], e1[1] + shear * e1[0])
        e1_shear = e1.copy()
        e1_shear[1] += shear * e1[0]
        
        # Apply rotation
        theta = np.radians(rot_deg)
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        e1_rot = R @ e1_shear
        e2_rot = R @ e2
        
        # Apply scale
        e1_final = scale * e1_rot
        e2_final = scale * e2_rot
        
        # Generate 12 centers in a 3x4 grid
        # Axial coords (q, r)
        # We want a compact cluster.
        # 3 columns, 4 rows?
        # Let's use q in [-1, 0, 1] and r in [-1, 0, 1, 2] ?
        # Or simply a rectangular block in axial coords.
        # Axial to Cartesian:
        # x = sqrt(3) * (q + r/2)
        # y = 1.5 * r
        
        # Let's pick 12 coords that form a nice hexagonal-ish or rectangular shape.
        # A 3x4 rectangle in axial coords:
        # q: -1, 0, 1
        # r: -1, 0, 1, 2
        # This gives 12 points.
        
        coords = []
        for q in [-1, 0, 1]:
            for r in [-1, 0, 1, 2]:
                coords.append((q, r))
        
        # Calculate centers using the deformed basis
        # Note: The basis vectors e1, e2 correspond to steps in q and r.
        # Standard axial to cartesian uses specific vectors.
        # Our e1 corresponds to step (1, 0) in axial?
        # x = sqrt(3)*(q + r/2), y = 1.5*r
        # Step (1,0): dx = sqrt(3), dy = 0. Matches e1.
        # Step (0,1): dx = sqrt(3)/2, dy = 1.5. Matches e2.
        
        centers = []
        for q, r in coords:
            c = q * e1_final + r * e2_final
            centers.append(c)
            
        centers = np.array(centers)
        
        # Check validity
        vertices_list = []
        for c in centers:
            verts = hexagon_vertices(c[0], c[1], 1.0, 0.0)
            vertices_list.append(verts)
            
        if not check_validity(vertices_list):
            return 1e10
            
        # Calculate required outer hexagon side length
        # We want to minimize the side length S of a regular hexagon centered at origin
        # that contains all vertices of the inner hexagons.
        # We can also rotate the outer hexagon to minimize S.
        # However, the inner hexagons are fixed at angle 0 in this parametric model.
        # The outer hexagon orientation is a free variable in the final output,
        # but here we just compute the min S for a fixed orientation (0) to guide optimization.
        # Actually, we can optimize the outer rotation too, but let's keep it simple first.
        # If we fix outer rotation to 0, we might miss the optimal.
        # But the inner lattice rotation `rot_deg` effectively rotates the whole cluster.
        # So we are effectively searching over relative orientations.
        
        # For a point (x,y), required S (outer angle 0) is:
        # S >= |x|
        # S >= |y| * 2 / sqrt(3)
        # S >= |x| + |y| / sqrt(3)
        
        s_vals = []
        for verts in vertices_list:
            for vx, vy in verts:
                s_req = max(abs(vx), abs(vy) * 2.0 / np.sqrt(3.0), abs(vx) + abs(vy) / np.sqrt(3.0))
                s_vals.append(s_req)
                
        if not s_vals:
            return 0.0
            
        return max(s_vals)

    # Initial guess
    # shear = 0, rot = 0, scale = 1.0
    x0 = np.array([0.0, 0.0, 1.0])
    
    # Bounds
    # Shear: -0.5 to 0.5
    # Rotation: -30 to 30 degrees
    # Scale: 0.9 to 1.1 (since we want them to touch, scale shouldn't vary too much from 1.0 if we want dense packing, 
    # but wait, the "scale" here scales the lattice spacing. If we scale up, they overlap. If we scale down, they separate.
    # We want them to be as close as possible to minimize bounding box, so scale should be 1.0 (touching).
    # However, due to shear/rotation, they might overlap if scale is 1.0.
    # So we might need to increase scale slightly to avoid overlap, which increases bounding box.
    # Or we can allow scale to be slightly less than 1.0 if the geometry allows? No, unit hexagons.
    # The distance between centers MUST be >= sqrt(3) to avoid overlap.
    # In the deformed lattice, the distance between adjacent centers is |e1_final| and |e2_final| and |e1_final - e2_final| etc.
    # We need min distance >= sqrt(3).
    # So scale should be adjusted to satisfy this constraint.
    # Let's just let the optimizer find the best scale, penalizing if invalid.
    
    bounds = [(-0.5, 0.5), (-30, 30), (1.0, 1.2)]
    
    best_val = 1e10
    best_x = x0
    
    # Try a few starting points
    starts = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.1, 0.0, 1.0]),
        np.array([-0.1, 0.0, 1.0]),
        np.array([0.0, 10.0, 1.0]),
        np.array([0.0, -10.0, 1.0]),
        np.array([0.2, 5.0, 1.0]),
        np.array([-0.2, -5.0, 1.0]),
    ]
    
    for s in starts:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(objective, s, bounds=bounds, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    # Final refinement with tighter bounds around best
    if best_val < 1e9:
        try:
            bounds_refined = [
                (best_x[0] - 0.1, best_x[0] + 0.1),
                (best_x[1] - 5.0, best_x[1] + 5.0),
                (max(1.0, best_x[2] - 0.05), best_x[2] + 0.05)
            ]
            res = minimize(objective, best_x, bounds=bounds_refined, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    # Construct final result
    shear = best_x[0]
    rot_deg = best_x[1]
    scale = best_x[2]
    
    e1 = np.array([np.sqrt(3), 0])
    e2 = np.array([np.sqrt(3)/2, 1.5])
    
    e1_shear = e1.copy()
    e1_shear[1] += shear * e1[0]
    
    theta = np.radians(rot_deg)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    e1_rot = R @ e1_shear
    e2_rot = R @ e2
    
    e1_final = scale * e1_rot
    e2_final = scale * e2_rot
    
    coords = []
    for q in [-1, 0, 1]:
        for r in [-1, 0, 1, 2]:
            coords.append((q, r))
            
    inner = []
    for q, r in coords:
        c = q * e1_final + r * e2_final
        inner.append([float(c[0]), float(c[1]), 0.0])
        
    # Outer hexagon
    # Center (0,0)
    # Side length best_val
    # Angle 0.0
    # Note: We might be able to rotate the outer hexagon to reduce side length further.
    # Let's do a quick 1D search for outer rotation.
    
    def outer_side_for_rotation(rot_outer_deg):
        # Rotate all inner vertices by -rot_outer_deg? No.
        # If outer hexagon rotates by alpha, the condition for a point (x,y) is:
        # Let p' = R(-alpha) @ p. Then check if p' is in hexagon(0, 0, S).
        # S = max( |x'|, |y'| * 2/sqrt(3), |x'| + |y'|/sqrt(3) )
        
        min_s = 1e10
        alpha_rad = np.radians(rot_outer_deg)
        R_inv = np.array([
            [np.cos(-alpha_rad), -np.sin(-alpha_rad)],
            [np.sin(-alpha_rad), np.cos(-alpha_rad)]
        ])
        
        for q, r in coords:
            c = q * e1_final + r * e2_final
            verts = hexagon_vertices(c[0], c[1], 1.0, 0.0)
            for vx, vy in verts:
                p = np.array([vx, vy])
                p_rot = R_inv @ p
                s_req = max(abs(p_rot[0]), abs(p_rot[1]) * 2.0 / np.sqrt(3.0), abs(p_rot[0]) + abs(p_rot[1]) / np.sqrt(3.0))
                if s_req > min_s:
                    min_s = s_req
        return min_s

    best_outer_rot = 0.0
    best_outer_side = best_val
    
    # Search outer rotation
    for rot in np.linspace(-30, 30, 61):
        s = outer_side_for_rotation(rot)
        if s < best_outer_side:
            best_outer_side = s
            best_outer_rot = rot
            
    # Refine outer rotation
    if best_outer_side < 1e9:
        for rot in np.linspace(best_outer_rot - 2, best_outer_rot + 2, 41):
            s = outer_side_for_rotation(rot)
            if s < best_outer_side:
                best_outer_side = s
                best_outer_rot = rot

    return inner, (0.0, 0.0), float(best_outer_side), float(best_outer_rot)

# EVOLVE-BLOCK-END
