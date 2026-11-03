import numpy as np
import time

N = 12

def hex_vertices(center, side, angle_deg):
    """
    Returns vertices of a regular hexagon.
    center: (x, y)
    side: length of side
    angle_deg: rotation in degrees
    """
    cx, cy = center
    angle_rad = np.deg2rad(angle_deg)
    # Vertices are at angles 0, 60, 120, 180, 240, 300 relative to angle_rad
    # Distance from center to vertex is 'side'
    # Wait, for a regular hexagon, side length = radius.
    angles = angle_rad + np.deg2rad(np.arange(0, 360, 60))
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def generate_topology(name):
    """
    Generates list of (x, y, angle_deg) for 12 inner hexagons.
    All inner hexagons have side=1, angle=0.
    """
    if name == "grid_3x4":
        # 3 rows, 4 columns
        # Offset coordinates
        # Row 0: x = 0, 2, 4, 6
        # Row 1: x = 1, 3, 5, 7, y = sqrt(3)
        # Row 2: x = 0, 2, 4, 6, y = 2*sqrt(3)
        hexes = []
        sqrt3 = np.sqrt(3)
        for r in range(3):
            y = r * sqrt3
            for c in range(4):
                x = c * 2.0
                if r % 2 == 1:
                    x += 1.0
                hexes.append([x, y, 0.0])
        return hexes
    elif name == "grid_2x6":
        hexes = []
        sqrt3 = np.sqrt(3)
        for r in range(2):
            y = r * sqrt3
            for c in range(6):
                x = c * 2.0
                if r % 2 == 1:
                    x += 1.0
                hexes.append([x, y, 0.0])
        return hexes
    elif name == "cluster_1_6_5":
        # 1 center, 6 ring, 5 more
        # Center: (0,0)
        # Ring 1: (1,0), (0.5, sqrt3/2), (-0.5, sqrt3/2), (-1,0), (-0.5, -sqrt3/2), (0.5, -sqrt3/2)
        # These are distance 1 from center.
        # Wait, hexagon side 1. Distance between centers of adjacent hexagons is sqrt(3)?
        # No.
        # If hexagons are packed, distance between centers is sqrt(3) * side?
        # Let's check.
        # Flat top hexagon.
        # Distance between parallel sides is sqrt(3).
        # Distance between centers of adjacent hexagons in a tiling is sqrt(3).
        # Let's verify.
        # Vertex at (1, 0). Next vertex (0.5, sqrt3/2).
        # Center (0,0).
        # Neighbor to the right: Center at (sqrt3, 0)?
        # If I place a hexagon at (sqrt3, 0), its left-most vertex is (sqrt3 - 1, 0).
        # My right-most vertex is (1, 0).
        # Gap = sqrt3 - 1 - 1 = sqrt3 - 2 < 0. They overlap.
        # Distance between centers must be such that they touch.
        # Distance between center and vertex is 1.
        # Distance between center and midpoint of side is sqrt3/2.
        # For two hexagons to share a side, the distance between centers is 2 * (sqrt3/2) = sqrt3.
        # So, lattice spacing is sqrt3 in the "flat" direction?
        # Let's use axial coordinates scaled by sqrt3.
        # Axial coord (q, r).
        # x = sqrt3 * (q + r/2)
        # y = 1.5 * r
        # This is for pointy-top?
        # Let's stick to simple Cartesian for the generator.
        # Neighbor vectors for side length 1 hexagons:
        # v1 = (sqrt(3), 0)
        # v2 = (sqrt(3)/2, 1.5)
        # v3 = (-sqrt(3)/2, 1.5)
        # v4 = (-sqrt(3), 0)
        # v5 = (-sqrt(3)/2, -1.5)
        # v6 = (sqrt(3)/2, -1.5)
        
        hexes = []
        hexes.append([0.0, 0.0, 0.0]) # Center
        # Ring 1
        hexes.append([np.sqrt(3), 0.0, 0.0])
        hexes.append([np.sqrt(3)/2, 1.5, 0.0])
        hexes.append([-np.sqrt(3)/2, 1.5, 0.0])
        hexes.append([-np.sqrt(3), 0.0, 0.0])
        hexes.append([-np.sqrt(3)/2, -1.5, 0.0])
        hexes.append([np.sqrt(3)/2, -1.5, 0.0])
        
        # Add 5 more to make 12.
        # Attach to the "outside" of the ring.
        # e.g. right of (sqrt3, 0) -> (2*sqrt3, 0)
        hexes.append([2*np.sqrt(3), 0.0, 0.0])
        # right of (sqrt3/2, 1.5) -> (3*sqrt3/2, 3.0)?
        # Vector (sqrt3, 0) added to (sqrt3/2, 1.5) -> (3sqrt3/2, 1.5)
        hexes.append([1.5*np.sqrt(3), 1.5, 0.0])
        # Vector (sqrt3/2, 1.5) added to (sqrt3/2, 1.5) -> (sqrt3, 3.0)
        hexes.append([np.sqrt(3), 3.0, 0.0])
        # Vector (-sqrt3/2, 1.5) added to (-sqrt3/2, 1.5) -> (-sqrt3, 3.0)
        hexes.append([-np.sqrt(3), 3.0, 0.0])
        # Vector (-sqrt3, 0) added to (-sqrt3, 0) -> (-2sqrt3, 0)
        hexes.append([-2*np.sqrt(3), 0.0, 0.0])
        
        return hexes
    else:
        return generate_topology("grid_3x4")

def construct():
    start_time = time.time()
    deadline = 120.0 # Safety margin for 150s limit
    
    topologies = ["grid_3x4", "grid_2x6", "cluster_1_6_5"]
    angles_to_try = [0.0, 30.0, 15.0, 45.0, 60.0] # Outer hexagon rotation
    
    best_L = float('inf')
    best_config = None
    
    for topo_name in topologies:
        if time.time() > deadline: break
        inner_hexes = generate_topology(topo_name)
        
        # Precompute vertices for all 12 hexagons
        # Side = 1.0
        all_vertices = []
        for hx, hy, ang in inner_hexes:
            verts = hex_vertices((hx, hy), 1.0, ang)
            all_vertices.append(verts)
        inner_vertices = np.vstack(all_vertices) # (72, 2)
        
        # Center of mass of the cluster to start search
        cx_start = np.mean(inner_vertices[:, 0])
        cy_start = np.mean(inner_vertices[:, 1])
        
        # Bounding box for center search
        x_min = np.min(inner_vertices[:, 0])
        x_max = np.max(inner_vertices[:, 0])
        y_min = np.min(inner_vertices[:, 1])
        y_max = np.max(inner_vertices[:, 1])
        
        # Search range for outer center
        # The outer center should be near the center of mass
        # Range: +/- 2.0 units
        search_range = 3.0
        cx_range = (cx_start - search_range, cx_start + search_range)
        cy_range = (cy_start - search_range, cy_start + search_range)
        
        # Coarse grid
        num_points = 50
        cx_vals = np.linspace(cx_range[0], cx_range[1], num_points)
        cy_vals = np.linspace(cy_range[0], cy_range[1], num_points)
        
        for outer_angle in angles_to_try:
            if time.time() > deadline: break
            
            # Normals for outer hexagon
            # Face normals are at angles outer_angle + 30, 90, ...
            # Wait, vertices are at angle. Faces are perpendicular to radius.
            # Radius to vertex is angle. Normal to face is angle + 30?
            # Let's check.
            # Vertex at 0 deg (1,0).
            # Face between 0 and 60 deg.
            # Midpoint of face is at 30 deg.
            # Normal points outwards at 30 deg.
            # So normals are at outer_angle + 30 + i*60.
            normal_angles = np.deg2rad(outer_angle + 30 + np.arange(0, 360, 60))
            normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles))) # (6, 2)
            
            # Precompute inner_vertices @ normals.T
            # (72, 2) @ (2, 6) -> (72, 6)
            proj_matrix = inner_vertices @ normals.T
            
            best_L_for_angle = float('inf')
            best_cx = 0
            best_cy = 0
            
            for cx in cx_vals:
                if time.time() > deadline: break
                for cy in cy_vals:
                    C = np.array([cx, cy])
                    # C @ normals.T -> (6,)
                    c_proj = C @ normals.T
                    
                    # max_proj[i] is max v . n_i
                    max_proj = np.max(proj_matrix, axis=0)
                    
                    # required_L[i] = (max_proj[i] - c_proj[i]) / (sqrt(3)/2)
                    # We need L >= max(required_L)
                    # Also, we must ensure the other side?
                    # The formula max_proj - c_proj gives the distance from C to the furthest point in direction n_i.
                    # The distance from C to the face of the outer hexagon is L * sqrt(3)/2.
                    # So we need L * sqrt(3)/2 >= max_proj - c_proj.
                    # This assumes the face is in the direction n_i.
                    # Since we check all 6 normals, this covers the whole hexagon.
                    
                    required_L = (max_proj - c_proj) / (np.sqrt(3)/2)
                    L_candidate = np.max(required_L)
                    
                    if L_candidate < best_L_for_angle:
                        best_L_for_angle = L_candidate
                        best_cx = cx
                        best_cy = cy
            
            if best_L_for_angle < best_L:
                best_L = best_L_for_angle
                # Store best config
                # We need to return inner, center, side, angle
                # center is (best_cx, best_cy)
                # side is best_L
                # angle is outer_angle
                # inner is the list of hexes
                best_config = (inner_hexes, (best_cx, best_cy), best_L, outer_angle)
                
                # Refine search around (best_cx, best_cy)
                # Smaller grid
                refine_range = 0.5
                cx_ref = np.linspace(best_cx - refine_range, best_cx + refine_range, 20)
                cy_ref = np.linspace(best_cy - refine_range, best_cy + refine_range, 20)
                
                best_L_refined = best_L_for_angle
                best_cx_ref = best_cx
                best_cy_ref = best_cy
                
                for cx in cx_ref:
                    if time.time() > deadline: break
                    for cy in cy_ref:
                        C = np.array([cx, cy])
                        c_proj = C @ normals.T
                        required_L = (max_proj - c_proj) / (np.sqrt(3)/2)
                        L_candidate = np.max(required_L)
                        if L_candidate < best_L_refined:
                            best_L_refined = L_candidate
                            best_cx_ref = cx
                            best_cy_ref = cy
                            
                if best_L_refined < best_L:
                    best_L = best_L_refined
                    best_config = (inner_hexes, (best_cx_ref, best_cy_ref), best_L, outer_angle)

    if best_config is None:
        # Fallback to seed
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    return best_config

# EVOLVE-BLOCK-END
