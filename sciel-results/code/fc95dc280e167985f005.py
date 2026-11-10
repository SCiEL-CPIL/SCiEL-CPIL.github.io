# EVOLVE-BLOCK-START
import numpy as np
import time

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    
    # Configuration: 2-4-4-2 rows
    # This shape is roughly square/hexagonal and compact.
    sqrt3 = np.sqrt(3)
    
    # Row definitions: (count, y_offset, x_start_offset)
    # We will center the cluster at the origin.
    
    # Generate centers for 2-4-4-2
    centers = []
    # Row 1 (top): 2 hexagons
    # y = 2.25
    # x = -0.5*sqrt3, 0.5*sqrt3
    centers.append([-0.5*sqrt3, 2.25])
    centers.append([0.5*sqrt3, 2.25])
    
    # Row 2: 4 hexagons
    # y = 0.75
    # x = -1.5*sqrt3, -0.5*sqrt3, 0.5*sqrt3, 1.5*sqrt3
    centers.append([-1.5*sqrt3, 0.75])
    centers.append([-0.5*sqrt3, 0.75])
    centers.append([0.5*sqrt3, 0.75])
    centers.append([1.5*sqrt3, 0.75])
    
    # Row 3: 4 hexagons
    # y = -0.75
    # x = -1.5*sqrt3, -0.5*sqrt3, 0.5*sqrt3, 1.5*sqrt3
    centers.append([-1.5*sqrt3, -0.75])
    centers.append([-0.5*sqrt3, -0.75])
    centers.append([0.5*sqrt3, -0.75])
    centers.append([1.5*sqrt3, -0.75])
    
    # Row 4 (bottom): 2 hexagons
    # y = -2.25
    # x = -0.5*sqrt3, 0.5*sqrt3
    centers.append([-0.5*sqrt3, -2.25])
    centers.append([0.5*sqrt3, -2.25])
    
    centers = np.array(centers)
    
    # Inner hexagons are aligned (angle 0)
    inner_hex_data = np.column_stack((centers[:, 0], centers[:, 1], np.zeros(12)))
    
    # Function to calculate minimum outer side length s for a given outer angle
    def calc_min_s(centers, outer_angle_deg):
        # Outer hexagon vertices/normal calculation
        # The outer hexagon has side s, centered at origin.
        # We need to find s such that all inner hexagon vertices are inside.
        
        # Inner hexagon vertices relative to center (0,0) with angle 0
        # v_k = (cos(60k), sin(60k))
        inner_verts_rel = np.array([
            [1, 0],
            [0.5, sqrt3/2],
            [-0.5, sqrt3/2],
            [-1, 0],
            [-0.5, -sqrt3/2],
            [0.5, -sqrt3/2]
        ])
        
        # Generate all absolute vertices for all inner hexagons
        all_verts = []
        for c in centers:
            all_verts.extend(c + inner_verts_rel)
        all_verts = np.array(all_verts)
        
        # Outer hexagon normals
        # If outer angle is a, the vertices are at a, a+60, ...
        # The normals to the edges are at a+30, a+90, ...
        theta = np.deg2rad(outer_angle_deg + 30)
        normals = []
        for i in range(6):
            phi = theta + i * np.deg2rad(60)
            normals.append([np.cos(phi), np.sin(phi)])
        normals = np.array(normals)
        
        # The condition for a point v to be inside a regular hexagon of side s (apothem A = s*sqrt(3)/2)
        # is |v . n_i| <= A for all i.
        # So A >= max(|v . n_i|) for all v, i.
        # s = 2/sqrt(3) * A.
        
        max_proj = 0
        for n in normals:
            projs = np.abs(all_verts @ n)
            max_proj = max(max_proj, np.max(projs))
            
        return (2 / sqrt3) * max_proj

    # Try outer angles 0 and 30
    s_0 = calc_min_s(centers, 0)
    s_30 = calc_min_s(centers, 30)
    
    if s_0 <= s_30:
        best_s = s_0
        best_outer_angle = 0
    else:
        best_s = s_30
        best_outer_angle = 30
        
    outer_hex_data = np.array([0, 0, best_outer_angle])
    
    return inner_hex_data, outer_hex_data, best_s

# EVOLVE-BLOCK-END
