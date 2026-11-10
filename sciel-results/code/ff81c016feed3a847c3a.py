# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _min_containing_hex_side(inner_hexes):
    """
    Given inner hexagons as list of (cx, cy, angle_deg), find the minimal side length s
    of a regular hexagon (centered at origin, angle 0) that contains all of them.
    Returns s.
    """
    all_verts = []
    for cx, cy, ang in inner_hexes:
        verts = _hex_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)  # (72, 2)
    
    # Normals for outer hexagon edges (pointing outwards)
    # Outer hexagon side s, angle 0.
    # Vertices at angles 0, 60, ...
    # Edges midpoints at 30, 90, ...
    # Normals at 30, 90, ...
    normals = np.array([[np.cos(np.radians(60 * (i + 0.5))), 
                         np.sin(np.radians(60 * (i + 0.5)))] for i in range(6)])
    
    dots = all_verts @ normals.T  # (72, 6)
    max_dots = np.max(dots, axis=1)
    s_required = max_dots / (np.sqrt(3) / 2)
    s = np.max(s_required)
    return s


def _objective(params):
    """
    Objective function: given rotation angle (radians), compute the containing hexagon side.
    Uses a compact 12-hexagon cluster.
    """
    theta = params[0]
    
    # Compact cluster of 12 hexagons in axial coordinates
    # Center: (0,0)
    # Ring 1: (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (0,1)
    # Ring 2 (partial): (2,-1), (2,-2), (1,-2), (0,-2), (-1,-2)
    axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (0, 1),
        (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2)
    ]
    
    sqrt3 = np.sqrt(3)
    centers = []
    for q, r in axial:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append([x, y])
    centers = np.array(centers)  # (12, 2)
    
    # Center the configuration
    center = np.mean(centers, axis=0)
    centers = centers - center
    
    # Scale to ensure separation > 1e-6
    # Base distance is sqrt(3). We need distance > sqrt(3) + 1e-6.
    # Let's add a safety margin.
    scale_factor = (sqrt3 + 5e-6) / sqrt3
    centers = centers * scale_factor
    
    # Rotate by theta
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    centers = centers @ R.T
    
    # All inner hexagons have angle 0 (aligned with outer)
    inner_hexes = [(cx, cy, 0.0) for cx, cy in centers]
    
    # Compute containing hexagon side
    s = _min_containing_hex_side(inner_hexes)
    return s


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    best_s = np.inf
    best_theta = 0.0
    
    # Grid search for rotation
    n_angles = 360
    for i in range(n_angles):
        if time.time() > deadline - 5.0:
            break
        theta = 2 * np.pi * i / n_angles
        s = _objective([theta])
        if s < best_s:
            best_s = s
            best_theta = theta
    
    # Refine
    if time.time() < deadline - 10.0:
        # Golden section search
        from math import pi
        phi = (1 + np.sqrt(5)) / 2
        inv_phi = 1.0 / phi
        
        a = best_theta - 2 * np.pi / n_angles
        b = best_theta + 2 * np.pi / n_angles
        
        c = b - inv_phi * (b - a)
        d = a + inv_phi * (b - a)
        fc = _objective([c])
        fd = _objective([d])
        
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - inv_phi * (b - a)
                fc = _objective([c])
            else:
                a = c
                c = d
                fc = fd
                d = a + inv_phi * (b - a)
                fd = _objective([d])
            
            mid = (a + b) / 2
            s_mid = _objective([mid])
            if s_mid < best_s:
                best_s = s_mid
                best_theta = mid

    # Final refinement
    if time.time() < deadline - 5.0:
        for offset in np.linspace(-0.1, 0.1, 21):
            if time.time() > deadline - 5.0:
                break
            theta = best_theta + offset
            s = _objective([theta])
            if s < best_s:
                best_s = s
                best_theta = theta
    
    # Construct the final answer
    axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (0, 1),
        (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2)
    ]
    
    sqrt3 = np.sqrt(3)
    centers = []
    for q, r in axial:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append([x, y])
    centers = np.array(centers)
    center = np.mean(centers, axis=0)
    centers = centers - center
    
    scale_factor = (sqrt3 + 5e-6) / sqrt3
    centers = centers * scale_factor
    
    R = np.array([[np.cos(best_theta), -np.sin(best_theta)],
                  [np.sin(best_theta), np.cos(best_theta)]])
    centers = centers @ R.T
    
    inner_hex_data = np.zeros((12, 3))
    for i, (cx, cy) in enumerate(centers):
        inner_hex_data[i] = [cx, cy, 0.0]
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = best_s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
