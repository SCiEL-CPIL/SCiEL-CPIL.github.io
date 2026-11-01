"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
import math
from scipy.optimize import minimize_scalar

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = SQRT3 / 2.0

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as (x, y) tuples."""
    angles = angle_rad + np.pi / 3.0 * np.arange(6)
    return [(cx + side * math.cos(a), cy + side * math.sin(a)) for a in angles]

def compute_required_side_for_centers(centers, outer_angle_deg=0.0):
    """
    Compute the minimum outer side length to contain unit hexagons centered at 'centers'.
    The outer hexagon is centered at (0,0) and rotated by outer_angle_deg.
    """
    if not centers:
        return 0.0
    
    ox, oy = 0.0, 0.0
    # Normals for the outer hexagon sides.
    # A regular hexagon with side S has apothem S * sqrt(3) / 2.
    # The vertices are at distance S from center.
    # The normals point to the midpoints of the sides.
    # For a "pointy-top" hexagon (vertex at 90 deg), the normals are at 0, 60, 120...
    # Let's stick to the convention: Outer hexagon vertices at angles 0, 60, 120...
    # Then normals are at 30, 90, 150...
    # Wait, the previous code used normals at 30 + i*60.
    # Let's use that.
    
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    max_projection = 0.0
    
    # For each inner hexagon center, the max projection of the hexagon itself
    # onto a direction n is center_proj + apothem_inner.
    # apothem_inner = sqrt(3)/2.
    
    for (px, py) in centers:
        # Rotate center by -outer_angle to align with outer normals?
        # No, it's easier to rotate the normals or the points.
        # Let's rotate the points by -outer_angle_deg so we can use fixed normals.
        lx = px * cos_a + py * sin_a
        ly = -px * sin_a + py * cos_a
        
        for nx, ny in normals:
            proj = lx * nx + ly * ny
            # The hexagon extends by APOTHEM in the normal direction.
            total_proj = proj + APOTHEM
            if total_proj > max_projection:
                max_projection = total_proj
    
    # The outer hexagon must have apothem >= max_projection.
    # Apothem_outer = S * sqrt(3) / 2.
    # S = 2 * max_projection / sqrt(3).
    return 2.0 * max_projection / SQRT3

def axial_to_cartesian(q, r, scale=1.0):
    """Convert axial coordinates to Cartesian. Scale=1.0 means touching hexagons."""
    # Distance between centers of touching unit hexagons is sqrt(3).
    # Standard axial to cartesian:
    # x = 1.5 * q * side
    # y = sqrt(3) * (r + q/2.0) * side
    # Here side=1.
    x = 1.5 * scale * q
    y = SQRT3 * scale * (r + q / 2.0)
    return (x, y)

def generate_candidate_shapes():
    """
    Generate a list of candidate configurations of 11 unit hexagons.
    Each configuration is a list of 11 (q, r) axial coordinates.
    """
    shapes = []
    
    # Base shape: Center + Ring 1 + 4 from Ring 2
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 candidates: (2,-1), (1,1), (-2,1), (-1,-1), (0,2), (2,0), etc.
    # We want to pick 4 from Ring 2 that are "compact".
    
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    center = (0,0)
    
    # All valid Ring 2 positions (distance 2 from center in hex metric)
    ring2_all = [
        (2, -1), (1, 1), (-2, 1), (-1, -1), (0, 2), (2, 0), 
        (-1, 2), (1, -2), (-2, -1), (0, -2), (-2, 0), (2, 1)
    ]
    
    # We need to choose 4 from ring2_all.
    # To keep it simple and fast, we'll try a few specific "compact" sets.
    
    # Set 1: 4 contiguous in one quadrant
    set1 = [(2, -1), (1, 1), (-2, 1), (-1, -1)]
    shapes.append([center] + ring1 + set1)
    
    # Set 2: 2 in one quadrant, 2 in opposite
    set2 = [(2, -1), (1, 1), (0, 2), (2, 0)]
    shapes.append([center] + ring1 + set2)
    
    # Set 3: Spread out more
    set3 = [(2, -1), (-2, 1), (0, 2), (0, -2)]
    shapes.append([center] + ring1 + set3)
    
    # Set 4: "Star" like
    set4 = [(2, -1), (1, 1), (-2, 1), (1, -2)]
    shapes.append([center] + ring1 + set4)
    
    # Set 5: All 4 in one half
    set5 = [(2, -1), (1, 1), (2, 0), (1, -2)] # Check validity/duplicates
    # (1,-2) is in ring2? 
    # Hex distance (q,r) from (0,0) is (|q| + |r| + |q+r|)/2.
    # (1,-2): |1|+|-2|+|-1| = 4 -> dist 2. Yes.
    shapes.append([center] + ring1 + set5)
    
    # Set 6: Another variation
    set6 = [(2, -1), (1, 1), (-2, 1), (2, 0)]
    shapes.append([center] + ring1 + set6)
    
    # Set 7: Try to make it more circular
    set7 = [(2, -1), (1, 1), (-1, 2), (-2, 1)]
    shapes.append([center] + ring1 + set7)
    
    # Set 8: 
    set8 = [(2, -1), (1, 1), (-1, -1), (-2, 1)] # Same as set1?
    # Set1 was (2,-1), (1,1), (-2,1), (-1,-1). Yes, duplicate.
    
    # Let's add some "perturbed" shapes by moving one hexagon to ring 3 if it fits better?
    # For now, stick to ring 2 to ensure compactness.
    
    return shapes

def construct():
    """Main entry point: find optimal packing of N unit hexagons in minimum outer hexagon."""
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_S = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    shapes = generate_candidate_shapes()
    
    for shape in shapes:
        if time.time() > deadline - 20:
            break
            
        # Convert axial to Cartesian centers (scale 1.0 = touching)
        centers = [axial_to_cartesian(q, r, 1.0) for q, r in shape]
        
        # We need to find the optimal rotation angle of the cluster relative to the outer hexagon.
        # Or equivalently, the optimal orientation of the outer hexagon.
        # Let's minimize S(alpha) where alpha is the rotation of the outer hexagon.
        # S(alpha) = compute_required_side_for_centers(centers, alpha)
        
        def obj_alpha(alpha_rad):
            alpha_deg = math.degrees(alpha_rad)
            return compute_required_side_for_centers(centers, alpha_deg)
        
        # Search for minimum in [0, 60 degrees] due to symmetry
        res = minimize_scalar(obj_alpha, bounds=(0, math.pi/3), method='bounded', 
                              options={'xatol': 1e-8})
        
        if res.success:
            S = res.fun
            alpha_opt = res.x
            alpha_opt_deg = math.degrees(alpha_opt)
            
            if S < best_S:
                best_S = S
                best_outer_angle = alpha_opt_deg
                # The inner hexagons are at 'centers' with rotation 0.
                # The outer hexagon is rotated by alpha_opt_deg.
                # We need to return the inner hexagons' coordinates and angles.
                # The inner hexagons are axis-aligned (angle 0) in this frame.
                # But the problem asks for (x, y, angle_degrees) for each inner hexagon.
                # If we fix the outer hexagon at 0 degrees, we should rotate the inner ones.
                # Or keep inner at 0 and report outer angle.
                # The interface: inner rows (x, y, angle_degrees), outer center, side, angle.
                # So we can keep inner at angle 0 and set outer angle to alpha_opt_deg.
                
                best_inner = [(px, py, 0.0) for px, py in centers]
                best_outer_center = (0.0, 0.0)
        
        # Also try rotating the inner hexagons themselves?
        # In a hexagonal lattice, rotating individual hexagons doesn't help if they are touching,
        # as they would overlap or create gaps. The optimal packing is usually a rigid body rotation
        # of the whole cluster.
        
        # However, we might be able to "squeeze" the cluster slightly.
        # Let's try scaling the cluster by 0.99, 0.98 etc. to see if it fits in a smaller outer hexagon.
        # This is a heuristic.
        for scale in [0.99, 0.98, 0.97]:
            if time.time() > deadline - 10:
                break
            centers_scaled = [axial_to_cartesian(q, r, scale) for q, r in shape]
            
            def obj_alpha_scaled(alpha_rad):
                alpha_deg = math.degrees(alpha_rad)
                return compute_required_side_for_centers(centers_scaled, alpha_deg)
            
            res_scaled = minimize_scalar(obj_alpha_scaled, bounds=(0, math.pi/3), method='bounded', 
                                      options={'xatol': 1e-8})
            
            if res_scaled.success:
                S_scaled = res_scaled.fun
                alpha_opt_scaled = res_scaled.x
                alpha_opt_deg_scaled = math.degrees(alpha_opt_scaled)
                
                if S_scaled < best_S:
                    best_S = S_scaled
                    best_outer_angle = alpha_opt_deg_scaled
                    best_inner = [(px, py, 0.0) for px, py in centers_scaled]
                    best_outer_center = (0.0, 0.0)

    # Fallback if no shape found or error
    if best_inner is None:
        # Use the first shape with scale 1.0 and angle 0
        shape0 = shapes[0]
        centers0 = [axial_to_cartesian(q, r, 1.0) for q, r in shape0]
        best_inner = [(px, py, 0.0) for px, py in centers0]
        best_S = compute_required_side_for_centers(centers0, 0.0)
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0

    return (best_inner, best_outer_center, best_S, best_outer_angle)

# EVOLVE-BLOCK-END
