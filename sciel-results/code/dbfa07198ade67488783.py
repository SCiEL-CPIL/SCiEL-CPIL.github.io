"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Strategy: Compact hexagonal lattice packing.
    # We place 12 unit hexagons in a "flower" pattern:
    # 1 at center, 6 in first ring, 5 in second ring (to make 12 total).
    # Actually, a standard hexagonal number packing for 12 items is:
    # Center (1)
    # Ring 1 (6)
    # Ring 2 (5 of 12) -> This is irregular.
    # Better: A 3x4 rectangular grid of hexagons? Or a hexagon of radius 2 is 19 hexagons.
    # Let's use a "bar" of 2 hexagons wide and 6 long?
    # Side length of unit hex is 1. Distance between centers in hex lattice is sqrt(3).
    # Let's try a 2x6 grid of hexagons aligned with the lattice.
    
    # Coordinates for a 2x6 grid of hexagons in a hex lattice:
    # Hex lattice basis: v1 = (sqrt(3), 0), v2 = (sqrt(3)/2, 1.5)
    # Wait, distance between centers is 2 * apothem = sqrt(3).
    # Let's define the center positions.
    # A 2x6 grid means 2 rows, 6 columns.
    # Row 0: y=0, x = 0, sqrt(3), 2*sqrt(3), ...
    # Row 1: y=1.5, x = sqrt(3)/2, 3*sqrt(3)/2, ... (offset)
    
    centers = []
    sqrt3 = np.sqrt(3)
    
    # 6 columns, 2 rows
    for row in range(2):
        for col in range(6):
            x = col * sqrt3
            y = row * 1.5
            if row == 1:
                x += sqrt3 / 2.0
            centers.append([x, y, 0.0])
            
    # Total 12 centers.
    inner = centers
    
    # Now, we need to find the minimum side length of a regular hexagon 
    # that can contain all these unit hexagons.
    # The outer hexagon can be rotated.
    # The bounding box of the inner cluster determines the size.
    # For a regular hexagon of side S, the width (distance between parallel sides) is S * sqrt(3).
    # The height (distance between opposite vertices) is 2 * S.
    
    # We need to find the minimum S such that there exists a rotation theta and translation (cx, cy)
    # where all inner hexagons are inside the outer hexagon.
    
    # Approximation:
    # The "width" of the cluster in any direction must be <= S * sqrt(3) if that direction is parallel to a side of the outer hex.
    # Or more generally, the support function of the cluster in the direction of the outer hex's normals must be <= S.
    
    # Let's compute the bounding box of the centers + the unit hexagon extents.
    # Since all inner hexagons are axis-aligned (angle 0), their vertices are fixed relative to their centers.
    # Vertices of a unit hexagon at (0,0) with angle 0 (pointy top? or flat top?):
    # Standard math: angle 0 is usually along x-axis.
    # Let's assume "flat top" or "pointy top"? 
    # The problem says "regular hexagon". Usually, angle 0 in these tasks implies one vertex is at (1,0) or similar.
    # Let's assume the standard: vertices at angles 0, 60, 120, 180, 240, 300 degrees.
    # So vertices are (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2).
    # This is a "pointy-top" hexagon if y is up? No, (1,0) is right. So it has a vertex on the x-axis.
    # The width in x is 2. The width in y is sqrt(3).
    
    # Let's compute the min/max x and y of the entire cluster.
    # But the outer hexagon can be rotated.
    # The optimal outer hexagon will likely align with the cluster's symmetry or bounding box.
    
    # Let's try a few outer hexagon rotations: 0, 30, 60, 90, 120, 150 degrees.
    # For a given outer rotation phi, the outer hexagon has 6 supporting lines.
    # The distance from the center of the outer hex to each supporting line is S.
    # We need to find the smallest S such that the cluster fits.
    
    # The cluster is a set of 12 unit hexagons.
    # Let's approximate the cluster by its bounding box of centers, plus the max radius of a unit hexagon in any direction.
    # Max radius of unit hexagon is 1 (vertex to center).
    # So if we find the center of mass of the cluster, and the max distance from that center to any point in the cluster (which is roughly max_center_dist + 1), 
    # that gives a circle bound. But we want a hexagon bound.
    
    # Let's use a more precise method:
    # 1. Compute all vertices of all inner hexagons.
    # 2. For a candidate outer hexagon orientation (angle), compute the required side length S.
    #    The outer hexagon is defined by 6 lines. 
    #    If the outer hexagon is centered at (0,0) for simplicity (we can translate later), 
    #    and rotated by angle alpha, its sides are at distances S from the center.
    #    Actually, the distance from center to side is the apothem a = S * sqrt(3)/2.
    #    So the constraint is: for each of the 6 outward normals n_k of the outer hexagon,
    #    max_{p in cluster} (p . n_k) <= a_outer + min_{p in cluster} (p . n_k) is not right.
    
    # Let's center the cluster first.
    # Compute centroid of all inner hexagon vertices.
    all_verts = []
    for x, y, ang in inner:
        # Vertices of unit hexagon at (x,y) with rotation ang
        base_angles = np.arange(6) * np.pi / 3 + np.radians(ang)
        vs = np.column_stack([x + np.cos(base_angles), y + np.sin(base_angles)])
        all_verts.append(vs)
    all_verts = np.vstack(all_verts)
    
    centroid = np.mean(all_verts, axis=0)
    
    # Translate cluster so centroid is at (0,0)
    all_verts_centered = all_verts - centroid
    
    # Now find min S for various outer rotations.
    # Outer hexagon normals are at angles: alpha, alpha+60, ..., alpha+300.
    # The condition for a point p to be inside a regular hexagon centered at origin with side S and rotation alpha is:
    # For each normal n_k (unit vector), p . n_k <= S * sqrt(3)/2.
    # And p . (-n_k) <= S * sqrt(3)/2.
    # So |p . n_k| <= S * sqrt(3)/2 for all k.
    # Thus, S >= max_{p, k} |p . n_k| / (sqrt(3)/2) = max_{p, k} |p . n_k| * 2 / sqrt(3).
    
    best_S = float('inf')
    best_alpha = 0
    
    # Try 12 orientations (every 30 degrees) to cover the symmetry
    for alpha_deg in range(0, 180, 30):
        alpha = np.radians(alpha_deg)
        normals = []
        for k in range(6):
            angle = alpha + k * np.pi / 3
            normals.append(np.array([np.cos(angle), np.sin(angle)]))
        normals = np.array(normals)
        
        # Compute max projection
        # all_verts_centered shape (N_pts, 2)
        # normals shape (6, 2)
        # Projections: (N_pts, 6)
        projs = all_verts_centered @ normals.T
        max_abs_proj = np.max(np.abs(projs))
        
        S_candidate = max_abs_proj * 2.0 / np.sqrt(3)
        
        if S_candidate < best_S:
            best_S = S_candidate
            best_alpha = alpha_deg
            
    # The outer hexagon should be centered at the centroid of the cluster.
    # We return the inner coordinates relative to the outer center.
    # Current `inner` is in world coords. 
    # We need to shift `inner` so that the outer hexagon center is at (0,0) or we return the outer center.
    # The interface: inner rows (x, y, angle), outer center (x, y), side, angle.
    # We can set outer center to (0,0) and shift inner coords by -centroid.
    
    shifted_inner = []
    for x, y, ang in inner:
        shifted_inner.append([x - centroid[0], y - centroid[1], ang])
        
    return shifted_inner, (0.0, 0.0), best_S, best_alpha
# EVOLVE-BLOCK-END
