# EVOLVE-BLOCK-START
import numpy as np

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses a hexagonal lattice arrangement and computes the minimal containing hexagon.
    """
    # For flat-top unit hexagons (angle=0), vertices are at angles 0,60,120,180,240,300
    # The hexagonal lattice for non-overlapping flat-top hexagons:
    # Adjacent hexagons in the "horizontal" direction have center distance = 3
    # Wait, let me recalculate. For two flat-top hexagons to touch:
    # Left hex rightmost vertex: (cx+1, cy)
    # Right hex leftmost vertex: (cx2-1, cy)  
    # They touch when cx2-1 = cx+1, so cx2 - cx = 2
    # But they also have slanted edges. Let me check if d=2 works.
    
    # Actually, the correct spacing for hexagonal lattice of flat-top hexagons:
    # The distance between centers of adjacent hexagons (sharing an edge) is 2*apothem = sqrt(3)
    # But for flat-top orientation, the "horizontal" neighbors are not edge-sharing.
    
    # Let me use a different approach: place hexagons in a pattern and verify no overlap.
    # Use a 3-row arrangement: 4, 4, 4 with proper hexagonal offset.
    
    # For flat-top hexagons, the lattice vectors for a hexagonal packing are:
    # v1 = (3, 0) - horizontal neighbor (vertex to vertex)
    # v2 = (1.5, sqrt(3)/2 * 2) = (1.5, sqrt(3)) - diagonal neighbor
    # Wait, I need to be more careful.
    
    # Let me just use a rectangular grid with sufficient spacing and then compute the bounding hexagon.
    # For flat-top unit hexagons:
    # - Width = 2 (left to right)
    # - Height = sqrt(3) (top to bottom)
    # To avoid overlap, horizontal spacing >= 2 + epsilon, vertical spacing >= sqrt(3) + epsilon
    # But with hexagonal offset, we can reduce this.
    
    # Actually, the simplest valid packing: use a rectangular grid with spacing 3 in x and 2.5 in y.
    # This guarantees no overlap. Then compute the minimal regular hexagon containing all.
    
    # Better: use the natural hexagonal lattice.
    # For flat-top hexagons, the centers of a hexagonal lattice are at:
    # (i * 3 + (j % 2) * 1.5, j * sqrt(3))
    # for i, j integers. This gives non-overlapping hexagons.
    
    # Let's use 4 columns and 3 rows:
    # Row 0 (j=0): i = 0,1,2,3 -> x = 0, 3, 6, 9; y = 0
    # Row 1 (j=1): i = 0,1,2,3 -> x = 1.5, 4.5, 7.5, 10.5; y = sqrt(3)
    # Row 2 (j=2): i = 0,1,2,3 -> x = 0, 3, 6, 9; y = 2*sqrt(3)
    
    # Center this arrangement at origin.
    # X range: 0 to 10.5, center at 5.25
    # Y range: 0 to 2*sqrt(3), center at sqrt(3)
    
    centers = []
    for j in range(3):
        for i in range(4):
            x = i * 3.0 + (j % 2) * 1.5
            y = j * np.sqrt(3)
            centers.append((x, y))
    
    # Center the arrangement
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    cx_mean = (min(xs) + max(xs)) / 2
    cy_mean = (min(ys) + max(ys)) / 2
    
    centers_centered = [(x - cx_mean, y - cy_mean) for x, y in centers]
    
    # All hexagons have angle 0 (flat-top)
    angles = [0.0] * 12
    
    # Now compute the minimal regular hexagon (centered at origin, angle 0) that contains all 12 hexagons.
    # A regular hexagon with side s, centered at origin, angle 0 has vertices at:
    # (s*cos(k*60°), s*sin(k*60°)) for k=0..5
    # The constraint is that all vertices of all inner hexagons must be inside this outer hexagon.
    
    # For a point (px, py) to be inside a regular hexagon with side s, centered at origin, angle 0:
    # The hexagon has 6 edges. For each edge, the point must be on the correct side.
    # Edge k connects vertex k and vertex (k+1)%6.
    # The inward normal of edge k points toward the center.
    
    # For a regular hexagon with side s, the apothem (distance from center to edge) is s*sqrt(3)/2.
    # The 6 edge normals (pointing inward) are at angles 30°, 90°, 150°, 210°, 270°, 330° (for flat-top).
    # Wait, for angle=0, vertices are at 0°,60°,120°,180°,240°,300°.
    # Edge 0: from (s,0) to (s/2, s*sqrt(3)/2)
    # The inward normal of this edge points toward the origin.
    
    # Actually, let me just compute the required s by checking all vertices of all inner hexagons.
    # For each inner hexagon vertex (vx, vy), find the minimum s such that (vx,vy) is inside the outer hexagon.
    
    # A point (vx,vy) is inside a regular hexagon with side s, centered at origin, angle 0 iff:
    # For each of the 6 edges, the cross product test passes.
    # Equivalently, the point satisfies 6 linear inequalities.
    
    # The 6 edges of the outer hexagon (CCW order):
    # Edge 0: from (s, 0) to (s/2, s*sqrt(3)/2)
    # Edge 1: from (s/2, s*sqrt(3)/2) to (-s/2, s*sqrt(3)/2)
    # Edge 2: from (-s/2, s*sqrt(3)/2) to (-s, 0)
    # Edge 3: from (-s, 0) to (-s/2, -s*sqrt(3)/2)
    # Edge 4: from (-s/2, -s*sqrt(3)/2) to (s/2, -s*sqrt(3)/2)
    # Edge 5: from (s/2, -s*sqrt(3)/2) to (s, 0)
    
    # For a point to be inside, it must be to the left of each directed edge (CCW polygon).
    # For edge from p1 to p2, the point v is inside if cross(p2-p1, v-p1) >= 0.
    
    # Let me derive the 6 constraints:
    # Edge 0: p1=(s,0), p2=(s/2, s*sqrt(3)/2)
    #   p2-p1 = (-s/2, s*sqrt(3)/2)
    #   v-p1 = (vx-s, vy)
    #   cross = (-s/2)*vy - (s*sqrt(3)/2)*(vx-s) >= 0
    #   => -vy/2 - sqrt(3)*(vx-s)/2 >= 0
    #   => -vy - sqrt(3)*vx + sqrt(3)*s >= 0
    #   => sqrt(3)*s >= sqrt(3)*vx + vy
    #   => s >= vx + vy/sqrt(3)
    
    # Edge 1: p1=(s/2, s*sqrt(3)/2), p2=(-s/2, s*sqrt(3)/2)
    #   p2-p1 = (-s, 0)
    #   v-p1 = (vx-s/2, vy-s*sqrt(3)/2)
    #   cross = (-s)*(vy-s*sqrt(3)/2) - 0 >= 0
    #   => -s*vy + s^2*sqrt(3)/2 >= 0
    #   => s^2*sqrt(3)/2 >= s*vy
    #   => s*sqrt(3)/2 >= vy  (assuming s>0)
    #   => s >= 2*vy/sqrt(3)
    
    # Edge 2: p1=(-s/2, s*sqrt(3)/2), p2=(-s, 0)
    #   p2-p1 = (-s/2, -s*sqrt(3)/2)
    #   v-p1 = (vx+s/2, vy-s*sqrt(3)/2)
    #   cross = (-s/2)*(vy-s*sqrt(3)/2) - (-s*sqrt(3)/2)*(vx+s/2) >= 0
    #   => -s*vy/2 + s^2*sqrt(3)/4 + s*sqrt(3)*vx/2 + s^2*sqrt(3)/4 >= 0
    #   => -s*vy/2 + s^2*sqrt(3)/2 + s*sqrt(3)*vx/2 >= 0
    #   => s^2*sqrt(3)/2 >= s*vy/2 - s*sqrt(3)*vx/2
    #   => s*sqrt(3)/2 >= vy/2 - sqrt(3)*vx/2
    #   => s*sqrt(3) >= vy - sqrt(3)*vx
    #   => s >= (vy - sqrt(3)*vx)/sqrt(3) = vy/sqrt(3) - vx
    
    # Edge 3: p1=(-s, 0), p2=(-s/2, -s*sqrt(3)/2)
    #   p2-p1 = (s/2, -s*sqrt(3)/2)
    #   v-p1 = (vx+s, vy)
    #   cross = (s/2)*vy - (-s*sqrt(3)/2)*(vx+s) >= 0
    #   => s*vy/2 + s*sqrt(3)*vx/2 + s^2*sqrt(3)/2 >= 0
    #   => s*vy/2 + s*sqrt(3)*vx/2 + s^2*sqrt(3)/2 >= 0
    #   => vy/2 + sqrt(3)*vx/2 + s*sqrt(3)/2 >= 0
    #   => s*sqrt(3)/2 >= -vy/2 - sqrt(3)*vx/2
    #   => s*sqrt(3) >= -vy - sqrt(3)*vx
    #   => s >= (-vy - sqrt(3)*vx)/sqrt(3) = -vy/sqrt(3) - vx
    
    # Edge 4: p1=(-s/2, -s*sqrt(3)/2), p2=(s/2, -s*sqrt(3)/2)
    #   p2-p1 = (s, 0)
    #   v-p1 = (vx+s/2, vy+s*sqrt(3)/2)
    #   cross = s*(vy+s*sqrt(3)/2) - 0 >= 0
    #   => s*vy + s^2*sqrt(3)/2 >= 0
    #   => s^2*sqrt(3)/2 >= -s*vy
    #   => s*sqrt(3)/2 >= -vy
    #   => s >= -2*vy/sqrt(3)
    
    # Edge 5: p1=(s/2, -s*sqrt(3)/2), p2=(s, 0)
    #   p2-p1 = (s/2, s*sqrt(3)/2)
    #   v-p1 = (vx-s/2, vy+s*sqrt(3)/2)
    #   cross = (s/2)*(vy+s*sqrt(3)/2) - (s*sqrt(3)/2)*(vx-s/2) >= 0
    #   => s*vy/2 + s^2*sqrt(3)/4 - s*sqrt(3)*vx/2 + s^2*sqrt(3)/4 >= 0
    #   => s*vy/2 + s^2*sqrt(3)/2 - s*sqrt(3)*vx/2 >= 0
    #   => s^2*sqrt(3)/2 >= -s*vy/2 + s*sqrt(3)*vx/2
    #   => s*sqrt(3)/2 >= -vy/2 + sqrt(3)*vx/2
    #   => s*sqrt(3) >= -vy + sqrt(3)*vx
    #   => s >= (sqrt(3)*vx - vy)/sqrt(3) = vx - vy/sqrt(3)
    
    # So the 6 constraints are:
    # s >= vx + vy/sqrt(3)
    # s >= 2*vy/sqrt(3)
    # s >= vy/sqrt(3) - vx
    # s >= -vy/sqrt(3) - vx
    # s >= -2*vy/sqrt(3)
    # s >= vx - vy/sqrt(3)
    
    # For a given point (vx, vy), the minimum s that contains it is:
    # s_min = max(vx + vy/sqrt(3), 2*vy/sqrt(3), vy/sqrt(3) - vx, -vy/sqrt(3) - vx, -2*vy/sqrt(3), vx - vy/sqrt(3))
    
    # We need s >= s_min for all vertices of all 12 inner hexagons.
    
    sqrt3 = np.sqrt(3)
    required_s = 0.0
    
    for idx, (cx, cy) in enumerate(centers_centered):
        ang = angles[idx]
        # Get vertices of this inner hexagon
        for k in range(6):
            angle_rad = np.deg2rad(ang + k * 60)
            vx = cx + np.cos(angle_rad)
            vy = cy + np.sin(angle_rad)
            
            # Compute the 6 constraints
            c1 = vx + vy / sqrt3
            c2 = 2 * vy / sqrt3
            c3 = vy / sqrt3 - vx
            c4 = -vy / sqrt3 - vx
            c5 = -2 * vy / sqrt3
            c6 = vx - vy / sqrt3
            
            s_min_point = max(c1, c2, c3, c4, c5, c6)
            required_s = max(required_s, s_min_point)
    
    # Add a small margin for numerical safety
    s = required_s + 1e-9
    
    # Ensure s >= sqrt(12)
    s = max(s, np.sqrt(12) + 1e-9)
    
    # Build output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = centers_centered[i][0]
        inner_hex_data[i, 1] = centers_centered[i][1]
        inner_hex_data[i, 2] = angles[i]
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s

# EVOLVE-BLOCK-END
