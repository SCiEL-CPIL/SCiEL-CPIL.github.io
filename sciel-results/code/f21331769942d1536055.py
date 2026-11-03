"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import itertools
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    # For N=11 unit hexagons, the optimal packing is the hexagonal cluster of 11 hexagons.
    # This forms a regular hexagon of "radius" 2 (in terms of hex cells), but since we have 11 hexagons,
    # it's actually a hexagon with side length 2 in terms of unit hexagons? No.
    # A hexagon with side length k (number of hexagons on each side) contains 1 + 6 * sum_{i=1}^{k-1} i = 1 + 3k(k-1) hexagons.
    # For k=1: 1 hex.
    # For k=2: 1 + 6*1 = 7 hexagons.
    # For k=3: 1 + 6*(1+2) = 19 hexagons.
    # So 11 hexagons don't form a perfect hexagonal cluster.
    
    # Let's think about the minimal enclosing hexagon.
    # The task asks to place N=11 unit hexagons inside a regular hexagon, minimizing the outer side length.
    # Each inner hexagon has side length 1.
    
    # The minimal enclosing hexagon for N hexagons is a known problem.
    # For N=11, the optimal arrangement is likely a compact cluster.
    
    # Let's use the known result for packing N unit hexagons in a hexagon.
    # The minimal side length S of the enclosing hexagon satisfies that the area of the enclosing hexagon
    # is at least the sum of areas of the inner hexagons.
    # Area of regular hexagon with side s is (3*sqrt(3)/2) * s^2.
    # So (3*sqrt(3)/2) * S^2 >= N * (3*sqrt(3)/2) * 1^2 => S^2 >= N => S >= sqrt(N).
    # For N=11, S >= sqrt(11) ≈ 3.3166.
    
    # But this is just the area bound. The actual minimal S is larger due to geometric constraints.
    
    # For hexagonal packing, the centers of the unit hexagons must be at distance at least 2*sqrt(3)/2 = sqrt(3) apart?
    # No, two unit hexagons with parallel sides are disjoint if their centers are at least sqrt(3) apart in the direction perpendicular to the sides?
    # Actually, the minimum distance between centers of two disjoint unit hexagons depends on their relative rotation.
    # For parallel hexagons, the minimum center distance is sqrt(3) (distance between opposite sides is sqrt(3), so centers must be at least sqrt(3) apart in the normal direction to avoid overlap... wait, no).
    # Two unit regular hexagons with the same orientation: the distance between parallel opposite sides is sqrt(3). So if centers are closer than sqrt(3) in any of the three principal directions, they overlap.
    # Actually, the condition for two parallel unit hexagons to be disjoint is that the center distance vector (dx, dy) satisfies |dx|/1 + |dy|/sqrt(3) ... no, it's simpler.
    # The hexagon is the intersection of 3 strips of width sqrt(3). So two parallel hexagons are disjoint if the center difference has at least one component (in the 3 axial directions) with magnitude > sqrt(3)/2 + sqrt(3)/2 = sqrt(3)? No.
    # Let me think again. A unit regular hexagon centered at origin with flat-top orientation has vertices at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2).
    # The distance from center to each side (apothem) is sqrt(3)/2.
    # Two such hexagons are disjoint if the distance between centers is greater than the sum of apothems in the direction perpendicular to the side... this is getting complicated.
    
    # Let's use a simpler approach: for N=11, try to find a good arrangement empirically or use known results.
    # Known: for N hexagons, the minimal enclosing hexagon side length is approximately sqrt(N) + some constant.
    
    # Let's try a hexagonal grid arrangement. Place 11 hexagons in a compact cluster.
    # Rows: 1, 2, 3, 3, 2, 1? That's 12. 
    # 2, 3, 3, 2, 1? That's 11. Or 1, 3, 3, 3, 1? That's 11.
    # Let's use the arrangement: row 0: 2 hexagons, row 1: 3 hexagons, row 2: 3 hexagons, row 3: 2 hexagons, row 4: 1 hexagon.
    # In axial coordinates for a flat-top hex grid, adjacent hexagons in a row are spaced by 1.5 in x and sqrt(3)/2 in y for odd/even rows.
    
    # Actually, let's just use the optimal known configuration for N=11.
    # According to literature, for N=11, the minimal enclosing hexagon has side length approximately 3.5 or so.
    
    # Let me try a different approach: use the fact that the optimal packing for hexagons in a hexagon is to use the hexagonal lattice.
    # The 11 hexagons can be placed in a hexagonal cluster shape.
    
    # Let's generate positions for 11 hexagons in a compact hexagonal arrangement.
    # Using axial coordinates (q, r) where the distance between adjacent hex centers is sqrt(3) for flat-top? No, for pointy-top it's different.
    
    # Let's use pointy-top hexagons. The distance between centers of adjacent hexagons in a hex grid is sqrt(3) for unit side length? 
    # No, for a unit regular hexagon (side length 1), the distance from center to vertex is 1, and center to side (apothem) is sqrt(3)/2.
    # In a hex grid, adjacent hexagons share a side, so their centers are 2 * apothem = sqrt(3) apart.
    
    # Let's place 11 hexagons in a hex cluster. The cluster with 11 hexagons is not a perfect hexagon, but a near-perfect one.
    # Let's use the following axial coordinates (q, r) for a pointy-top hex grid:
    # (0,0), (1,0), (2,0), (3,0), (4,0) - 5 in a row
    # (0,1), (1,1), (2,1), (3,1) - 4 in the next row
    # (0,2), (1,2), (2,2) - 3 in the next row
    # Total: 5+4+3 = 12. Too many.
    # Let's use 4+4+3 = 11.
    # Row 0: (0,0), (1,0), (2,0), (3,0)
    # Row 1: (0,1), (1,1), (2,1), (3,1)
    # Row 2: (0,2), (1,2), (2,2)
    
    # Convert axial to pixel coordinates for pointy-top:
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    
    positions = []
    for r in range(3):
        for q in range(4 if r < 2 else 3):
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            positions.append((x, y))
    
    # Center the positions
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    positions = [(x - cx, y - cy) for x, y in positions]
    
    # Now find the minimal enclosing hexagon.
    # For a set of points, the minimal enclosing regular hexagon with a given orientation can be found by projecting points onto the 3 principal axes.
    # Let's try orientation 0 degrees (flat-top) and 30 degrees (pointy-top) and find the best.
    
    def enclosing_hex_side(points, angle_deg):
        """Compute the minimal side length of a regular hexagon enclosing all points, with the hexagon rotated by angle_deg."""
        angle = math.radians(angle_deg)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        # The 3 principal axes of a regular hexagon are at angles 0, 60, 120 degrees (for flat-top).
        # For a hexagon rotated by angle_deg, the axes are at angle_deg, angle_deg+60, angle_deg+120.
        max_proj = [0.0, 0.0, 0.0]
        min_proj = [0.0, 0.0, 0.0]
        
        for i in range(3):
            axis_angle = angle + math.radians(60 * i)
            ca = math.cos(axis_angle)
            sa = math.sin(axis_angle)
            projs = [x * ca + y * sa for x, y in points]
            max_proj[i] = max(projs)
            min_proj[i] = min(projs)
        
        # The width in each direction is max_proj[i] - min_proj[i].
        # For a regular hexagon with side length S, the width in each principal direction is 2 * S * cos(0) = 2S? 
        # No, the width of a regular hexagon (distance between opposite sides) is 2 * apothem = 2 * S * sqrt(3)/2 = S * sqrt(3).
        # So S * sqrt(3) >= max(width_i) for all i.
        # Thus S >= max(width_i) / sqrt(3).
        
        widths = [max_proj[i] - min_proj[i] for i in range(3)]
        max_width = max(widths)
        return max_width / math.sqrt(3)
    
    best_side = float('inf')
    best_angle = 0.0
    for angle_deg in [0, 30, 60, 90, 120, 150]:
        side = enclosing_hex_side(positions, angle_deg)
        if side < best_side:
            best_side = side
            best_angle = angle_deg
    
    # Refine the angle
    best_side_refined = best_side
    best_angle_refined = best_angle
    for _ in range(3):
        step = 1.0
        for _ in range(5):
            step /= 2.0
            improved = False
            for da in [-step, step]:
                new_angle = best_angle_refined + da
                side = enclosing_hex_side(positions, new_angle)
                if side < best_side_refined:
                    best_side_refined = side
                    best_angle_refined = new_angle
                    improved = True
            if not improved:
                break
    
    inner = [[x, y, 0.0] for x, y in positions]
    return inner, (0.0, 0.0), best_side_refined, best_angle_refined
# EVOLVE-BLOCK-END
