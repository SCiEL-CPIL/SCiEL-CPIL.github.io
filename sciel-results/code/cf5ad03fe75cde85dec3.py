"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from math import sqrt, cos, sin, pi, floor, ceil

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Place N=11 unit regular hexagons inside a regular hexagon, minimizing outer side length.
    
    Strategy:
    1. Analytical lower bound: for N hexagons of side 1, the area of outer hexagon with side s is (3*sqrt(3)/2)*s^2.
       Inner area is N*(3*sqrt(3)/2)*1^2 = N*3*sqrt(3)/2.
       So s^2 >= N => s >= sqrt(N) ≈ 3.317.
    2. We use a grid search over outer side length s, and for each s, try to pack N hexagons.
    3. For aligned orientations (angle=0), two unit hexagons don't overlap iff distance between centers >= sqrt(3) ≈ 1.732.
       Actually, for two regular hexagons of side 1, the minimum distance between centers to avoid overlap is 2 (if flat-to-flat) 
       or sqrt(3) (if point-to-point)? Let me think...
       
       A regular hexagon of side 1 has inradius r = sqrt(3)/2 ≈ 0.866 and circumradius R = 1.
       The Minkowski sum of two such hexagons (for non-overlap) gives a constraint.
       For two identical regular hexagons of side 1, the minimum center-to-center distance for non-overlap 
       depends on relative orientation. For aligned orientations, the minimum distance is 2 (the sum of two inradii 
       in the direction perpendicular to a flat side is 2*sqrt(3)/2 = sqrt(3) ≈ 1.732, but in the vertex direction it's 2*1=2).
       
       Actually, for two regular hexagons of side 1 with the same orientation, the non-overlap condition is that 
       the distance between centers is at least the width of the hexagon in that direction. The hexagon has width 
       2 (vertex to opposite vertex) and sqrt(3) (flat to flat). The Minkowski difference (feasible region for one 
       center given another) is a hexagon scaled by 2. So the minimum distance is 2 in the vertex directions and 
       sqrt(3) in the flat directions. The overall constraint is that the vector between centers must lie outside 
       the Minkowski sum of the hexagon and its reflection, which is a regular hexagon of side 2.
       
       For simplicity and a conservative estimate, I'll use distance >= 2.0 as the non-overlap constraint for 
       aligned hexagons. This is sufficient (not necessary) but ensures no overlap.
       
    4. Use a grid-based placement: try placing hexagons on a triangular lattice or square grid within the outer hexagon.
    
    Let me use a simpler approach: try a honeycomb-like arrangement.
    """
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Best result so far
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Lower bound: sqrt(N) for area, but geometrically we need more
    # For N=11, let's search s from 3.5 to 6.0
    
    # Try different outer side lengths
    s_values = np.linspace(3.5, 6.0, 50)
    
    for s in s_values:
        if time.time() > deadline:
            break
            
        # Outer hexagon centered at origin with side s, flat-top orientation
        # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
        # Inradius of outer hexagon: s * sqrt(3)/2
        # Circumradius: s
        
        # Feasible region for inner hexagon centers: 
        # The center of an inner hexagon must be at least the "radius" of the inner hexagon 
        # away from the outer boundary. For a unit hexagon, the inradius is sqrt(3)/2.
        # So the feasible region is a smaller hexagon with inradius = s*sqrt(3)/2 - sqrt(3)/2 = (s-1)*sqrt(3)/2
        # Which corresponds to a hexagon of side (s-1).
        
        # So we need to place N points in a regular hexagon of side (s-1) such that 
        # pairwise distances are >= 2.0 (conservative non-overlap for aligned unit hexagons).
        
        feasible_side = s - 1.0
        if feasible_side < 0:
            continue
        
        # Try to pack N points in a hexagon of side feasible_side with min distance 2.0
        # Use a grid search on a triangular lattice
        
        # For a hexagon of side L, the number of points on a triangular lattice with spacing d
        # that fit is roughly (3/2) * (L/d)^2 for large L/d.
        # We need this >= N.
        
        # Try different lattice spacings
        # The min distance is 2.0, so lattice spacing should be >= 2.0
        # Let's try to place points on a triangular lattice with spacing 2.0
        
        # A triangular lattice with spacing d has points at:
        # (i*d, j*d*sqrt(3)) for integer i, j, with offset for even j
        # Actually, let me use a simpler grid approach.
        
        # For a hexagon of side L, the max number of points with min distance d is approximately
        # (2*L/d)^2 * 3/4 for a square grid, or similar for triangular.
        
        # Let me just try a few configurations:
        # 1. A single row: N points in a line, length = (N-1)*2 + 2 = 2N. Needs 2N <= 2*L => L >= N.
        #    So s-1 >= N => s >= N+1 = 12. Too large.
        
        # 2. Two rows: 6 and 5 points. 
        #    Row spacing: sqrt(3) (for triangular lattice with spacing 2).
        #    Actually, for triangular lattice with spacing 2, row spacing is 2*sqrt(3)/2 = sqrt(3) ≈ 1.732.
        #    6 points in a row: length = 5*2 = 10. Plus margins.
        #    Total width: ~10, height: ~1.732. 
        #    Hexagon of side L has width 2L and height sqrt(3)*L.
        #    So we need 2L >= 10 => L >= 5, and sqrt(3)*L >= 1.732 => L >= 1.
        #    So L >= 5, s >= 6.
        
        # 3. Three rows: 4, 4, 3 points.
        #    Width: 3*2 = 6, height: 2*sqrt(3) ≈ 3.46.
        #    Need 2L >= 6 => L >= 3, sqrt(3)*L >= 3.46 => L >= 2.
        #    So L >= 3, s >= 4.
        
        # 4. Four rows: 3, 3, 3, 2 points.
        #    Width: 2*2 = 4, height: 3*sqrt(3) ≈ 5.2.
        #    Need 2L >= 4 => L >= 2, sqrt(3)*L >= 5.2 => L >= 3.
        #    So L >= 3, s >= 4.
        
        # Let me try to construct a valid packing for s=4.0 (feasible_side=3.0)
        # Hexagon of side 3.0, need 11 points with min distance 2.0.
        
        # Let me use a more systematic approach: try to place points on a triangular lattice 
        # with spacing 2.0 inside the hexagon of side feasible_side.
        
        # Triangular lattice points: 
        # x = i*2 + (j%2)*1, y = j*sqrt(3)
        # for i, j in some range.
        
        # Let me generate candidate points and check if they're inside the hexagon.
        
        # A point (x, y) is inside a regular hexagon of side L centered at origin (flat-top) if:
        # |x| <= L and |y| <= L*sqrt(3)/2 and |x| + |y|/sqrt(3) <= L
        # Wait, let me get the correct inequality.
        # For a flat-top hexagon with side L, the vertices are at (L, 0), (L/2, L*sqrt(3)/2), 
        # (-L/2, L*sqrt(3)/2), (-L, 0), (-L/2, -L*sqrt(3)/2), (L/2, -L*sqrt(3)/2).
        # The hexagon is the intersection of 6 half-planes.
        # The inequalities are:
        # |x| <= L
        # |y| <= L*sqrt(3)/2
        # |x|/L + |y|/(L*sqrt(3)/2) <= 1  =>  |x|/L + 2|y|/(L*sqrt(3)) <= 1
        
        def in_hexagon(x, y, L):
            if L <= 0:
                return False
            if abs(x) > L:
                return False
            if abs(y) > L * sqrt(3) / 2:
                return False
            if abs(x) / L + 2 * abs(y) / (L * sqrt(3)) > 1 + 1e-9:
                return False
            return True
        
        # Generate triangular lattice points
        # Spacing 2.0 in x, sqrt(3) in y, with offset 1.0 in x for odd rows
        max_coord = feasible_side + 1
        points = []
        
        j_range = int(2 * max_coord / sqrt(3)) + 2
        for j in range(-j_range, j_range + 1):
            y = j * sqrt(3)
            i_range = int(max_coord / 2) + 1
            for i in range(-i_range, i_range + 1):
                x = i * 2.0 + (1.0 if j % 2 != 0 else 0.0)
                if in_hexagon(x, y, feasible_side):
                    points.append((x, y))
        
        # Now we have a set of candidate points. We need to select N=11 of them 
        # such that pairwise distances are >= 2.0.
        # Since the lattice spacing is 2.0, adjacent points are exactly 2.0 apart, 
        # so any two distinct lattice points are at least 2.0 apart. 
        # Wait, is that true? In a triangular lattice with spacing 2.0, the nearest neighbors 
        # are at distance 2.0. So yes, any two distinct points are >= 2.0 apart.
        
        if len(points) >= N:
            # We can pick any N points
            selected = points[:N]
            # Construct inner hexagons
            inner = []
            for (x, y) in selected:
                inner.append([x, y, 0.0])
            
            # Verify non-overlap (should be fine by construction)
            # Verify containment (should be fine by construction)
            
            if s < best_side:
                best_side = s
                best_inner = inner
                best_center = (0.0, 0.0)
                best_angle = 0.0
        
        # Also try a denser packing with smaller spacing? No, spacing must be >= 2.0.
        # Try square lattice?
        # Square lattice with spacing 2.0: points at (2i, 2j).
        # Nearest neighbors at distance 2.0. Diagonal at 2*sqrt(2) ≈ 2.828.
        # This is also valid.
        
        points_sq = []
        i_range_sq = int(max_coord / 2) + 1
        j_range_sq = int(max_coord / 2) + 1
        for i in range(-i_range_sq, i_range_sq + 1):
            for j in range(-j_range_sq, j_range_sq + 1):
                x = i * 2.0
                y = j * 2.0
                if in_hexagon(x, y, feasible_side):
                    points_sq.append((x, y))
        
        if len(points_sq) >= N:
            selected = points_sq[:N]
            inner = []
            for (x, y) in selected:
                inner.append([x, y, 0.0])
            
            if s < best_side:
                best_side = s
                best_inner = inner
                best_center = (0.0, 0.0)
                best_angle = 0.0
    
    # If we found a valid packing, return it
    if best_inner is not None:
        return best_inner, best_center, best_side, best_angle
    
    # Fallback: return a trivial valid solution (even if not optimal)
    # Place hexagons in a line with spacing 2.0
    inner = []
    for k in range(N):
        x = (k - N // 2) * 2.0
        inner.append([x, 0.0, 0.0])
    
    # Outer side needed: the line has length (N-1)*2 = 20.
    # Hexagon of side s has width 2s. So 2s >= 20 => s >= 10.
    # But we also need to account for the hexagon's own width.
    # The leftmost hexagon center is at x = -10, rightmost at x = 10.
    # The hexagon extends 1 unit from its center in the x-direction.
    # So total extent is from -11 to 11, width 22.
    # 2s >= 22 => s >= 11.
    # Let's use s = 12 for safety.
    
    return inner, (0.0, 0.0), 12.0, 0.0
# EVOLVE-BLOCK-END
