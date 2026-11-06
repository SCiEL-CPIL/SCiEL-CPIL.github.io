"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    """
    Construct 32 distinct lattice points on the triangular lattice (a + b*omega, omega=exp(i*pi/3))
    maximizing the number of pairs at distance exactly 1.
    
    The optimum is 76 edges (Harborth's bound). We use a hexagonal spiral arrangement
    which achieves this bound for n=32.
    """
    # The triangular lattice: point (a,b) corresponds to a + b*omega
    # Distance squared between (a1,b1) and (a2,b2):
    # Let da = a1-a2, db = b1-b2
    # The complex difference is da + db*omega = da + db*(1/2 + i*sqrt(3)/2)
    # = (da + db/2) + i*(db*sqrt(3)/2)
    # |diff|^2 = (da + db/2)^2 + 3*(db/2)^2 = da^2 + da*db + db^2
    # So distance^2 = da^2 + da*db + db^2
    # Distance exactly 1 means da^2 + da*db + db^2 = 1
    # The 6 neighbors of (a,b) are:
    # (a+1,b), (a-1,b), (a,b+1), (a,b-1), (a+1,b-1), (a-1,b+1)
    # Check: (1,0): 1+0+0=1 ✓
    # (0,1): 0+0+1=1 ✓
    # (1,-1): 1-1+1=1 ✓
    # (-1,1): 1-1+1=1 ✓
    # (0,-1): 0+0+1=1 ✓
    # (-1,0): 1+0+0=1 ✓
    
    # A hexagonal spiral centered at origin:
    # Ring 0: (0,0) - 1 point
    # Ring 1: 6 points
    # Ring 2: 12 points
    # Ring 3: 18 points
    # ...
    # Ring k (k>=1): 6k points
    # Total up to ring r: 1 + 6*(1+2+...+r) = 1 + 3*r*(r+1)
    # For n=32: we need rings 0,1,2,3 -> 1+6+12+18 = 37, too many
    # Rings 0,1,2 -> 1+6+12 = 19, need 13 more from ring 3
    
    # Let's generate a hexagonal spiral and take the first 32 points.
    # The spiral order should be such that we get a compact hexagon.
    
    # Actually, let me think about this more carefully.
    # A hexagonal number arrangement: the points form a hexagon.
    # For n=32, the best is a near-hexagonal shape.
    
    # Let's use a different approach: start with a hexagonal spiral and
    # use local search to maximize edges.
    
    # Generate hexagonal spiral points
    def hex_spiral(n):
        """Generate n points in a hexagonal spiral on the triangular lattice."""
        points = [(0, 0)]
        if n <= 1:
            return points[:n]
        
        # Ring directions: for ring k, we go along 6 sides
        # Each side has k steps
        # The 6 directions for the spiral:
        # We need to trace the hexagonal spiral in order
        
        # Standard hex spiral: 
        # Start at center, go right 1, then turn 60 degrees clockwise each step
        # Actually, let me use the standard algorithm:
        
        # For ring r (r>=1), the points are at distance r from center
        # The ring has 6r points
        # We trace around the hexagon
        
        # Let's use a simpler approach: generate all points in a hexagon of radius r
        # and sort by distance from center, breaking ties by angle
        
        # For n=32, we need a hexagon of radius 2 (19 points) plus 13 from radius 3
        
        # Let me just generate points in order of a hex spiral:
        # Position in spiral: 
        # The hex spiral goes: center, then ring 1 (6 pts), ring 2 (12 pts), etc.
        # Within each ring, we go around the hexagon.
        
        # Directions for the 6 sides of a hexagon (in the triangular lattice):
        # Side 1: from (r, 0) going in direction (-1, 1) for r steps -> to (0, r)
        # Wait, let me think in terms of the 6 neighbor directions.
        
        # The 6 unit directions: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        
        # Hex spiral algorithm:
        # Start at (0,0)
        # For each ring r = 1, 2, 3, ...:
        #   Start at (r, 0)  [east point of the ring]
        #   Then walk 6 sides, each of length r
        #   Side directions: (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (1,0)
        #   Wait, that's not quite right for a spiral.
        
        # Let me use a cleaner method:
        # Generate all lattice points with |a|+|b|+|a+b| <= 2*r (hexagonal region of radius r)
        # Sort by (distance from center, angle)
        
        def hex_distance(a, b):
            # Hex distance in axial coordinates (a,b) for triangular lattice
            # The hex distance is max(|a|, |b|, |a+b|) ... wait, that's for cube coords
            # In our system, point is a + b*omega
            # The 6 neighbors are at (±1,0), (0,±1), (1,-1), (-1,1)
            # Hex distance: we can convert to cube coordinates
            # x = a, y = b, z = -(a+b) ... but that's for hex grid
            # Actually for triangular lattice with omega = exp(i*pi/3):
            # The distance from origin in hex metric:
            # d = max(|a|, |b|, |a+b|) ... let me verify
            # (1,0): max(1,0,1) = 1 ✓
            # (0,1): max(0,1,1) = 1 ✓  
            # (1,-1): max(1,-1,0) = 1 ✓
            # (1,1): max(1,1,2) = 2 ✓ (distance 2 from origin)
            # (2,0): max(2,0,2) = 2 ✓
            return max(abs(a), abs(b), abs(a + b))
        
        # Generate points in hexagonal region, sorted by distance then angle
        all_points = []
        max_r = 10
        for a in range(-max_r, max_r + 1):
            for b in range(-max_r, max_r + 1):
                all_points.append((a, b))
        
        # Sort by hex distance, then by angle
        def sort_key(p):
            a, b = p
            d = hex_distance(a, b)
            # Angle: atan2 of the complex number a + b*omega
            # a + b*(1/2 + i*sqrt(3)/2) = (a + b/2) + i*b*sqrt(3)/2
            x = a + b / 2.0
            y = b * math.sqrt(3) / 2.0
            angle = math.atan2(y, x)
            return (d, angle)
        
        all_points.sort(key=sort_key)
        
        # Take the first n points
        return all_points[:n]
    
    points = hex_spiral(32)
    
    # Verify: count edges
    def count_edges(pts):
        n = len(pts)
        edges = 0
        for i in range(n):
            for j in range(i + 1, n):
                da = pts[i][0] - pts[j][0]
                db = pts[i][1] - pts[j][1]
                if da * da + da * db + db * db == 1:
                    edges += 1
        return edges
    
    edges = count_edges(points)
    
    # If we already have 76, return immediately
    if edges >= 76:
        return [[a, b] for a, b in points]
    
    # Local search to improve
    # Try moving points to nearby positions to increase edge count
    
    def try_improve(pts, max_iters=50000):
        pts = list(pts)
        n = len(pts)
        best = count_edges(pts)
        
        # Precompute neighbor offsets
        neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for i in range(n):
                for da, db in neighbors:
                    new_a = pts[i][0] + da
                    new_b = pts[i][1] + db
                    if abs(new_a) > 64 or abs(new_b) > 64:
                        continue
                    # Check if position is already occupied
                    if (new_a, new_b) in pts:
                        continue
                    # Check minimum distance constraint
                    valid = True
                    old_a, old_b = pts[i]
                    for j in range(n):
                        if j == i:
                            continue
                        da2 = new_a - pts[j][0]
                        db2 = new_b - pts[j][1]
                        if da2 * da2 + da2 * db2 + db2 * db2 == 0:
                            valid = False
                            break
                    if not valid:
                        continue
                    # Try the move
                    pts[i] = (new_a, new_b)
                    new_edges = count_edges(pts)
                    if new_edges > best:
                        best = new_edges
                        improved = True
                    else:
                        pts[i] = (old_a, old_b)
        
        return pts, best
    
    points, edges = try_improve(points)
    
    return [[a, b] for a, b in points]
# EVOLVE-BLOCK-END
