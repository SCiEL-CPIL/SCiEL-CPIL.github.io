import time
from fractions import Fraction
from itertools import combinations

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def count_ordinary_lines(points):
    """
    Count the number of lines containing exactly two of the points.
    """
    n = len(points)
    if n < 3:
        return 0
    
    lines = {}
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        
        g = _gcd(_gcd(abs(A), abs(B)), abs(C))
        if g != 0:
            A //= g
            B //= g
            C //= g
            
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
            
        return (A, B, C)

    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            lines[key] = lines.get(key, 0) + 1
            
    ordinary = 0
    for key, pair_count in lines.items():
        if pair_count == 1:
            ordinary += 1
            
    return ordinary

def _rational_to_int_point(rx, ry, scale_limit=10**9):
    """Convert rational point to integer point by scaling."""
    if isinstance(rx, Fraction) and isinstance(ry, Fraction):
        # Find LCM of denominators
        denom = rx.denominator * ry.denominator // _gcd(rx.denominator, ry.denominator)
        ix = rx.numerator * (denom // rx.denominator)
        iy = ry.numerator * (denom // ry.denominator)
        # Scale down if needed
        g = _gcd(abs(ix), abs(iy))
        if g > 0:
            ix //= g
            iy //= g
        # Scale up to stay within bounds
        factor = 1
        while abs(ix * factor) <= scale_limit and abs(iy * factor) <= scale_limit:
            if abs(ix * factor) > 10**8 or abs(iy * factor) > 10**8:
                break
            factor += 1
        ix *= factor
        iy *= factor
        if abs(ix) <= scale_limit and abs(iy) <= scale_limit:
            return [ix, iy]
    return None

def _construct_boroczky_12():
    """
    Construct a 12-point configuration based on the Böröczky-type arrangement.
    
    The Böröczky configuration for n=12 achieves the minimum of 6 ordinary lines.
    It consists of:
    - 6 points on a conic (parabola y = x^2)
    - 6 points at infinity (directions of tangents at those points)
    
    For integer coordinates, we use a finite approximation:
    - Points on a parabola
    - Points arranged to create many 3-point collinearities
    """
    # Known optimal structure: 4 lines of 3 points each + 4 points on a line
    # This gives: total lines = C(12,2) - (4*3 + 1*6) = 66 - 18 = 48 lines
    # Lines with >=3 points: 4 (3-point) + 1 (4-point) = 5 lines
    # Ordinary lines: 48 - 5 = 43... that's not right.
    
    # Let me think differently. The Böröczky config for n=12:
    # 6 points on a conic, 6 "points at infinity"
    # Each pair of conic points determines a line; the tangent at each point 
    # meets the conic at that point with multiplicity 2.
    # 
    # For a practical integer construction, let's use:
    # A 4x3 grid-like structure with strategic modifications
    
    # Configuration A: Based on affine plane of order 3 + extensions
    # 9 points in 3x3 grid, 3 more points creating extra collinearities
    
    configs = []
    
    # Config: 3x3 grid + 3 points on the "infinity" lines
    # In projective plane, the 3x3 grid has 12 lines (4 rows, 4 cols, 4 diagonals)
    # Each line has 3 points. Adding 3 points at infinity...
    
    # Let's try a direct construction:
    # 12 points where we maximize 3-collinearities
    
    # Structure: 2 parallel lines with 4 points each, 
    #           2 parallel lines with 2 points each,
    #           plus 2 more points creating cross-collinearities
    
    # Try: points at (0,0),(1,0),(2,0),(3,0) - 4 on x-axis
    #      (0,1),(1,1),(2,1),(3,1) - 4 on y=1
    #      (0,2),(1,2) - 2 on y=2
    #      (0,3),(1,3) - 2 on y=3
    # This gives 4+4+2+2 = 12 points
    # Lines: x-axis(4), y=1(4), y=2(2), y=3(2)
    # Verticals: x=0(4), x=1(4), x=2(2), x=3(2)
    # Diagonals and other lines...
    
    cfg_a = [[0,0],[1,0],[2,0],[3,0],
             [0,1],[1,1],[2,1],[3,1],
             [0,2],[1,2],
             [0,3],[1,3]]
    configs.append(cfg_a)
    
    # Structure B: 3x4 grid
    cfg_b = [[x,y] for x in range(3) for y in range(4)]
    configs.append(cfg_b)
    
    # Structure C: 4x3 grid
    cfg_c = [[x,y] for x in range(4) for y in range(3)]
    configs.append(cfg_c)
    
    # Structure D: Hexagonal/symmetric
    # 6 points on a regular hexagon + 6 on a smaller hexagon
    # Approximate with integer coords
    cfg_d = [[1,0],[0,1],[-1,1],[-1,0],[0,-1],[1,-1],
             [2,0],[0,2],[-2,2],[-2,0],[0,-2],[2,-2]]
    configs.append(cfg_d)
    
    # Structure E: 3 lines of 4 points (pencil-like)
    # Line 1: y=0, x=0,1,2,3
    # Line 2: y=x, points at x=0,1,2,3 -> (0,0),(1,1),(2,2),(3,3)
    # Line 3: y=2x, points at x=0,1,2,3 -> (0,0),(2,2),(4,4),(6,6) - overlaps!
    # Need non-overlapping
    
    # Structure F: Use the fact that for 12 points, optimal is 6 ordinary lines
    # This means 66 - 6 = 60 pairs are on non-ordinary lines
    # If we have k_3 lines of 3 points and k_4 lines of 4 points:
    # 3*k_3 + 4*k_4 pairs are "used" by multi-point lines
    # But pairs can be shared... 
    # 
    # Böröczky: 6 on conic, 6 at infinity
    # Each conic point connects to 5 other conic points (5 lines)
    # Each conic point has a tangent (1 line at infinity)
    # Lines: 15 conic-conic + 6 tangent lines + lines through infinity points
    # 
    # For integer approximation, let's try:
    # Parabola y = x^2: points (0,0),(1,1),(2,4),(3,9),(4,16),(5,25)
    # And 6 more points chosen to create collinearities
    
    para = [[0,0],[1,1],[2,4],[3,9],[4,16],[5,25]]
    # Add points that create 3-collinearities with pairs on parabola
    # Line through (0,0) and (1,1): y=x, add (2,2)? No, (2,4) is on parabola
    # Line through (0,0) and (2,4): y=2x, add (3,6)? 
    # Line through (1,1) and (2,4): slope 3, y=3x-2, add (3,7)?
    
    cfg_f = para + [[3,6],[4,10],[5,14]]  # These might not be great
    configs.append(cfg_f)
    
    # Structure G: Known good - 12 points with high collinearity
    # Use a "net" structure
    cfg_g = [[0,0],[1,0],[2,0],[3,0],
             [0,1],[1,1],[2,1],[3,1],
             [0,2],[1,2],[2,2],[3,2],
             [0,3],[1,3]]
    # That's 13, fix
    cfg_g = [[0,0],[1,0],[2,0],[3,0],
             [0,1],[1,1],[2,1],[3,1],
             [0,2],[1,2],[2,2],[3,2],
             [0,3]]
    # Still 13... 
    cfg_g = [[0,0],[1,0],[2,0],[3,0],
             [0,1],[1,1],[2,1],[3,1],
             [0,2],[1,2],[2,2],
             [0,3],[1,3],[2,3]]
    configs.append(cfg_g)
    
    return configs

def _line_partition_search(deadline):
    """
    Search for configurations by trying to realize specific line partitions.
    A line partition specifies how many points are on each line.
    """
    best_points = None
    best_score = float('inf')
    
    # Try all configurations from the constructive method
    for cfg in _construct_boroczky_12():
        if time.time() > deadline:
            break
        if len(set(map(tuple, cfg))) != 12:
            continue
        score = count_ordinary_lines(cfg)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in cfg]
    
    return best_points, best_score

def _smart_local_search(points, deadline, max_iter=200):
    """Improved local search with larger moves and restart capability."""
    n = len(points)
    best_points = [p[:] for p in points]
    best_score = count_ordinary_lines(best_points)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            x, y = best_points[i]
            # Try various move sizes
            for step in [1, 2, 3, 5, 8, 13]:
                for dx in range(-step, step+1, max(1, step//2)):
                    for dy in range(-step, step+1, max(1, step//2)):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if abs(nx) > 10**9 or abs(ny) > 10**9:
                            continue
                        new_pt = [nx, ny]
                        if any(new_pt == p for p in best_points):
                            continue
                        new_points = best_points[:]
                        new_points[i] = new_pt
                        new_score = count_ordinary_lines(new_points)
                        if new_score < best_score:
                            best_score = new_score
                            best_points = new_points
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_points = None
    best_score = float('inf')
    
    # Phase 1: Constructive search
    pts, score = _line_partition_search(deadline)
    if pts is not None:
        best_points = pts
        best_score = score
    
    # Phase 2: Local search from best
    if best_points is not None and time.time() < deadline - 5:
        best_points, best_score = _smart_local_search(best_points, deadline)
    
    # Phase 3: Try more initial configs if time permits
    if time.time() < deadline - 20:
        # Generate more diverse initial configs
        import random
        random.seed(42)
        
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Random grid-like structure
            if attempt % 3 == 0:
                # 3x4 grid variant
                xs = random.sample(range(-5, 6), 3)
                ys = random.sample(range(-5, 6), 4)
                cfg = [[x, y] for x in xs for y in ys]
            elif attempt % 3 == 1:
                # 4x3 grid variant
                xs = random.sample(range(-5, 6), 4)
                ys = random.sample(range(-5, 6), 3)
                cfg = [[x, y] for x in xs for y in ys]
            else:
                # Random with structure
                base = [[0,0],[1,0],[2,0],[3,0]]
                extra = []
                for _ in range(8):
                    extra.append([random.randint(-3, 3), random.randint(-3, 3)])
                cfg = base + extra
            
            if len(set(map(tuple, cfg))) != 12:
                continue
            
            score = count_ordinary_lines(cfg)
            if score < best_score:
                best_score = score
                best_points = [p[:] for p in cfg]
                # Local search from this
                if time.time() < deadline - 10:
                    best_points, best_score = _smart_local_search(best_points, deadline, max_iter=50)
    
    # Final fallback
    if best_points is None:
        best_points = [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2]]
    
    return best_points

# EVOLVE-BLOCK-END
