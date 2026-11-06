import time
import math
import random
from collections import defaultdict

def _gcd3(a, b, c):
    g = math.gcd(a, b)
    g = math.gcd(g, c)
    return g

def count_ordinary_lines(points):
    """
    Counts the number of lines containing exactly two of the given points.
    Optimized for speed using normalized line equations.
    """
    n = len(points)
    if n < 3:
        return 0
    
    lines = defaultdict(set)
    
    for i in range(n):
        x1, y1 = points[i][0], points[i][1]
        for j in range(i + 1, n):
            x2, y2 = points[j][0], points[j][1]
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize line equation: a*x + b*y + c = 0
            # Direction vector (dx, dy) is normal to (dy, -dx)
            # So a = dy, b = -dx
            a = dy
            b = -dx
            c = dx * y1 - dy * x1
            
            # Normalize (a, b, c)
            g = _gcd3(abs(a), abs(b), abs(c))
            if g:
                a //= g
                b //= g
                c //= g
                
            # Ensure canonical sign
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
            
            lines[(a, b, c)].update((i, j))
            
    return sum(1 for s in lines.values() if len(s) == 2)

def _is_valid(points):
    if len(points) != 12:
        return False
    if len(set(map(tuple, points))) < 12:
        return False
    if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in points):
        return False
    # Check not all collinear
    (x0, y0), (x1, y1), (x2, y2) = points[0], points[1], points[2]
    for k in range(3, 12):
        xi, yi = points[k]
        if (x1 - x0) * (yi - y0) != (y1 - y0) * (xi - x0):
            return True
    return False

def _build_grid_3x4():
    return [[x, y] for x in range(3) for y in range(4)]

def _build_hollow_4x4():
    # 4x4 grid minus corners
    pts = []
    for x in range(4):
        for y in range(4):
            if (x == 0 and y == 0) or (x == 0 and y == 3) or (x == 3 and y == 0) or (x == 3 and y == 3):
                continue
            pts.append([x, y])
    return pts

def _build_random(seed):
    rng = random.Random(seed)
    pts = []
    while len(pts) < 12:
        x = rng.randint(-10, 10)
        y = rng.randint(-10, 10)
        if [x, y] not in pts:
            pts.append([x, y])
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_points = None
    best_score = float('inf')
    
    def consider(pts):
        nonlocal best_points, best_score
        if not _is_valid(pts):
            return
        s = count_ordinary_lines(pts)
        if s < best_score:
            best_score = s
            best_points = [[int(x), int(y)] for x, y in pts]
            if best_score <= 6:
                # Theoretical minimum found
                return True
        return False

    # 1. Initialize with diverse strong candidates
    candidates = [
        _build_grid_3x4(),
        _build_hollow_4x4(),
        _build_random(1),
        _build_random(42),
        _build_random(123)
    ]
    
    for cfg in candidates:
        if consider(cfg):
            return best_points
            
    # 2. Simulated Annealing
    # We operate on a list of 12 points.
    # Moves:
    # A. Move point i to a random nearby integer coordinate (local search)
    # B. "Line Snap": Pick i, j, k. Move i to a point collinear with j, k.
    
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    # Parameters for Simulated Annealing
    T_start = 10.0
    T_end = 0.1
    cooling_rate = 0.99
    
    rng = random.Random(999)
    
    # Pre-compute a set of candidate lattice points for "Line Snap" and "Local" moves
    # We restrict to a reasonable box to keep moves "local" enough for SA to work well,
    # but large enough to allow reconfiguration.
    # The points are integers. We can search in a box around the current centroid.
    
    def get_centroid(pts):
        cx = sum(p[0] for p in pts) / 12.0
        cy = sum(p[1] for p in pts) / 12.0
        return cx, cy

    cx, cy = get_centroid(current_points)
    
    # Define a local search box around the centroid
    # If points drift too far, we re-center the box
    box_size = 20
    
    def generate_local_move(i, cx, cy):
        """Generate a new position for point i within a box around (cx, cy)"""
        for _ in range(10): # Try a few times to find a unique point
            nx = int(cx + rng.randint(-box_size, box_size))
            ny = int(cy + rng.randint(-box_size, box_size))
            # Check bounds
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
            # Check uniqueness
            if [nx, ny] not in current_points:
                return [nx, ny]
        return None

    def generate_line_snap_move(i, cx, cy):
        """
        Pick two other points j, k.
        Find a point collinear with j, k, not currently used, within bounds.
        """
        j, k = rng.sample(range(12), 2)
        if j == i or k == i:
            return None
            
        xj, yj = current_points[j]
        xk, yk = current_points[k]
        
        dx = xk - xj
        dy = yk - yj
        
        # Line equation: dy * (x - xj) - dx * (y - yj) = 0
        # dy * x - dx * y = dy * xj - dx * yj
        # c = dy * xj - dx * yj
        
        # We want to find integer points (x, y) on this line.
        # If dx == 0: x = xj. y can be anything.
        # If dy == 0: y = yj. x can be anything.
        # Else: parametric or direct search.
        
        # To keep it fast, we don't scan the whole line.
        # We pick a random parameter t and compute x, y.
        # x = xj + t * dx_g, y = yj + t * dy_g where (dx_g, dy_g) is primitive direction.
        
        g = math.gcd(abs(dx), abs(dy))
        if g == 0:
            return None
        dx_g = dx // g
        dy_g = dy // g
        
        # Try a few random t values
        for _ in range(20):
            t = rng.randint(-10, 10)
            if t == 0:
                continue # This would be xj, yj which is occupied
            nx = xj + t * dx_g
            ny = yj + t * dy_g
            
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
            
            # Check if this point is already in the set
            # Optimization: O(N) check is fine for N=12
            if [nx, ny] in current_points:
                continue
                
            return [nx, ny]
            
        return None

    steps = 0
    T = T_start
    
    while time.time() < deadline and T > T_end:
        steps += 1
        
        # Choose move type
        move_type = rng.random()
        new_points = None
        
        if move_type < 0.5:
            # Local random move
            i = rng.randint(0, 11)
            new_pos = generate_local_move(i, cx, cy)
            if new_pos:
                new_points = [p[:] for p in current_points]
                new_points[i] = new_pos
        elif move_type < 0.9:
            # Line Snap Move
            i = rng.randint(0, 11)
            new_pos = generate_line_snap_move(i, cx, cy)
            if new_pos:
                new_points = [p[:] for p in current_points]
                new_points[i] = new_pos
        else:
            # Double swap: move two points to snap to a line formed by others?
            # Or just two random local moves
            i1, i2 = rng.sample(range(12), 2)
            new_points = [p[:] for p in current_points]
            p1 = generate_local_move(i1, cx, cy)
            p2 = generate_local_move(i2, cx, cy)
            if p1 and p2 and p1 != p2 and p1 not in new_points and p2 not in new_points:
                new_points[i1] = p1
                new_points[i2] = p2
            else:
                new_points = None

        if new_points is None:
            T *= cooling_rate
            continue
            
        if not _is_valid(new_points):
            T *= cooling_rate
            continue
            
        new_score = count_ordinary_lines(new_points)
        delta = new_score - current_score
        
        # Acceptance criterion
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_points = new_points
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
                if best_score <= 6:
                    break
                    
            # Re-center if we've moved far
            if steps % 100 == 0:
                cx, cy = get_centroid(current_points)
        
        T *= cooling_rate
        
    # Final polish: Hill climbing on the best found
    if time.time() < deadline - 5:
        cur = [p[:] for p in best_points]
        cur_score = best_score
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(12):
                # Try all nearby points in a small box
                x0, y0 = cur[i]
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx == 0 and dy == 0: continue
                        nx, ny = x0 + dx, y0 + dy
                        if abs(nx) > 10**9 or abs(ny) > 10**9: continue
                        if [nx, ny] in cur: continue
                        
                        new_cur = [p[:] for p in cur]
                        new_cur[i] = [nx, ny]
                        if not _is_valid(new_cur): continue
                        
                        s = count_ordinary_lines(new_cur)
                        if s < cur_score:
                            cur_score = s
                            cur = new_cur
                            improved = True
                            if cur_score < best_score:
                                best_score = cur_score
                                best_points = [p[:] for p in cur]
                                if best_score <= 6:
                                    return best_points

    return best_points

if __name__ == "__main__":
    # Quick test
    pts = construct()
    print(f"Score: {count_ordinary_lines(pts)}")
    print(pts)
