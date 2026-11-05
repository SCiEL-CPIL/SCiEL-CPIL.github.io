"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

_start_time = time.time()
_DEADLINE = 100  # seconds, safety margin for 120s kill

def _count_four_lines(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    count = 0
    # Use a set of canonical line representations to avoid double counting
    seen = set()
    for i in range(n):
        for j in range(i+1, n):
            p1, p2 = points[i], points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            # Canonical direction
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            # Reduce by gcd
            g = 0
            a, b = abs(dx), abs(dy)
            while b:
                a, b = b, a % b
            g = a if a else 1
            if g > 1:
                dx //= g
                dy //= g
            # Line: all points (x,y) such that (x - p1[0])*dy - (y - p1[1])*dx == 0
            # To make canonical, use the normalized direction and a reference point
            # Count points on this line
            line_pts = set()
            for k in range(n):
                px, py = points[k]
                if (px - p1[0]) * dy == (py - p1[1]) * dx:
                    line_pts.add(k)
            if len(line_pts) == 4:
                # Canonical form: sort the indices
                key = tuple(sorted(line_pts))
                if key not in seen:
                    seen.add(key)
                    count += 1
    return count

def _is_valid(points):
    """Check that all points are distinct, |x|,|y| <= 1000, and no five collinear."""
    if len(points) != 16:
        return False
    # Check bounds
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    # Check distinct
    if len(set(tuple(p) for p in points)) != 16:
        return False
    # Check no five collinear
    n = len(points)
    seen_lines = set()
    for i in range(n):
        for j in range(i+1, n):
            p1, p2 = points[i], points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            g = 0
            a, b = abs(dx), abs(dy)
            while b:
                a, b = b, a % b
            g = a if a else 1
            if g > 1:
                dx //= g
                dy //= g
            line_pts = set()
            for k in range(n):
                px, py = points[k]
                if (px - p1[0]) * dy == (py - p1[1]) * dx:
                    line_pts.add(k)
            if len(line_pts) >= 5:
                return False
            key = tuple(sorted(line_pts))
            seen_lines.add(key)
    return True

def _initial_grid():
    return [[x, y] for x in range(4) for y in range(4)]

def _try_move(points, idx, dx, dy):
    """Try moving point idx by (dx, dy)."""
    new_pts = [list(p) for p in points]
    new_pts[idx] = [new_pts[idx][0] + dx, new_pts[idx][1] + dy]
    # Check bounds
    x, y = new_pts[idx]
    if abs(x) > 1000 or abs(y) > 1000:
        return None
    # Check distinct
    for j in range(16):
        if j != idx and new_pts[j] == new_pts[idx]:
            return None
    # Check no five collinear
    n = 16
    for i in range(n):
        for j in range(i+1, n):
            p1, p2 = new_pts[i], new_pts[j]
            ddx = p2[0] - p1[0]
            ddy = p2[1] - p1[1]
            if ddx < 0 or (ddx == 0 and ddy < 0):
                ddx, ddy = -ddx, -ddy
            a, b = abs(ddx), abs(ddy)
            while b:
                a, b = b, a % b
            g = a if a else 1
            if g > 1:
                ddx //= g
                ddy //= g
            cnt = 0
            for k in range(n):
                px, py = new_pts[k]
                if (px - p1[0]) * ddy == (py - p1[1]) * ddx:
                    cnt += 1
                    if cnt >= 5:
                        return None
    return new_pts

def construct():
    global _start_time
    random.seed(42)
    
    # Start with a good initial configuration
    # 4x4 grid gives 10 lines of 4.
    # We can try to perturb it to get more.
    best = _initial_grid()
    best_score = _count_four_lines(best)
    
    # Simulated Annealing
    current = [list(p) for p in best]
    current_score = best_score
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # Precompute a faster check for validity (no 5 collinear)
    # We'll rely on _try_move for validity, but make it faster by checking only necessary constraints
    
    while time.time() - _start_time < _DEADLINE and T > T_min:
        # Pick a random point to move
        idx = random.randint(0, 15)
        # Random move, larger steps to escape local minima
        dx = random.randint(-10, 10)
        dy = random.randint(-10, 10)
        if dx == 0 and dy == 0:
            continue
            
        new_pts = _try_move(current, idx, dx, dy)
        if new_pts is None:
            T *= cooling_rate
            continue
            
        new_score = _count_four_lines(new_pts)
        
        # Accept if better, or with probability based on temperature if worse
        delta = new_score - current_score
        if delta > 0:
            current = new_pts
            current_score = new_score
            if current_score > best_score:
                best = [list(p) for p in current]
                best_score = current_score
        else:
            # Metropolis criterion
            # Since score is integer, we can use exp(delta/T)
            # But delta is negative.
            p = math.exp(delta / T)
            if random.random() < p:
                current = new_pts
                current_score = new_score
        
        T *= cooling_rate
        
        # Occasionally, if stuck, do a random restart or larger perturbation
        if random.random() < 0.01:
            # Try a few random large moves
            for _ in range(5):
                idx = random.randint(0, 15)
                dx = random.randint(-50, 50)
                dy = random.randint(-50, 50)
                new_pts = _try_move(current, idx, dx, dy)
                if new_pts is not None:
                    ns = _count_four_lines(new_pts)
                    if ns > current_score:
                        current = new_pts
                        current_score = ns
                        if current_score > best_score:
                            best = [list(p) for p in current]
                            best_score = current_score
                        break
                        
    return [[int(x), int(y)] for x, y in best]

# EVOLVE-BLOCK-END
