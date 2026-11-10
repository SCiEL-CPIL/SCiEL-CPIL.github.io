"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    best_points = None
    best_score = -1

    def score_points(pts):
        """Score: number of lines with exactly 4 points, penalty for 5+ collinear."""
        n = len(pts)
        if len(set(tuple(p) for p in pts)) != n:
            return -1
        # Check bounds
        for x, y in pts:
            if abs(x) > 1000 or abs(y) > 1000:
                return -1
        # Count lines by direction
        line_counts = defaultdict(int)
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                # Normalize direction
                g = gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx //= g
                dy //= g
                # Ensure canonical form: dx > 0 or dx == 0 and dy > 0
                if dx < 0 or (dx == 0 and dy < 0):
                    dx = -dx
                    dy = -dy
                # Compute line offset (b in ax + by = c, where a=dy, b=-dx)
                a = dy
                b = -dx
                c = a * x1 + b * y1
                line_counts[(a, b, c)] += 1
        
        score = 0
        for key, count in line_counts.items():
            if count == 6:  # C(4,2)=6 pairs for 4 points
                score += 1
            elif count >= 10:  # C(5,2)=10 pairs for 5 points -> penalty
                score -= 100
        return score

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def try_improve(pts, time_limit):
        """Hill-climbing with random moves."""
        nonlocal best_points, best_score
        current = [list(p) for p in pts]
        current_score = score_points(current)
        if current_score > best_score:
            best_score = current_score
            best_points = [list(p) for p in current]
        
        for _ in range(5000):
            if time.time() > time_limit:
                break
            # Random move: pick a point, change one coordinate
            idx = random.randint(0, 15)
            axis = random.randint(0, 1)
            old_val = current[idx][axis]
            new_val = random.randint(-1000, 1000)
            if new_val == old_val:
                continue
            current[idx][axis] = new_val
            new_score = score_points(current)
            if new_score >= current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [list(p) for p in current]
            else:
                current[idx][axis] = old_val
        return current

    # Start with a good initial configuration
    # Try a few different starting points
    random.seed(42)
    
    # Initial: 4x4 grid
    init1 = [[x, y] for x in range(4) for y in range(4)]
    
    # Try improving from grid
    result1 = try_improve(init1, deadline)
    
    if time.time() > deadline:
        return [list(p) for p in best_points] if best_points else init1
    
    # Try a few more random starts
    for attempt in range(20):
        if time.time() > deadline:
            break
        # Random points
        pts = []
        while len(pts) < 16:
            p = [random.randint(-50, 50), random.randint(-50, 50)]
            if p not in pts:
                pts.append(p)
        result = try_improve(pts, deadline)
        if time.time() > deadline:
            break

    return [list(p) for p in best_points] if best_points else init1
# EVOLVE-BLOCK-END
