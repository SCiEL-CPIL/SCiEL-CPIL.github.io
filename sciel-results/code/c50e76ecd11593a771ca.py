import random
import time

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 300s kill
    
    # Precompute all grid points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    def sq_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + dy * dy
    
    def is_isosceles(a, b, c):
        """Check if triangle with vertices a, b, c is isosceles (including degenerate)."""
        d_ab = sq_dist(a, b)
        d_bc = sq_dist(b, c)
        d_ca = sq_dist(c, a)
        # Isosceles if any two sides are equal
        if d_ab == d_bc or d_ab == d_ca or d_bc == d_ca:
            return True
        return False
    
    def can_add(points, p):
        """Check if adding point p to existing points creates an isosceles triangle."""
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                a = points[i]
                b = points[j]
                if is_isosceles(p, a, b):
                    return False
        return True
    
    def greedy_random(seed=0):
        """Greedy algorithm with randomization."""
        random.seed(seed)
        points = []
        remaining = all_points[:]
        random.shuffle(remaining)
        
        for p in remaining:
            if time.time() > deadline:
                break
            if can_add(points, p):
                points.append(p)
        
        return points
    
    def greedy_deterministic():
        """Deterministic greedy: try points in a fixed order."""
        points = []
        for p in all_points:
            if can_add(points, p):
                points.append(p)
        return points
    
    # First, try the simple line-based approach from the seed
    def line_sidon():
        xs, used = [], set()
        x = 0
        while x < N:
            diffs = {abs(x - y) for y in xs}
            if not (diffs & used):
                used |= diffs
                xs.append(x)
            x += 1
        return [(x, 0) for x in xs]
    
    best = line_sidon()
    best_len = len(best)
    
    # Try deterministic greedy
    if time.time() < deadline:
        det_result = greedy_deterministic()
        if len(det_result) > best_len:
            best = det_result
            best_len = len(det_result)
    
    # Try multiple random seeds
    seed = 0
    while time.time() < deadline - 5.0:
        result = greedy_random(seed)
        if len(result) > best_len:
            best = result
            best_len = len(result)
        seed += 1
    
    return best

# EVOLVE-BLOCK-END
