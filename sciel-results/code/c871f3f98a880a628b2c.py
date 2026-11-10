import random
import time
import sys
from itertools import combinations

_START_TIME = None
DEADLINE = 150.0
MARGIN = 15.0

def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME

def _deadline_left():
    return DEADLINE - MARGIN - _elapsed()

def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def are_collinear(p1, p2, p3):
    return cross(p1, p2, p3) == 0

def has_four_collinear(points):
    """Check if any 4 points are collinear. Returns True if invalid."""
    n = len(points)
    # For each pair, count points on the line
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = points[i], points[j]
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                if are_collinear(p1, p2, points[k]):
                    count += 1
            if count >= 4:
                return True
    return False

def get_collinear_triples(points):
    """Return list of (i, j, k) tuples for collinear triples."""
    n = len(points)
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if are_collinear(points[i], points[j], points[k]):
                    triples.append((i, j, k))
    return triples

def estimate_max_disjoint_triples(points, iterations=100):
    """
    Estimate the maximum number of disjoint collinear triples using randomized greedy.
    This provides an upper bound on the maximum 3-free subset size: 20 - k.
    We want to maximize k.
    """
    n = len(points)
    triples = get_collinear_triples(points)
    
    if not triples:
        return 0
        
    best_k = 0
    # To speed up, we can just run a few random greedy passes
    for _ in range(iterations):
        random.shuffle(triples)
        used = [False] * n
        count = 0
        for t in triples:
            i, j, k = t
            if not used[i] and not used[j] and not used[k]:
                used[i] = True
                used[j] = True
                used[k] = True
                count += 1
        if count > best_k:
            best_k = count
            
    return best_k

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    best_k = -1
    best_points = None
    
    def random_point():
        return (random.randint(-100, 100), random.randint(-100, 100))

    # Strategy:
    # 1. Initialize with structures that naturally have many disjoint triples (e.g., grids, lines).
    # 2. Local search to maximize the estimated max disjoint triples count (k).
    # 3. The upper bound on the score (max 3-free subset) is 20 - k.
    #    Minimizing the score is equivalent to maximizing k.
    
    # Common structures:
    # - 4x5 grid: has many horizontal/vertical triples.
    # - 5x4 grid.
    # - Random points on a few lines.
    
    for restart in range(1000):
        if _deadline_left() < 5.0:
            break
            
        # Choose initialization strategy
        strategy = random.randint(0, 2)
        points = []
        
        if strategy == 0:
            # Grid-like
            ox = random.randint(-80, 80)
            oy = random.randint(-80, 80)
            sx = random.randint(1, 10)
            sy = random.randint(1, 10)
            rows = random.choice([4, 5])
            cols = random.choice([4, 5])
            
            candidates = []
            for i in range(rows):
                for j in range(cols):
                    x = ox + i * sx + random.randint(-1, 1)
                    y = oy + j * sy + random.randint(-1, 1)
                    if -100 <= x <= 100 and -100 <= y <= 100:
                        candidates.append((x, y))
                        
            seen = set()
            unique_candidates = []
            for p in candidates:
                if p not in seen:
                    seen.add(p)
                    unique_candidates.append(p)
                    
            while len(unique_candidates) < 20:
                p = random_point()
                if p not in seen:
                    seen.add(p)
                    unique_candidates.append(p)
                    
            random.shuffle(unique_candidates)
            points = unique_candidates[:20]
            
        elif strategy == 1:
            # Points clustered on lines
            # Pick 4-5 lines, put 4-5 points on each
            lines = []
            for _ in range(random.randint(4, 5)):
                # Random line ax + by + c = 0
                a = random.randint(-10, 10)
                b = random.randint(-10, 10)
                if a == 0 and b == 0:
                    a = 1
                c = random.randint(-50, 50)
                lines.append((a, b, c))
                
            candidates = []
            for a, b, c in lines:
                # Sample points on the line
                # x = t, y = -(a*t + c)/b if b != 0
                if b != 0:
                    for _ in range(5):
                        t = random.randint(-100, 100)
                        # y must be integer
                        # a*t + c must be divisible by b
                        # Just find integer solutions
                        # This is hard to do quickly for random lines.
                        # Let's just pick random integer points and check if they are close to the line?
                        # No, we want them ON the line to form triples.
                        # Let's pick 2 random points on the line and interpolate?
                        # Simpler: Pick random integer x, check if y is integer.
                        # This might yield few points.
                        pass
                # Fallback for this strategy: just use random points and hope for the best?
                # Or use a simpler line definition: y = mx + c
                # Let's restart this strategy with y = mx + c
                pass
            
            # Let's simplify strategy 1: Use y = mx + c lines
            candidates = []
            num_lines = random.randint(4, 6)
            used_points = set()
            for _ in range(num_lines):
                m = random.randint(-3, 3)
                c = random.randint(-50, 50)
                # Find integer points on y = mx + c within bounds
                line_points = []
                x_min = max(-100, (-100 - c) // m if m > 0 else (100 - c) // m if m < 0 else -100)
                # This logic is messy. Just iterate x.
                for x in range(-100, 101):
                    y = m * x + c
                    if -100 <= y <= 100:
                        p = (x, y)
                        if p not in used_points:
                            line_points.append(p)
                            
                # Pick 4-5 points from this line
                if len(line_points) >= 4:
                    chosen = random.sample(line_points, random.randint(4, 5))
                    candidates.extend(chosen)
                    for p in chosen:
                        used_points.add(p)
                        
            while len(candidates) < 20:
                p = random_point()
                if p not in used_points:
                    candidates.append(p)
                    used_points.add(p)
            points = candidates[:20]
            
        else:
            # Pure random
            seen = set()
            while len(points) < 20:
                p = random_point()
                if p not in seen:
                    seen.add(p)
                    points.append(p)

        # Ensure no 4 collinear
        valid = False
        attempts = 0
        while not valid and attempts < 100:
            if not has_four_collinear(points):
                valid = True
                break
            idx = random.randint(0, 19)
            new_p = random_point()
            if new_p not in points:
                points[idx] = new_p
            attempts += 1
            
        if not valid:
            continue
            
        current_k = estimate_max_disjoint_triples(points, iterations=50)
        
        if current_k > best_k:
            best_k = current_k
            best_points = points.copy()
            
        # Local Search
        current_points = points.copy()
        current_k = current_k
        
        steps = 50 if restart < 50 else 20
        for step in range(steps):
            if _deadline_left() < 10.0:
                break
                
            idx = random.randint(0, 19)
            old_p = current_points[idx]
            
            if random.random() < 0.5:
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                new_p = (old_p[0] + dx, old_p[1] + dy)
                if new_p[0] < -100 or new_p[0] > 100 or new_p[1] < -100 or new_p[1] > 100:
                    continue
            else:
                new_p = random_point()
            
            if new_p in current_points:
                continue
                
            candidate_points = current_points.copy()
            candidate_points[idx] = new_p
            
            if has_four_collinear(candidate_points):
                continue
                
            new_k = estimate_max_disjoint_triples(candidate_points, iterations=20)
            
            if new_k > current_k:
                current_points = candidate_points
                current_k = new_k
                if current_k > best_k:
                    best_k = current_k
                    best_points = current_points.copy()

    if best_points is None:
        # Fallback
        points = []
        seen = set()
        while len(points) < 20:
            p = random_point()
            if p not in seen:
                seen.add(p)
                points.append(p)
        for _ in range(100):
            if not has_four_collinear(points):
                break
            idx = random.randint(0, 19)
            new_p = random_point()
            if new_p not in points:
                points[idx] = new_p
        best_points = points

    return [list(p) for p in best_points]
