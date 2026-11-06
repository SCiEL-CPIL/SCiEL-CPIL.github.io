import random
import time
from itertools import combinations

def _collinear(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

def _is_valid(points):
    n = len(points)
    # Check for 4 collinear
    # O(n^4) is 20^4 = 160,000. Fast enough.
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear(points[i], points[j], points[k]):
                    for l in range(k+1, n):
                        if _collinear(points[i], points[j], points[l]):
                            return False
    return True

def _get_triples(points):
    n = len(points)
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear(points[i], points[j], points[k]):
                    triples.append((i, j, k))
    return triples

def _calculate_score(points):
    n = len(points)
    triples = _get_triples(points)
    
    # Precompute pair to thirds
    pair_to_thirds = {}
    for i, j, k in triples:
        pair = (min(i, j), max(i, j))
        if pair not in pair_to_thirds:
            pair_to_thirds[pair] = set()
        pair_to_thirds[pair].add(k)
        
    # Degrees for heuristic
    degrees = [0] * n
    for i, j, k in triples:
        degrees[i] += 1
        degrees[j] += 1
        degrees[k] += 1
        
    best_score = 0
    
    def can_add(v, selected_mask):
        # Check if v can be added to selected_mask
        # Selected mask is a bitmask of indices
        # We need to check if there are any two bits u, w in selected_mask
        # such that (u, w, v) is a triple.
        # Iterate through set bits in selected_mask
        # Since n=20, we can just iterate 0..19
        for u in range(n):
            if not (selected_mask & (1 << u)):
                continue
            for w in range(u+1, n):
                if not (selected_mask & (1 << w)):
                    continue
                pair = (u, w)
                if pair in pair_to_thirds:
                    if v in pair_to_thirds[pair]:
                        return False
        return True

    def search(selected_mask, available_mask):
        nonlocal best_score
        if available_mask == 0:
            if bin(selected_mask).count('1') > best_score:
                best_score = bin(selected_mask).count('1')
            return
            
        # Upper bound
        current_size = bin(selected_mask).count('1')
        ub = current_size + bin(available_mask).count('1')
        if ub <= best_score:
            return
            
        # Pick a vertex from available_mask
        # Heuristic: Pick one with highest degree in the original graph?
        # Or just the first one.
        # Let's pick the one with highest degree among available
        best_v = -1
        best_deg = -1
        temp_mask = available_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            v = lsb.bit_length() - 1
            if degrees[v] > best_deg:
                best_deg = degrees[v]
                best_v = v
            temp_mask ^= lsb
            
        if best_v == -1:
            return
            
        new_available = available_mask & ~(1 << best_v)
        
        # Option 1: Exclude best_v
        search(selected_mask, new_available)
        
        # Option 2: Include best_v
        if can_add(best_v, selected_mask):
            search(selected_mask | (1 << best_v), new_available)
            
    search(0, (1 << n) - 1)
    return best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial points: 5x4 grid
    # x in 0..4, y in 0..3
    # Scale to fit in -100..100
    # Let's use x in -20..20 step 10, y in -15..15 step 10
    # 5 points: -20, -10, 0, 10, 20
    # 4 points: -15, -5, 5, 15
    xs = [-20, -10, 0, 10, 20]
    ys = [-15, -5, 5, 15]
    
    initial_points = []
    for x in xs:
        for y in ys:
            initial_points.append((x, y))
            
    # This grid has 4 collinear points (rows and cols).
    # We need to repair it.
    
    def repair(points):
        # Move points until valid
        for _ in range(1000):
            if _is_valid(points):
                return True
            # Find a violation
            # Just pick a random point and move it
            idx = random.randint(0, 19)
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            if (new_x, new_y) not in points:
                points[idx] = (new_x, new_y)
        return False

    best_score = 20
    best_points = initial_points[:]
    
    # Initial repair
    current_points = initial_points[:]
    if not repair(current_points):
        # Fallback to random
        while not repair(current_points):
             current_points = [(random.randint(-100, 100), random.randint(-100, 100)) for _ in range(20)]
             
    current_score = _calculate_score(current_points)
    if current_score < best_score:
        best_score = current_score
        best_points = current_points[:]
        
    while time.time() < deadline:
        # Local Search
        idx = random.randint(0, 19)
        old_x, old_y = current_points[idx]
        new_x = random.randint(-100, 100)
        new_y = random.randint(-100, 100)
        
        if (new_x, new_y) == (old_x, old_y):
            continue
        if (new_x, new_y) in current_points:
            continue
            
        current_points[idx] = (new_x, new_y)
        
        if _is_valid(current_points):
            new_score = _calculate_score(current_points)
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_points = current_points[:]
            else:
                # Revert if not better (greedy)
                # Or accept with prob 0.1 to explore?
                # Let's stick to greedy for now, with occasional restarts
                current_points[idx] = (old_x, old_y)
        else:
            # Revert
            current_points[idx] = (old_x, old_y)
            
        # Occasional restart from a slightly perturbed grid
        if random.random() < 0.01:
            new_pts = initial_points[:]
            for i in range(20):
                if random.random() < 0.5:
                    new_pts[i] = (new_pts[i][0] + random.randint(-5, 5), new_pts[i][1] + random.randint(-5, 5))
            if repair(new_pts):
                s = _calculate_score(new_pts)
                if s < current_score:
                    current_points = new_pts
                    current_score = s
                    if s < best_score:
                        best_score = s
                        best_points = new_pts[:]

    return [list(p) for p in best_points]
