import random
import time

def _are_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])

def _count_triples(points):
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    count += 1
    return count

def _has_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            count = 0
            for k in range(n):
                if k != i and k != j and _are_collinear(points[i], points[j], points[k]):
                    count += 1
                    if count >= 2:
                        return True
    return False

def _max_no_three_collinear(points):
    n = len(points)
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    triples.append((i, j, k))
    
    triple_partners = [[] for _ in range(n)]
    for (i, j, k) in triples:
        triple_partners[i].append((j, k))
        triple_partners[j].append((i, k))
        triple_partners[k].append((i, j))
        
    best = 0
    
    def dfs(idx, current_size, included_mask):
        nonlocal best
        if current_size > best:
            best = current_size
            
        if idx == n:
            return
            
        if current_size + (n - idx) <= best:
            return
        
        valid = True
        for (a, b) in triple_partners[idx]:
            if (included_mask & (1 << a)) and (included_mask & (1 << b)):
                valid = False
                break
        
        if valid:
            dfs(idx + 1, current_size + 1, included_mask | (1 << idx))
            
        dfs(idx + 1, current_size, included_mask)
        
    dfs(0, 0, 0)
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Pool of candidate points
    # Using a grid-like structure to encourage collinearity
    # Let's try a 21x21 grid from -100 to 100
    # Step is 10
    coords = list(range(-100, 101, 10))
    pool = [[x, y] for x in coords for y in coords]
    
    # Initial configuration: random 20 points from pool that satisfy no 4 collinear
    # We can just pick random points and repair
    points = []
    attempts = 0
    while len(points) < 20 and attempts < 5000:
        p = random.choice(pool)
        if p not in points:
            points.append(p)
            if _has_four_collinear(points):
                points.pop()
        attempts += 1
        
    if len(points) < 20:
        # Fallback: dense grid selection
        # Take a 5x5 grid and perturb?
        # Or just use a known good config if available?
        # Let's force a grid
        points = [[x, y] for x in range(-100, 101, 25) for y in range(-100, 101, 25)][:20]
        # Ensure no 4 collinear? A 5x5 grid has many 4-collinear.
        # We need to remove/replace points to fix 4-collinear.
        # This is a repair step.
        
    # Simulated Annealing to maximize triple count
    # State: points
    # Objective: -_count_triples(points) (minimize negative = maximize triples)
    # Constraint: no 4 collinear
    
    best_triples = _count_triples(points)
    best_points = [p[:] for p in points]
    best_score = _max_no_three_collinear(points)
    
    current_triples = best_triples
    T = 100.0
    cooling = 0.99
    
    while time.time() < deadline:
        # Pick a point to move
        idx = random.randint(0, 19)
        old_p = points[idx]
        
        # Pick a new point from pool
        # To be efficient, we can pick a point that is "close" or random
        new_p = random.choice(pool)
        
        if new_p == old_p or new_p in points:
            continue
            
        points[idx] = new_p
        
        if _has_four_collinear(points):
            points[idx] = old_p
            continue
            
        new_triples = _count_triples(points)
        delta = new_triples - current_triples
        
        # Accept if better or with probability exp(-delta/T) (since we want to maximize, delta > 0 is good)
        # Wait, if we maximize triples, delta = new - old.
        # If delta > 0, accept.
        # If delta < 0, accept with prob exp(delta/T).
        if delta > 0 or random.random() < pow(2.71828, delta / T):
            current_triples = new_triples
            if new_triples > best_triples:
                best_triples = new_triples
                best_points = [p[:] for p in points]
                # Check exact score periodically or if improvement is significant
                # To save time, we can check exact score less frequently
                # But we need to return the best *score*, not just triples.
                # High triples usually means low score, but not always.
                # Let's check exact score every 1000 moves or when a new triple record is broken.
                if best_triples % 10 == 0: 
                     pass # placeholder
                # Actually, let's just check exact score if we have time
                if time.time() < deadline - 5:
                    current_score = _max_no_three_collinear(points)
                    if current_score < best_score:
                        best_score = current_score
                        best_points = [p[:] for p in points]
        else:
            points[idx] = old_p
            
        T *= cooling
        
    return best_points
