import time
import random
import itertools

def _is_collinear(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

def _has_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _is_collinear(points[i], points[j], points[k]) and \
                       _is_collinear(points[i], points[j], points[l]):
                        return True
    return False

def _count_triples(points):
    count = 0
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _is_collinear(points[i], points[j], points[k]):
                    count += 1
    return count

def _max_3_free_subset(points):
    # Branch and bound for maximum independent set in 3-uniform hypergraph
    # Nodes are points. Edges are triples.
    # We want max set with no edges.
    n = len(points)
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _is_collinear(points[i], points[j], points[k]):
                    triples.append((i, j, k))
    
    # Bitmask representation for speed
    # state: (current_mask, remaining_mask, count)
    # Pruning: if count + popcount(remaining_mask) <= best, prune
    
    best = 0
    # Simple backtracking
    def backtrack(remaining_mask, current_count):
        nonlocal best
        if current_count + bin(remaining_mask).count('1') <= best:
            return
        
        if remaining_mask == 0:
            if current_count > best:
                best = current_count
            return
        
        # Pick a point
        # Heuristic: pick the point with the most constraints? Or just the first available?
        # Let's pick the lowest set bit.
        lsb = remaining_mask & -remaining_mask
        idx = (lsb.bit_length() - 1)
        
        # Option 1: Include idx
        # If we include idx, we must remove all points that form a triple with idx
        # Find all points j, k such that (idx, j, k) is a triple
        remove_mask = lsb
        for (i, j, k) in triples:
            if i == idx:
                remove_mask |= (1 << j) | (1 << k)
            elif j == idx:
                remove_mask |= (1 << i) | (1 << k)
            elif k == idx:
                remove_mask |= (1 << i) | (1 << j)
        
        new_remaining = remaining_mask & ~remove_mask
        backtrack(new_remaining, current_count + 1)
        
        # Option 2: Exclude idx
        new_remaining2 = remaining_mask & ~lsb
        backtrack(new_remaining2, current_count)
        
    backtrack((1 << n) - 1, 0)
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    # Initial Configuration: Two Parabolas
    # 10 points on y=x^2, 10 on y=x^2+1
    # Choose x values to spread them out
    pts1 = [[x, x*x] for x in range(-5, 5)] # 10 pts
    pts2 = [[x, x*x+1] for x in range(-5, 5)] # 10 pts
    points = pts1 + pts2
    
    # Ensure distinct
    if len(set(map(tuple, points))) < 20:
        # Fallback to random if collision (unlikely with these)
        points = [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(20)]
    
    # Validate initial
    if _has_four_collinear(points):
        # Perturb until valid
        while _has_four_collinear(points):
            i = random.randint(0, 19)
            points[i] = [random.randint(-100, 100), random.randint(-100, 100)]
    
    best_points = [p[:] for p in points]
    best_triples = _count_triples(points)
    
    # Local Search
    temp = 50.0
    cool = 0.999
    current_triples = best_triples
    
    while time.time() < deadline:
        # Pick a point to move
        i = random.randint(0, 19)
        old_p = points[i][:]
        
        # Generate new point
        # Small step
        if random.random() < 0.5:
            points[i][0] += random.randint(-5, 5)
        else:
            points[i][1] += random.randint(-5, 5)
        
        # Keep in bounds
        points[i][0] = max(-100, min(100, points[i][0]))
        points[i][1] = max(-100, min(100, points[i][1]))
        
        # Check validity
        if _has_four_collinear(points):
            points[i] = old_p
            continue
        
        # Check distinctness
        if len(set(map(tuple, points))) < 20:
            points[i] = old_p
            continue
            
        new_triples = _count_triples(points)
        delta = new_triples - current_triples
        
        # Accept if better, or with probability exp(delta/temp)
        if delta > 0 or random.random() < pow(2.71828, delta / temp):
            current_triples = new_triples
            if new_triples > best_triples:
                best_triples = new_triples
                best_points = [p[:] for p in points]
        else:
            points[i] = old_p
            
        temp *= cool
        if temp < 1.0:
            temp = 1.0

    # Final Verification and Score Calculation
    # The score is the max 3-free subset.
    # We optimized triples, which correlates inversely with score.
    # Return the points. The evaluator will calculate the score.
    
    return best_points
