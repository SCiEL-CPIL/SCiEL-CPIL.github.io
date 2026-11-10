import itertools
import random
import numpy as np

def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def _collinear3(p, q, r):
    return _cross(p, q, r) == 0

def _has_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if not _collinear3(points[i], points[j], points[k]):
                    continue
                for l in range(k+1, n):
                    if _collinear3(points[i], points[j], points[l]):
                        return True
    return False

def _count_new_triples(S, p):
    """Count collinear triples formed by p and two points in S."""
    count = 0
    n = len(S)
    for i in range(n):
        for j in range(i+1, n):
            if _collinear3(S[i], S[j], p):
                count += 1
    return count

def _is_valid_addition(S, p):
    """Check if adding p to S creates 4 collinear points."""
    n = len(S)
    for i in range(n):
        for j in range(i+1, n):
            if _collinear3(S[i], S[j], p):
                # Check if any other point in S is on this line
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if _collinear3(S[i], S[j], S[k]):
                        return False
    return True

def _greedy_construct(pool, target_size=20, seed=42):
    """
    Greedily construct a set of points from pool to maximize collinear triples.
    """
    random.seed(seed)
    S = []
    remaining = list(pool)
    random.shuffle(remaining)
    
    for _ in range(target_size):
        best_p = None
        best_triples = -1
        # We can limit the number of candidates considered to speed up
        # Or just iterate all
        for p in remaining:
            if not _is_valid_addition(S, p):
                continue
            triples = _count_new_triples(S, p)
            if triples > best_triples:
                best_triples = triples
                best_p = p
            elif triples == best_triples:
                # Tie-breaking: random or distance
                if random.random() < 0.5:
                    best_p = p
                    best_triples = triples
        
        if best_p is None:
            # No valid point found? This shouldn't happen with a large enough pool
            # If it does, break
            break
        
        S.append(best_p)
        remaining.remove(best_p)
        
    return S

def _max_independent_set(points):
    """
    Exact branch and bound for largest subset with no three collinear.
    """
    n = len(points)
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear3(points[i], points[j], points[k]):
                    triples.append((i, j, k))
    
    if not triples:
        return n
    
    point_triples = [[] for _ in range(n)]
    for t in triples:
        for idx in t:
            point_triples[idx].append(t)
    
    best = [0]
    
    def _search(remaining, current_size):
        if not remaining:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        if current_size + len(remaining) <= best[0]:
            return
        
        active = [t for t in triples if all(p in remaining for p in t)]
        
        if not active:
            new_size = current_size + len(remaining)
            if new_size > best[0]:
                best[0] = new_size
            return
        
        t = active[0]
        for p in t:
            new_remaining = remaining - {p}
            _search(new_remaining, current_size)
            
    _search(set(range(n)), 0)
    return best[0]

def _score(points):
    if _has_four_collinear(points):
        return 100
    return _max_independent_set(points)

def construct():
    # Define pools
    pools = []
    # Grid 1: -5 to 5
    for x in range(-5, 6):
        for y in range(-5, 6):
            pools.append([x, y])
            
    # Grid 2: -10 to 10, step 2 (to have more spread)
    pool2 = []
    for x in range(-10, 11, 2):
        for y in range(-10, 11, 2):
            pool2.append([x, y])
            
    # Grid 3: Random points in [-50, 50]
    random.seed(123)
    pool3 = []
    for _ in range(200):
        x = random.randint(-50, 50)
        y = random.randint(-50, 50)
        pool3.append([x, y])
        
    pools.append(pool2)
    pools.append(pool3)
    
    best_score = 100
    best_points = None
    
    for pool in pools:
        # Try multiple seeds for each pool
        for seed in range(10):
            pts = _greedy_construct(pool, 20, seed)
            if len(pts) < 20:
                continue
            if _has_four_collinear(pts):
                continue
            score = _score(pts)
            if score < best_score:
                best_score = score
                best_points = pts
                # Early exit if we get a very low score?
                # Minimum possible is 1 (if all 20 are on one line, but we can't have 4 collinear).
                # Actually, if no 4 collinear, we can have 3 on a line.
                # Max independent set in a set of 20 with no 4 collinear...
                # If we group them in lines of 3, we have 6 lines of 3 + 2 points.
                # Independent set would pick at most 2 from each line of 3?
                # No, independent set means NO 3 collinear.
                # So from a line of 3, we can pick at most 2.
                # If we have 6 disjoint lines of 3, we can pick 2 from each = 12.
                # Plus 2 single points = 14.
                # So score 14 is possible?
                # If we can get score < 14, it's good.
                
    if best_points is None:
        # Fallback
        return [[x, x * x] for x in range(-10, 10)]
        
    return best_points
