import random
import time
import math
from itertools import combinations

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def _are_collinear(p1, p2, p3):
    return (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]) == 0

def _no_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    for l in range(k+1, n):
                        if _are_collinear(points[i], points[j], points[l]):
                            return False
    return True

def _build_triple_graph(points):
    """Build list of collinear triples (indices)."""
    n = len(points)
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    triples.append((i, j, k))
    return triples

def _greedy_max_3free(points):
    """Greedy: find a large subset with no three collinear.
    Returns the size of the subset found (lower bound on max 3-free)."""
    n = len(points)
    triples = _build_triple_graph(points)
    if not triples:
        return n
    
    # Build point -> triples incidence
    point_triples = {i: [] for i in range(n)}
    for idx, (a, b, c) in enumerate(triples):
        point_triples[a].append(idx)
        point_triples[b].append(idx)
        point_triples[c].append(idx)
    
    # Greedy: remove point in most remaining triples, repeat
    active = set(range(n))
    active_triples = set(range(len(triples)))
    removed = 0
    
    while active_triples:
        # Find point in active that appears in most active triples
        best_point = None
        best_count = -1
        for p in active:
            cnt = sum(1 for t in point_triples[p] if t in active_triples)
            if cnt > best_count:
                best_count = cnt
                best_point = p
        if best_point is None or best_count == 0:
            break
        # Remove best_point
        active.remove(best_point)
        removed += 1
        # Remove all triples containing best_point
        for t in point_triples[best_point]:
            if t in active_triples:
                active_triples.discard(t)
    
    return len(active)

def _random_valid_points():
    """Generate 20 random points with no four collinear."""
    while True:
        points = []
        seen = set()
        for _ in range(20):
            while True:
                x = random.randint(-50, 50)
                y = random.randint(-50, 50)
                if (x, y) not in seen:
                    seen.add((x, y))
                    break
            points.append([x, y])
        if _no_four_collinear(points):
            return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Initial population
    best_points = _random_valid_points()
    best_score = _greedy_max_3free(best_points)
    
    # Local search with simulated annealing
    temperature = 2.0
    cooling = 0.9995
    
    points = best_points[:]
    current_score = best_score
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 19)
        old_x, old_y = points[idx][0], points[idx][1]
        
        # Try a random new position nearby
        new_x = old_x + random.randint(-8, 8)
        new_y = old_y + random.randint(-8, 8)
        
        if abs(new_x) > 100 or abs(new_y) > 100:
            continue
        
        # Check distinctness
        if any(points[i][0] == new_x and points[i][1] == new_y for i in range(20) if i != idx):
            continue
        
        # Try the move
        points[idx][0] = new_x
        points[idx][1] = new_y
        
        if not _no_four_collinear(points):
            points[idx][0] = old_x
            points[idx][1] = old_y
            continue
        
        new_score = _greedy_max_3free(points)
        
        if new_score < current_score:
            # Better (lower is better)
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = points[:]
        else:
            # Accept with probability
            delta = new_score - current_score
            if delta > 0 and random.random() < math.exp(-delta / temperature):
                current_score = new_score
            else:
                points[idx][0] = old_x
                points[idx][1] = old_y
        
        temperature *= cooling
        if temperature < 0.1:
            temperature = 0.1
    
    return best_points
