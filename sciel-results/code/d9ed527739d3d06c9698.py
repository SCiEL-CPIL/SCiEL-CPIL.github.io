import random
import itertools
import time

def _is_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

def _find_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _is_collinear(points[i], points[j], points[k]):
                        if _is_collinear(points[i], points[j], points[l]):
                            return [i, j, k, l]
    return None

def _count_triples(points):
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _is_collinear(points[i], points[j], points[k]):
                    count += 1
    return count

def _estimate_score(points):
    # Greedy removal of points involved in triples
    # This gives an upper bound on the largest subset with no 3 collinear.
    # We want to minimize this upper bound.
    current_set = list(range(len(points)))
    changed = True
    while changed:
        changed = False
        # Find a triple in the current set
        for i in range(len(current_set)):
            for j in range(i + 1, len(current_set)):
                for k in range(j + 1, len(current_set)):
                    p1 = points[current_set[i]]
                    p2 = points[current_set[j]]
                    p3 = points[current_set[k]]
                    if _is_collinear(p1, p2, p3):
                        # Remove one point, say the last one
                        current_set.pop(k)
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
    return len(current_set)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    best_score = 20
    best_points = None
    
    # Initial guess: try to form many triples
    # A known good structure is often related to grids with perturbations.
    # Let's try a few random restarts and local optimizations.
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Generate initial points
        # Try to cluster points on lines of 3
        # Randomly pick 6-7 lines, put 3 points on each?
        # Or just random points and repair.
        
        points = []
        while len(points) < 20:
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            p = [x, y]
            if p not in points:
                points.append(p)
        
        # Repair 4-collinearities
        valid = False
        for _ in range(1000):
            four = _find_four_collinear(points)
            if four is None:
                valid = True
                break
            # Move one of the four points
            idx = random.choice(four)
            old_p = points[idx]
            new_p = None
            for _ in range(100):
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                cand = [x, y]
                if cand == old_p or cand in points:
                    continue
                points[idx] = cand
                # Check if this creates a new 4-collinearity involving idx
                # Simple check: just re-run _find_four_collinear
                if _find_four_collinear(points) is None:
                    new_p = cand
                    break
                else:
                    points[idx] = old_p
            if new_p is None:
                # Failed to repair easily, restart this configuration
                break
        
        if not valid:
            continue
            
        score = _estimate_score(points)
        if score < best_score:
            best_score = score
            best_points = points[:]
            
        # Local search: try to improve score by moving points
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_score = _estimate_score(points)
            for i in range(20):
                if time.time() > deadline:
                    break
                old_p = points[i]
                # Try moving point i to a random location
                for _ in range(50):
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    cand = [x, y]
                    if cand == old_p or cand in points:
                        continue
                    points[i] = cand
                    if _find_four_collinear(points) is None:
                        new_score = _estimate_score(points)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            break
                        else:
                            points[i] = old_p
                    else:
                        points[i] = old_p
                if improved:
                    break
            
    if best_points is None:
        # Fallback to parabola if no valid config found (should not happen)
        return [[x, x * x] for x in range(-10, 10)]
        
    return best_points
