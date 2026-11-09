import random
import time

def _collinear(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p1[1]) == (p1[0] - p2[0]) * (p3[1] - p1[1])

def _count_collinear_triples(points):
    """Count number of collinear triples."""
    count = 0
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear(points[i], points[j], points[k]):
                    count += 1
    return count

def _has_four_collinear(points):
    """Check if any 4 points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _collinear(points[i], points[j], points[k]):
                        if _collinear(points[i], points[j], points[l]):
                            return True
    return False

def _generate_random_valid():
    """Generate 20 random points with no four collinear."""
    while True:
        points = []
        seen = set()
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in seen:
                seen.add((x, y))
                points.append([x, y])
        if not _has_four_collinear(points):
            return points

def _local_search(points, deadline):
    """Local search minimizing collinear triples while maintaining no-4-collinear."""
    current_triples = _count_collinear_triples(points)
    best_triples = current_triples
    best_points = points[:]
    n = len(points)
    
    while time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for _ in range(30):
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
                new_pt = [new_x, new_y]
                
                if any(new_pt == points[j] for j in range(n) if j != i):
                    continue
                
                candidate = points[:]
                candidate[i] = new_pt
                if _has_four_collinear(candidate):
                    continue
                
                new_triples = _count_collinear_triples(candidate)
                if new_triples < best_triples:
                    best_triples = new_triples
                    best_points = candidate
                    points = candidate
                    current_triples = new_triples
                    improved = True
                    break
        if not improved:
            break
    
    return best_points, best_triples

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    random.seed(12345)
    
    best_points = None
    best_triples = float('inf')
    
    # Multiple restarts
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        points = _generate_random_valid()
        triples = _count_collinear_triples(points)
        
        if triples < best_triples:
            best_points = points[:]
            best_triples = triples
        
        # Local search
        if time.time() < deadline - 10:
            improved_points, improved_triples = _local_search(points, min(deadline - 5, time.time() + 20))
            if improved_triples < best_triples:
                best_points = improved_points
                best_triples = improved_triples
    
    # Final validation
    assert len(best_points) == 20
    assert len(set(tuple(p) for p in best_points)) == 20
    assert all(abs(p[0]) <= 100 and abs(p[1]) <= 100 for p in best_points)
    assert not _has_four_collinear(best_points)
    
    return best_points
