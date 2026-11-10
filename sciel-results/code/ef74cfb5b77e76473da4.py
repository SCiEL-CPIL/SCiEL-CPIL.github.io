"""Minimize largest 3-collinear-free subset for 20 points (no four collinear)."""
import time
import random
import sys
from itertools import combinations

def _check_no_four_collinear(points):
    """Check if no four points are collinear. Returns True if valid."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            count = 2
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    count += 1
            if count >= 4:
                return False
    return True

def _build_triple_conflicts(points):
    """
    Build a list of all collinear triples.
    Returns a list of sets of indices, each set being a triple {i, j, k}.
    """
    n = len(points)
    triples = []
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            collinear = []
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                if dx * (y3 - y1) == dy * (x3 - x1):
                    collinear.append(k)
            # Any subset of 2 from collinear + {i,j} forms a triple with i,j
            # Actually, any 3 points on this line form a collinear triple.
            # The line contains {i, j} + collinear. Total m = 2 + len(collinear)
            # Number of triples on this line = C(m, 3)
            # But we need unique triples. Let's just collect all triples on this line.
            all_points_on_line = [i, j] + collinear
            for tri in combinations(all_points_on_line, 3):
                t = frozenset(tri)
                if t not in seen:
                    seen.add(t)
                    triples.append(set(t))
    return triples

def _max_independent_set(points):
    """
    Find the maximum subset of points with no three collinear.
    Uses branch and bound.
    """
    n = len(points)
    triples = _build_triple_conflicts(points)
    
    # Convert triples to a conflict graph for the independent set problem
    # We want a subset S such that no triple is fully contained in S.
    # This is equivalent to: for every triple {a,b,c}, not (a in S and b in S and c in S).
    # This is a 3-uniform hypergraph independent set problem.
    
    # Branch and bound:
    # State: current set S, available points A, upper bound on size.
    # If S is valid (no triple fully in S) and we can't add more, it's a candidate.
    # Actually, we want to maximize |S| such that S is independent.
    
    # Let's use a recursive approach.
    # Order points to improve pruning.
    # Sort points by degree in the hypergraph? Or just use index order.
    
    # Precompute for each point, which triples it belongs to.
    point_triples = [[] for _ in range(n)]
    for idx, tri in enumerate(triples):
        for p in tri:
            point_triples[p].append(idx)
    
    best_size = 0
    best_set = set()
    
    def backtrack(current_set, available, idx):
        nonlocal best_size, best_set
        if len(current_set) + len(available) <= best_size:
            return
        
        # If no available points, update best if current is valid (it is, by construction)
        if not available:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
        
        # Try to add points from available
        # Heuristic: try points with fewer conflicts first? Or just iterate.
        # To ensure we find the MAX, we need to explore all.
        # But we can prune if even taking all remaining doesn't beat best.
        
        # Pick a point to branch on.
        # Let's pick the first available point.
        p = available[0]
        rest = available[1:]
        
        # Option 1: Include p
        # Check if including p creates a conflict.
        # A conflict is created if there exist two other points in current_set that form a triple with p.
        # Or if there are two points in current_set, say a, b, such that {a, b, p} is a triple.
        # Actually, we just need to ensure no triple is fully in current_set U {p}.
        # Since current_set is independent, the only new triples formed are those containing p and two points from current_set.
        
        valid_include = True
        for tri_idx in point_triples[p]:
            tri = triples[tri_idx]
            # Count how many of tri are in current_set
            count = 0
            for q in tri:
                if q == p:
                    continue
                if q in current_set:
                    count += 1
            if count == 2:
                valid_include = False
                break
        
        if valid_include:
            new_set = set(current_set)
            new_set.add(p)
            backtrack(new_set, rest, idx + 1)
        
        # Option 2: Exclude p
        backtrack(current_set, rest, idx + 1)
    
    backtrack(set(), list(range(n)), 0)
    return best_size

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Initial configuration: two parabolas
    # y = x^2 and y = x^2 + C, or similar.
    # Let's try a mix to avoid obvious collinearities.
    
    # Known good heuristic: points on a convex curve like a parabola have no 3 collinear.
    # But we want to MINIMIZE the max independent set.
    # So we want MANY collinear triples.
    # A grid has many, but also 4-collinear.
    # A "no 4 collinear" configuration with many 3-collinear triples.
    
    # Strategy: Start with a random configuration, then local search.
    # Move a point to a new position to reduce the max independent set size.
    
    random.seed(42)
    
    # Generate a valid starting configuration.
    # Try random points and check validity.
    points = []
    attempts = 0
    while len(points) < 20 and attempts < 1000:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        p = [x, y]
        if p not in points:
            points.append(p)
        attempts += 1
    
    # If we didn't get 20, fill in with a known valid set
    if len(points) < 20:
        points = [[x, x * x] for x in range(-10, 10)]
    
    # Ensure no 4 collinear
    while not _check_no_four_collinear(points):
        # Perturb one point
        idx = random.randint(0, 19)
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        points[idx] = [x, y]
        # Check for duplicates
        if len(set(map(tuple, points))) < 20:
            # Revert and try again
            points[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
    
    current_score = _max_independent_set(points)
    best_points = [p[:] for p in points]
    best_score = current_score
    
    # Local search
    while time.time() < deadline:
        improved = False
        for i in range(20):
            if time.time() > deadline:
                break
            # Try moving point i to a random new position
            original = points[i][:]
            for _ in range(10):  # 10 random moves per point
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                points[i] = [x, y]
                
                # Check validity
                if len(set(map(tuple, points))) < 20:
                    continue
                if not _check_no_four_collinear(points):
                    continue
                
                # Calculate score
                score = _max_independent_set(points)
                if score < current_score:
                    current_score = score
                    if current_score < best_score:
                        best_score = current_score
                        best_points = [p[:] for p in points]
                    improved = True
                    break
                else:
                    # Revert
                    points[i] = original
            
            if not improved:
                # Maybe try a few more random restarts or perturbations
                pass
        
        # If no improvement in a full pass, do a random restart or larger perturbation
        if not improved:
            # Perturb 2-3 random points
            for _ in range(3):
                idx = random.randint(0, 19)
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                points[idx] = [x, y]
            
            if len(set(map(tuple, points))) < 20 or not _check_no_four_collinear(points):
                points = [p[:] for p in best_points]
                current_score = best_score
            else:
                current_score = _max_independent_set(points)
                if current_score < best_score:
                    best_score = current_score
                    best_points = [p[:] for p in points]
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    print("Score:", _max_independent_set(pts))
