"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _are_collinear(p1, p2, p3):
    return (p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) == 0

def _build_collinear_graph(points):
    """Build adjacency: for each point, set of other points that form a collinear triple with some third."""
    n = len(points)
    # collinear_pairs[i] = set of j such that there exists k != i,j with i,j,k collinear
    # Actually we need: for the 3-free subset problem, we need to know which pairs are "conflicting"
    # A pair (i,j) is conflicting if there exists k such that i,j,k are collinear.
    # Then a 3-free subset cannot contain any conflicting pair along with its witness.
    # Actually simpler: build the set of all collinear triples, then do B&B on the hypergraph.
    
    # For each point i, store the set of (j,k) pairs where i,j,k are collinear
    # Better: for each point i, store the set of points j such that i and j are on a line with some third point
    # Even better for B&B: for each point i, store which other points conflict with it given current subset
    
    # Let's just precompute all collinear triples
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    triples.append((i, j, k))
    return triples

def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    for i, j, k, l in combinations(range(n), 4):
        if _are_collinear(points[i], points[j], points[k]) and \
           _are_collinear(points[i], points[j], points[l]) and \
           _are_collinear(points[i], points[k], points[l]):
            return True
    return False

def _max_no_three_collinear(points):
    """
    Find the largest subset with no three collinear using branch and bound
    on the 3-uniform hypergraph of collinear triples.
    
    We use a recursive B&B that picks points one at a time.
    Key insight: when we add point p to our subset, any other point q that
    forms a collinear triple (p, q, r) with some r NOT in our subset is fine,
    but if r IS in our subset, we can't add q.
    
    Actually, the constraint is: no three points in the subset are collinear.
    So for a candidate subset S, we need: for every triple (i,j,k) in the hypergraph,
    not all of i,j,k are in S.
    
    B&B strategy:
    - State: current subset S (as a bitmask or list), remaining candidate points R
    - Upper bound: |S| + |R|
    - When considering adding point p to S: check that for every q in S, 
      there's no r in S such that (p,q,r) is a collinear triple.
      More precisely: for every pair (q, r) in S, check that (p, q, r) is not collinear.
    
    To make this efficient, precompute for each point p, the set of pairs (q,r) 
    that are collinear with p. Then when adding p, check if any such pair is fully in S.
    """
    n = len(points)
    if n == 0:
        return 0
    
    # Precompute: for each point p, the set of frozensets {q,r} such that p,q,r are collinear
    # This allows fast checking: when adding p to S, check if any {q,r} subset of S exists in collinear_pairs[p]
    collinear_pairs = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    collinear_pairs[i].add(frozenset((j, k)))
                    collinear_pairs[j].add(frozenset((i, k)))
                    collinear_pairs[k].add(frozenset((i, j)))
    
    best = [0]
    
    def can_add(p, S):
        """Check if point p can be added to subset S without creating a collinear triple."""
        # S is a frozenset of indices
        for pair in collinear_pairs[p]:
            if pair.issubset(S):
                return False
        return True
    
    def bnb(S, R):
        """
        S: frozenset of points in current subset
        R: frozenset of remaining candidate points
        """
        # Upper bound
        if len(S) + len(R) <= best[0]:
            return
        
        if not R:
            if len(S) > best[0]:
                best[0] = len(S)
            return
        
        # Pick a point from R to branch on
        # Heuristic: pick the point that conflicts most with S (most restrictive)
        # or just pick the first one
        p = min(R)  # deterministic choice
        
        # Option 1: Add p to S
        if can_add(p, S):
            new_S = S | frozenset([p])
            new_R = R - frozenset([p])
            bnb(new_S, new_R)
        
        # Option 2: Don't add p
        new_R = R - frozenset([p])
        bnb(S, new_R)
    
    bnb(frozenset(), frozenset(range(n)))
    return best[0]

def _find_valid_initial():
    """Find a valid initial 20-point set with no four collinear."""
    while True:
        points = []
        for _ in range(20):
            while True:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                p = [x, y]
                if p not in points:
                    break
            points.append(p)
        if not _has_four_collinear(points):
            return points

def _count_collinear_triples(points):
    """Count the number of collinear triples."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    count += 1
    return count

def _local_search(points, deadline):
    """
    Local search: try to reduce the size of the maximum 3-free subset.
    Strategy: for each point, try moving it to a position that increases
    the number of collinear triples (which tends to reduce the 3-free subset size).
    """
    best_points = points[:]
    best_score = _max_no_three_collinear(points)
    current_points = points[:]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(20):
            if time.time() > deadline:
                break
            
            # Try moving point idx to various positions
            best_new = None
            best_new_score = best_score
            
            for attempt in range(30):
                if time.time() > deadline:
                    break
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                new_p = [x, y]
                if new_p in current_points:
                    continue
                
                new_points = current_points[:idx] + [new_p] + current_points[idx+1:]
                if _has_four_collinear(new_points):
                    continue
                
                # Quick heuristic: count collinear triples
                new_triples = _count_collinear_triples(new_points)
                old_triples = _count_collinear_triples(current_points)
                
                if new_triples <= old_triples:
                    continue
                
                # Full evaluation
                new_score = _max_no_three_collinear(new_points)
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new = new_points[:]
            
            if best_new is not None:
                current_points = best_new
                if best_new_score < best_score:
                    best_score = best_new_score
                    best_points = current_points[:]
                improved = True
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s safety margin for the evaluator's 150s limit
    
    # Start with a valid initial configuration
    points = _find_valid_initial()
    best_points, best_score = _local_search(points, deadline)
    
    # Try a few more random restarts if time permits
    while time.time() < deadline - 20:
        points = _find_valid_initial()
        candidate_points, candidate_score = _local_search(points, min(deadline, time.time() + 30))
        
        if candidate_score < best_score:
            best_score = candidate_score
            best_points = candidate_points[:]
    
    return best_points
# EVOLVE-BLOCK-END
