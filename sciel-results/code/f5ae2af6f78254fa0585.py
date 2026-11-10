"""Improved program: systematic search for larger rational-distance point sets."""
import time
import math
import random
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_square(n):
    """Check if n (int) is a perfect square."""
    if not isinstance(n, int):
        n = int(n)
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n


def _is_squarefree(n):
    """Check if n is squarefree."""
    if not isinstance(n, int):
        n = int(n)
    if n <= 1:
        return True
    for p in range(2, math.isqrt(n) + 1):
        if n % (p * p) == 0:
            return False
    return True


def _sq_dist_sq(p1, p2, D):
    """Return squared distance as a Fraction between (x1,y1) and (x2,y2) in the (x, y*sqrt(D)) space."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + D * dy * dy


def _is_rational_square(fr):
    """Check if Fraction fr is a perfect square of a rational number."""
    if fr < 0:
        return False
    num = fr.numerator
    den = fr.denominator
    return _is_square(num) and _is_square(den)


def _check_all_distances_rational(points, D):
    """Check that all pairwise squared distances are perfect squares of rationals."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            d2 = _sq_dist_sq(points[i], points[j], D)
            if not _is_rational_square(d2):
                return False
    return True


def _check_no_three_collinear(points):
    """Check no three points are collinear using exact arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                det = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if det == 0:
                    return False
    return True


def _check_no_four_concyclic(points, D):
    """Check no four points are concyclic using exact arithmetic."""
    n = len(points)
    for quad in combinations(range(n), 4):
        i, j, k, l = quad
        x1, y1 = points[i]
        x2, y2 = points[j]
        x3, y3 = points[k]
        x4, y4 = points[l]
        
        # Translate so p1 is at origin
        x2p = x2 - x1
        y2p = y2 - y1
        x3p = x3 - x1
        y3p = y3 - y1
        x4p = x4 - x1
        y4p = y4 - y1
        
        # Determinant of:
        # [x2p, y2p, -(x2p^2 + D*y2p^2)]
        # [x3p, y3p, -(x3p^2 + D*y3p^2)]
        # [x4p, y4p, -(x4p^2 + D*y4p^2)]
        q2 = x2p * x2p + D * y2p * y2p
        q3 = x3p * x3p + D * y3p * y3p
        q4 = x4p * x4p + D * y4p * y4p
        
        det = (x2p * (y3p * (-q4) - y4p * (-q3))
               - y2p * (x3p * (-q4) - x4p * (-q3))
               + (-q2) * (x3p * y4p - x4p * y3p))
        
        if det == 0:
            return False
    return True


def _validate(points_frac, D):
    """Full validation of a point set."""
    if len(points_frac) < 3:
        return False
    if not _is_squarefree(D):
        return False
    # Check distinctness
    seen = set()
    for p in points_frac:
        key = (p[0], p[1])
        if key in seen:
            return False
        seen.add(key)
    if not _check_all_distances_rational(points_frac, D):
        return False
    if not _check_no_three_collinear(points_frac):
        return False
    if not _check_no_four_concyclic(points_frac, D):
        return False
    return True


def _build_known_5points():
    """Return the known 5-point configuration with D=1."""
    D = 1
    pts = [(Fraction(0), Fraction(0)),
           (Fraction(-56), Fraction(33)),
           (Fraction(-16), Fraction(63)),
           (Fraction(0), Fraction(33)),
           (Fraction(16), Fraction(63))]
    return pts, D


def _find_max_clique(graph, deadline):
    """
    Find the maximum clique in a graph using a greedy heuristic.
    graph: dict mapping node index to set of neighbor indices.
    Returns list of indices in the clique.
    """
    nodes = list(graph.keys())
    if not nodes:
        return []
    
    # Sort nodes by degree descending
    nodes.sort(key=lambda x: len(graph[x]), reverse=True)
    
    best_clique = []
    
    # Greedy: try to build a clique starting from each node
    for start in nodes:
        if time.time() > deadline:
            break
        current_clique = [start]
        candidates = set(graph[start])
        
        # Try to add nodes from candidates that are connected to all in current_clique
        while candidates:
            if time.time() > deadline:
                break
            # Pick candidate with max degree in the induced subgraph
            # For simplicity, just pick the first one that fits, or one with high degree
            # A better heuristic: pick the node in candidates with highest degree among candidates
            best_cand = None
            best_deg = -1
            for c in candidates:
                # Check if c is connected to all in current_clique
                # Since c is in candidates (intersection of neighbors of all in clique so far),
                # it is connected to all in current_clique by construction if we maintain candidates correctly.
                # Wait, the standard greedy is:
                # 1. Start with node u.
                # 2. Candidates = neighbors(u).
                # 3. Pick v in candidates with max degree.
                # 4. Add v to clique.
                # 5. Candidates = candidates intersect neighbors(v).
                
                # We need to maintain candidates as the set of nodes connected to ALL in current_clique.
                # The loop above is slightly wrong if not careful. Let's re-implement properly.
                pass
            
            # Re-implementing the step properly
            if not candidates:
                break
            
            # Pick node with highest degree in the whole graph (or induced subgraph)
            # For speed, just pick one with high global degree
            # A simple heuristic: pick the one with the most neighbors in `candidates`
            best_cand = None
            max_common = -1
            for c in candidates:
                # Count how many neighbors c has in candidates
                common = len(candidates.intersection(graph[c]))
                if common > max_common:
                    max_common = common
                    best_cand = c
            
            if best_cand is None:
                break
                
            current_clique.append(best_cand)
            # Update candidates: must be neighbors of best_cand AND already in candidates
            candidates = candidates.intersection(graph[best_cand])
            
        if len(current_clique) > len(best_clique):
            best_clique = current_clique
            
    return best_clique

def _search_clique(D, deadline):
    """
    Search for a large clique of integer points with rational distances.
    """
    # Define search range
    # We want to find points with integer coordinates (x, y) such that (x1-x2)^2 + D(y1-y2)^2 is a square.
    # For D=1, this is Pythagorean triples.
    # Let's search in a box [-R, R] x [-R, R]
    
    best_set = []
    
    # Try increasing radii
    for R in [20, 30, 50, 80]:
        if time.time() > deadline:
            break
            
        # Generate candidates
        candidates = []
        for x in range(-R, R+1):
            for y in range(-R, R+1):
                candidates.append((Fraction(x), Fraction(y)))
                
        n = len(candidates)
        if n == 0:
            continue
            
        # Build graph
        # graph[i] = set of j such that dist(i,j) is rational
        graph = {i: set() for i in range(n)}
        
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i+1, n):
                d2 = _sq_dist_sq(candidates[i], candidates[j], D)
                if _is_rational_square(d2):
                    graph[i].add(j)
                    graph[j].add(i)
                    
        # Find max clique
        clique_indices = _find_max_clique(graph, deadline)
        clique_points = [candidates[i] for i in clique_indices]
        
        # Check validity (collinearity, concyclicity)
        # Note: The clique ensures rational distances.
        # We need to filter out invalid configurations.
        # Since we want a valid set, we might need to prune the clique.
        # Greedy clique might contain collinear points.
        
        # Let's try to extract a valid subset from the clique.
        # Start with empty set, add points one by one if valid.
        valid_set = []
        for p in clique_points:
            if p in valid_set:
                continue
            new_set = valid_set + [p]
            if _validate(new_set, D):
                valid_set = new_set
        
        if len(valid_set) > len(best_set):
            best_set = valid_set
            
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Start with known 5-point set
    base_points, D = _build_known_5points()
    
    if not _validate(base_points, D):
        raise RuntimeError("Base configuration failed validation")
    
    best = list(base_points)
    
    # Strategy 1: Clique search
    # This might find a larger set, but it's expensive.
    # We limit the search time.
    clique_start = time.time()
    clique_deadline = clique_start + 30 # 30 seconds for clique search
    clique_result = _search_clique(D, clique_deadline)
    if len(clique_result) > len(best):
        best = clique_result
        
    # Strategy 2: Greedy expansion from best found so far
    # If we have a set of size K, try to add more points from a larger grid.
    # We can use the grid from the clique search but focused on the existing points.
    
    # Let's try a direct expansion on the best set found.
    # Generate candidates in a larger box.
    # If best is small, search a larger box.
    
    # For now, if we found something better, return it.
    # If not, we might try a few random restarts or larger grids.
    
    # If we have time, try a larger grid for clique search
    if time.time() < deadline - 5 and len(best) < 6:
        # Try a slightly larger search or different D?
        # The task says D <= 10^12 squarefree.
        # D=1 is the most common for integer distances.
        # Let's stick to D=1 for now as it's the easiest to generate.
        
        # Try one more clique search with a larger radius but shorter time
        clique_deadline2 = time.time() + (deadline - time.time() - 2)
        clique_result2 = _search_clique(D, clique_deadline2)
        if len(clique_result2) > len(best):
            best = clique_result2

    # Convert to required format
    result_points = []
    for x, y in best:
        result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': D, 'points': result_points}

# EVOLVE-BLOCK-END
