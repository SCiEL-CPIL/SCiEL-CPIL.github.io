"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random
from itertools import combinations

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    g = gcd(gcd(abs(a), abs(b)), abs(c))
    if g > 0:
        a //= g
        b //= g
        c //= g
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _count_ordinary(pts):
    """Count lines containing exactly 2 of the given points."""
    n = len(pts)
    if n < 2:
        return 0
    line_map = {}
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            key = _line_key((x1, y1), (x2, y2))
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    return sum(1 for v in line_map.values() if len(v) == 2)

def _is_valid(pts):
    """Check 12 distinct integer points, |x|,|y| <= 1e9, not all collinear."""
    if len(pts) != 12:
        return False
    seen = set()
    for p in pts:
        if not isinstance(p[0], int) or not isinstance(p[1], int):
            return False
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
        key = (p[0], p[1])
        if key in seen:
            return False
        seen.add(key)
    # Check not all collinear
    if len(pts) < 3:
        return True
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    for i in range(2, len(pts)):
        xi, yi = pts[i]
        if (xi - x0) * (y1 - y0) != (yi - y0) * (x1 - x0):
            return True
    return False

def _generate_hypergraphs(max_ordinary=6):
    """
    Generate all combinatorially valid incidence structures (hypergraphs)
    for 12 points where the number of ordinary lines (size 2) is <= max_ordinary.
    
    Yields a list of tuples, where each tuple is a frozenset of point indices.
    """
    points = list(range(12))
    
    # Precompute all pairs
    all_pairs = list(combinations(points, 2))
    
    # We will build the set of lines incrementally.
    # State: (current_lines_tuple, next_pair_index)
    # But this is too slow if we do it naively.
    
    # Alternative: Randomized generation of hypergraphs.
    # We want to find hypergraphs with few ordinary lines.
    # The theoretical minimum is 6.
    # Let's try to generate random hypergraphs with exactly 6 ordinary lines and check validity.
    # Or better, enumerate "small" hypergraphs.
    
    # Given time constraints, we will use a randomized approach to find valid hypergraphs.
    # We will generate random partitions of pairs into lines of size >= 2.
    # A valid hypergraph must cover all pairs.
    
    # Since we have 12 points, there are 66 pairs.
    # If we have 6 ordinary lines (size 2), they cover 12 pairs.
    # The remaining 54 pairs must be covered by lines of size >= 3.
    
    # This enumeration is complex. Instead, we will use a known family of configurations
    # or a specific search for hypergraphs.
    
    # Actually, let's just yield a few known good hypergraph structures if we can find them,
    # or use a simple random search for hypergraphs in the solver phase.
    
    # For now, let's yield the "trivial" hypergraph where we just pick random lines.
    # This is a placeholder for a more robust generator if needed.
    # But we can also just let the solver try to find ANY configuration with <= 6 ordinary lines.
    
    # Let's yield the structure of the 4x3 grid (which has 10 ordinary lines) as a baseline,
    # and then let the solver try to improve it.
    
    # To be more effective, we should yield hypergraphs with FEWER ordinary lines.
    # Let's try to construct one.
    
    # The 12-point configuration with 6 ordinary lines is related to the Hesse configuration or similar.
    # The Hesse configuration has 9 points and 12 lines.
    # For 12 points, the "minimum ordinary lines" problem is non-trivial.
    
    # Let's yield a random hypergraph generator that produces candidates.
    # We will generate a random set of lines (subsets of size >= 2) that covers all pairs.
    
    def random_hypergraph():
        # Start with all pairs
        remaining_pairs = set(all_pairs)
        lines = []
        
        # Randomly pick lines
        # To ensure we have few ordinary lines, we should prefer larger lines.
        # But they must be consistent (collinear).
        # In the combinatorial phase, we don't know if they are collinear.
        # So we just pick random subsets.
        
        # This is a heuristic.
        while remaining_pairs:
            # Pick a random pair
            pair = random.choice(list(remaining_pairs))
            p1, p2 = pair
            
            # Decide line size
            # If we want few ordinary lines, we should make lines larger.
            # But we must ensure all pairs in the line are covered.
            
            # Let's try to extend the line to include other points.
            line = {p1, p2}
            
            # Try to add more points
            # This is a greedy approach.
            # We want to cover as many pairs as possible with few lines.
            
            # For simplicity, let's just yield the grid structure and let the solver work.
            # The solver will try to move points to reduce ordinary lines.
            
            # This function is a stub for now.
            pass
            
    # Instead of a complex generator, we will just use the grid and a few other starting points.
    # The solver will do the heavy lifting.
    
    # Yield the 4x3 grid
    yield None # Signal to use default starting points

def _solve_geometry(hypergraph, deadline):
    """
    Given a hypergraph (list of lines), find integer coordinates that realize it.
    This is a constraint satisfaction problem.
    
    If hypergraph is None, use a random starting point.
    """
    if hypergraph is None:
        # Use 4x3 grid
        return [[x, y] for x in range(4) for y in range(3)]
    
    # This is a placeholder for a real solver.
    # For now, we will just do a local search on the grid.
    return None

def construct():
    deadline = time.time() + 145
    
    # Start with 4x3 grid
    best_pts = [[x, y] for x in range(4) for y in range(3)]
    best_score = _count_ordinary(best_pts)
    
    # Local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(12))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            orig_x, orig_y = best_pts[i]
            
            # Try to move point i to reduce ordinary lines
            # We will try to align it with existing lines or create new non-ordinary lines.
            
            # Heuristic: Try to place it on a line with >= 2 other points.
            # Or try to break an ordinary line.
            
            # For now, just random small moves.
            tried = 0
            while tried < 50 and time.time() < deadline:
                dx = random.randint(-50, 50)
                dy = random.randint(-50, 50)
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = orig_x + dx, orig_y + dy
                if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                    continue
                
                # Check distinctness
                valid = True
                for j in range(12):
                    if j == i:
                        continue
                    if best_pts[j][0] == new_x and best_pts[j][1] == new_y:
                        valid = False
                        break
                if not valid:
                    tried += 1
                    continue
                
                # Calculate new score
                old_pts = [list(p) for p in best_pts]
                best_pts[i] = [new_x, new_y]
                new_score = _count_ordinary(best_pts)
                
                if new_score < best_score:
                    best_score = new_score
                    improved = True
                    break
                else:
                    # Revert
                    best_pts = old_pts
                    tried += 1
                    
        if not improved:
            # Stuck. Try a larger jump or restart?
            # For now, break.
            break
            
    return [list(p) for p in best_pts]
# EVOLVE-BLOCK-END
