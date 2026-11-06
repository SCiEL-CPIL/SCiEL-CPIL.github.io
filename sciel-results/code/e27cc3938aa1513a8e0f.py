"""
Structural change: model as Maximum Weight Independent Set on a conflict graph.
Build a graph where vertices are 1..256 and edges connect pairs that cannot coexist
in a 3-AP-free set. Then solve MWIS with a branch-and-bound using:
  - Greedy coloring for upper bound (Tomita-style maximum clique on complement)
  - Pivot-based branching
  - Weight-aware ordering
This exploits the graph structure directly rather than local search on a hypergraph.
"""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import sys

N = 256
TIME_LIMIT = 85.0  # Safety margin before 90s kill

def build_conflict_graph():
    """
    Build the conflict graph: two numbers x, y are adjacent if they cannot both be
    in a 3-AP-free set. That is, there exists z in [1, N] such that {x, y, z} is a
    3-AP.
    
    A 3-AP is {a, a+d, a+2d} for d >= 1. So x, y conflict if:
    - y = x + d and x + 2d <= N (x is first, y is second)
    - y = x - d and x - 2d >= 1 (x is second, y is first)  
    - y = x + 2d (x is first, y is third)
    - y = x - 2d (x is third, y is first)
    - y = x + d and x - d >= 1 (x is middle, y is second) -> wait, this is x-d, x, x+d
    - y = x - d and x + d <= N (x is middle, y is first)
    
    More simply: x and y conflict if there exists z such that {x, y, z} forms a 3-AP.
    For distinct x, y, z, they form a 3-AP iff one of them is the average of the other two.
    So x, y conflict if:
    - (x + y) / 2 is an integer in [1, N] and (x+y)/2 != x and (x+y)/2 != y
      (i.e., x + y is even and |x - y| > 0, and the midpoint is valid)
    - x is the midpoint: y = 2x - z for some z, i.e., x is midpoint of y and some z.
      This means 2x - y is in [1, N] and 2x - y != x and 2x - y != y.
    - y is the midpoint: 2y - x is in [1, N] and 2y - x != x and 2y - x != y.
    
    Actually, for any pair {x, y}, they conflict if the set {x, y, z} for some z is a 3-AP.
    The three cases:
    1. z = (x + y) / 2 (x and y are endpoints, z is midpoint). Valid if x+y even, z != x, z != y, 1 <= z <= N.
    2. z = 2x - y (y and z are endpoints, x is midpoint). Valid if 1 <= 2x-y <= N, 2x-y != x, 2x-y != y.
       Note: 2x-y != x means x != y (true). 2x-y != y means x != y (true). So just check range.
    3. z = 2y - x (x and z are endpoints, y is midpoint). Valid if 1 <= 2y-x <= N.
    
    So x, y conflict iff any of:
    a) (x + y) % 2 == 0 and 1 <= (x+y)//2 <= N and (x+y)//2 != x and (x+y)//2 != y
    b) 1 <= 2*x - y <= N
    c) 1 <= 2*y - x <= N
    
    But wait: case (b) says z = 2x - y. If z = x, then 2x-y = x => x = y, excluded.
    If z = y, then 2x-y = y => x = y, excluded.
    So for distinct x, y, case (b) is valid iff 1 <= 2x-y <= N.
    Similarly case (c) is valid iff 1 <= 2y-x <= N.
    
    But case (a): if (x+y)//2 = x, then y = x, excluded. If (x+y)//2 = y, then x = y, excluded.
    So case (a) is valid iff (x+y) % 2 == 0 and 1 <= (x+y)//2 <= N.
    
    Note: case (a) is actually a subset of cases (b) and (c)? Let's check.
    If z = (x+y)/2, then x, z, y is an AP (or y, z, x). Here z is the midpoint.
    Case (b) with x as midpoint: z' = 2x - y. This is different from z = (x+y)/2 unless x = y.
    So they are distinct.
    
    Let me re-examine. For {x, y, z} to be a 3-AP with distinct elements:
    - x is min: x, x+d, x+2d. So y = x+d or y = x+2d.
    - x is max: x-2d, x-d, x. So y = x-d or y = x-2d.
    - x is mid: x-d, x, x+d. So y = x-d or y = x+d.
    
    So for a fixed pair (x, y) with x < y:
    - y = x + d: then we need x + 2d = x + 2(y-x) = 2y - x <= N. So conflict if 2y - x <= N.
    - y = x + 2d: then d = (y-x)/2, must be integer. So y - x must be even. And x + d = x + (y-x)/2 = (x+y)/2 must be in [1, N] (always true if x, y in [1,N] and x<y). So conflict if (y-x) % 2 == 0.
    - x = y - d: symmetric to y = x + d.
    - x = y - 2d: symmetric to y = x + 2d.
    - x is mid: y = x + d, so we need x - d = 2x - y >= 1. So conflict if 2x - y >= 1.
    
    So for x < y, conflict iff:
    a) 2y - x <= N  (y = x + d, third term exists)
    b) (y - x) % 2 == 0  (y = x + 2d, midpoint exists)
    c) 2x - y >= 1  (x is midpoint, first term exists)
    
    Let's verify: 
    - (a) covers y = x+d where the AP is x, y, 2y-x.
    - (b) covers y = x+2d where the AP is x, (x+y)/2, y.
    - (c) covers y = x+d where the AP is 2x-y, x, y.
    
    Yes, these are the three ways a pair can be part of a 3-AP.
    """
    # Build adjacency as bitmasks for fast operations
    adj = [0] * (N + 1)  # adj[x] is bitmask of neighbors of x
    for x in range(1, N + 1):
        for y in range(x + 1, N + 1):
            # Check if x and y conflict
            conflict = False
            # Case a: 2y - x <= N
            if 2 * y - x <= N:
                conflict = True
            # Case b: (y - x) % 2 == 0
            if (y - x) % 2 == 0:
                conflict = True
            # Case c: 2x - y >= 1
            if 2 * x - y >= 1:
                conflict = True
            if conflict:
                adj[x] |= (1 << (y - 1))
                adj[y] |= (1 << (x - 1))
    return adj

# Precompute
ADJ = build_conflict_graph()

def greedy_color_bound(candidates, weights):
    """
    Greedy coloring of the candidate subgraph to get an upper bound on the
    maximum weight independent set. Returns the color classes as a list of
    lists of vertices, ordered by color.
    """
    # We'll use a simple greedy coloring: process vertices in order, assign
    # the smallest available color.
    n = len(candidates)
    if n == 0:
        return []
    
    color_of = [0] * n  # 0 means uncolored
    colors = []  # colors[i] is list of vertex indices with color i+1
    
    for i in range(n):
        v = candidates[i]
        # Find smallest color not used by any neighbor of v that's already colored
        used = set()
        for j in range(i):
            if color_of[j] > 0 and ((ADJ[v] >> (candidates[j] - 1)) & 1):
                used.add(color_of[j])
        # Assign smallest available color
        c = 1
        while c in used:
            c += 1
        color_of[i] = c
        while len(colors) < c:
            colors.append([])
        colors[c - 1].append(v)
    
    return colors

def mwis_branch_and_bound(initial_candidates, weights, time_limit):
    """
    Branch and bound for Maximum Weight Independent Set.
    initial_candidates: list of vertex numbers (1-indexed)
    weights: dict or array where weights[x] = 1/x
    Returns the best independent set found.
    """
    start_time = time.time()
    best_set = []
    best_weight = 0.0
    
    # Sort candidates by weight descending (smaller x first)
    candidates = sorted(initial_candidates, key=lambda x: -weights[x])
    
    # Precompute weights as a list for faster access
    w = [0.0] * (N + 1)
    for i in range(1, N + 1):
        w[i] = 1.0 / i
    
    # We'll use a recursive approach with explicit stack to avoid recursion limit
    # State: (candidates_list, current_set, current_weight)
    # But for efficiency, we use the standard Tomita-style approach.
    
    def _mwis(cands, cur_set, cur_weight):
        nonlocal best_set, best_weight
        nonlocal start_time
        
        if time.time() - start_time > time_limit:
            return
        
        if not cands:
            if cur_weight > best_weight:
                best_weight = cur_weight
                best_set = list(cur_set)
            return
        
        # Upper bound: sum of weights of all remaining candidates
        # This is a loose bound; let's use greedy coloring for a tighter bound
        # For speed, we'll use the sum bound first, and only do coloring if needed
        upper = cur_weight + sum(w[v] for v in cands)
        if upper <= best_weight:
            return
        
        # Greedy coloring bound
        color_classes = greedy_color_bound(cands, w)
        if not color_classes:
            if cur_weight > best_weight:
                best_weight = cur_weight
                best_set = list(cur_set)
            return
        
        # The upper bound is cur_weight + sum of max weight in each color class
        color_upper = cur_weight
        for cc in color_classes:
            if cc:
                color_upper += max(w[v] for v in cc)
        if color_upper <= best_weight:
            return
        
        # Branch: pick a vertex to branch on
        # Use pivot-based branching: pick a vertex v, then branch on
        # include v (and remove its neighbors) vs exclude v
        # For simplicity, we branch on the first vertex in the sorted list
        
        # Actually, let's use a better strategy: pick the vertex with highest weight
        # among those that appear in the first color class (or just the first vertex)
        
        v = cands[0]
        
        # Branch 1: Include v
        # Remove v and all its neighbors from candidates
        new_cands = [u for u in cands[1:] if not ((ADJ[v] >> (u - 1)) & 1)]
        cur_set.append(v)
        _mwis(new_cands, cur_set, cur_weight + w[v])
        cur_set.pop()
        
        if time.time() - start_time > time_limit:
            return
        
        # Branch 2: Exclude v
        _mwis(cands[1:], cur_set, cur_weight)
    
    _mwis(candidates, [], 0.0)
    
    return best_set

def greedy_mwis(candidates, weights):
    """
    Greedy heuristic for MWIS: repeatedly pick the highest-weight vertex
    that doesn't conflict with already-picked vertices.
    """
    selected = set()
    remaining = list(candidates)
    remaining.sort(key=lambda x: -weights[x])
    
    for v in remaining:
        if all(not ((ADJ[v] >> (u - 1)) & 1) for u in selected):
            selected.add(v)
    
    return list(selected)

def local_improvement(initial_set, weights, time_limit):
    """
    Local search: try to add vertices (in order of weight) that don't conflict,
    then try swaps.
    """
    start_time = time.time()
    current = set(initial_set)
    current_weight = sum(weights[v] for v in current)
    best = list(current)
    best_weight = current_weight
    
    # Add-only phase
    all_vertices = list(range(1, N + 1))
    all_vertices.sort(key=lambda x: -weights[x])
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        for v in all_vertices:
            if v in current:
                continue
            if all(not ((ADJ[v] >> (u - 1)) & 1) for u in current):
                current.add(v)
                current_weight += weights[v]
                improved = True
                if current_weight > best_weight:
                    best_weight = current_weight
                    best = list(current)
    
    # Swap phase: try removing one vertex and adding one or two others
    for _ in range(50):  # Limit iterations
        if time.time() - start_time > time_limit:
            break
        made_swap = False
        # Try removing each vertex in current (sorted by weight ascending, i.e., remove low-weight first)
        verts = sorted(current, key=lambda x: weights[x])
        for v_out in verts:
            if time.time() - start_time > time_limit:
                break
            temp = current - {v_out}
            temp_weight = current_weight - weights[v_out]
            # Try to add any vertex not in temp
            for v_in in all_vertices:
                if v_in in temp:
                    continue
                if all(not ((ADJ[v_in] >> (u - 1)) & 1) for u in temp):
                    new_weight = temp_weight + weights[v_in]
                    if new_weight > current_weight:
                        current = temp | {v_in}
                        current_weight = new_weight
                        made_swap = True
                        if current_weight > best_weight:
                            best_weight = current_weight
                            best = list(current)
                        break
            if made_swap:
                break
    
    return best

def construct():
    start_time = time.time()
    
    weights = {i: 1.0 / i for i in range(1, N + 1)}
    
    # Phase 1: Greedy construction
    greedy_set = greedy_mwis(range(1, N + 1), weights)
    
    # Phase 2: Local improvement
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 1.0:
        improved_set = local_improvement(greedy_set, weights, remaining - 1.0)
        if sum(weights[v] for v in improved_set) > sum(weights[v] for v in greedy_set):
            greedy_set = improved_set
    
    # Phase 3: Branch and bound on a reduced problem
    # To make B&B tractable, we focus on improving the solution by considering
    # a subset of vertices: those not in the current solution plus some in the solution
    # that might be swapped out.
    remaining = TIME_LIMIT - (time.time() - start_time)
    if remaining > 5.0:
        # We'll try B&B on a small set of promising vertices
        # Take the top 60 vertices by weight that are not in the current set,
        # plus the current set itself
        in_set = set(greedy_set)
        outside = [v for v in range(1, N + 1) if v not in in_set]
        outside.sort(key=lambda x: -weights[x])
        # Take top 40 outside vertices
        focus_outside = outside[:40]
        # Also include all vertices in the current set (they might be swapped)
        focus = list(in_set) + focus_outside
        # Remove duplicates
        focus = list(set(focus))
        
        bnb_set = mwis_branch_and_bound(focus, weights, remaining - 2.0)
        
        # Combine: the B&B set is an independent set on the focus subgraph.
        # But it may not be valid for the full graph if it excludes some in_set vertices.
        # Actually, since we included all in_set vertices in focus, the B&B solution
        # is a valid independent set for the full graph (any conflict would be within focus).
        # Wait, no: if B&B excludes a vertex from in_set, that's fine. If it includes
        # an outside vertex, we need to make sure it doesn't conflict with any vertex
        # NOT in focus. But since focus includes all of in_set and the top outside,
        # and the full graph has more vertices, a vertex in focus_outside might conflict
        # with a vertex not in focus. So we need to verify.
        
        # Verify the B&B solution is valid for the full graph
        bnb_set = set(bnb_set)
        valid = True
        for v in bnb_set:
            for u in bnb_set:
                if v < u and ((ADJ[v] >> (u - 1)) & 1):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            bnb_weight = sum(weights[v] for v in bnb_set)
            current_weight = sum(weights[v] for v in greedy_set)
            if bnb_weight > current_weight:
                greedy_set = list(bnb_set)
    
    # Final: return sorted list
    return sorted(greedy_set)

# EVOLVE-BLOCK-END
