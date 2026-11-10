"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd
from functools import reduce
from itertools import combinations
from fractions import Fraction

# EVOLVE-BLOCK-START
def _lcm(a, b):
    return a * b // gcd(a, b)

def _check_reciprocal_subsum(target_x, others):
    """Check if 1/target_x equals sum of 1/y for some subset of others."""
    if not others:
        return False
    all_vals = [target_x] + list(others)
    lcm_val = reduce(_lcm, all_vals)
    target_int = lcm_val // target_x
    candidate_vals = [lcm_val // y for y in others]
    n = len(candidate_vals)
    if n == 0:
        return False
    half = n // 2
    left = candidate_vals[:half]
    right = candidate_vals[half:]
    left_sums = {0}
    for v in left:
        left_sums = left_sums | {s + v for s in left_sums}
    right_sums = {0}
    for v in right:
        right_sums = right_sums | {s + v for s in right_sums}
    for s in left_sums:
        if target_int - s in right_sums:
            return True
    return False

def _is_valid_set(selected):
    """Check if selected set is reciprocal-subsum-free."""
    selected = list(selected)
    for i, x in enumerate(selected):
        others = selected[:i] + selected[i+1:]
        if _check_reciprocal_subsum(x, others):
            return False
    return True

def _precompute_pairwise_conflicts():
    """
    Precompute pairwise conflicts: edge (i,j) exists if selecting both i and j
    creates a violation with some already-selected element.
    
    For the MIS formulation: we build a conflict graph where an edge between i and j
    means they cannot both be in a valid set. This is conservative (only catches
    2-element conflicts), but we'll verify final sets with full check.
    
    Actually, the proper approach: precompute for each element x, which subsets of
    other elements sum to 1/x. Then build the conflict graph considering all such
    subset relationships.
    
    For efficiency, we'll use a different approach: build the conflict graph where
    edge (i,j) exists if there's a subset S containing both i and j such that for
    some x in S, 1/x = sum of 1/s for s in S\{x}. We approximate by checking
    small subsets (size 2, 3, 4).
    """
    n = 24
    # adj[i] = bitmask of j such that i and j conflict
    adj = [0] * (n + 1)  # 1-indexed
    
    # For each potential target x, find all subsets of [1..24]\{x} that sum to 1/x
    # Then all pairs within such a subset conflict with x
    # This is expensive, so we focus on small subsets
    
    # Method: for each x, enumerate subsets of size 2,3,4 from [1..24]\{x}
    # Check if sum of reciprocals equals 1/x
    
    for x in range(1, 25):
        others = [y for y in range(1, 25) if y != x]
        
        # Size 2: 1/x = 1/a + 1/b
        for i in range(len(others)):
            a = others[i]
            for j in range(i + 1, len(others)):
                b = others[j]
                # 1/x = 1/a + 1/b => a*b = x*(a+b)
                if a * b == x * (a + b):
                    # Conflict: x, a, b all can't be together
                    # Mark edges: (x,a), (x,b), (a,b) all conflict
                    adj[x] |= (1 << a)
                    adj[a] |= (1 << x)
                    adj[x] |= (1 << b)
                    adj[b] |= (1 << x)
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
        
        # Size 3: 1/x = 1/a + 1/b + 1/c
        for i in range(len(others)):
            a = others[i]
            for j in range(i + 1, len(others)):
                b = others[j]
                for k in range(j + 1, len(others)):
                    c = others[k]
                    # 1/x = 1/a + 1/b + 1/c
                    # abc = x*(bc + ac + ab)
                    lhs = a * b * c
                    rhs = x * (b * c + a * c + a * b)
                    if lhs == rhs:
                        # All pairs in {x, a, b, c} conflict
                        for p in [x, a, b, c]:
                            for q in [x, a, b, c]:
                                if p != q:
                                    adj[p] |= (1 << q)
        
        # Size 4: 1/x = 1/a + 1/b + 1/c + 1/d
        # This is O(n^4) which is ~100k per x, so ~2.4M total. Might be too slow.
        # Let's skip size 4 for now and rely on final verification.
    
    return adj

def _mis_branch_and_bound(adj, candidates_mask, selected_mask, best_size, best_set, deadline):
    """
    Maximum independent set via branch and bound.
    candidates_mask: bitmask of remaining candidates
    selected_mask: bitmask of currently selected
    best_size: size of best independent set found
    best_set: list of elements in best set
    """
    if time.time() > deadline:
        return best_size, best_set
    
    # Count candidates
    cand_count = bin(candidates_mask).count('1')
    sel_count = bin(selected_mask).count('1')
    
    if sel_count + cand_count <= best_size:
        return best_size, best_set
    
    if candidates_mask == 0:
        if sel_count > best_size:
            # Extract set
            s = []
            for i in range(1, 25):
                if selected_mask & (1 << i):
                    s.append(i)
            best_size = sel_count
            best_set = s
        return best_size, best_set
    
    # Pick a vertex with highest degree in the candidate subgraph
    # Greedy coloring bound: color the candidates to get upper bound
    # For now, use simple bound: number of candidates
    
    # Select a vertex to branch on (highest degree among candidates)
    best_v = -1
    best_deg = -1
    temp_mask = candidates_mask
    while temp_mask:
        lsb = temp_mask & (-temp_mask)
        v = lsb.bit_length() - 1
        # Degree in candidate subgraph
        deg = bin(adj[v] & candidates_mask).count('1')
        if deg > best_deg:
            best_deg = deg
            best_v = v
        temp_mask ^= lsb
    
    if best_v == -1:
        return best_size, best_set
    
    # Branch 1: Include best_v
    new_selected = selected_mask | (1 << best_v)
    # Remove best_v and its neighbors from candidates
    new_candidates = candidates_mask & ~(1 << best_v) & ~adj[best_v]
    
    result_size, result_set = _mis_branch_and_bound(
        adj, new_candidates, new_selected, best_size, best_set, deadline
    )
    best_size = result_size
    best_set = result_set
    
    # Branch 2: Exclude best_v
    new_candidates2 = candidates_mask & ~(1 << best_v)
    result_size, result_set = _mis_branch_and_bound(
        adj, new_candidates2, selected_mask, best_size, best_set, deadline
    )
    best_size = result_size
    best_set = result_set
    
    return best_size, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Greedy initial solution: start with large elements (fewer conflicts)
    greedy_set = []
    for x in range(24, 0, -1):
        test_set = greedy_set + [x]
        if _is_valid_set(test_set):
            greedy_set = test_set
    
    best_set = greedy_set
    best_size = len(greedy_set)
    
    # Precompute pairwise conflicts
    adj = _precompute_pairwise_conflicts()
    
    # Full candidate mask: bits 1-24 set
    full_mask = 0
    for i in range(1, 25):
        full_mask |= (1 << i)
    
    # MIS B&B
    result_size, result_set = _mis_branch_and_bound(
        adj, full_mask, 0, best_size, list(best_set), deadline
    )
    if result_size > best_size:
        best_size = result_size
        best_set = result_set
    
    # Verify and if time permits, try to improve
    if time.time() < deadline - 3:
        # Try greedy with different orderings
        for order in [list(range(1, 25)), list(range(24, 0, -1)), 
                      [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                      [24, 12, 23, 13, 22, 14, 21, 15, 20, 16, 19, 17, 18, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]:
            g = []
            for x in order:
                test = g + [x]
                if _is_valid_set(test):
                    g = test
            if len(g) > best_size:
                best_size = len(g)
                best_set = g
    
    return best_set
# EVOLVE-BLOCK-END
