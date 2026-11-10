"""
Strategy: Precomputed Conflict Matrix + Randomized Greedy + Local Search (Add/Swap)

Key structural change:
1. Precompute a (192, 192) boolean conflict matrix where conflict[i][j] = True
   if adding element j to a set that already contains i would create a violation
   in at least one scenario. Actually, the constraint is not pairwise.
   
   Revised structural change:
   - Precompute for every triple (a,b,c) from [1,192], the set of divisors d such
     that d divides (a+b+c). If d is in the selected set, and {a,b,c} are three
     DISTINCT OTHER selected elements, that's a violation.
     
   More precisely: For a selected set S, the constraint is:
     For every d in S, there is no triple (x,y,z) in S\{d} such that d | (x+y+z).
     
   We can precompute:
     For each d in [1,192] and each pair (x,y) with x != y, the values z such that
     d | (x+y+z). That is, z ≡ (-x-y) mod d.
     
   This is complex. Let's use a different approach:
   
   APPROACH:
   1. Precompute all valid triples: For each triple (a,b,c) of distinct values in [1,192],
      compute s = a+b+c. For each divisor d of s, if d is not in {a,b,c}, then d
      "conflicts" with this triple in the sense that d cannot be in the selected set
      if all of a,b,c are in the selected set.
      
   2. Build a hypergraph: vertices are numbers 1..192, hyperedges are triples
      {a,b,c} that are "blocked" by d. Actually, for each d and each triple {a,b,c}
      not containing d where d | (a+b+c), we have a constraint: not (d in S and a in S and b in S and c in S).
      
   3. Use a different formulation: We want the largest S such that for all d in S,
      and for all triples {x,y,z} in S\{d}, d does not divide x+y+z.
      
   Let me just implement an efficient incremental check and use randomized greedy + local search.
"""

import time
import random
import sys

sys.setrecursionlimit(10000)

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin for 90s limit


def _check_valid_addition(candidate, selected):
    """
    Check if 'candidate' can be added to 'selected' without violating constraints.
    Constraint: No selected integer may divide the sum of any 3 DISTINCT OTHER selected integers.
    
    When we add 'candidate' to 'selected':
    1. candidate must not divide sum of any 3 distinct elements in selected.
    2. For each a in selected, a must not divide (candidate + x + y) for any 
       distinct x, y in selected \\ {a}.
    """
    n = len(selected)
    
    # 1. Check candidate against existing triples in selected
    if n >= 3:
        sel = selected
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    s = sel[i] + sel[j] + sel[k]
                    if s % candidate == 0:
                        return False
    
    # 2. Check existing 'a' against new sums involving candidate
    # For each a in selected:
    #   Check if a divides (candidate + x + y) for distinct x, y in selected \ {a}
    for idx_a, a in enumerate(selected):
        target = (-candidate) % a
        # We need distinct x, y from selected \ {a} such that (x+y) % a == target
        # And x != y.
        
        # Use a set of residues for efficiency
        residues_seen = set()
        for i in range(n):
            if i == idx_a:
                continue
            x = selected[i]
            x_res = x % a
            needed = (target - x_res) % a
            
            # Check if we've already seen an element with residue 'needed'
            if needed in residues_seen:
                return False
            residues_seen.add(x_res)
        
        # We also need to handle the case where x and y could be the same element
        # Wait, x and y must be DISTINCT. The above loop only checks pairs where
        # y was seen BEFORE x in the iteration. This correctly handles distinctness
        # because we add x_res AFTER checking if needed was already present.
        # But we need to be careful: if needed == x_res, that would mean x + x is valid,
        # but x and y must be distinct elements, so x + x is not allowed unless there
        # are two different elements with the same residue. The set approach handles this:
        # if needed == x_res, we check if x_res was in residues_seen BEFORE adding x_res,
        # which would mean a PREVIOUS element had that residue. Correct.
        
    return True


def _is_valid_set(selected):
    """Full validation of a set."""
    n = len(selected)
    for i in range(n):
        d = selected[i]
        # Check if d divides sum of any 3 distinct other elements
        others = [selected[j] for j in range(n) if j != i]
        m = len(others)
        for x in range(m):
            for y in range(x + 1, m):
                for z in range(y + 1, m):
                    if (others[x] + others[y] + others[z]) % d == 0:
                        return False
    return True


def _greedy_construct(order=None):
    """Greedy construction with given order (default: descending)."""
    if order is None:
        order = list(range(192, 0, -1))
    
    selected = []
    for c in order:
        if _check_valid_addition(c, selected):
            selected.append(c)
    return selected


def _local_search(initial_set, max_time=None):
    """
    Local search with add and swap moves to improve set size.
    Returns the best valid set found.
    """
    if max_time is None:
        max_time = 5.0
    
    start = time.time()
    best_set = list(initial_set)
    best_size = len(best_set)
    
    # Build a set for O(1) lookup
    best_set_set = set(best_set)
    
    improvement = True
    while improvement and time.time() - start < max_time:
        improvement = False
        
        # Try adding elements
        for c in range(192, 0, -1):
            if c in best_set_set:
                continue
            if _check_valid_addition(c, best_set):
                new_set = best_set + [c]
                if _is_valid_set(new_set):
                    best_set = new_set
                    best_set_set = set(best_set)
                    best_size += 1
                    improvement = True
                    break
        
        if improvement:
            continue
        
        # Try swap moves: remove one element, add another
        for i in range(len(best_set)):
            for c in range(192, 0, -1):
                if c in best_set_set:
                    continue
                # Remove best_set[i], try adding c
                reduced = [x for j, x in enumerate(best_set) if j != i]
                if _check_valid_addition(c, reduced):
                    new_set = reduced + [c]
                    if _is_valid_set(new_set):
                        best_set = new_set
                        best_set_set = set(best_set)
                        # Size stays the same, but maybe enables further additions
                        # We don't count this as improvement for loop termination,
                        # but it might enable additions later
                        # For now, just accept it and continue
                        break
            if time.time() - start > max_time:
                break
        
        # After a swap, try adding again
        if time.time() - start < max_time:
            for c in range(192, 0, -1):
                if c in best_set_set:
                    continue
                if _check_valid_addition(c, best_set):
                    new_set = best_set + [c]
                    if _is_valid_set(new_set):
                        best_set = new_set
                        best_set_set = set(best_set)
                        best_size += 1
                        improvement = True
                        break
    
    return best_set


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    best_set = []
    best_size = 0
    
    # Try multiple random orders for greedy construction
    random.seed(42)
    
    # First, try natural descending order
    set1 = _greedy_construct(list(range(192, 0, -1)))
    if len(set1) > best_size:
        best_set = set1
        best_size = len(set1)
    
    # Try ascending order
    set2 = _greedy_construct(list(range(1, 193)))
    if len(set2) > best_size:
        best_set = set2
        best_size = len(set2)
    
    # Try random orders
    num_random = 20
    for i in range(num_random):
        if time.time() > deadline - 10:
            break
        order = list(range(1, 193))
        random.shuffle(order)
        s = _greedy_construct(order)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
    
    # Local search from the best set
    if time.time() < deadline - 5:
        improved = _local_search(best_set, max_time=min(30.0, deadline - time.time() - 3))
        if len(improved) > best_size:
            best_set = improved
            best_size = len(improved)
    
    # Final validation
    if not _is_valid_set(best_set):
        # Fallback: return a small valid set
        best_set = [192, 191, 190]
    
    return best_set
