"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from itertools import combinations
import random
import time

def _precompute_conflicts():
    """Precompute which pairs of 4-subsets form a sunflower with any third set.
    Actually, we need to check: for each pair (A,B), determine if there exists C
    such that A∩B = A∩C = B∩C (all three pairwise intersections equal).
    
    For 4-sets, the pairwise intersection sizes can be 0,1,2,3,4.
    A sunflower with core K means A∩B = A∩C = B∩C = K.
    
    We'll precompute for each pair (i,j) the set of intersection patterns.
    Actually, let's think differently: for each pair (A,B) with intersection I,
    we need to know if adding a third set C would create a sunflower.
    
    Better approach: precompute for each 4-subset its bitmask, and for each pair,
    store the intersection bitmask. Then when checking if C can be added to {A,B,...},
    we check if A∩B == A∩C == B∩C.
    
    Let's precompute all 4-subsets as bitmasks and their pairwise intersections.
    """
    all_sets = list(combinations(range(16), 4))
    n = len(all_sets)
    
    # Convert to bitmasks
    masks = []
    for s in all_sets:
        m = 0
        for x in s:
            m |= (1 << x)
        masks.append(m)
    
    # Precompute pairwise intersections
    # inter[i][j] = masks[i] & masks[j]
    # Store as list of lists for fast access
    inter = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            ij = masks[i] & masks[j]
            inter[i][j] = ij
            inter[j][i] = ij
    
    return all_sets, masks, inter

def _greedy_with_min_degree(all_sets, masks, inter, n, deadline):
    """Greedy: repeatedly pick the set with minimum number of conflicts with already chosen sets.
    A conflict means: for already-chosen pair (A,B), if A∩B == A∩new == B∩new, it's a sunflower.
    Also need to check: for each chosen set A, and each other chosen set B, 
    the triple (A, B, new) must not form a sunflower.
    
    Actually, the condition is: no THREE distinct sets have all three pairwise intersections equal.
    So when adding new set C, we need to check all pairs (A,B) already in the family:
    A∩B == A∩C == B∩C must NOT hold.
    """
    chosen = set()
    chosen_list = []
    
    # For efficiency, maintain for each candidate the number of conflicts
    # conflict_count[c] = number of pairs (a,b) in chosen such that a∩b == a∩c == b∩c
    
    # We'll use a different approach: maintain a "blocked" status
    # Actually, let's just do the greedy with a time check
    
    remaining = list(range(n))
    random.shuffle(remaining)
    
    best_size = 0
    best_chosen = []
    
    while remaining:
        if time.time() > deadline:
            break
        
        # Find the candidate with minimum conflicts
        best_cand = -1
        best_conflicts = float('inf')
        
        for idx in remaining:
            c = masks[idx]
            conflicts = 0
            for i in range(len(chosen_list)):
                a = chosen_list[i]
                ac = a & c
                for j in range(i+1, len(chosen_list)):
                    b = chosen_list[j]
                    bc = b & c
                    ab = a & b
                    if ab == ac == bc:
                        conflicts += 1
                        break
            if conflicts < best_conflicts:
                best_conflicts = conflicts
                best_cand = idx
                if conflicts == 0:
                    break
        
        if best_cand == -1 or best_conflicts > 0:
            # Try to find any with 0 conflicts
            found = False
            for idx in remaining:
                c = masks[idx]
                ok = True
                for i in range(len(chosen_list)):
                    a = chosen_list[i]
                    ac = a & c
                    for j in range(i+1, len(chosen_list)):
                        b = chosen_list[j]
                        bc = b & c
                        ab = a & b
                        if ab == ac == bc:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    best_cand = idx
                    found = True
                    break
            if not found:
                break
        
        # Add best_cand
        chosen_list.append(masks[best_cand])
        remaining.remove(best_cand)
        chosen.add(best_cand)
        
        if len(chosen_list) > best_size:
            best_size = len(chosen_list)
            best_chosen = list(chosen_list)
    
    return best_chosen

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    all_sets, masks, inter = _precompute_conflicts()
    n = len(all_sets)
    
    best_result = []
    
    # Multiple randomized restarts
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        random.seed(r * 7919 + 42)  # Deterministic per restart
        result = _greedy_with_min_degree(all_sets, masks, inter, n, deadline)
        
        if len(result) > len(best_result):
            best_result = result
        
        if time.time() > deadline - 10:
            break
    
    # Convert back to sorted lists
    result = []
    for m in best_result:
        s = []
        for i in range(16):
            if m & (1 << i):
                s.append(i + 1)
        result.append(sorted(s))
    
    return result
# EVOLVE-BLOCK-END
