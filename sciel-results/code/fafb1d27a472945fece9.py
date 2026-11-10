# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square modulo 205.
    
    Strategy:
    1. Compute forbidden differences (nonzero quadratic residues mod 205).
    2. Build conflict graph: edge between i and j if (i-j) % 205 is a forbidden difference.
    3. Use bitmask-accelerated local search with multiple random restarts.
       - Start with a random greedy construction.
       - Refine using 1-add and 1-swap moves.
    4. Return the best solution found within time budget.
    """
    m = 205
    deadline = time.time() + 620  # 10s safety margin before 630s kill
    
    # Compute forbidden differences: nonzero squares mod 205
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Build conflict matrix: conflict[i][j] = True if i and j cannot both be in set
    # conflict[i][j] is True if (i-j) % m in forbidden or (j-i) % m in forbidden
    # Since forbidden is symmetric (if d in forbidden then -d mod m may or may not be in forbidden)
    # Actually, we need: (a-b) % m not in forbidden AND (b-a) % m not in forbidden
    # Note: (b-a) % m = (- (a-b)) % m. So we need neither d nor -d (mod m) to be in forbidden.
    # Let's define: bad_diff(d) = (d % m in forbidden) or ((-d) % m in forbidden)
    # But since we check both directions, it's equivalent to: (a-b)%m in forbidden OR (b-a)%m in forbidden
    
    # Precompute: for each pair (i, j), is there a conflict?
    # conflict[i] is a bitmask of all j that conflict with i
    conflict = [0] * m
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                conflict[i] |= (1 << j)
    
    def is_valid_selection(selected_mask):
        """Check if the selection (bitmask) is valid."""
        # For each set bit i, check that no other set bit j conflicts with i
        # This is O(n^2) but n is small
        bits = []
        temp = selected_mask
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            bits.append(idx)
            temp ^= lsb
        for i in range(len(bits)):
            for j in range(i+1, len(bits)):
                a, b = bits[i], bits[j]
                d1 = (a - b) % m
                d2 = (b - a) % m
                if d1 in forbidden or d2 in forbidden:
                    return False
        return True
    
    def greedy_random(seed):
        """Random greedy construction."""
        rng = random.Random(seed)
        candidates = list(range(m))
        rng.shuffle(candidates)
        selected_mask = 0
        selected_list = []
        for cand in candidates:
            # Check if cand conflicts with any already selected
            ok = True
            for s in selected_list:
                d1 = (cand - s) % m
                d2 = (s - cand) % m
                if d1 in forbidden or d2 in forbidden:
                    ok = False
                    break
            if ok:
                selected_mask |= (1 << cand)
                selected_list.append(cand)
        return selected_mask, selected_list
    
    def local_search(selected_mask, selected_list, rng, max_iters=5000):
        """1-add and 1-swap local search."""
        best_mask = selected_mask
        best_list = list(selected_list)
        best_size = len(best_list)
        
        # Build a set for fast lookup
        best_set = set(best_list)
        
        no_improve = 0
        iters = 0
        while iters < max_iters and no_improve < 200:
            iters += 1
            improved = False
            
            # Try 1-add: add a candidate not in set
            candidates_to_add = [c for c in range(m) if c not in best_set]
            rng.shuffle(candidates_to_add)
            for cand in candidates_to_add:
                # Check if cand can be added
                ok = True
                for s in best_list:
                    d1 = (cand - s) % m
                    d2 = (s - cand) % m
                    if d1 in forbidden or d2 in forbidden:
                        ok = False
                        break
                if ok:
                    best_mask |= (1 << cand)
                    best_list.append(cand)
                    best_set.add(cand)
                    best_size += 1
                    improved = True
                    no_improve = 0
                    break
                else:
                    no_improve += 1
            
            if improved:
                continue
            
            # Try 1-swap: remove one, add another
            # For each element in set, try removing it and adding a new element
            rng.shuffle(best_list)
            for remove_idx, rem in enumerate(best_list):
                # Temporarily remove rem
                temp_list = best_list[:remove_idx] + best_list[remove_idx+1:]
                temp_set = set(temp_list)
                
                # Try adding each candidate not in temp_set
                candidates_to_add = [c for c in range(m) if c not in temp_set]
                rng.shuffle(candidates_to_add)
                for cand in candidates_to_add:
                    # Check if cand can be added to temp_list
                    ok = True
                    for s in temp_list:
                        d1 = (cand - s) % m
                        d2 = (s - cand) % m
                        if d1 in forbidden or d2 in forbidden:
                            ok = False
                            break
                    if ok:
                        # Perform the swap
                        best_list = temp_list + [cand]
                        best_set = temp_set | {cand}
                        best_mask = 0
                        for x in best_list:
                            best_mask |= (1 << x)
                        improved = True
                        no_improve = 0
                        break
                    else:
                        no_improve += 1
                if improved:
                    break
            
            if not improved:
                no_improve += 1
        
        return best_mask, best_list, best_size
    
    # Main search loop
    best_mask = 0
    best_list = []
    best_size = 0
    
    # Start with deterministic greedy (in order)
    selected_mask, selected_list = greedy_random(0)
    # Actually, let's do a simple forward greedy first for a baseline
    sel = []
    for cand in range(m):
        ok = True
        for s in sel:
            d1 = (cand - s) % m
            d2 = (s - cand) % m
            if d1 in forbidden or d2 in forbidden:
                ok = False
                break
        if ok:
            sel.append(cand)
    best_list = sel
    best_size = len(sel)
    best_mask = 0
    for x in best_list:
        best_mask |= (1 << x)
    
    # Try multiple random restarts
    seed = 1
    while time.time() < deadline:
        # Random greedy start
        start_mask, start_list = greedy_random(seed)
        
        # Local search
        res_mask, res_list, res_size = local_search(start_mask, start_list, random.Random(seed))
        
        if res_size > best_size:
            best_mask = res_mask
            best_list = res_list
            best_size = res_size
        
        seed += 1
        
        # Also try local search on the current best
        if seed % 5 == 0:
            res_mask, res_list, res_size = local_search(best_mask, best_list, random.Random(seed + 1000))
            if res_size > best_size:
                best_mask = res_mask
                best_list = res_list
                best_size = res_size
    
    return np.array(best_list, dtype=np.int64)

# EVOLVE-BLOCK-END
