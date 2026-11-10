"""Local search for max 3-APs with no 4-APs in [1,256]."""
import time
import random
import numpy as np

N = 256

def construct():
    t0 = time.time()
    deadline = t0 + 85  # safety margin below 90s
    
    # Precompute 3-AP and 4-AP bitmasks
    # For each x in [1, N], 3ap_mask[x] has bits set for y such that {x,y} is in some 3-AP
    # Actually: for each x, we need to know which pairs (a,c) with x as middle or endpoint
    # Simpler: for each x, 3ap_partners[x] = set of y where x,y are both in some 3-AP
    # But we need: adding x to A creates a 3-AP iff there exist a,c in A with {a,x,c} forming a 3-AP
    
    # Let's precompute for each x: the list of (a, c) pairs such that a, x, c form a 3-AP (x is middle)
    # and for each x: pairs where x is an endpoint
    # Actually simpler: for each x, precompute a bitmask of all y such that x and y are in a 3-AP together
    # Then count_3aps_involving(x, A) = popcount(A & 3ap_mask[x]) / ... no that double counts
    
    # Better approach: enumerate all 3-APs and all 4-APs, store them
    # N=256, so number of 3-APs is about sum_{d=1}^{128} (256-2d) ≈ 128*256/2 ≈ 16000
    # Number of 4-APs is about sum_{d=1}^{85} (256-3d) ≈ 85*256/2 ≈ 10000
    
    # Precompute all 3-APs as sorted tuples (a, b, c) where b-a = c-b = d > 0
    three_aps = []
    three_aps_by_elem = [[] for _ in range(N+1)]
    for a in range(1, N+1):
        for d in range(1, (N - a) // 2 + 1):
            b = a + d
            c = a + 2*d
            if c > N:
                break
            three_aps.append((a, b, c))
            three_aps_by_elem[a].append(len(three_aps)-1)
            three_aps_by_elem[b].append(len(three_aps)-1)
            three_aps_by_elem[c].append(len(three_aps)-1)
    
    # Precompute all 4-APs
    four_aps = []
    four_aps_by_elem = [[] for _ in range(N+1)]
    for a in range(1, N+1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e > N:
                break
            four_aps.append((a, b, c, e))
            four_aps_by_elem[a].append(len(four_aps)-1)
            four_aps_by_elem[b].append(len(four_aps)-1)
            four_aps_by_elem[c].append(len(four_aps)-1)
            four_aps_by_elem[e].append(len(four_aps)-1)
    
    # For each element x, precompute a bitmask of elements that with x form part of a 3-AP
    # Specifically: 3ap_partner_mask[x] has bit y set if x and y appear together in some 3-AP
    # This is useful for counting: # of 3-APs fully in A involving x = for each (a,c) pair...
    # Actually let's use a different representation:
    # For each x, store the list of indices into three_aps that involve x
    # Then count_3aps(x, A) = number of those APs where all 3 elements are in A
    
    # For 4-AP feasibility: x can be added iff no 4-AP involving x has all other 3 elements in A
    # four_ap_check(x, A): for each 4-AP idx in four_aps_by_elem[x], check if the other 3 are all in A
    
    # Use numpy bitmasks for speed
    # Represent A as a numpy boolean array of length N+1 (index 0 unused)
    
    def count_3aps(A_mask):
        """Count distinct 3-APs fully contained in A_mask (numpy bool array)."""
        count = 0
        for a, b, c in three_aps:
            if A_mask[a] and A_mask[b] and A_mask[c]:
                count += 1
        return count
    
    def has_4ap(A_mask):
        """Check if A_mask contains any 4-AP."""
        for a, b, c, e in four_aps:
            if A_mask[a] and A_mask[b] and A_mask[c] and A_mask[e]:
                return True
        return False
    
    def count_3aps_involving(x, A_mask):
        """Count 3-APs that include x and are fully in A_mask."""
        count = 0
        for idx in three_aps_by_elem[x]:
            a, b, c = three_aps[idx]
            if A_mask[a] and A_mask[b] and A_mask[c]:
                count += 1
        return count
    
    def four_ap_violations_on_add(x, A_mask):
        """Check if adding x to A_mask creates any 4-AP. Returns True if violation."""
        for idx in four_aps_by_elem[x]:
            a, b, c, e = four_aps[idx]
            # x is one of a,b,c,e; check if the other three are all in A_mask
            others = [v for v in (a, b, c, e) if v != x]
            if all(A_mask[o] for o in others):
                return True
        return False
    
    def new_3aps_on_add(x, A_mask):
        """Count new 3-APs created by adding x to A_mask."""
        count = 0
        for idx in three_aps_by_elem[x]:
            a, b, c = three_aps[idx]
            others = [v for v in (a, b, c) if v != x]
            if all(A_mask[o] for o in others):
                count += 1
        return count
    
    # Phase 1: Greedy construction - build a 4-AP-free set maximizing 3-APs
    # Start with empty set, add elements in order of "desirability"
    
    # Strategy: start with a random 4-AP-free set, then do local search
    # Or: greedy add elements that maximize 3-AP gain without creating 4-AP
    
    # Let's try: greedy construction
    def greedy_construct():
        A = np.zeros(N+1, dtype=bool)
        A_count = 0
        # Try adding elements in a randomized order, but prioritize those with many 3-AP potential
        order = list(range(1, N+1))
        random.shuffle(order)
        for x in order:
            if not four_ap_violations_on_add(x, A):
                A[x] = True
                A_count += 1
        return A, A_count
    
    # Phase 2: Local search - try swaps (remove one, add another) to improve 3-AP count
    def local_search(A, max_iter=50000):
        A = A.copy()
        best_score = count_3aps(A)
        no_improve = 0
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            # Try single removals that don't reduce score too much, then additions
            # Try swaps: for each x in A, for each y not in A, if swap improves
            # This is O(|A| * (N-|A|)) per iteration which is too slow
            # Instead: try adding elements that give positive gain
            for y in range(1, N+1):
                if A[y]:
                    continue
                if four_ap_violations_on_add(y, A):
                    continue
                gain = new_3aps_on_add(y, A)
                if gain > 0:
                    A[y] = True
                    best_score += gain
                    improved = True
            # Try removals + re-additions (swap-like)
            for x in range(1, N+1):
                if not A[x]:
                    continue
                # What's the cost of removing x?
                cost = count_3aps_involving(x, A)
                if cost == 0:
                    # Free to remove, try to add something else
                    A[x] = False
                    # Try to add back a different element
                    for y in range(1, N+1):
                        if A[y]:
                            continue
                        if not four_ap_violations_on_add(y, A):
                            gain = new_3aps_on_add(y, A)
                            if gain > 0:
                                A[y] = True
                                best_score += gain
                                improved = True
                                break
                    if not improved:
                        A[x] = True  # put back if nothing better found
                    else:
                        best_score -= cost
                        # Actually this accounting is getting complex, let's just recompute periodically
            # Periodic full recompute to avoid drift
            if it % 100 == 0:
                current_score = count_3aps(A)
                if current_score > best_score:
                    best_score = current_score
                else:
                    best_score = current_score
            if not improved:
                no_improve += 1
                if no_improve > 50:
                    break
            else:
                no_improve = 0
        return A, best_score
    
    # Phase 3: More aggressive local search with swaps
    def swap_search(A, max_iter=20000):
        A = A.copy()
        best_score = count_3aps(A)
        no_improve = 0
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            # Try each pair (x in A, y not in A)
            # To speed up, only try promising swaps
            for x in range(1, N+1):
                if not A[x]:
                    continue
                cost_x = count_3aps_involving(x, A)
                if cost_x > 15:  # skip expensive removals
                    continue
                for y in range(1, N+1):
                    if A[y]:
                        continue
                    if four_ap_violations_on_add(y, A):
                        continue
                    gain_y = new_3aps_on_add(y, A)
                    if gain_y > cost_x:
                        # Do the swap
                        A[x] = False
                        A[y] = True
                        best_score += gain_y - cost_x
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                no_improve += 1
                if no_improve > 30:
                    break
            else:
                no_improve = 0
        # Verify
        actual_score = count_3aps(A)
        if has_4ap(A):
            # Shouldn't happen, but fix it
            while has_4ap(A):
                # Remove element in most 4-APs
                elem_4ap_count = [0]*(N+1)
                for a,b,c,e in four_aps:
                    if A[a] and A[b] and A[c] and A[e]:
                        for v in (a,b,c,e):
                            elem_4ap_count[v] += 1
                worst = max(range(1,N+1), key=lambda v: (elem_4ap_count[v], 0) if A[v] else (-1, 0))
                A[worst] = False
            actual_score = count_3aps(A)
        return A, actual_score
    
    # Run multiple restarts
    best_A = None
    best_final_score = -1
    
    # First, a deterministic greedy
    A0, _ = greedy_construct()
    A0, s0 = local_search(A0)
    A0, s0 = swap_search(A0)
    if s0 > best_final_score:
        best_final_score = s0
        best_A = A0.copy()
    
    # Randomized restarts
    attempts = 0
    while time.time() < deadline - 10:
        A_r = np.zeros(N+1, dtype=bool)
        order = list(range(1, N+1))
        random.shuffle(order)
        for x in order:
            if not four_ap_violations_on_add(x, A_r):
                A_r[x] = True
        A_r, s_r = local_search(A_r)
        A_r, s_r = swap_search(A_r)
        if s_r > best_final_score:
            best_final_score = s_r
            best_A = A_r.copy()
        attempts += 1
        if attempts >= 30 and time.time() > deadline - 20:
            break
    
    # Final verification and extraction
    if best_A is None:
        # Fallback: use the seed algorithm
        selected = set()
        for x in range(1, 257):
            if all(not all(x - j*d in selected for j in (1,2,3)) for d in range(1, (x-1)//3+1)):
                selected.add(x)
        return sorted(selected)
    
    result = [x for x in range(1, N+1) if best_A[x]]
    
    # Final safety check: no 4-AP
    if has_4ap(best_A):
        print("WARNING: 4-AP detected, fixing")
        while has_4ap(best_A):
            elem_4ap_count = [0]*(N+1)
            for a,b,c,e in four_aps:
                if best_A[a] and best_A[b] and best_A[c] and best_A[e]:
                    for v in (a,b,c,e):
                        elem_4ap_count[v] += 1
            candidates = [v for v in range(1, N+1) if best_A[v]]
            if not candidates:
                break
            worst = max(candidates, key=lambda v: elem_4ap_count[v])
            best_A[worst] = False
        result = [x for x in range(1, N+1) if best_A[x]]
    
    if not result:
        return [1]
    
    return sorted(result)

