import time
import random

def construct():
    start_time = time.time()
    n = 256
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Precompute all 3-APs as (a, b, c) with a < b < c, b-a = c-b > 0
    aps_3 = []
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            aps_3.append((a, b, c))

    # Precompute for each element x, the list of 3-AP indices where x appears
    # and the positions (0=a, 1=b, 2=c)
    ap_index_by_elem = {x: [] for x in range(1, n + 1)}
    for idx, (a, b, c) in enumerate(aps_3):
        ap_index_by_elem[a].append((idx, 0))
        ap_index_by_elem[b].append((idx, 1))
        ap_index_by_elem[c].append((idx, 2))

    # Precompute 4-AP conflict checks using bitmasks
    # For each x, precompute the set of (y1, y2, y3) triples that with x form a 4-AP
    # i.e., adding x to a set containing {y1,y2,y3} creates a 4-AP
    # We'll use a different approach: for each x, compute which existing elements block it
    # Actually, let's precompute for each x the list of "forbidden triples" - 
    # triples (p,q,r) such that {p,q,r,x} is a 4-AP
    # There are at most ~3*85 = 255 such triples per x (for each d, up to 4 positions)
    
    # For efficiency, let's use a different strategy:
    # Represent S as a set, and for checking if x can be added, check all 4-APs containing x
    # A 4-AP is (s, s+d, s+2d, s+3d). x can be at position 0,1,2,3.
    # Precompute for each x, the list of (d, pos) where x = s + pos*d
    # and the other three elements
    
    # Actually, simplest: for each x, precompute all 4-APs that contain x
    four_aps_with_x = {x: [] for x in range(1, n + 1)}
    for d in range(1, (n - 1) // 3 + 1):
        for s in range(1, n - 3 * d + 1):
            ap = (s, s + d, s + 2 * d, s + 3 * d)
            for elem in ap:
                four_aps_with_x[elem].append(ap)
    
    # For fast 4-AP check when adding x to set S:
    # Check if any 4-AP containing x has the other 3 elements in S
    def can_add(x, S):
        for ap in four_aps_with_x[x]:
            # Check if the other 3 elements are in S
            if all(elem in S for elem in ap if elem != x):
                return False
        return True

    # Score function: count 3-APs in S
    def score_set(S):
        cnt = 0
        for a, b, c in aps_3:
            if a in S and b in S and c in S:
                cnt += 1
        return cnt

    # Incremental score: given S and element x not in S, compute how many new 3-APs are formed
    def gain_if_add(x, S):
        gain = 0
        for idx, pos in ap_index_by_elem[x]:
            a, b, c = aps_3[idx]
            if pos == 0:  # x is a
                if b in S and c in S:
                    gain += 1
            elif pos == 1:  # x is b
                if a in S and c in S:
                    gain += 1
            else:  # x is c
                if a in S and b in S:
                    gain += 1
        return gain

    # Greedy construction with score-awareness
    def greedy_construct(order, use_score=False):
        S = set()
        for x in order:
            if not can_add(x, S):
                continue
            if use_score:
                # Only add if it gives positive gain, or if we haven't added many yet
                g = gain_if_add(x, S)
                if g > 0 or len(S) < 30:
                    S.add(x)
            else:
                S.add(x)
        return S

    # Local search: add, remove, swap
    def local_search(S, max_iters=50):
        S = set(S)
        current_score = score_set(S)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            
            # Phase 1: Try adding elements
            for x in range(1, n + 1):
                if x in S:
                    continue
                if can_add(x, S):
                    g = gain_if_add(x, S)
                    if g > 0:
                        S.add(x)
                        current_score += g
                        improved = True
                        if time.time() > deadline - 5:
                            break
            
            if time.time() > deadline - 5:
                break
            
            # Phase 2: Try removing elements that are "expensive" (block many additions)
            # For each y in S, check if removing y allows adding elements with net positive gain
            # This is expensive, so do it sparingly
            if iteration % 3 == 0:
                for y in list(S):
                    if time.time() > deadline - 5:
                        break
                    S.remove(y)
                    # Try to add back elements that were blocked
                    added = []
                    for x in range(1, n + 1):
                        if x in S:
                            continue
                        if can_add(x, S):
                            g = gain_if_add(x, S)
                            if g > 0:
                                S.add(x)
                                added.append((x, g))
                    
                    new_score = current_score
                    # Recalculate properly
                    new_score = score_set(S)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                    else:
                        # Revert
                        S.discard(y)
                        for x, _ in added:
                            S.discard(x)
                        # Re-add y
                        S.add(y)
            
            if not improved:
                break
        
        return S, current_score

    # Main: run multiple strategies and keep the best
    best_S = set()
    best_score = 0
    
    # Strategy 1: Natural order greedy
    if time.time() < deadline - 10:
        S1 = greedy_construct(range(1, n + 1), use_score=False)
        S1, sc1 = local_search(S1, max_iters=20)
        if sc1 > best_score:
            best_score = sc1
            best_S = S1
    
    # Strategy 2: Random order greedy (multiple restarts)
    random.seed(42)
    for restart in range(10):
        if time.time() > deadline - 15:
            break
        random_order = list(range(1, n + 1))
        random.shuffle(random_order)
        S_r = greedy_construct(random_order, use_score=True)
        S_r, sc_r = local_search(S_r, max_iters=15)
        if sc_r > best_score:
            best_score = sc_r
            best_S = S_r
    
    # Strategy 3: Start from a dense initial set and prune 4-APs
    if time.time() < deadline - 10:
        # Start with all elements, then remove to break 4-APs
        S_p = set(range(1, n + 1))
        # Remove elements to break all 4-APs
        # Greedy removal: for each 4-AP, if all 4 present, remove one
        four_aps_all = []
        for d in range(1, (n - 1) // 3 + 1):
            for s in range(1, n - 3 * d + 1):
                four_aps_all.append((s, s + d, s + 2 * d, s + 3 * d))
        
        for ap in four_aps_all:
            if all(elem in S_p for elem in ap):
                # Remove the element with the lowest "value" (fewest 3-APs)
                # For simplicity, remove the last one
                S_p.discard(ap[3])
        
        S_p, sc_p = local_search(S_p, max_iters=20)
        if sc_p > best_score:
            best_score = sc_p
            best_S = S_p
    
    # Final local search on best
    if time.time() < deadline - 5 and best_S:
        best_S, best_score = local_search(best_S, max_iters=30)
    
    return sorted(best_S)

# EVOLVE-BLOCK-END
