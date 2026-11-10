import time
import random

N = 256
DEADLINE = 55.0  # Safety margin for 90s limit

def construct():
    start_time = time.time()
    
    # Precompute all 3-APs as (a, b, c) triples
    three_aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))
    
    # Precompute 4-APs: map each element to list of (other three elements)
    four_aps_by_v = {i: [] for i in range(1, N + 1)}
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            for v in [a, b, c, e]:
                others = [x for x in [a, b, c, e] if x != v]
                four_aps_by_v[v].append(others)
    
    # Fast score: count 3-APs using difference array
    # A 3-AP (a, a+d, a+2d) exists iff the middle element a+d is in S
    # and both a and a+2d are in S.
    # We can count by: for each element m in S (as middle), count pairs (m-d, m+d) both in S.
    def count_3aps(S):
        """S is a set or boolean list (index 1..N). Returns number of 3-APs."""
        if isinstance(S, set):
            s = S
        else:
            s = set(S)
        count = 0
        for m in s:
            d = 1
            while m - d >= 1 and m + d <= N:
                if (m - d) in s and (m + d) in s:
                    count += 1
                d += 1
        return count
    
    # Fast score using array representation
    def count_3aps_arr(arr):
        """arr is a list of 0/1 of length N+1 (index 1..N)."""
        count = 0
        for m in range(1, N + 1):
            if not arr[m]:
                continue
            d = 1
            while m - d >= 1 and m + d <= N:
                if arr[m - d] and arr[m + d]:
                    count += 1
                d += 1
        return count
    
    # Check if adding v to S creates a 4-AP
    def creates_4ap(S_set, v):
        for others in four_aps_by_v[v]:
            if all(o in S_set for o in others):
                return True
        return False
    
    # Check if swap u->v is valid: S' = S - {u} + {v}
    # Invalid if there exists a 4-AP containing v where the other 3 are in S'
    # Since u is removed, we need: the other 3 don't include u (or if they do, u was in S but is removed)
    # Actually: 4-AP {v, x, y, z}. In S', v is present. The AP is in S' iff x,y,z all in S'.
    # S' = S - {u} + {v}. So x,y,z in S' iff (x,y,z in S) and (x,y,z != u for all, OR u is one of x,y,z but then it's removed)
    # Simpler: the 4-AP is fully contained in S' iff {v,x,y,z} ⊆ S'. Since v ∈ S', we need {x,y,z} ⊆ S'.
    # {x,y,z} ⊆ S' means: for each w in {x,y,z}, w ∈ S'. w ∈ S' iff (w ∈ S and w ≠ u) or w == v.
    # But w ≠ v (since w is from others, which excludes v). So w ∈ S' iff w ∈ S and w ≠ u.
    # So: for each 4-AP containing v, check if the other 3 are all in S and none of them is u.
    def valid_swap(S_set, u, v):
        for others in four_aps_by_v[v]:
            # Check if all others are in S and none is u
            if u in others:
                # u is being removed, so this 4-AP can't be completed (one element missing)
                # But wait: we need all of {others} to be in S'. u is not in S', so if u in others, that 4-AP is broken.
                # So this 4-AP does NOT create a problem.
                continue
            if all(o in S_set for o in others):
                return False
        return True
    
    # Initial set: use a Salem-type construction
    # In Z_p, the set of quadratic residues has no 4-AP for certain p.
    # For [1, 256], we can try: take numbers whose representation has certain properties.
    # Alternative: start with a random greedy set, then optimize.
    
    def greedy_construct():
        S = set()
        for x in range(1, N + 1):
            if not creates_4ap(S, x):
                S.add(x)
        return S
    
    S = greedy_construct()
    current_score = count_3aps(S)
    best_S = set(S)
    best_score = current_score
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Local search with incremental score updates
    # For swap u -> v:
    # Delta score = (3-APs involving v that are now complete) - (3-APs involving u that are now broken)
    # A 3-AP involving v: v can be first (v, v+d, v+2d), middle (v-d, v, v+d), or last (v-2d, v-d, v)
    
    def score_delta(S, u, v):
        """Compute change in 3-AP count when swapping u for v in S."""
        # 3-APs that will be gained (contain v, not u, and other elements in S)
        gained = 0
        for (a, b, c) in three_aps:
            if v in (a, b, c) and u not in (a, b, c):
                others = [x for x in (a, b, c) if x != v]
                if all(o in S for o in others):
                    gained += 1
        # 3-APs that will be lost (contain u, not v, and other elements in S)
        lost = 0
        for (a, b, c) in three_aps:
            if u in (a, b, c) and v not in (a, b, c):
                others = [x for x in (a, b, c) if x != u]
                if all(o in S for o in others):
                    lost += 1
        return gained - lost
    
    # Precompute 3-APs containing each element for faster delta
    aps_by_elem = {i: [] for i in range(1, N + 1)}
    for (a, b, c) in three_aps:
        aps_by_elem[a].append((b, c))  # a is first
        aps_by_elem[b].append((a, c))  # b is middle
        aps_by_elem[c].append((a, b))  # c is last
    
    def score_delta_fast(S, u, v):
        """Faster delta using precomputed APs."""
        gained = 0
        for (x, y) in aps_by_elem[v]:
            if u not in (x, y) and x in S and y in S:
                gained += 1
        lost = 0
        for (x, y) in aps_by_elem[u]:
            if v not in (x, y) and x in S and y in S:
                lost += 1
        return gained - lost
    
    # Main local search loop
    no_improve_rounds = 0
    
    while time_left() > 1.0:
        elements = list(S)
        non_elements = [i for i in range(1, N + 1) if i not in S]
        
        if not elements or not non_elements:
            break
        
        random.shuffle(elements)
        random.shuffle(non_elements)
        
        improved = False
        for u in elements:
            if time_left() < 1.0:
                break
            for v in non_elements:
                if time_left() < 1.0:
                    break
                if not valid_swap(S, u, v):
                    continue
                delta = score_delta_fast(S, u, v)
                if delta > 0:
                    S.discard(u)
                    S.add(v)
                    current_score += delta
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(S)
                    break
            if improved:
                break
        
        if not improved:
            no_improve_rounds += 1
            # Random kick to escape local optimum
            if no_improve_rounds >= 3:
                # Try a few random swaps, accept even if score drops slightly
                for attempt in range(20):
                    if time_left() < 1.0:
                        break
                    u = random.choice(elements)
                    v = random.choice(non_elements)
                    if valid_swap(S, u, v):
                        delta = score_delta_fast(S, u, v)
                        if delta >= -3:  # Accept small drops
                            S.discard(u)
                            S.add(v)
                            current_score += delta
                            if current_score > best_score:
                                best_score = current_score
                                best_S = set(S)
                            no_improve_rounds = 0
                            break
                if no_improve_rounds >= 3:
                    no_improve_rounds = 0
                    # More aggressive: try multiple random kicks
                    for attempt in range(50):
                        if time_left() < 1.0:
                            break
                        u = random.choice(elements)
                        v = random.choice(non_elements)
                        if valid_swap(S, u, v):
                            delta = score_delta_fast(S, u, v)
                            if delta >= -5:
                                S.discard(u)
                                S.add(v)
                                current_score += delta
                                if current_score > best_score:
                                    best_score = current_score
                                    best_S = set(S)
                                break
    # Return the best set found
    return sorted(best_S)
