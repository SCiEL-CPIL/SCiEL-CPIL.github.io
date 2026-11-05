import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # 1. Precompute GCDs
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]

    # 2. Identify Bad Triples using integer bitmasks for speed
    pair_index = {}
    idx = 0
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            pair_index[(a, b)] = idx
            idx += 1
    num_pairs = len(pair_index)
    
    conflict_mask = [0] * (N + 1)
    
    for a, b, c in combinations(numbers, 3):
        g_ab = gcd_table[a][b]
        g_ac = gcd_table[a][c]
        g_bc = gcd_table[b][c]
        if g_ab == g_ac == g_bc:
            conflict_mask[a] |= (1 << pair_index[(b, c)])
            conflict_mask[b] |= (1 << pair_index[(a, c)])
            conflict_mask[c] |= (1 << pair_index[(a, b)])

    # Precompute degrees
    degrees = [bin(conflict_mask[v]).count('1') for v in range(N + 1)]
    
    # 3. Greedy Construction - pick vertices with lowest conflict degree first
    candidates = sorted(numbers, key=lambda x: degrees[x])
    
    S = set()
    active_mask = 0
    
    def can_add(v, active):
        return (conflict_mask[v] & active) == 0

    def add_to_S(v):
        nonlocal active_mask
        S.add(v)
        for u in S:
            if u != v:
                a, b = (u, v) if u < v else (v, u)
                active_mask |= (1 << pair_index[(a, b)])

    def rebuild_active(S_set):
        mask = 0
        S_list = sorted(S_set)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                mask |= (1 << pair_index[(S_list[i], S_list[j])])
        return mask

    for v in candidates:
        if time.time() > deadline:
            break
        if can_add(v, active_mask):
            add_to_S(v)

    # 4. Local Search with add, 1-swap, and 2-swap phases
    remaining = set(numbers) - S
    best_S = set(S)
    best_size = len(S)
    
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        
        # Phase 1: Try to add a vertex
        for v in sorted(remaining, key=lambda x: degrees[x]):
            if time.time() > deadline - 3.0:
                break
            if can_add(v, active_mask):
                add_to_S(v)
                remaining.remove(v)
                improved = True
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
                break
        
        if improved:
            continue
        
        # Phase 2: Try 1-swap (remove u from S, add v from remaining)
        found_swap = False
        for u in list(S):
            if time.time() > deadline - 3.0:
                break
            # Build mask for S without u
            u_mask = 0
            for w in S:
                if w != u:
                    a, b = (u, w) if u < w else (w, u)
                    u_mask |= (1 << pair_index[(a, b)])
            active_minus_u = active_mask & ~u_mask
            
            for v in sorted(remaining, key=lambda x: degrees[x]):
                if time.time() > deadline - 3.0:
                    break
                if can_add(v, active_minus_u):
                    S.remove(u)
                    S.add(v)
                    remaining.remove(v)
                    remaining.add(u)
                    active_mask = rebuild_active(S)
                    found_swap = True
                    improved = True
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)
                    break
            if found_swap:
                break
        
        if improved:
            continue
        
        # Phase 3: Try 2-swap (remove u1, u2 from S, add v1, v2 from remaining)
        if time.time() < deadline - 8.0:
            found_2swap = False
            S_list = sorted(S)
            rem_list = sorted(remaining, key=lambda x: degrees[x])
            
            for i in range(len(S_list)):
                if time.time() > deadline - 5.0:
                    break
                u1 = S_list[i]
                for j in range(i + 1, len(S_list)):
                    if time.time() > deadline - 5.0:
                        break
                    u2 = S_list[j]
                    
                    S_minus = S - {u1, u2}
                    active_minus = rebuild_active(S_minus)
                    
                    # Find remaining elements addable to S_minus
                    addable = [v for v in rem_list if can_add(v, active_minus)]
                    
                    if len(addable) < 2:
                        continue
                    
                    for k in range(len(addable)):
                        if time.time() > deadline - 5.0:
                            break
                        v1 = addable[k]
                        for l in range(k + 1, len(addable)):
                            if time.time() > deadline - 5.0:
                                break
                            v2 = addable[l]
                            
                            a, b = (v1, v2) if v1 < v2 else (v2, v1)
                            pair_v1v2_bit = 1 << pair_index[(a, b)]
                            ok_both = True
                            for w in S_minus:
                                if conflict_mask[w] & pair_v1v2_bit:
                                    ok_both = False
                                    break
                            
                            if ok_both:
                                S.discard(u1)
                                S.discard(u2)
                                S.add(v1)
                                S.add(v2)
                                remaining.discard(v1)
                                remaining.discard(v2)
                                remaining.add(u1)
                                remaining.add(u2)
                                active_mask = rebuild_active(S)
                                found_2swap = True
                                improved = True
                                if len(S) > best_size:
                                    best_S = set(S)
                                    best_size = len(S)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

    if len(best_S) >= len(S):
        return list(best_S)
    return list(S)
