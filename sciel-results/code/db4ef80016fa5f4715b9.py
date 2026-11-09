import numpy as np
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    
    # 1. Decompose using CRT
    # 205 = 5 * 41
    # Forbidden differences mod 205 are non-zero squares.
    # d is a non-zero square mod 205 iff:
    # d_5 is a square mod 5 (0, 1, 4) AND d_41 is a square mod 41 (0, 1, 4, 5, 9, 16, 20, 25, 36)
    # AND not (d_5 == 0 and d_41 == 0)
    
    # We construct a set S such that for any a, b in S:
    # (a-b)_5 is NOT in {0, 1, 4} OR (a-b)_41 is NOT in {0, 1, 4, 5, 9, 16, 20, 25, 36}
    
    # Strategy: Use residues 0, 2, 4 mod 5.
    # Differences between these residues:
    # 0-2=3 (non-square), 0-4=1 (square), 2-4=3 (non-square)
    # So:
    # - Pairs from S_0 and S_2: Free (diff 3 or 2, non-square)
    # - Pairs from S_2 and S_4: Free (diff 3 or 2, non-square)
    # - Pairs from S_0 and S_4: Constrained (diff 1 or 4, square) -> must have non-square diff mod 41
    # - Internal pairs (S_i, S_i): Constrained (diff 0, square) -> must have non-square diff mod 41
    
    # So we need:
    # T_0: Independent set in G_41
    # T_2: Independent set in G_41
    # T_4: Independent set in G_41 AND T_4 is "far" from T_0 (no diff in Q'_41)
    
    # We can pick T_0 = T_opt (MIS of G_41)
    # T_2 = T_opt
    # T_4 = T_opt (since T_opt is independent, and T_opt is not in N(T_opt) because 0 is not in Q'_41? 
    # Wait. If u, v in T_opt, u-v is not in Q'_41.
    # The constraint for T_4 vs T_0 is: for all u in T_4, v in T_0, u-v not in Q'_41.
    # If T_4 = T_0, then for u, v in T_0, u-v not in Q'_41. This holds.
    # So T_4 = T_0 is valid.
    
    # Thus, we can just find one Maximum Independent Set T_opt in G_41.
    # Then S = {x in Z_205 | x mod 5 in {0, 2, 4} and x mod 41 in T_opt}
    # Size = 3 * |T_opt|
    
    # 2. Compute Q_41
    Q_41 = set()
    for x in range(1, 41):
        Q_41.add(pow(x, 2, 41))
    # Q_41 contains non-zero squares.
    # Note: 0 is not in Q_41 because x=1..40.
    
    # 3. Build Graph G_41
    n = 41
    adj = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and (i - j) % n in Q_41:
                mask |= (1 << j)
        adj[i] = mask
        
    # 4. Find MIS of G_41
    # Use B&B
    best_size = 0
    best_set = 0
    deadline = time.time() + 5 # 5 seconds is plenty for n=41
    
    # Ordering: sort by degree descending
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    order = list(range(n))
    order.sort(key=lambda x: degrees[x], reverse=True)
    
    def greedy_color_upper_bound(candidates):
        if candidates == 0:
            return 0
        remaining = candidates
        colors = 0
        while remaining:
            colors += 1
            v = -1
            for u in order:
                if (remaining >> u) & 1:
                    v = u
                    break
            if v == -1:
                break
            clique = 1 << v
            compatible = remaining & adj[v]
            while compatible:
                u = -1
                for w in order:
                    if (compatible >> w) & 1:
                        u = w
                        break
                if u == -1:
                    break
                clique |= (1 << u)
                compatible &= adj[u]
            remaining &= ~clique
        return colors

    def branch_and_bound(candidates, current_set, current_size):
        nonlocal best_size, best_set, deadline
        if time.time() > deadline:
            return
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return
        ub = greedy_color_upper_bound(candidates)
        if current_size + ub <= best_size:
            return
        v = -1
        for u in order:
            if (candidates >> u) & 1:
                v = u
                break
        if v == -1:
            return
        include_candidates = candidates & ~(adj[v] | (1 << v))
        branch_and_bound(include_candidates, current_set | (1 << v), current_size + 1)
        exclude_candidates = candidates & ~(1 << v)
        branch_and_bound(exclude_candidates, current_set, current_size)

    all_mask = (1 << n) - 1
    branch_and_bound(all_mask, 0, 0)
    
    # Extract T_opt
    T_opt = []
    for i in range(n):
        if (best_set >> i) & 1:
            T_opt.append(i)
            
    # 5. Construct S in Z_205
    result = []
    # Inverse of 5 mod 41
    inv_5 = 33 # 5 * 33 = 165 = 4 * 41 + 1
    
    for t in T_opt:
        for r in [0, 2, 4]:
            # x = r + 5k
            # r + 5k = t mod 41
            # 5k = t - r mod 41
            k = ((t - r) * inv_5) % 41
            x = r + 5 * k
            result.append(x)
            
    # Convert to numpy array
    return np.array(result, dtype=np.int64)
