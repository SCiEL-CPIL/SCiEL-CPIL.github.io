import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # Numbers in [1,255] that share a factor with 255=3*5*17
    candidates = [i for i in range(1, 256) if math.gcd(i, 255) > 1]
    if 255 in candidates:
        candidates.remove(255)
    n = len(candidates)

    # Build adjacency bitmasks: adj[i] has bit j set if gcd(candidates[i], candidates[j])>1
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    def popcount(x):
        return bin(x).count("1")

    # Initial clique: all multiples of 3 (they all share factor 3)
    init_mask = 0
    for i in range(n):
        if candidates[i] % 3 == 0:
            init_mask |= (1 << i)
    best_size = popcount(init_mask)
    best_mask = init_mask

    # Bron-Kerbosch with pivot to find max clique
    def bron_kerbosch(R_mask, P_mask, X_mask):
        nonlocal best_size, best_mask
        if time.time() > DEADLINE:
            return
        if P_mask == 0 and X_mask == 0:
            sz = popcount(R_mask)
            if sz > best_size:
                best_size = sz
                best_mask = R_mask
            return

        # Pivot: choose vertex with max degree in P
        # Try a few candidates for pivot
        p = 0
        max_deg = -1
        tmp = P_mask
        while tmp:
            lsb = tmp & (-tmp)
            idx = lsb.bit_length() - 1
            d = popcount(adj[idx] & P_mask)
            if d > max_deg:
                max_deg = d
                p = idx
            tmp ^= lsb

        # Branch on P \ N(p)
        candidates_p = P_mask & ~(adj[p] | (1 << p))
        if candidates_p == 0:
            candidates_p = P_mask

        tmp = candidates_p
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            new_R = R_mask | lsb
            new_P = P_mask & adj[v]
            new_X = X_mask & adj[v]
            bron_kerbosch(new_R, new_P, new_X)
            P_mask ^= lsb
            X_mask |= lsb
            tmp ^= lsb

    # Run Bron-Kerbosch starting from empty, but use init as lower bound
    # To speed up, we can start with the init clique and try to extend
    # Actually, let's just run full B-K but it might be slow for ~170 nodes
    # Let's use a greedy approach first to get a good bound, then B-K on a reduced graph

    # Greedy: iteratively add vertex with max connections to current clique
    greedy_mask = 0
    remaining = (1 << n) - 1
    while remaining:
        if time.time() > DEADLINE:
            break
        best_v = -1
        best_conn = -1
        tmp = remaining
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            conn = popcount(adj[v] & greedy_mask)
            if conn > best_conn:
                best_conn = conn
                best_v = v
            tmp ^= lsb
        if best_conn < 0:
            break
        greedy_mask |= (1 << best_v)
        remaining &= adj[best_v]

    if popcount(greedy_mask) > best_size:
        best_size = popcount(greedy_mask)
        best_mask = greedy_mask

    # Now try B-K to improve, but limit depth/time
    # For efficiency, only run B-K if n is manageable or use it carefully
    if n <= 100:
        bron_kerbosch(0, (1 << n) - 1, 0)
    else:
        # Use a heuristic: try to extend the best found clique
        # Find vertices not in best_mask that connect to all in best_mask
        ext_candidates = 0
        tmp = (1 << n) - 1
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            if (adj[v] & best_mask) == best_mask:
                ext_candidates |= lsb
            tmp ^= lsb

        if ext_candidates:
            # Try adding one more vertex
            tmp = ext_candidates
            while tmp:
                lsb = tmp & (-tmp)
                v = lsb.bit_length() - 1
                new_mask = best_mask | lsb
                # Check if we can add more
                further = ext_candidates & adj[v]
                if popcount(new_mask) > best_size:
                    best_size = popcount(new_mask)
                    best_mask = new_mask
                tmp ^= lsb

    # Convert best_mask back to list, ensure 255 is included
    result = [255]
    tmp = best_mask
    while tmp:
        lsb = tmp & (-tmp)
        idx = lsb.bit_length() - 1
        result.append(candidates[idx])
        tmp ^= lsb

    return result
# EVOLVE-BLOCK-END
