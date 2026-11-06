import time
import sys
from math import gcd

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin before 60s limit

    N = 256
    vertices = list(range(2, N + 1))
    n = len(vertices)
    # index i corresponds to vertex i+2

    # Precompute incompatibility mask:
    # incompat[i] has bit j set if lcm(vertices[i], vertices[j]) <= N
    # We only need to know if a candidate conflicts with the current set
    incompat = [0] * n
    for i in range(n):
        a = vertices[i]
        for j in range(i + 1, n):
            b = vertices[j]
            g = gcd(a, b)
            if a * b // g <= N:
                incompat[i] |= (1 << j)
                incompat[j] |= (1 << i)

    weights = [1.0 / v for v in vertices]

    def get_value(mask):
        s = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += weights[idx]
            m ^= lsb
        return s

    def greedy_add(S, order, forbidden=0):
        """Add vertices in order that are compatible with S and not forbidden."""
        new_mask = S
        for v in order:
            if forbidden & (1 << v):
                continue
            if new_mask & (1 << v):
                continue
            if (new_mask & incompat[v]) == 0:
                new_mask |= (1 << v)
        return new_mask

    def local_search_1swap(S):
        """Try removing each vertex and greedily refilling. Return improved set."""
        best_val = get_value(S)
        best_mask = S
        m = S
        while m:
            lsb = m & -m
            u_idx = lsb.bit_length() - 1
            m ^= lsb

            S_candidate = S & ~lsb
            # Greedy refill
            new_mask = greedy_add(S_candidate, range(n))
            val_candidate = get_value(new_mask)

            if val_candidate > best_val:
                best_val = val_candidate
                best_mask = new_mask

        return best_mask

    def local_search_1for2(S):
        """Try removing one vertex and adding two others."""
        best_val = get_value(S)
        best_mask = S
        m = S
        while m:
            lsb = m & -m
            u_idx = lsb.bit_length() - 1
            m ^= lsb

            S_candidate = S & ~lsb
            # Try adding pairs
            # First, find candidates compatible with S_candidate
            candidates = []
            for v in range(n):
                if S_candidate & (1 << v):
                    continue
                if (S_candidate & incompat[v]) == 0:
                    candidates.append(v)

            # Try all pairs of candidates
            for i, a in enumerate(candidates):
                for j in range(i + 1, len(candidates)):
                    b = candidates[j]
                    if (1 << a) & (1 << b):
                        continue
                    # Check a and b are compatible with each other
                    if (incompat[a] & (1 << b)) or (incompat[b] & (1 << a)):
                        continue
                    new_mask = S_candidate | (1 << a) | (1 << b)
                    val_candidate = get_value(new_mask)
                    if val_candidate > best_val:
                        best_val = val_candidate
                        best_mask = new_mask

        return best_mask

    best_mask = 0
    best_val = 0.0

    # Phase 1: Multiple greedy starts with different exclusion prefixes
    max_prefix = 15
    for prefix_len in range(max_prefix + 1):
        if time.time() > deadline:
            break

        forbidden_mask = 0
        for i in range(prefix_len):
            forbidden_mask |= (1 << i)

        S = greedy_add(0, range(n), forbidden_mask)
        val = get_value(S)
        if val > best_val:
            best_val = val
            best_mask = S

    # Phase 2: Local search with 1-swap and 1-for-2 moves
    improved = True
    iteration = 0
    while improved and time.time() < deadline:
        improved = False
        iteration += 1

        # 1-swap local search
        new_best = local_search_1swap(best_mask)
        if get_value(new_best) > best_val:
            best_mask = new_best
            best_val = get_value(best_mask)
            improved = True
            continue

        # 1-for-2 local search
        new_best = local_search_1for2(best_mask)
        if get_value(new_best) > best_val:
            best_mask = new_best
            best_val = get_value(best_mask)
            improved = True

    # Extract result
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(vertices[idx])
        m ^= lsb

    if not result:
        return [2]

    return result
