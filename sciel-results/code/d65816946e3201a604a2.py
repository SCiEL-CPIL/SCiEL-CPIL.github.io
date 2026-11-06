def construct():
    import time
    start = time.time()
    DEADLINE = 85.0  # safety margin before 90s
    
    N = 256
    # Numbers 2..256
    # Build incompatibility: inc[a][b] = (lcm(a,b) <= 256)
    # We'll use arrays indexed by value.
    
    from math import gcd, lcm as _lcm
    # Precompute lcm for pairs? 255*255/2 ~ 32500 pairs, fast.
    
    # Actually, let's index by the actual values for clarity.
    # S_small = 2..128, B = 129..256.
    
    # Precompute for each a in 2..256:
    #   inc_small[a] = list of b in 2..128 with lcm(a,b) <= 256 (for a in 2..128)
    #   multB[a] = list of b in 129..256 with a | b
    
    # Let me just build full incompatibility among 2..256 for the final validity check,
    # but for the algorithm use the S/B decomposition.
    
    small = list(range(2, 129))  # 2..128
    big = list(range(129, 257))  # 129..256
    small_set = set(small)
    big_set = set(big)
    
    # inc_small: for a in small, which small b are incompatible
    inc_small = {a: [] for a in small}
    for a in small:
        for b in small:
            if a < b and _lcm(a, b) <= 256:
                inc_small[a].append(b)
                inc_small[b].append(a)
    
    # multB: for a in small, which big b does a divide
    multB = {a: [] for a in small}
    for a in small:
        for b in big:
            if b % a == 0:
                multB[a].append(b)
    
    # weights
    w = {a: 1.0/a for a in small}
    wB = {b: 1.0/b for b in big}
    
    # value of a set A (subset of small): sum_{a in A} 1/a - sum_{b in union multB[a]} 1/b
    # We'll track A as a set, and N as the set of big multiples.
    
    best_val = -1.0
    best_A = None
    best_N = None
    
    # Greedy initial solution
    def compute_value(A, N):
        v = sum(1.0/a for a in A)
        v -= sum(1.0/b for b in N)
        return v
    
    # Greedy: repeatedly add the compatible a with best marginal gain
    A = set()
    N = set()
    remaining = set(small)
    improved = True
    while improved:
        improved = False
        best_gain = 0.0
        best_a = None
        for a in remaining:
            # check compatibility with all in A
            if any(b in inc_small[a] for b in A):
                continue
            # marginal gain
            new_N = N | set(multB[a])
            gain = 1.0/a - sum(1.0/b for b in new_N - N)
            if gain > best_gain:
                best_gain = gain
                best_a = a
        if best_a is not None and best_gain > 1e-12:
            A.add(best_a)
            N |= set(multB[best_a])
            remaining.remove(best_a)
            improved = True
    
    val = compute_value(A, N)
    if val > best_val:
        best_val = val
        best_A = set(A)
        best_N = set(N)
    
    # Local search: try adding any compatible a with positive gain (already done in greedy until none), 
    # try swaps: remove a, add b
    # ... (implement a few passes)
    
    # B&B with time limit
    # ...
    
    # Build final solution
    S_final = set(best_A) | (set(big) - set(best_N))
    # Validity check
    # ...
    return list(S_final)
