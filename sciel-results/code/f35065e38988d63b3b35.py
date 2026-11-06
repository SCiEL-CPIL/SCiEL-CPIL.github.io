import time
import math

_START = time.time()
_HARD_DEADLINE = _START + 85.0


def _prime_factors(num):
    if num <= 1:
        return frozenset()
    out = []
    x = num
    if x % 2 == 0:
        out.append(2)
        while x % 2 == 0:
            x //= 2
    f = 3
    while f * f <= x:
        if x % f == 0:
            out.append(f)
            while x % f == 0:
                x //= f
        f += 2
    if x > 1:
        out.append(x)
    return frozenset(out)


def _smooth_numbers(max_val, P):
    """Generate all P-smooth numbers up to max_val."""
    result = set()
    
    def gen(idx, current):
        if current > max_val:
            return
        if idx == len(P):
            result.add(current)
            return
        p = P[idx]
        gen(idx + 1, current)
        val = current
        while val * p <= max_val:
            val *= p
            gen(idx, val)
    
    gen(0, 1)
    return sorted(result)


def construct():
    deadline = time.time() + 85.0
    n = 12
    MAXV = 10000
    MAX_SUM = 2 * MAXV

    # Precompute prime factors for all possible sums
    pf_cache = {s: _prime_factors(s) for s in range(2, MAX_SUM + 1)}

    def count_distinct_primes(arr):
        union = set()
        for i in range(n - 1):
            ai = arr[i]
            for j in range(i + 1, n):
                union |= pf_cache[ai + arr[j]]
        return len(union)

    best_result = None
    best_score = 10**9

    # Strategy: Try small prime sets, find 12 numbers with all pairwise sums P-smooth
    prime_sets_to_try = [
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
    ]

    for P in prime_sets_to_try:
        if time.time() > deadline - 30.0:
            break
        
        # Get all P-smooth sums up to MAX_SUM
        smooth_sums = _smooth_numbers(MAX_SUM, P)
        smooth_sums_set = set(smooth_sums)
        
        # Build candidate pool: numbers that have at least one smooth-sum partner
        candidate_pool = []
        for a in range(1, MAXV + 1):
            has_partner = False
            for b in range(max(1, a - 200), min(MAXV, a + 200) + 1):
                if b == a:
                    continue
                if (a + b) in smooth_sums_set:
                    has_partner = True
                    break
            if has_partner:
                candidate_pool.append(a)
        
        if len(candidate_pool) < n:
            continue
        
        # Limit pool size
        if len(candidate_pool) > 500:
            def partner_count(a):
                cnt = 0
                for b in candidate_pool:
                    if b != a and (a + b) in smooth_sums_set:
                        cnt += 1
                return cnt
            candidate_pool.sort(key=partner_count, reverse=True)
            candidate_pool = candidate_pool[:500]
        
        # Build adjacency
        adj = {a: set() for a in candidate_pool}
        for i, a in enumerate(candidate_pool):
            for j in range(i + 1, len(candidate_pool)):
                b = candidate_pool[j]
                if (a + b) in smooth_sums_set:
                    adj[a].add(b)
                    adj[b].add(a)
        
        # Bron-Kerbosch with pivot
        found_clique = []
        
        def bron_kerbosch(R, P_set, X):
            if time.time() > deadline - 5.0:
                return
            if len(R) >= n:
                found_clique.extend(list(R))
                return
            if not P_set and not X:
                return
            U = P_set | X
            if not U:
                pivot = None
            else:
                pivot = max(U, key=lambda v: len(adj[v] & P_set))
            candidates = P_set - (adj[pivot] if pivot else set())
            for v in list(candidates):
                bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v])
                P_set.discard(v)
                X.add(v)
                if len(found_clique) >= n:
                    return
        
        bron_kerbosch(set(), set(candidate_pool), set())
        
        if len(found_clique) >= n:
            arr = sorted(found_clique[:n])
            sc = count_distinct_primes(arr)
            if sc < best_score:
                best_score = sc
                best_result = list(arr)
                if best_score <= 5:
                    return list(best_result)
    
    # Fallback: local search from good seeds
    if best_result is None:
        seeds = []
        seeds.append([30 * i for i in range(1, 13)])
        pool = set()
        for a in range(0, 14):
            v2 = 2 ** a
            if v2 > MAXV:
                break
            for b in range(0, 9):
                v3 = v2 * (3 ** b)
                if v3 > MAXV:
                    break
                for c in range(0, 7):
                    v5 = v3 * (5 ** c)
                    if v5 > MAXV:
                        break
                    for d in range(0, 6):
                        v7 = v5 * (7 ** d)
                        if v7 > MAXV:
                            break
                        if v7 >= 1:
                            pool.add(v7)
        pool = sorted(pool)
        if len(pool) >= 12:
            seeds.append(pool[:12])
        seeds.append(list(range(1, 13)))
        
        for s in seeds:
            if len(set(s)) != n:
                continue
            if any(x > MAXV or x <= 0 for x in s):
                continue
            sc = count_distinct_primes(s)
            if sc < best_score:
                best_score = sc
                best_result = list(s)
    
    # Local search refinement
    if best_result is not None and time.time() < deadline - 10.0:
        current = list(best_result)
        improved = True
        iters = 0
        max_iters = 200
        
        local_pool = set()
        for a in range(0, 14):
            v2 = 2 ** a
            if v2 > MAXV:
                break
            for b in range(0, 9):
                v3 = v2 * (3 ** b)
                if v3 > MAXV:
                    break
                for c in range(0, 7):
                    v5 = v3 * (5 ** c)
                    if v5 > MAXV:
                        break
                    for d in range(0, 6):
                        v7 = v5 * (7 ** d)
                        if v7 > MAXV:
                            break
                        if v7 >= 1:
                            local_pool.add(v7)
        for i in range(1, 201):
            local_pool.add(i)
        local_pool = sorted(local_pool)
        
        while improved and iters < max_iters and time.time() < deadline - 5.0:
            improved = False
            iters += 1
            cur_score = count_distinct_primes(current)
            
            for idx in range(n):
                if time.time() > deadline - 5.0:
                    return list(best_result)
                
                old_val = current[idx]
                existing = set(current)
                
                best_cand = None
                best_cand_score = cur_score
                
                for cand in local_pool:
                    if cand == old_val or cand in existing:
                        continue
                    if cand > MAXV:
                        break
                    
                    current[idx] = cand
                    sc = count_distinct_primes(current)
                    
                    if sc < best_cand_score:
                        best_cand_score = sc
                        best_cand = cand
                    else:
                        current[idx] = old_val
                
                if best_cand is not None:
                    current[idx] = best_cand
                    cur_score = best_cand_score
                    improved = True
                    if cur_score < best_score:
                        best_score = cur_score
                        best_result = list(current)
                    if best_score <= 5:
                        return list(best_result)
                    break
    
    if best_result is None:
        return list(range(1, 13))
    
    return list(best_result)
