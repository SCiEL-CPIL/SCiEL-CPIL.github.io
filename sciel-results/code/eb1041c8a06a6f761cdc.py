import time
import itertools
import math

_START = time.time()

def _spf_sieve(limit):
    spf = list(range(limit+1))
    if limit >= 0: spf[0] = 0
    if limit >= 1: spf[1] = 1
    for i in range(2, int(limit**0.5)+1):
        if spf[i] == i:
            for j in range(i*i, limit+1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

_SPF = _spf_sieve(20000)

PRIMES30 = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113]
P2IDX = {p:i for i,p in enumerate(PRIMES30)}

def _build_M(limit):
    M = [-1]*(limit+1)
    M[1] = 0
    for x in range(2, limit+1):
        t = x
        m = 0
        ok = True
        while t > 1:
            p = _SPF[t]
            idx = P2IDX.get(p)
            if idx is None:
                ok = False
                break
            m |= 1 << idx
            while t % p == 0:
                t //= p
        if ok:
            M[x] = m
    return M

_M = _build_M(20000)

def _is_smooth(x, Pmask):
    m = _M[x]
    if m < 0:
        return False
    return (m & ~Pmask) == 0

def _smooth_list(Pmask, limit=10000):
    return [x for x in range(1, limit+1) if _is_smooth(x, Pmask)]

def _find_clique(A, Pmask, k, node_budget, deadline):
    n = len(A)
    if n < k:
        return None
    # Build adjacency and compute degrees
    adj = [0]*n
    for i in range(n):
        ai = A[i]
        for j in range(i+1, n):
            if _is_smooth(ai + A[j], Pmask):
                adj[i] |= (1<<j)
                adj[j] |= (1<<i)
    
    # Compute degree of each vertex
    deg = [adj[i].bit_count() for i in range(n)]
    
    # Only keep vertices with sufficient degree
    min_deg_needed = k - 1
    active = [i for i in range(n) if deg[i] >= min_deg_needed]
    if len(active) < k:
        return None
    
    # Remap to 0..m-1 where m = len(active)
    m = len(active)
    old_to_new = {old: new for new, old in enumerate(active)}
    
    # Build new adjacency in compact form
    new_adj = [0]*m
    for i in range(m):
        for j in range(i+1, m):
            oi, oj = active[i], active[j]
            if adj[oi] & (1 << oj):
                new_adj[i] |= (1 << j)
                new_adj[j] |= (1 << i)
    
    # Precompute higher masks
    higher = [0]*m
    for v in range(m-2, -1, -1):
        higher[v] = higher[v+1] | (1 << (v+1))
    
    nodes = [0]
    aborted = [False]
    import time as _t
    
    def dfs(cand_mask, need, path):
        if need == 0:
            return list(path)
        if cand_mask.bit_count() < need:
            return None
        nodes[0] += 1
        if nodes[0] > node_budget or _t.time() > deadline:
            aborted[0] = True
            return None
        
        # Get candidates sorted by degree (within current candidate set)
        msk = cand_mask
        cands = []
        while msk:
            lsb = msk & (-msk)
            v = lsb.bit_length() - 1
            nc_count = (cand_mask & new_adj[v] & higher[v]).bit_count()
            if nc_count + 1 >= need:
                cands.append((nc_count, v))
            msk &= msk - 1
        
        # Sort by degree descending (try high-degree vertices first)
        cands.sort(key=lambda x: -x[0])
        
        for nc_count, v in cands:
            nc = cand_mask & new_adj[v] & higher[v]
            res = dfs(nc, need-1, path + [v])
            if res is not None:
                return res
        
        return None
    
    res = dfs((1<<m)-1, k, [])
    if res is None:
        return None
    vals = [A[active[v]] for v in res]
    # verify
    for i in range(k):
        for j in range(i+1, k):
            if not _is_smooth(vals[i]+vals[j], Pmask):
                return None
    return vals

def _greedy_clique(A, Pmask, k):
    """Fast greedy heuristic to find a clique of size k."""
    n = len(A)
    if n < k:
        return None
    
    # Compute degrees
    adj = [0]*n
    for i in range(n):
        ai = A[i]
        for j in range(i+1, n):
            if _is_smooth(ai + A[j], Pmask):
                adj[i] |= (1<<j)
                adj[j] |= (1<<i)
    
    deg = [adj[i].bit_count() for i in range(n)]
    
    # Greedy: repeatedly pick vertex with max degree in remaining set
    selected = []
    remaining_mask = (1 << n) - 1
    
    while len(selected) < k and remaining_mask:
        # Find vertex with max degree within remaining set
        best_v = -1
        best_deg = -1
        m = remaining_mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            d = (remaining_mask & adj[v]).bit_count()
            if d > best_deg:
                best_deg = d
                best_v = v
            m &= m - 1
        
        if best_v == -1 or best_deg < k - 1 - len(selected):
            break
        
        selected.append(best_v)
        # Remove best_v and its non-neighbors from remaining
        remaining_mask &= adj[best_v]
        remaining_mask &= ~(1 << best_v)
    
    if len(selected) < k:
        return None
    
    vals = [A[v] for v in selected]
    # Verify
    for i in range(k):
        for j in range(i+1, k):
            if not _is_smooth(vals[i]+vals[j], Pmask):
                return None
    return vals

def _try_arithmetic_progression(Pmask, k=12, max_val=10000):
    """Try to find k distinct values in an AP where all pair sums are Pmask-smooth.
    If all values are congruent to r mod m, then all pair sums are 2r mod m.
    We want 2r to be divisible by all primes in Pmask, i.e., 2r ≡ 0 mod (product of odd primes).
    """
    # Extract the set of primes from Pmask
    primes_in_mask = [p for p in PRIMES30 if (Pmask >> P2IDX[p]) & 1]
    if not primes_in_mask:
        return None
    
    # For pair sums to be smooth, we want them divisible by all primes in Pmask
    # If x_i = a + i*d, then x_i + x_j = 2a + (i+j)*d
    # We need 2a + (i+j)*d to be Pmask-smooth for all i≠j in [0,k-1]
    
    # Strategy: pick d as a multiple of the product of primes (or lcm)
    # and a such that 2a is also a multiple
    
    # Simple approach: try small multipliers
    prod_odd = 1
    for p in primes_in_mask:
        if p != 2:
            prod_odd *= p
    
    # d should be a multiple of prod_odd (or at least of each prime)
    # a should be such that 2a is a multiple of prod_odd
    
    # Try: a = prod_odd, d = prod_odd
    # Then x_i = prod_odd * (1 + i), sums = prod_odd * (2 + i + j)
    # Need (2+i+j) to be Pmask-smooth for all i≠j in [0,k-1]
    
    base = prod_odd
    # Check if all values <= max_val
    if base * (1 + k - 1) > max_val:
        # Try smaller: use lcm instead of product
        from math import gcd
        lcm_val = 1
        for p in primes_in_mask:
            lcm_val = lcm_val * p // gcd(lcm_val, p)
        base = lcm_val
        if base * (1 + k - 1) > max_val:
            return None
    
    # Try multiple bases: base * t for small t
    for t in range(1, 20):
        b = base * t
        if b * (1 + k - 1) > max_val:
            break
        
        # Values: b, 2b, 3b, ..., k*b
        # Sums: b*(i+j) for i≠j, i,j in [1,k]
        # Need b*(i+j) to be Pmask-smooth
        # Since b is already smooth (product of primes in Pmask),
        # we need (i+j) to be Pmask-smooth for all i≠j in [1,k]
        
        ok = True
        for i in range(1, k+1):
            for j in range(i+1, k+1):
                s = i + j
                if not _is_smooth(s, Pmask):
                    ok = False
                    break
            if not ok:
                break
        
        if ok:
            vals = [b * i for i in range(1, k+1)]
            # Verify all are distinct and <= max_val
            if len(set(vals)) == k and all(v <= max_val for v in vals):
                return vals
    
    # Alternative: try AP with different structure
    # x_i = a + i*d where a and d are chosen so that 2a+(i+j)d is smooth
    # Try: make 2a+(i+j)d = b * f(i,j) where f is smooth
    # Simpler: try a=0 equivalent (shifted)
    for t in range(1, 10):
        b = base * t
        if b * k > max_val:
            break
        # Try values: b, 2b, ..., kb but shifted
        # Actually try: (t+1)*b, (t+2)*b, ...
        for offset in range(0, 5):
            vals = [b * (i + 1 + offset) for i in range(k)]
            if vals[-1] > max_val:
                break
            ok = True
            for i in range(k):
                for j in range(i+1, k):
                    s = (i+1+offset) + (j+1+offset)
                    if not _is_smooth(s, Pmask):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return vals
    
    return None

def construct():
    deadline = time.time() + 75.0
    primes_first = [2,3,5,7,11,13,17,19,23,29,31,37,41,43]
    
    def pmask_from_list(P):
        m = 0
        for p in P:
            m |= 1 << P2IDX[p]
        return m
    
    best_result = None
    best_prime_count = 99
    
    # Strategy 0: Try arithmetic progression construction for small prime sets
    for k in range(4, 9):
        if time.time() > deadline - 10:
            break
        P = primes_first[:k]
        pm = pmask_from_list(P)
        res = _try_arithmetic_progression(pm, 12, 10000)
        if res is not None:
            # Verify
            valid = True
            for i in range(12):
                for j in range(i+1, 12):
                    if not _is_smooth(res[i] + res[j], pm):
                        valid = False
                        break
                if not valid:
                    break
            if valid and len(set(res)) == 12 and all(1 <= x <= 10000 for x in res):
                best_result = res
                best_prime_count = k
                return list(res)
    
    # Strategy 1: Try smallest prime sets first with prefix primes
    for k in range(3, 11):
        if time.time() > deadline - 8:
            break
        P = primes_first[:k]
        pm = pmask_from_list(P)
        
        # Try AP first
        res = _try_arithmetic_progression(pm, 12, 10000)
        if res is not None:
            valid = True
            for i in range(12):
                for j in range(i+1, 12):
                    if not _is_smooth(res[i] + res[j], pm):
                        valid = False
                        break
                if not valid:
                    break
            if valid and len(set(res)) == 12 and all(1 <= x <= 10000 for x in res):
                return list(res)
        
        L = _smooth_list(pm)
        if len(L) < 12:
            continue
        
        # Try different subset sizes, starting with smaller for speed
        Ts = [50, 100, 150, 200, 300, 400, 500, 600, 800, 1000]
        Ts = [t for t in Ts if t <= len(L)]
        if len(L) not in Ts and len(L) > 1000:
            Ts.append(len(L))
        
        for T in Ts:
            if time.time() > deadline - 8:
                break
            A = L[:T]
            
            # First try fast greedy
            res = _greedy_clique(A, pm, 12)
            if res is not None:
                if best_result is None or k < best_prime_count:
                    best_result = res
                    best_prime_count = k
                return list(res)
            
            remaining = deadline - time.time()
            sub_deadline = min(time.time() + max(3.0, remaining * 0.3), deadline - 8.0)
            budget = 500000
            res = _find_clique(A, pm, 12, budget, sub_deadline)
            if res is not None:
                if best_result is None or k < best_prime_count:
                    best_result = res
                    best_prime_count = k
                return list(res)
    
    # Strategy 2: Alternative 7-prime sets - try more combinations
    alts7 = [
        [2,3,5,7,11,13,17],
        [2,3,5,7,11,13,19],
        [2,3,5,7,11,13,23],
        [2,3,5,7,11,17,19],
        [2,3,5,7,13,17,19],
        [2,3,5,11,13,17,19],
        [2,3,7,11,13,17,19],
        [2,5,7,11,13,17,19],
        [2,3,5,7,11,13,29],
        [2,3,5,7,11,17,23],
        [2,3,5,7,13,17,23],
        [2,3,5,7,11,19,23],
        [2,3,5,7,13,19,23],
        [2,3,5,7,11,13,31],
        [2,3,5,7,11,17,29],
        [2,3,5,7,11,13,37],
        [2,3,5,7,11,17,31],
        [2,3,5,7,13,17,29],
        [2,3,5,7,11,19,29],
    ]
    for P in alts7:
        if time.time() > deadline - 8:
            break
        pm = pmask_from_list(P)
        
        # Try AP first
        res = _try_arithmetic_progression(pm, 12, 10000)
        if res is not None:
            valid = True
            for i in range(12):
                for j in range(i+1, 12):
                    if not _is_smooth(res[i] + res[j], pm):
                        valid = False
                        break
                if not valid:
                    break
            if valid and len(set(res)) == 12 and all(1 <= x <= 10000 for x in res):
                return list(res)
        
        L = _smooth_list(pm)
        if len(L) < 12:
            continue
        for T in [150, 300, 500, 700, len(L)]:
            if T > len(L): T = len(L)
            if time.time() > deadline - 8:
                break
            A = L[:T]
            
            # Try greedy first
            res = _greedy_clique(A, pm, 12)
            if res is not None:
                return list(res)
            
            remaining = deadline - time.time()
            sub_deadline = min(time.time() + max(2.0, remaining * 0.2), deadline - 8.0)
            res = _find_clique(A, pm, 12, 400000, sub_deadline)
            if res is not None:
                return list(res)
    
    # Strategy 3: Alternative 8-prime sets
    alts8 = [
        [2,3,5,7,11,13,17,19],
        [2,3,5,7,11,13,17,23],
        [2,3,5,7,11,13,19,23],
        [2,3,5,7,11,17,19,23],
        [2,3,5,7,13,17,19,23],
        [2,3,5,11,13,17,19,23],
        [2,3,5,7,11,13,17,29],
        [2,3,5,7,11,13,19,29],
        [2,3,5,7,11,13,17,31],
        [2,3,5,7,11,13,23,29],
        [2,3,5,7,11,17,19,29],
        [2,3,5,7,13,17,19,29],
    ]
    for P in alts8:
        if time.time() > deadline - 8:
            break
        pm = pmask_from_list(P)
        
        # Try AP first
        res = _try_arithmetic_progression(pm, 12, 10000)
        if res is not None:
            valid = True
            for i in range(12):
                for j in range(i+1, 12):
                    if not _is_smooth(res[i] + res[j], pm):
                        valid = False
                        break
                if not valid:
                    break
            if valid and len(set(res)) == 12 and all(1 <= x <= 10000 for x in res):
                return list(res)
        
        L = _smooth_list(pm)
        if len(L) < 12:
            continue
        for T in [200, 400, 600, len(L)]:
            if T > len(L): T = len(L)
            if time.time() > deadline - 8:
                break
            A = L[:T]
            
            # Try greedy first
            res = _greedy_clique(A, pm, 12)
            if res is not None:
                return list(res)
            
            remaining = deadline - time.time()
            sub_deadline = min(time.time() + max(2.0, remaining * 0.2), deadline - 8.0)
            res = _find_clique(A, pm, 12, 400000, sub_deadline)
            if res is not None:
                return list(res)
    
    # Fallback known-good 8-prime set
    if best_result is not None:
        return list(best_result)
    return [1,2,3,5,6,9,10,15,18,25,27,30]
