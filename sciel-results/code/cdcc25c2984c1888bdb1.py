"""Strong algorithm for bounded-lcm set maximization on [1,512]."""
import time
from math import gcd, isqrt

N = 512
DEADLINE = 85.0  # safety margin before 90s kill

def _lcm(a, b):
    return a * b // gcd(a, b)

def _validate(s):
    s = set(s)
    if not s:
        return False
    for x in s:
        if x < 1 or x > N:
            return False
    lst = sorted(s)
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if _lcm(lst[i], lst[j]) > N:
                return False
    return True

def _divisors_of(n, limit=N):
    res = set()
    for i in range(1, isqrt(n) + 1):
        if n % i == 0:
            if i <= limit:
                res.add(i)
            j = n // i
            if j != i and j <= limit:
                res.add(j)
    return res

def construct():
    start = time.time()

    # --- Precompute LCM compatibility matrix using bitmasks ---
    compat_mask = [0] * (N + 1)
    for v in range(1, N + 1):
        mask = 0
        for u in range(1, N + 1):
            if _lcm(v, u) <= N:
                mask |= (1 << (u - 1))
        compat_mask[v] = mask

    def is_compatible(v, S_mask):
        return (S_mask & ~compat_mask[v]) == 0

    def set_to_mask(s):
        m = 0
        for v in s:
            m |= (1 << (v - 1))
        return m

    def mask_to_set(m):
        s = set()
        while m:
            lb = m & (-m)
            v = lb.bit_length()
            s.add(v)
            m &= m - 1
        return s

    def greedy_augment(seed_set):
        S = set(seed_set)
        S_mask = set_to_mask(S)
        changed = True
        while changed:
            changed = False
            for v in range(1, N + 1):
                if v in S:
                    continue
                if is_compatible(v, S_mask):
                    S.add(v)
                    S_mask |= (1 << (v - 1))
                    changed = True
        return S

    def greedy_by_degree(seed_set):
        S = set(seed_set)
        S_mask = set_to_mask(S)
        remaining_mask = 0
        for v in range(1, N + 1):
            if v not in S and is_compatible(v, S_mask):
                remaining_mask |= (1 << (v - 1))
        
        while remaining_mask:
            best_v = None
            best_deg = -1
            r = remaining_mask
            while r:
                lb = r & (-r)
                v = lb.bit_length()
                deg = (compat_mask[v] & remaining_mask).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                r &= r - 1
            
            if best_v is None:
                break
            S.add(best_v)
            S_mask |= (1 << (best_v - 1))
            remaining_mask &= compat_mask[best_v]
            remaining_mask &= ~(1 << (best_v - 1))
        
        return S

    def local_search_improve(S, max_iterations=1000):
        """Try to improve S by swapping elements or adding/removing."""
        S = set(S)
        S_mask = set_to_mask(S)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations and (time.time() - start < DEADLINE - 5):
            improved = False
            iterations += 1
            
            # Try adding any compatible element
            for v in range(1, N + 1):
                if v not in S and is_compatible(v, S_mask):
                    S.add(v)
                    S_mask |= (1 << (v - 1))
                    improved = True
                    break
            
            # Try swapping: remove one element, add a different one
            if not improved:
                for remove_v in list(S):
                    S_remove_mask = S_mask & ~(1 << (remove_v - 1))
                    # Find best candidate to add
                    best_add = None
                    best_score = 0
                    for add_v in range(1, N + 1):
                        if add_v in S or add_v == remove_v:
                            continue
                        if is_compatible(add_v, S_remove_mask):
                            # Score: how many new compatible elements this enables
                            new_compat = compat_mask[add_v] & ~S_remove_mask
                            score = new_compat.bit_count()
                            if score > best_score:
                                best_score = score
                                best_add = add_v
                    if best_add is not None and best_score > 0:
                        S.discard(remove_v)
                        S.add(best_add)
                        S_mask = set_to_mask(S)
                        improved = True
                        break
        
        return S

    # --- Find good starting seeds ---
    candidate_Ms = list(range(1, N + 1))
    
    def div_count(m):
        return len(_divisors_of(m, N))
    
    M_sorted = sorted(candidate_Ms, key=div_count, reverse=True)
    
    best_S = set()
    best_size = 0
    
    # Try top M values as seeds
    tried_M = set()
    for M in M_sorted:
        if time.time() - start > DEADLINE - 15:
            break
        if M in tried_M:
            continue
        tried_M.add(M)
        
        d = _divisors_of(M, N)
        if not d:
            continue
        
        S1 = greedy_augment(d)
        if len(S1) > best_size:
            best_size = len(S1)
            best_S = S1
        
        S2 = greedy_by_degree(d)
        if len(S2) > best_size:
            best_size = len(S2)
            best_S = S2
        
        if len(tried_M) >= 30:
            break
    
    # Try from empty set with different orderings
    if time.time() - start < DEADLINE - 20:
        S3 = greedy_by_degree(set())
        if len(S3) > best_size:
            best_size = len(S3)
            best_S = S3
    
    # Try increasing-order greedy from empty
    if time.time() - start < DEADLINE - 15:
        S = set()
        S_mask = 0
        for v in range(1, N + 1):
            if is_compatible(v, S_mask):
                S.add(v)
                S_mask |= (1 << (v - 1))
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    # Try decreasing-order greedy from empty
    if time.time() - start < DEADLINE - 15:
        S = set()
        S_mask = 0
        for v in range(N, 0, -1):
            if is_compatible(v, S_mask):
                S.add(v)
                S_mask |= (1 << (v - 1))
        if len(S) > best_size:
            best_size = len(S)
            best_S = S

    # --- Local Search Polish ---
    if time.time() - start < DEADLINE - 5:
        polished_S = local_search_improve(best_S, max_iterations=200)
        if len(polished_S) > best_size:
            best_size = len(polished_S)
            best_S = polished_S

    # Final polish: try to add one more element to best_S
    if time.time() - start < DEADLINE - 2:
        S_mask = set_to_mask(best_S)
        for v in range(1, N + 1):
            if v not in best_S and is_compatible(v, S_mask):
                best_S.add(v)
                break
    
    # Validate
    if not _validate(best_S):
        # Fallback
        best_S = set(range(1, 21))
    
    return list(best_S)
