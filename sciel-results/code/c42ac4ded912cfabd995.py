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
            s.add(lb.bit_length())
            m &= m - 1
        return s

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

    def fast_local_search(S):
        """Efficient local search using bitmask operations."""
        S_mask = set_to_mask(S)
        S_list = sorted(S)
        improved = True
        while improved:
            improved = False
            # Try 1-add: find any compatible element not in S
            all_mask = (1 << N) - 1
            candidates = all_mask & ~S_mask
            for v in range(1, N + 1):
                if v not in S and is_compatible(v, S_mask):
                    S.add(v)
                    S_mask |= (1 << (v - 1))
                    improved = True
                    break
            if improved:
                continue
            # Try 1-swap: remove x, add y where y is compatible with S\{x}
            for x in S_list:
                x_mask = 1 << (x - 1)
                S_minus_x_mask = S_mask & ~x_mask
                # Find y not in S, compatible with S\{x}
                for y in range(1, N + 1):
                    if y in S:
                        continue
                    if is_compatible(y, S_minus_x_mask):
                        # Accept swap if it doesn't reduce size (it won't) 
                        # Prefer swaps that increase future compatibility
                        S.discard(x)
                        S.add(y)
                        S_mask = S_mask & ~x_mask | (1 << (y - 1))
                        S_list = sorted(S)
                        improved = True
                        break
                if improved:
                    break
        return S

    def two_opt_search(S):
        """Try removing two elements and adding two different ones."""
        S_mask = set_to_mask(S)
        S_list = sorted(S)
        improved = True
        while improved:
            improved = False
            # First try 1-add
            for v in range(1, N + 1):
                if v not in S and is_compatible(v, S_mask):
                    S.add(v)
                    S_mask |= (1 << (v - 1))
                    improved = True
                    break
            if improved:
                continue
            # Try 2-remove, 2-add
            n = len(S_list)
            for i in range(n):
                for j in range(i + 1, n):
                    x, y = S_list[i], S_list[j]
                    xy_mask = (1 << (x - 1)) | (1 << (y - 1))
                    S_minus_xy_mask = S_mask & ~xy_mask
                    # Try to find two new elements a, b compatible with S\{x,y} and with each other
                    for a in range(1, N + 1):
                        if a in S:
                            continue
                        if not is_compatible(a, S_minus_xy_mask):
                            continue
                        new_mask_a = S_minus_xy_mask | (1 << (a - 1))
                        for b in range(1, N + 1):
                            if b in S or b == a:
                                continue
                            if is_compatible(b, new_mask_a) and is_compatible(a, (1 << (b - 1))):
                                # Accept
                                S.discard(x)
                                S.discard(y)
                                S.add(a)
                                S.add(b)
                                S_mask = set_to_mask(S)
                                S_list = sorted(S)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return S

    # --- Find good starting seeds using divisor structure ---
    M_sorted = sorted(range(1, N + 1), key=lambda m: len(_divisors_of(m, N)), reverse=True)
    
    best_S = set()
    best_size = 0
    
    # Try top M values as seeds with degree-based greedy
    for M in M_sorted[:50]:
        if time.time() - start > DEADLINE - 20:
            break
        d = _divisors_of(M, N)
        if not d:
            continue
        
        S2 = greedy_by_degree(d)
        if len(S2) > best_size:
            best_size = len(S2)
            best_S = S2
    
    # Also try from empty set with different orderings
    if time.time() - start < DEADLINE - 25:
        S3 = greedy_by_degree(set())
        if len(S3) > best_size:
            best_size = len(S3)
            best_S = S3
    
    # Try increasing-order greedy
    if time.time() - start < DEADLINE - 20:
        S = set()
        S_mask = 0
        for v in range(1, N + 1):
            if is_compatible(v, S_mask):
                S.add(v)
                S_mask |= (1 << (v - 1))
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    # Try decreasing-order greedy
    if time.time() - start < DEADLINE - 18:
        S = set()
        S_mask = 0
        for v in range(N, 0, -1):
            if is_compatible(v, S_mask):
                S.add(v)
                S_mask |= (1 << (v - 1))
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    # Try power-of-2 based greedy
    if time.time() - start < DEADLINE - 16:
        S = set()
        S_mask = 0
        powers = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        for p in powers:
            for v in range(1, N + 1):
                if v not in S and (v & (v - 1)) == 0 or True:
                    if is_compatible(v, S_mask):
                        S.add(v)
                        S_mask |= (1 << (v - 1))
                        break
        # Then fill in with degree-based
        S = greedy_by_degree(S)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S

    # Local search polish: fast 1-swap search
    if time.time() - start < DEADLINE - 10:
        polished_S = fast_local_search(best_S)
        if len(polished_S) > best_size:
            best_size = len(polished_S)
            best_S = polished_S
    
    # 2-opt search
    if time.time() - start < DEADLINE - 5:
        polished_S = two_opt_search(best_S)
        if len(polished_S) > best_size:
            best_size = len(polished_S)
            best_S = polished_S
    
    # Final polish: try to add one more element
    if time.time() - start < DEADLINE - 2:
        S_mask = set_to_mask(best_S)
        for v in range(1, N + 1):
            if v not in best_S and is_compatible(v, S_mask):
                best_S.add(v)
                break
    
    # Validate
    if not _validate(best_S):
        best_S = set(range(1, 21))
    
    return list(best_S)
