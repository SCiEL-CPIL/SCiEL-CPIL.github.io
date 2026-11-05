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

    # --- Precompute LCM compatibility matrix (512x512 boolean) ---
    # compat[i][j] = True iff lcm(i+1, j+1) <= 512
    # Use a list of bitmasks for speed: compat_mask[v] has bit w set iff lcm(v, w+1) <= 512
    # Actually let's index 1..512 directly.
    # compat_mask[v]: bitmask where bit u is set iff u is in [1,512] and lcm(v,u) <= 512
    compat_mask = [0] * (N + 1)
    for v in range(1, N + 1):
        mask = 0
        for u in range(1, N + 1):
            if _lcm(v, u) <= N:
                mask |= (1 << (u - 1))  # bit position (u-1) represents number u
        compat_mask[v] = mask

    # Helper: check if v is compatible with all members of set S (represented as bitmask)
    # S_mask: bit u-1 set iff u in S
    def is_compatible(v, S_mask):
        # v must be compatible with every member of S
        # That means for all u in S, bit (u-1) of compat_mask[v] must be set
        # i.e., S_mask & ~compat_mask[v] == 0
        return (S_mask & ~compat_mask[v]) == 0

    # Helper: get the set of numbers from a bitmask
    def mask_to_set(mask):
        s = set()
        for u in range(1, N + 1):
            if mask & (1 << (u - 1)):
                s.add(u)
        return s

    def set_to_mask(s):
        m = 0
        for v in s:
            m |= (1 << (v - 1))
        return m

    def greedy_augment(seed_set):
        """Start from seed_set, greedily add elements that are compatible with all current members."""
        S = set(seed_set)
        S_mask = set_to_mask(S)
        changed = True
        while changed:
            changed = False
            # Try to add each number not yet in S
            for v in range(1, N + 1):
                if v in S:
                    continue
                if is_compatible(v, S_mask):
                    S.add(v)
                    S_mask |= (1 << (v - 1))
                    changed = True
        return S

    def greedy_by_degree(seed_set):
        """Greedy: repeatedly add the element (compatible with all in S) that is compatible with the most other candidates."""
        S = set(seed_set)
        S_mask = set_to_mask(S)
        remaining_mask = 0
        for v in range(1, N + 1):
            if v not in S and is_compatible(v, S_mask):
                remaining_mask |= (1 << (v - 1))
        
        while remaining_mask:
            # Find the element in remaining that has the highest degree (compatible with most other remaining elements)
            best_v = None
            best_deg = -1
            # Iterate over bits in remaining_mask
            r = remaining_mask
            while r:
                lb = r & (-r)  # lowest set bit
                v = lb.bit_length()  # position of lowest set bit (1-indexed: bit position = v)
                # bit_length of (1<<k) is k+1, so v = (lb.bit_length())
                # Wait: 1 << (u-1) has bit_length = u. So v = lb.bit_length()
                deg = 0
                # Count compatible elements in remaining
                # compat_mask[v] & remaining_mask gives all remaining compatible with v
                # Exclude v itself
                comp_with_remaining = compat_mask[v] & remaining_mask
                deg = comp_with_remaining.bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                r &= r - 1  # clear lowest set bit
            
            if best_v is None:
                break
            S.add(best_v)
            S_mask |= (1 << (best_v - 1))
            # Remove incompatible elements from remaining
            # New remaining: those in remaining_mask that are compatible with best_v
            remaining_mask &= compat_mask[best_v]
            # Also remove best_v from remaining (it's now in S)
            remaining_mask &= ~(1 << (best_v - 1))
        
        return S

    # --- Find good starting seeds ---
    # Seed type 1: divisor sets of numbers with many divisors
    # Focus on M that are multiples of 16*3*5=240 or 32*3*5=480 or 16*9*5=720(>512, skip)
    # Actually, for M <= 512, the most composite numbers:
    # 240 = 2^4 * 3 * 5 -> 5*2*2 = 20 divisors
    # 360 = 2^3 * 3^2 * 5 -> 4*3*2 = 24 divisors
    # 420 = 2^2 * 3 * 5 * 7 -> 3*2*2*2 = 24 divisors
    # 480 = 2^5 * 3 * 5 -> 6*2*2 = 24 divisors
    # 504 = 2^3 * 3^2 * 7 -> 4*3*2 = 24 divisors
    # 512 = 2^9 -> 10 divisors
    # 384 = 2^7 * 3 -> 8*2 = 16 divisors
    # 432 = 2^4 * 3^3 -> 5*4 = 20 divisors
    # 448 = 2^6 * 7 -> 7*2 = 14 divisors
    # 464 = 2^4 * 29 -> 5*2 = 10 divisors
    
    # Let's scan all M from 1 to 512 and find those with many divisors
    # But to save time, focus on "smooth" numbers
    candidate_Ms = []
    for M in range(1, N + 1):
        candidate_Ms.append(M)
    
    # Sort by number of divisors (descending)
    def div_count(m):
        c = 0
        for d in _divisors_of(m, N):
            c += 1
        return c
    
    # Precompute divisor counts for sorting
    M_sorted = sorted(candidate_Ms, key=lambda m: len(_divisors_of(m, N)), reverse=True)
    
    best_S = set()
    best_size = 0
    
    # Try top 50 M values as seeds
    tried_M = set()
    for M in M_sorted:
        if time.time() - start > DEADLINE - 10:
            break
        if M in tried_M:
            continue
        tried_M.add(M)
        
        d = _divisors_of(M, N)
        if not d:
            continue
        
        # Try greedy augmentation from this seed
        S1 = greedy_augment(d)
        if len(S1) > best_size:
            best_size = len(S1)
            best_S = S1
        
        # Also try degree-based greedy from this seed
        S2 = greedy_by_degree(d)
        if len(S2) > best_size:
            best_size = len(S2)
            best_S = S2
        
        # Only need to try a handful of top M's since they're similar
        if len(tried_M) >= 30:
            break
    
    # Also try from empty set with degree-based greedy
    if time.time() - start < DEADLINE - 15:
        S3 = greedy_by_degree(set())
        if len(S3) > best_size:
            best_size = len(S3)
            best_S = S3
    
    # Try increasing-order greedy from empty
    if time.time() - start < DEADLINE - 10:
        S = set()
        S_mask = 0
        for v in range(1, N + 1):
            if is_compatible(v, S_mask):
                S.add(v)
                S_mask |= (1 << (v - 1))
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
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
