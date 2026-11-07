# EVOLVE-BLOCK-START
import time
import bisect
import math

MAXV = 10000000
WALL_LIMIT = 85.0
SAFETY = 0.95

def construct():
    t0 = time.time()
    hard_stop = t0 + WALL_LIMIT * SAFETY

    def time_left():
        return hard_stop - time.time()

    # Fallback: a known valid pair with sigma(11)=12, sigma(6)=12.
    # |11/6 - 1.5| = 1/3.
    # Score = 1 / (1/3) = 3. This is a very poor score, but valid.
    fallback = [11, 6]

    try:
        N = MAXV

        # ---------- Linear Sieve for sigma ----------
        # We need sigma(n) for n up to N.
        # Linear sieve is O(N).
        
        is_prime = bytearray(b'\x01') * (N + 1)
        is_prime[0] = 0
        is_prime[1] = 0
        
        sigma = [0] * (N + 1)
        sigma[1] = 1
        primes = []
        # max_pow[i] = p^k where p^k is the highest power of p dividing i
        # sigma_pow[i] = 1 + p + ... + p^k
        max_pow = [0] * (N + 1)
        sigma_pow = [0] * (N + 1)
        
        # Local variables for speed
        _is_prime = is_prime
        _sigma = sigma
        _max_pow = max_pow
        _sigma_pow = sigma_pow
        _primes = primes
        _N = N
        
        # Linear Sieve
        for i in range(2, _N + 1):
            if _is_prime[i]:
                _primes.append(i)
                _sigma[i] = 1 + i
                _max_pow[i] = i
                _sigma_pow[i] = 1 + i
            
            for p in _primes:
                ip = i * p
                if ip > _N:
                    break
                _is_prime[ip] = 0
                if i % p == 0:
                    _max_pow[ip] = _max_pow[i] * p
                    _sigma_pow[ip] = _sigma_pow[i] + _max_pow[ip]
                    _sigma[ip] = _sigma[i] // _sigma_pow[i] * _sigma_pow[ip]
                    break
                else:
                    _max_pow[ip] = p
                    _sigma_pow[ip] = 1 + p
                    _sigma[ip] = _sigma[i] * (1 + p)
            
            if (i & 0xFFFFF) == 0 and time_left() < 30.0:
                return fallback

        if time_left() < 20.0:
            return fallback

        # ---------- Group indices by sigma value ----------
        # We want to find a, b such that sigma(a) = sigma(b) and a/b approx 3/2.
        # This implies a approx 1.5 b.
        # So a > b.
        # We can iterate over b, find target a = 1.5 b.
        # We need to quickly find if there is an a in the same sigma group close to 1.5 b.
        
        # Strategy:
        # 1. Build a dict: sigma_val -> list of n's.
        # 2. For each group, sort the list.
        # 3. For each b in the group, look for a in the same group closest to 1.5 b.
        #    Use binary search.
        
        # Memory concern: Storing lists for 10^7 items.
        # Dict of lists is okay.
        
        groups = {}
        for n in range(2, _N + 1):
            s = _sigma[n]
            if s in groups:
                groups[s].append(n)
            else:
                groups[s] = [n]
        
        # We only care about groups with size >= 2.
        # Also, to minimize |a/b - 1.5|, we want large b?
        # No, the error is |a/b - 1.5|.
        # Let a = 1.5 b + eps.
        # |eps/b| is the error.
        # To minimize error, we want small |eps| and large b.
        # So large b is better if |eps| is small.
        
        # Iterate through groups.
        best_val = float('inf')
        best_pair = None
        
        # To speed up, we can process groups in some order?
        # Or just iterate all.
        # Number of groups is roughly N / log N? No, sigma(n) is roughly n.
        # Number of distinct sigma values is smaller.
        
        for s, gn in groups.items():
            L = len(gn)
            if L < 2:
                continue
            
            gn.sort()
            
            # For each b in gn, find a in gn closest to 1.5 b.
            # We can do this in O(L log L) or O(L) with two pointers?
            # Since we want a > b (approx 1.5b), and both are in sorted list.
            # Let's use bisect.
            
            for b in gn:
                target = 1.5 * b
                # Find a in gn closest to target.
                # a must be distinct from b.
                # Since a approx 1.5b, and b >= 2, a >= 3.
                # If b is small, a is small.
                
                # Bisect to find insertion point of target in gn.
                idx = bisect.bisect_left(gn, target)
                
                # Check idx and idx-1.
                # We need to ensure a != b.
                # Since target = 1.5 b, and b is in gn, target > b (for b>0).
                # So a will be > b usually.
                # If a == b, it means 1.5b == b => b=0, impossible.
                # Or if the closest value is b itself?
                # If target is between b and next element, and b is the closest?
                # Distance to b is |1.5b - b| = 0.5b.
                # Distance to next element a_next is |a_next - 1.5b|.
                # If a_next is very close to 1.5b, it's better.
                
                candidates = []
                if idx < L:
                    candidates.append(gn[idx])
                if idx > 0:
                    candidates.append(gn[idx-1])
                
                for a in candidates:
                    if a == b:
                        continue
                    # Calculate error |a/b - 1.5| = |2a - 3b| / (2b)
                    # We want to minimize this.
                    # Equivalent to minimizing |2a - 3b| / b.
                    # Or minimizing |2a - 3b| / (2b).
                    
                    num = abs(2*a - 3*b)
                    if num == 0:
                        continue
                        # But "raw value of exactly 0 ... invalid".
                        # So we should skip this?
                        # "A raw value of exactly 0 cannot be scored and counts as invalid."
                        # So we must NOT return a pair where 2a == 3b.
                        # If we find one, we should look for the next best?
                        # Or just return a different one?
                        # If 2a == 3b, then a/b = 1.5 exactly.
                        # We need a/b != 1.5.
                        # So skip this candidate.
                    
                    val = num / (2.0 * b)
                    if val < best_val:
                        best_val = val
                        best_pair = (a, b)
            
            if time_left() < 5.0:
                break
        
        if best_pair is None:
            return fallback
        
        a, b = int(best_pair[0]), int(best_pair[1])
        if not (1 <= a <= _N and 1 <= b <= _N):
            return fallback
        if 2 * a == 3 * b:
            return fallback
        
        return [a, b]

    except Exception:
        return fallback

# EVOLVE-BLOCK-END
