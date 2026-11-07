# EVOLVE-BLOCK-START
import time
import bisect
import numpy as np

MAXV = 10000000
WALL_LIMIT = 90.0
SAFETY = 0.85  # hard stop for search at 85% of wall limit

def construct():
    t0 = time.time()
    hard_stop = t0 + WALL_LIMIT * SAFETY

    def time_left():
        return hard_stop - time.time()

    try:
        N = MAXV

        # ---------- Linear Sieve for sigma ----------
        # Using Python lists for speed in the inner loop, then converting to np arrays if needed for sorting?
        # Or just keep as lists.
        # sigma[1] = 1
        # is_prime as bytearray for memory efficiency
        is_prime = bytearray(b'\x01') * (N + 1)
        is_prime[0] = 0
        is_prime[1] = 0
        
        sigma = [0] * (N + 1)
        max_pow = [0] * (N + 1)
        sigma_pow = [0] * (N + 1)
        
        sigma[1] = 1
        primes = []
        
        # Local variables for speed
        _is_prime = is_prime
        _sigma = sigma
        _max_pow = max_pow
        _sigma_pow = sigma_pow
        _primes = primes
        _N = N
        
        for i in range(2, _N + 1):
            if _is_prime[i]:
                _primes.append(i)
                _sigma[i] = 1 + i
                _max_pow[i] = i
                _sigma_pow[i] = 1 + i
            
            # Inner loop
            # Optimization: break early if i * p > N
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
            
            if (i & 0x7FFFF) == 0 and time_left() < 30.0:
                return [11, 6]

        if time_left() < 15.0:
            return [11, 6]

        # ---------- Group indices by sigma value ----------
        # Create dictionary of sigma -> list of indices
        # This might be memory heavy if we store lists of ints for 10^7 items.
        # Total items is 10^7. Average group size is small.
        # Using a dict of lists is okay.
        
        groups = {}
        # Iterate from 2 to N
        # We can skip 1? sigma(1)=1. Only one number has sigma 1? No, sigma(1)=1.
        # Are there other n with sigma(n)=1? No.
        # So group 1 has size 1. We can skip it.
        
        for n in range(2, _N + 1):
            s = _sigma[n]
            if s in groups:
                groups[s].append(n)
            else:
                groups[s] = [n]
                
        # Sort the lists in the groups
        # We only care about groups with size >= 2
        best_val = None
        best_pair = None
        max_b = (2 * _N) // 3
        
        # Iterate over groups
        # To optimize, we can filter groups first?
        # Or just iterate.
        
        for s, gn in groups.items():
            L = len(gn)
            if L < 2:
                continue
            
            gn.sort()
            
            # Iterate b large -> small
            for bi in range(L - 1, -1, -1):
                b = gn[bi]
                if b > max_b:
                    continue
                
                # Find a closest to 1.5 * b
                target = 1.5 * b
                # bisect_left
                j = bisect.bisect_left(gn, target)
                
                # Check j
                if j < L:
                    a = gn[j]
                    if a != b:
                        num = abs(2 * a - 3 * b)
                        if num:
                            val = num / (2 * b)
                            if best_val is None or val < best_val:
                                best_val = val
                                best_pair = (a, b)
                
                # Check j-1
                jm = j - 1
                if jm >= 0:
                    a = gn[jm]
                    if a != b:
                        num = abs(2 * a - 3 * b)
                        if num:
                            val = num / (2 * b)
                            if best_val is None or val < best_val:
                                best_val = val
                                best_pair = (a, b)
                                
            if time_left() < 5.0:
                break

        if best_pair is None:
            return [11, 6]
        a, b = int(best_pair[0]), int(best_pair[1])
        if not (1 <= a <= _N and 1 <= b <= _N):
            return [11, 6]
        if 2 * a == 3 * b:
            return [11, 6]
        return [a, b]
    except Exception:
        return [11, 6]
# EVOLVE-BLOCK-END
