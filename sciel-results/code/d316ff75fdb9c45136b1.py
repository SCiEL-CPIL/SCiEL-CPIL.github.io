import time
import math

def construct():
    deadline = time.time() + 115

    def sieve_primes(limit):
        if limit < 2:
            return []
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]

    best_k = 0
    best_p = 2

    SIEVE_LIMIT = 2 * 10**6
    primes_list = sieve_primes(SIEVE_LIMIT)
    primes_set = set(primes_list)

    def check_run_smooth(base, p, max_k=200):
        """Check how many consecutive numbers base+1..base+max_k are p-smooth.
        Uses segmented approach: for each prime <= p, divide it out of all numbers."""
        # Initialize array with the numbers to check
        vals = [base + i for i in range(1, max_k + 1)]
        
        for pr in primes_list:
            if pr > p:
                break
            # Find which positions are divisible by pr
            # First occurrence: smallest i in [1, max_k] such that (base+i) % pr == 0
            # base+i ≡ 0 (mod pr) => i ≡ -base (mod pr)
            rem = (-base) % pr
            if rem == 0:
                start = pr
            else:
                start = rem
            if start > max_k:
                continue
            # Divide out pr from all vals[start-1], vals[start-1+pr], ...
            idx = start - 1
            while idx < max_k:
                v = vals[idx]
                while v % pr == 0:
                    v //= pr
                vals[idx] = v
                idx += pr
        
        # Now check how many are 1 (fully factored) or <= p
        k = 0
        for i in range(max_k):
            v = vals[i]
            if v == 1 or v <= p:
                k += 1
            else:
                break
        return k

    # Phase 1: Quick scan of smaller primes
    for p in primes_list:
        if time.time() > deadline - 10:
            break
        base = p * p
        k = check_run_smooth(base, p)
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 200:
                break

    return {'p': best_p, 'k': best_k}
