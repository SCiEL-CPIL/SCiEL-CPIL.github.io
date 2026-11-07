import math
import time

def _sieve(n):
    if n < 2:
        return []
    is_prime = bytearray([1]) * (n + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start::step] = bytearray(len(range(start, n + 1, step)))
    return [i for i in range(2, n + 1) if is_prime[i]]

MAX_VAL = 10210
_PRIMES = _sieve(102)

def _squarefree_kernel(n):
    if n <= 1:
        return 1
    result = 1
    f = n
    for p in _PRIMES:
        if p * p > f:
            break
        if f % p == 0:
            exp = 0
            while f % p == 0:
                f //= p
                exp += 1
            if exp % 2 == 1:
                result *= p
    if f > 1:
        result *= f
    return result

_KER = [0] * (MAX_VAL + 1)
for i in range(211, MAX_VAL + 1):
    _KER[i] = _squarefree_kernel(i)

def _is_perfect_square(n):
    if n < 0:
        return False
    x = math.isqrt(n)
    return x * x == n

def _validate(result, N=210):
    if not result or len(result) > 100:
        return False
    if len(set(result)) != len(result):
        return False
    for x in result:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x <= 210 or x > MAX_VAL:
            return False
    prod = 1
    for x in result:
        prod *= x
    return _is_perfect_square(prod * N)

def construct():
    N = 210
    start_time = time.time()
    deadline = start_time + 80.0
    
    # 1. Check for 1-element solutions
    # We want K(x) == 210.
    # Iterate x from 211 upwards. The first one found minimizes max(S).
    for x in range(211, MAX_VAL + 1):
        if _KER[x] == 210:
            res = [x]
            if _validate(res):
                return res
        if time.time() > deadline:
            break

    # 2. Check for 2-element solutions
    # We want K(a) ^ K(b) == 210 => K(b) == K(a) ^ 210.
    # To minimize max(a, b), we should check pairs in a way that prioritizes smaller max.
    # A simple approach: for each max_val from 212 to MAX_VAL, check if there exists a pair (a, b) with a < b <= max_val.
    # This is O(N^2) in the worst case but we can optimize by iterating a and checking if the required b exists in [a+1, max_val].
    # Since we want the global minimum max, we can just find any valid pair and then refine? 
    # No, the first valid pair found with minimal max is the best.
    # Let's iterate max_val from 212 to MAX_VAL.
    # For a given max_val, we need a < b <= max_val.
    # We can maintain a set of seen kernels for a < current_b.
    # Actually, simpler:
    # For b from 212 to MAX_VAL:
    #   For a from 211 to b-1:
    #      if K(a) ^ K(b) == 210: return [a, b]
    # This is O(N^2) which is ~100M operations, might be too slow in Python.
    
    # Optimized 2-element search:
    # Map kernel -> smallest number with that kernel.
    # We want min(max(a,b)).
    # Let's collect all numbers and their kernels.
    # We can iterate b from 212 to MAX_VAL.
    # We need a < b such that K(a) = K(b) ^ 210.
    # If we have a map ker_to_min_num, we can check if K(b)^210 exists and if the min_num is < b.
    # But we want the pair with the smallest max. So iterating b upwards and returning the first valid pair is correct.
    # Because if a valid pair (a,b) exists with max=b, and we haven't found a pair with max < b, then this is the best so far.
    # We must ensure a < b.
    
    ker_to_min_num = {}
    # Initialize with 211
    ker_to_min_num[_KER[211]] = 211
    
    for b in range(212, MAX_VAL + 1):
        kb = _KER[b]
        need = kb ^ 210
        if need in ker_to_min_num:
            a = ker_to_min_num[need]
            if a < b:
                res = [a, b]
                if _validate(res):
                    return res
        # Update map with b if b is smaller than existing (it is, since we go up, so we only set if not present? 
        # Wait, we want the smallest a for a given kernel to minimize the chance of conflict? 
        # Actually, for a fixed b, any valid a < b works. The max is b. 
        # So we just need to know if ANY a < b exists with the required kernel.
        # Storing the minimum a is sufficient to check existence of a < b? 
        # If min_a < b, then yes.
        if need not in ker_to_min_num or b < ker_to_min_num[need]:
             # This logic is for finding min a. But we iterate b upwards.
             # We just need to add b to the pool for future b's.
             # If need not in ker_to_min_num, we add it.
             # If it is already there, the existing one is smaller, so we don't need to update for the purpose of "does a exist < current_b".
             # Actually, we just need to add b to the map if it's the first time we see this kernel?
             # No, if we see kernel K at 211, and then at 250, min is 211.
             # So we only set if not present.
             pass 
        
        if kb not in ker_to_min_num:
            ker_to_min_num[kb] = b
            
        if time.time() > deadline:
            break
            
    # Fallback if no 1 or 2 element solution found quickly
    # Try 3 elements? 
    # Given the nature of squarefree kernels, 1 or 2 element solutions are highly likely to exist for small max.
    # If we reach here, we return a known valid solution to avoid timeout/invalid.
    # [840] is valid: 840 * 210 = 176400 = 420^2.
    return [840]
