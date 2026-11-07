"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve_primes(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    # Deadline guard: return best result before 85s to be safe within 90s
    deadline = time.time() + 85.0
    
    gaps = P['gaps']
    n = len(gaps)
    result = [None] * n
    
    # Determine the maximum gap size to know which small primes we need to check
    max_gap = max(hi - lo for lo, hi in gaps)
    
    # Precompute primes less than max_gap (these are the "forbidden" small factors)
    # A witness m in (lo, hi) is valid if its least prime factor >= (hi - lo)
    # This means m is NOT divisible by any prime p < (hi - lo)
    # Equivalently, for each gap with size g, m must not be divisible by any prime p < g
    
    # Precompute all primes up to max_gap - 1
    forbidden_primes = _sieve_primes(max_gap - 1)
    
    # For efficiency, group gaps by their gap size
    # For each distinct gap size g, we need primes p < g
    # Let's precompute for each possible gap size which primes to check
    
    # Actually, let's just iterate through each gap and find a valid witness
    # Strategy: for each gap (lo, hi) with gap size g = hi - lo,
    #   find an integer m in (lo, hi) such that m is not divisible by any prime p < g
    
    # Optimization: precompute a "bad" sieve for each distinct gap size
    # For a given gap size g, create a boolean array marking which residues mod lcm(2..g-1) are bad
    # But that's complex. Instead, for each m, just check divisibility by primes < g.
    
    # Even better: for each gap, we can check candidates m = lo+1, lo+2, ..., hi-1
    # and test if any prime p < g divides m. If no such prime divides m, m is valid.
    
    # Let's optimize by precomputing, for each distinct gap size, the list of forbidden primes
    gap_sizes = {}
    for lo, hi in gaps:
        g = hi - lo
        if g not in gap_sizes:
            gap_sizes[g] = [p for p in forbidden_primes if p < g]
    
    # Now process each gap
    for i, (lo, hi) in enumerate(gaps):
        g = hi - lo
        forbidden = gap_sizes[g]
        
        # Try candidates from lo+1 to hi-1
        # We want the first valid one (or any valid one)
        found = False
        for m in range(lo + 1, hi):
            # Check if m is divisible by any forbidden prime
            valid = True
            for p in forbidden:
                if m % p == 0:
                    valid = False
                    break
            if valid:
                result[i] = m
                found = True
                break
        
        # Check deadline
        if time.time() > deadline:
            break
    
    # Ensure types are correct: Python int or None
    final_result = []
    for val in result:
        if val is None:
            final_result.append(None)
        else:
            final_result.append(int(val))
    
    return final_result

# EVOLVE-BLOCK-END
