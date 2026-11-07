"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve_primes(limit):
    """Generate all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]

# Precompute primes up to a reasonable limit
_MAX_GAP = max(hi - lo for lo, hi in P['gaps'])
_PRIMES_CACHE = _sieve_primes(_MAX_GAP + 10)  # primes up to max_gap+10

# Precompute a bitset marking "smooth" numbers (divisible by any prime < max_gap)
# We only need to check primes < _MAX_GAP for the strict condition lpf >= gap
# For a number m to have lpf >= gap, it must not be divisible by any prime < gap.
# Since gap <= _MAX_GAP, we mark numbers divisible by any prime < _MAX_GAP.
# But we need finer granularity: for a specific gap, we need primes < gap.
# Strategy: precompute for each prime p < _MAX_GAP a bitset of multiples of p in [min_lo, max_hi].
# Then for a query (lo, hi, gap), valid m must NOT be in any bitset for p < gap.

_MIN_LO = min(lo for lo, hi in P['gaps'])
_MAX_HI = max(hi for lo, hi in P['gaps'])
_RANGE_SIZE = _MAX_HI - _MIN_LO + 1

# For each prime p < _MAX_GAP, create a bytearray of length _RANGE_SIZE
# where index (m - _MIN_LO) is 1 if m is divisible by p
# We'll store these as bytearrays for efficient bitwise operations
_PRIME_BITS = {}
for p in _PRIMES_CACHE:
    if p >= _MAX_GAP:
        break
    bits = bytearray(_RANGE_SIZE)
    # Mark multiples of p in [_MIN_LO, _MAX_HI]
    # First multiple of p >= _MIN_LO
    first = (( _MIN_LO + p - 1) // p) * p
    for m in range(first, _MAX_HI + 1, p):
        bits[m - _MIN_LO] = 1
    _PRIME_BITS[p] = bits

# For efficient lookup, sort primes < _MAX_GAP
_SMALL_PRIMES = [p for p in _PRIMES_CACHE if p < _MAX_GAP]

def _find_witness(lo, hi, gap):
    """Find an integer m in (lo, hi) with lpf(m) >= gap using precomputed bitsets."""
    if gap <= 2:
        # Any m works
        return lo + 1
    
    # We need m not divisible by any prime < gap
    # Collect bitsets for primes < gap
    # Use bitwise OR to combine, then check which positions are 0
    # For small gaps, this is fast; for larger gaps, still manageable
    
    # Get the relevant primes
    relevant_primes = [p for p in _SMALL_PRIMES if p < gap]
    
    if not relevant_primes:
        return lo + 1
    
    # Start with all-ones mask for the range [lo+1, hi-1]
    # We'll build a combined "invalid" mask
    # For efficiency, only process the subrange [lo+1, hi-1]
    start_offset = lo + 1 - _MIN_LO
    end_offset = hi - 1 - _MIN_LO  # inclusive
    length = end_offset - start_offset + 1
    
    if length <= 0:
        return None
    
    # Initialize invalid mask as all zeros
    invalid = bytearray(length)
    
    for p in relevant_primes:
        bits = _PRIME_BITS[p]
        # OR the relevant portion into invalid
        for i in range(length):
            if bits[start_offset + i]:
                invalid[i] = 1
    
    # Find first position where invalid[i] == 0
    for i in range(length):
        if not invalid[i]:
            return lo + 1 + i
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50 seconds margin
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        
        witness = _find_witness(lo, hi, gap)
        
        result[i] = witness
    
    return result

# EVOLVE-BLOCK-END
