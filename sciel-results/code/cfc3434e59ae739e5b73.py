"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import bisect

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

def _build_residue_table(forbidden_primes):
    """
    Build a lookup table mapping residue mod (product of forbidden primes)
    to whether it's valid (not divisible by any forbidden prime).
    Returns (modulus, valid_residues_list) where valid_residues_list is sorted.
    """
    if not forbidden_primes:
        return 1, None  # None means all residues are valid
    
    modulus = 1
    for p in forbidden_primes:
        modulus *= p
    
    # Build sorted list of valid residues
    valid_residues = []
    for r in range(modulus):
        valid = True
        for p in forbidden_primes:
            if r % p == 0:
                valid = False
                break
        if valid:
            valid_residues.append(r)
    
    return modulus, valid_residues

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    gaps = P['gaps']
    n = len(gaps)
    result = [None] * n
    
    # Determine the maximum gap size
    max_gap = max(hi - lo for lo, hi in gaps)
    
    # Precompute all primes up to max_gap - 1
    all_primes = _sieve_primes(max_gap - 1)
    
    # For each distinct gap size, precompute the wheel data
    gap_data = {}
    for lo, hi in gaps:
        g = hi - lo
        if g not in gap_data:
            forbidden = [p for p in all_primes if p < g]
            if not forbidden:
                gap_data[g] = {'forbidden': forbidden, 'modulus': 1, 'residues': None}
            else:
                modulus, residues = _build_residue_table(forbidden)
                gap_data[g] = {'forbidden': forbidden, 'modulus': modulus, 'residues': residues}
    
    # Process each gap
    for i, (lo, hi) in enumerate(gaps):
        if time.time() > deadline:
            break
        
        g = hi - lo
        data = gap_data[g]
        forbidden = data['forbidden']
        modulus = data['modulus']
        residues = data['residues']
        
        if not forbidden:
            # No forbidden primes: any integer in (lo, hi) works
            if lo + 1 < hi:
                result[i] = int(lo + 1)
            continue
        
        # Find the smallest m in (lo, hi) such that m mod modulus is in residues
        start = lo + 1
        end = hi  # exclusive
        
        # Find the first valid candidate >= start
        r_start = start % modulus
        
        # Use bisect to find the smallest residue >= r_start
        idx = bisect.bisect_left(residues, r_start)
        
        if idx < len(residues):
            delta = residues[idx] - r_start
            candidate = start + delta
        else:
            # Wrap around: use the smallest residue in the next cycle
            delta = residues[0] + modulus - r_start
            candidate = start + delta
        
        # Check if candidate is strictly between lo and hi
        if candidate < hi:
            result[i] = int(candidate)
        else:
            # Fallback: linear scan (should be rare)
            for m in range(start, hi):
                valid = True
                for p in forbidden:
                    if m % p == 0:
                        valid = False
                        break
                if valid:
                    result[i] = int(m)
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
