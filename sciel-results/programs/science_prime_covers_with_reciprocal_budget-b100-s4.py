"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
import numpy as np

_CAP = 100000
_SEARCH_CAP = 2000 # We only need to verify coverage up to a reasonable limit for search speed

def _primes_upto(n):
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    import math
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _calc_L(covered, cap):
    L = 0
    for i in range(1, cap + 1):
        if covered[i]:
            L = i
        else:
            break
    return L

def _update_coverage(covered, p, r, cap, sign):
    # sign 1 to add, -1 to remove
    start = r if r != 0 else p
    # We need to know how many times each position is covered to handle removals correctly
    # This is tricky with a simple bool array.
    # For local search, we can just recalculate the whole coverage array from scratch if it's fast enough.
    # 168 primes, cap 2000. 168 * 2000/p ~ 168 * 2000/100 ~ 3360 ops. Very fast.
    pass

def construct():
    deadline = time.time() + 110.0
    primes = _primes_upto(1000)
    
    # 1. Deterministic Seed Construction (Inspiration 1)
    # Primes >= 13
    seed_primes = [p for p in primes if p >= 13]
    seed_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_primes)]
    
    # Calculate initial L
    covered = np.zeros(_SEARCH_CAP + 1, dtype=bool)
    for p, r in seed_classes:
        start = r if r != 0 else p
        covered[start:_SEARCH_CAP+1:p] = True
        
    best_L = _calc_L(covered, _SEARCH_CAP)
    best_classes = [c[:] for c in seed_classes] # Deep copy
    
    # 2. Local Search to improve L
    # We will try to change residues of primes to extend the prefix.
    # We maintain a list of (p, r) and a coverage array.
    # To make removals easy, we'll just rebuild coverage array on every successful change.
    # Rebuild cost is low.
    
    rng = random.Random(42)
    current_classes = [c[:] for c in best_classes]
    current_L = best_L
    
    # Map prime to index in current_classes for quick lookup
    # primes are distinct, so we can use a dict
    prime_to_idx = {p: i for i, (p, r) in enumerate(current_classes)}
    
    while time.time() < deadline - 5.0:
        # Pick a random prime to modify
        # Bias towards primes that cover the current frontier?
        # Or just random.
        idx = rng.randint(0, len(current_classes) - 1)
        p, r_old = current_classes[idx]
        
        # Try a new residue
        # Maybe try residues that cover current_L + 1?
        # If we can cover current_L + 1, L might jump.
        target = current_L + 1
        r_new = target % p
        
        if r_new == r_old:
            continue
            
        # Apply change
        current_classes[idx] = [p, r_new]
        
        # Recalculate coverage
        covered = np.zeros(_SEARCH_CAP + 1, dtype=bool)
        for p_c, r_c in current_classes:
            start = r_c if r_c != 0 else p_c
            covered[start:_SEARCH_CAP+1:p_c] = True
            
        new_L = _calc_L(covered, _SEARCH_CAP)
        
        if new_L > current_L:
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_classes = [c[:] for c in current_classes]
        else:
            # Revert
            current_classes[idx] = [p, r_old]
            
        # If we reached the search cap, we might want to extend the cap?
        # Or just stop if we are high.
        if current_L >= _SEARCH_CAP:
            # If we covered the whole search cap, we can try to extend it
            # But for now, let's just break if we are well above seed
            # Actually, let's allow it to keep trying, maybe we can go higher
            pass

    # 3. Final Verification with full cap?
    # The score is based on exact L.
    # If best_L is from _SEARCH_CAP, we should verify if it's actually that high.
    # But _SEARCH_CAP=2000 is likely high enough for a jump from 168.
    
    # Fallback if something went wrong
    if best_L < 100:
        best_classes = seed_classes

    return best_classes
# EVOLVE-BLOCK-END
