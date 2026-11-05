import time
import math
import random

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)

def _is_admissible(S, primes):
    """Check if S is admissible: for every prime p <= len(S), some residue mod p is missing."""
    S_set = set(S)
    k = len(S_set)
    for p in primes:
        if p > k:
            break
        # Check if all residues mod p are covered
        residues = set()
        for x in S_set:
            residues.add(x % p)
        if len(residues) == p:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # Initial: all numbers 0..N
    S = set(range(N + 1))
    
    best_set = set(S)
    best_size = len(S)
    
    # Phase 1: Greedy removal to make admissible
    # We remove residue classes for violating primes, choosing the one with minimum count
    while time.time() < deadline - 30:
        M = len(S)
        if M == 0:
            break
        
        # Find violating primes (p <= M where all residues are present)
        violating = []
        for p in PRIMES:
            if p > M:
                break
            residues = set()
            for x in S:
                residues.add(x % p)
            if len(residues) == p:
                # All residues present -> violation
                # Compute count for each residue to find min
                counts = {}
                for x in S:
                    r = x % p
                    counts[r] = counts.get(r, 0) + 1
                min_count = min(counts.values())
                # Find which residue has min count
                min_r = min(counts, key=lambda r: counts[r])
                violating.append((min_count, p, min_r))
        
        if not violating:
            break
        
        # Remove the residue class with minimum removal cost
        violating.sort()
        min_count, p_star, r_star = violating[0]
        
        # Remove all x in S where x % p_star == r_star
        to_remove = set(x for x in S if x % p_star == r_star)
        S -= to_remove
    
    # Update best
    if len(S) > best_size:
        best_size = len(S)
        best_set = set(S)
    
    # Phase 2: Local search - try to add back removed elements
    # We'll do several passes of adding back elements that don't break admissibility
    
    # For efficiency, we'll track which primes are "tight" (have exactly p-1 residues)
    # Adding an element with the missing residue for a tight prime would break admissibility
    
    removed = set(range(N + 1)) - S
    removed_list = list(removed)
    random.shuffle(removed_list)
    
    # Multiple rounds of local search
    max_rounds = 10
    for round_num in range(max_rounds):
        if time.time() > deadline - 10:
            break
        
        # Recompute tight primes for current S
        current_k = len(S)
        tight_primes = {}  # p -> missing residue
        for p in PRIMES:
            if p > current_k + 1:
                break
            residues = set()
            for x in S:
                residues.add(x % p)
            if len(residues) == p - 1:
                missing = set(range(p)) - residues
                if len(missing) == 1:
                    tight_primes[p] = missing.pop()
        
        added_any = False
        for x in removed_list:
            if time.time() > deadline - 5:
                break
            
            # Check if adding x breaks admissibility
            can_add = True
            for p, missing_r in tight_primes.items():
                if x % p == missing_r:
                    can_add = False
                    break
            
            if can_add:
                S.add(x)
                added_any = True
                # Update tight_primes for the new element
                new_k = len(S)
                # We need to recheck which primes are now tight
                # For efficiency, just recompute all tight primes
                tight_primes = {}
                for p in PRIMES:
                    if p > new_k + 1:
                        break
                    residues = set()
                    for x_s in S:
                        residues.add(x_s % p)
                    if len(residues) == p - 1:
                        missing = set(range(p)) - residues
                        if len(missing) == 1:
                            tight_primes[p] = missing.pop()
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
        
        if not added_any:
            break
    
    # Final verification and return
    return list(best_set)
