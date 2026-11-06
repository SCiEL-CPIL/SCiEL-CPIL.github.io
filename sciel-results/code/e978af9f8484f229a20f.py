"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def _is_admissible(S, all_primes):
    """Check if S is admissible: for every prime p <= |S|, some residue mod p is missed."""
    k = len(S)
    if k == 0:
        return True
    s_arr = np.array(S, dtype=np.int64)
    for p in all_primes:
        if p > k:
            break
        residues = s_arr % p
        counts = np.bincount(residues, minlength=p)
        if counts.min() > 0:
            return False
    return True


def _greedy_construct(S, all_primes, deadline):
    """Greedy removal of smallest residue classes for violating primes."""
    S = list(S)
    while True:
        if time.time() > deadline:
            break
        
        k = len(S)
        if k == 0:
            break
        
        s_arr = np.array(S, dtype=np.int64)
        
        # Get primes <= k
        primes_to_check = [p for p in all_primes if p <= k]
        
        violation_found = False
        for p in primes_to_check:
            residues = s_arr % p
            counts = np.bincount(residues, minlength=p)
            
            # Check if any residue is missing
            if counts.min() == 0:
                continue  # This prime is satisfied
            
            # Violation: all residues covered.
            # Remove the class with smallest count
            target_residue = int(np.argmin(counts))
            
            # Remove all elements == target_residue mod p
            mask = residues != target_residue
            S = [int(x) for x in s_arr[mask]]
            s_arr = s_arr[mask]
            
            violation_found = True
            break  # Restart with new S
        
        if not violation_found:
            break
    
    return S


def _local_search(S, all_primes, deadline):
    """Try to add back elements that were removed, if admissibility is maintained."""
    S_set = set(S)
    S_list = list(S)
    k = len(S_list)
    
    # Try to add elements from [0, N] not in S
    candidates = [x for x in range(N + 1) if x not in S_set]
    
    # Sort candidates by how "safe" they are (fewer new residue conflicts)
    for candidate in candidates:
        if time.time() > deadline:
            break
        
        # Try adding this candidate
        new_S = S_list + [candidate]
        new_k = len(new_S)
        
        # Check admissibility with the new size
        new_s_arr = np.array(new_S, dtype=np.int64)
        admissible = True
        for p in all_primes:
            if p > new_k:
                break
            residues = new_s_arr % p
            counts = np.bincount(residues, minlength=p)
            if counts.min() > 0:
                admissible = False
                break
        
        if admissible:
            S_list = new_S
            S_set.add(candidate)
            k = new_k
    
    return S_list


def construct():
    start_time = time.time()
    deadline = start_time + 285.0
    
    # Precompute primes up to N
    all_primes = _primes(N)
    
    # Start with all numbers in [0, N]
    S = list(range(N + 1))
    
    # Phase 1: Greedy removal to get an admissible set
    S = _greedy_construct(S, all_primes, deadline)
    
    # Phase 2: Local search to try to add back elements
    if time.time() < deadline:
        S = _local_search(S, all_primes, deadline)
    
    return sorted(S)
# EVOLVE-BLOCK-END
