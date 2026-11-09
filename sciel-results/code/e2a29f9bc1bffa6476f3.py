"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from math import isqrt

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p:limit+1:p] = False
    return np.where(sieve)[0]

def _is_admissible(S_mask, primes):
    """Check if S_mask represents an admissible tuple."""
    size = np.count_nonzero(S_mask)
    if size == 0:
        return True
    S_indices = np.where(S_mask)[0]
    for p in primes:
        if p > size:
            break
        residues = S_indices % p
        if len(np.unique(residues)) == p:
            return False
    return True

def _try_strategy(primes, sieve_threshold, use_least_populated_for_small=True):
    """Try a specific strategy and return the resulting set size and mask."""
    S_mask = np.ones(N + 1, dtype=bool)
    
    # Sieve primes up to threshold
    small_primes = primes[primes <= sieve_threshold]
    for p in small_primes:
        if use_least_populated_for_small:
            # Find least populated residue among current S
            S_indices = np.where(S_mask)[0]
            if len(S_indices) == 0:
                return 0, S_mask
            residues = S_indices % p
            counts = np.bincount(residues, minlength=p)
            best_r = np.argmin(counts)
            to_remove = S_indices[residues == best_r]
            S_mask[to_remove] = False
        else:
            # Remove residue 0
            multiples = np.arange(0, N + 1, p)
            S_mask[multiples] = False
    
    remaining_primes = primes[primes > sieve_threshold]
    
    changed = True
    while changed:
        current_size = np.count_nonzero(S_mask)
        if current_size == 0:
            break
        
        S_indices = np.where(S_mask)[0]
        candidates = remaining_primes[remaining_primes <= current_size]
        
        bad_p = None
        min_removal_count = float('inf')
        best_r = -1
        best_p = -1
        
        for p in candidates:
            residues = S_indices % p
            unique_res = np.unique(residues)
            if len(unique_res) == p:
                counts = np.bincount(residues, minlength=p)
                min_count = np.min(counts)
                if min_count < min_removal_count:
                    min_removal_count = min_count
                    best_r = np.argmin(counts)
                    best_p = p
                    bad_p = p
        
        if bad_p is not None:
            to_remove_mask = (S_indices % best_p) == best_r
            values_to_remove = S_indices[to_remove_mask]
            S_mask[values_to_remove] = False
            changed = True
        else:
            changed = False
    
    return np.count_nonzero(S_mask), S_mask

def _greedy_conflict_graph(primes, K, start_time, deadline):
    """
    Conflict-graph formulation: vertices are integers in [0, N].
    Two vertices conflict if their difference is divisible by some prime p <= K
    AND they occupy the same residue class mod p (i.e., one blocks the other
    for the admissibility constraint at p).
    
    Actually, the admissibility constraint says: for each prime p <= |S|,
    S must NOT cover all p residue classes mod p. So S is "bad" for prime p
    iff it hits every residue class mod p.
    
    We want to maximize |S| subject to: for all p <= |S|, S does not cover all
    p residue classes mod p.
    
    Greedy minimum-degree approach:
    - Start with all N+1 elements.
    - While there exists a prime p <= current_size such that S covers all p residues,
      remove the residue class (mod p) with the fewest elements.
    - This is essentially what _try_strategy does, but we frame it as
      "remove the vertex with minimum conflict degree."
    
    A better greedy: iteratively pick the element whose removal resolves the
    most "tight" constraints. We approximate conflict degree as: for each
    element x in S, count how many primes p <= |S| have x in a residue class
    that is "critical" (i.e., removing x's residue class mod p would make
    that class empty). Elements in critical classes have higher "degree."
    We remove the element with highest degree first.
    """
    S_mask = np.ones(N + 1, dtype=bool)
    
    # First, sieve small primes up to K to get a good starting point
    small_primes = primes[primes <= K]
    for p in small_primes:
        S_indices = np.where(S_mask)[0]
        if len(S_indices) == 0:
            break
        residues = S_indices % p
        counts = np.bincount(residues, minlength=p)
        best_r = np.argmin(counts)
        to_remove = S_indices[residues == best_r]
        S_mask[to_remove] = False
    
    # Now apply minimum-degree greedy for remaining primes
    remaining_primes = primes[primes > K]
    
    while True:
        if time.time() - start_time > deadline:
            break
        
        current_size = np.count_nonzero(S_mask)
        if current_size == 0:
            break
        
        S_indices = np.where(S_mask)[0]
        candidates = remaining_primes[remaining_primes <= current_size]
        
        if len(candidates) == 0:
            break
        
        # Find all "bad" primes (those where S covers all residues)
        bad_primes_info = []
        for p in candidates:
            residues = S_indices % p
            if len(np.unique(residues)) == p:
                counts = np.bincount(residues, minlength=p)
                bad_primes_info.append((p, counts))
        
        if not bad_primes_info:
            break
        
        # For each bad prime, identify the least-populated residue class.
        # Build a "conflict degree" score for each element: sum over bad primes
        # of whether the element is in the least-populated residue class.
        # Elements in the least-populated class of the most bad primes get removed first.
        
        degree = np.zeros(N + 1, dtype=np.int32)
        
        for p, counts in bad_primes_info:
            # The least populated residue class
            min_count = np.min(counts)
            # All residue classes with minimum count are "critical"
            critical_residues = np.where(counts == min_count)[0]
            
            for r in critical_residues:
                # Add to degree of all elements in this residue class
                mask_r = (S_indices % p) == r
                degree[S_indices[mask_r]] += 1
        
        # Find the element with maximum degree
        max_deg = np.max(degree)
        if max_deg == 0:
            # No critical elements found; fallback to removing from the
            # bad prime with smallest min_count
            p, counts = bad_primes_info[0]
            best_r = np.argmin(counts)
            to_remove = S_indices[(S_indices % p) == best_r]
            S_mask[to_remove] = False
        else:
            # Remove all elements with maximum degree (batch removal)
            to_remove = np.where(degree == max_deg)[0]
            to_remove = to_remove[S_mask[to_remove]]  # only those still in S
            S_mask[to_remove] = False
    
    return S_mask

def _local_search_improve(S_mask, primes, start_time, deadline):
    """Try to add back removed elements to improve the set size while maintaining admissibility."""
    S_mask = S_mask.copy()
    improved = True
    
    while improved and time.time() - start_time < deadline:
        improved = False
        
        # Get current size and check which primes are constraints
        current_size = np.count_nonzero(S_mask)
        if current_size == 0:
            break
        
        # Find all removed elements
        removed = np.where(~S_mask)[0]
        if len(removed) == 0:
            break
        
        # Try adding back each removed element
        for val in removed:
            if time.time() - start_time > deadline:
                break
            
            # Try adding this element
            S_mask[val] = True
            new_size = np.count_nonzero(S_mask)
            
            # Check if still admissible
            if _is_admissible(S_mask, primes):
                improved = True
                current_size = new_size
            else:
                S_mask[val] = False
    
    return S_mask

def construct():
    start_time = time.time()
    deadline = 310.0  # Safety margin (330s kill, leave 20s buffer)
    
    primes = _primes(N)
    
    best_size = 0
    best_mask = None
    
    # Strategy 1: Original sieve-based approach with various thresholds
    thresholds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
    
    for threshold in thresholds:
        if time.time() - start_time > deadline - 60:
            break
        
        # Try with least populated for small primes
        size, mask = _try_strategy(primes, threshold, use_least_populated_for_small=True)
        if size > best_size:
            best_size = size
            best_mask = mask.copy()
        
        # Try with residue 0 for small primes
        size, mask = _try_strategy(primes, threshold, use_least_populated_for_small=False)
        if size > best_size:
            best_size = size
            best_mask = mask.copy()
    
    # Strategy 2: Conflict-graph minimum-degree greedy with various K
    for K in [30, 50, 70, 90, 100, 120, 150, 200]:
        if time.time() - start_time > deadline - 30:
            break
        
        mask = _greedy_conflict_graph(primes, K, start_time, deadline)
        size = np.count_nonzero(mask)
        if size > best_size:
            best_size = size
            best_mask = mask.copy()
    
    # Local search improvement on the best result
    if best_mask is not None and time.time() - start_time < deadline - 5:
        improved_mask = _local_search_improve(best_mask, primes, start_time, deadline)
        improved_size = np.count_nonzero(improved_mask)
        if improved_size > best_size:
            best_size = improved_size
            best_mask = improved_mask
    
    if best_mask is None:
        # Fallback: just return empty or all
        return list(range(N + 1))
    
    return np.where(best_mask)[0].tolist()

# EVOLVE-BLOCK-END
