"""
Strategy: Deterministic Greedy Construction + Fast Local Search.
1. Greedily select primes based on "gap filling" potential: 
   Prioritize small primes that cover currently uncovered small integers.
2. Use float sums for fast budget tracking during search, with exact 
   Fraction validation only for the final candidate.
3. Optimize residues using a fast local search that iterates through 
   active primes and picks the residue that maximizes the prefix coverage.
4. Use NumPy for O(1) amortized coverage updates via boolean array slicing.
"""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _get_primes(limit=1000):
    return [p for p in range(2, limit + 1) if _is_prime(p)]

# Precompute primes
ALL_PRIMES = _get_primes(1000)
P_ARRAY = np.array(ALL_PRIMES)
N_PRIMES = len(ALL_PRIMES)
PRIME_TO_IDX = {p: i for i, p in enumerate(ALL_PRIMES)}

# Precompute reciprocal floats for fast budget checking
RECIP_FLOATS = np.array([1.0 / p for p in ALL_PRIMES])

def _compute_coverage(active_indices, residues, cap=100000):
    """
    Returns the largest L such that 1..L are covered.
    """
    if not active_indices:
        return 0
    
    covered = np.zeros(cap + 1, dtype=bool)
    
    for idx, r in zip(active_indices, residues):
        p = ALL_PRIMES[idx]
        if r == 0:
            start = p
        else:
            start = r
        if start < 1:
            start += p * ((1 - start + p - 1) // p)
        if start <= cap:
            covered[start::p] = True
            
    if not covered[1]:
        return 0
        
    inv = ~covered[1:]
    idx_false = np.argmax(inv)
    if inv[idx_false]:
        L = 1 + idx_false
    else:
        L = cap
        
    return L

def _validate_budget_exact(active_indices):
    """Check if sum(1/p) <= 1 using exact fractions."""
    total = Fraction(0, 1)
    for idx in active_indices:
        total += Fraction(1, ALL_PRIMES[idx])
    return total <= 1

def _validate_budget_float(active_indices):
    """Fast float check for budget."""
    return np.sum(RECIP_FLOATS[active_indices]) <= 1.0 + 1e-9

def construct():
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: return a known valid solution (Seed-like)
        primes = [p for p in range(13, 1001) if _is_prime(p)]
        return [[p, (i + 1) % p] for i, p in enumerate(primes)]

    rng = random.Random(42)
    
    # 1. Greedy Prime Selection
    # Goal: Cover 1..L with minimal budget.
    # Start with empty set.
    active_indices = []
    active_set = set()
    current_budget_float = 0.0
    
    # We want to cover small numbers first.
    # A prime p is "good" if it covers numbers that are currently uncovered.
    # However, checking coverage for every prime is expensive.
    # Heuristic: Small primes are generally better.
    # Let's try to build a set that covers 1..1000 efficiently.
    
    # Initialize with a strong candidate set known to be good for small L.
    # {2, 3, 7, 41} covers a lot.
    # Let's start with {2, 3, 7} and add others greedily.
    
    # Candidate primes to consider in order of preference (smaller is better usually)
    # But we must respect budget.
    
    # Let's try a specific greedy strategy:
    # 1. Start with 2, 3.
    # 2. Add 7 (1/2+1/3+1/7 = 0.975).
    # 3. Try to add more primes that fit in the remaining 0.025 budget?
    #    1/41 = 0.024. 1/43 = 0.023.
    #    So {2, 3, 7, 41} is a very tight, strong set.
    
    # Let's initialize with {2, 3, 7, 41}
    initial_primes = [2, 3, 7, 41]
    initial_indices = [PRIME_TO_IDX[p] for p in initial_primes]
    
    if not _validate_budget_float(initial_indices):
         # Fallback if something weird
         initial_primes = [2, 3, 7]
         initial_indices = [PRIME_TO_IDX[p] for p in initial_primes]
         
    active_indices = initial_indices[:]
    active_set = set(active_indices)
    
    # Initialize residues
    # Heuristic: For each prime, pick residue that covers the most small uncovered numbers.
    # Let's start with 0 for all, then optimize.
    current_residues = [0] * len(active_indices)
    
    # 2. Local Search on Residues
    # Iterate: for each prime in the set, try all residues and pick the one
    # that maximizes coverage L.
    
    best_L = _compute_coverage(active_indices, current_residues)
    best_active = active_indices[:]
    best_residues = current_residues[:]
    
    # Fast Local Search Loop
    # We can afford many iterations because _compute_coverage is fast (NumPy).
    # Let's run a hill-climbing loop.
    
    max_iterations = 20000
    no_improve_count = 0
    plateau_limit = 500
    
    for _ in range(max_iterations):
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            break
            
        improved = False
        
        # 1. Optimize residues for current set
        # Iterate through primes. For each, try all residues.
        # To speed up, we can just try a few random ones or the best one.
        # Let's try the best one by scanning all residues?
        # If p is small (e.g., 2, 3), scanning is cheap.
        # If p is large (e.g., 41), scanning 41 residues is still cheap.
        
        for i in range(len(active_indices)):
            idx = active_indices[i]
            p = ALL_PRIMES[idx]
            best_r_for_p = current_residues[i]
            best_L_for_p = best_L
            
            # Try all residues?
            # For p <= 100, this is at most 100 evaluations.
            # For p > 100, maybe just sample?
            # Let's sample if p > 50 to save time, else scan.
            if p <= 50:
                candidates = range(p)
            else:
                candidates = [current_residues[i]] + [rng.randrange(p) for _ in range(10)]
                # Also try 0
                if 0 not in candidates:
                    candidates.append(0)
                    
            for r in candidates:
                if r == current_residues[i]:
                    continue
                old_r = current_residues[i]
                current_residues[i] = r
                L = _compute_coverage(active_indices, current_residues)
                if L > best_L_for_p:
                    best_L_for_p = L
                    best_r_for_p = r
                else:
                    current_residues[i] = old_r
            # Apply best found
            if best_r_for_p != current_residues[i]:
                current_residues[i] = best_r_for_p
                best_L = best_L_for_p
                improved = True
        
        # 2. Try to swap primes
        # If we have budget to spare, or if swapping a large prime for a smaller one helps.
        # Let's try to swap the largest prime in the set with a smaller one not in the set.
        
        if len(active_indices) > 1:
            # Find index of largest prime
            max_idx_pos = np.argmax([ALL_PRIMES[i] for i in active_indices])
            max_idx = active_indices[max_idx_pos]
            max_p = ALL_PRIMES[max_idx]
            
            # Find candidates to swap in
            # Must be smaller than max_p to potentially reduce budget or improve coverage?
            # Actually, we want to maintain budget.
            # Let's just try swapping with a random smaller prime.
            
            # Get list of primes not in active set
            active_set_local = set(active_indices)
            candidates = [i for i in range(N_PRIMES) if i not in active_set_local]
            
            # Filter candidates: must be such that budget is still valid
            # Current budget sum
            current_sum = np.sum(RECIP_FLOATS[active_indices])
            # Remove max_p
            new_sum_base = current_sum - RECIP_FLOATS[max_idx]
            
            # Find candidates where new_sum_base + 1/p <= 1
            valid_candidates = []
            for i in candidates:
                if new_sum_base + RECIP_FLOATS[i] <= 1.0 + 1e-9:
                    # Prefer smaller primes?
                    valid_candidates.append(i)
            
            if valid_candidates:
                # Try a few random swaps
                for _ in range(5):
                    new_idx = rng.choice(valid_candidates)
                    new_p = ALL_PRIMES[new_idx]
                    
                    # Swap
                    old_idx = active_indices[max_idx_pos]
                    old_r = current_residues[max_idx_pos]
                    
                    active_indices[max_idx_pos] = new_idx
                    # Reset residue to 0 or random
                    current_residues[max_idx_pos] = rng.randrange(new_p)
                    
                    L = _compute_coverage(active_indices, current_residues)
                    
                    if L > best_L:
                        best_L = L
                        best_active = active_indices[:]
                        best_residues = current_residues[:]
                        improved = True
                    else:
                        # Revert
                        active_indices[max_idx_pos] = old_idx
                        current_residues[max_idx_pos] = old_r
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > plateau_limit:
                break
        else:
            no_improve_count = 0
            best_active = active_indices[:]
            best_residues = current_residues[:]

    # Final validation and return
    # Ensure best solution is valid with exact fractions
    if not _validate_budget_exact(best_active):
        # Fallback
        primes = [p for p in range(13, 1001) if _is_prime(p)]
        return [[p, (i + 1) % p] for i, p in enumerate(primes)]
        
    result = [[ALL_PRIMES[i], r] for i, r in zip(best_active, best_residues)]
    return result
# EVOLVE-BLOCK-END
