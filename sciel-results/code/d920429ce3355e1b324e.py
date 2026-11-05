"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 5000
DEADLINE = 325  # seconds, safety margin before 330s kill

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]

def construct():
    start_time = time.time()
    
    # Start with the full set [0, 5000]
    S = set(range(N + 1))
    
    # Get primes up to N+1 (max possible size)
    primes = _primes(N + 1)
    
    # Convert to numpy array for faster residue counting
    S_arr = np.array(sorted(S), dtype=np.int64)
    
    for p in primes:
        # Check deadline
        if time.time() - start_time > DEADLINE:
            break
            
        current_size = len(S_arr)
        
        # If p is greater than the current size, we don't need to satisfy this constraint
        if p > current_size:
            break
            
        # Count residues mod p using numpy
        residues = S_arr % p
        counts = np.bincount(residues, minlength=p)
        
        # Check if any class is empty
        if counts.min() == 0:
            continue
            
        # All classes are covered. We must remove elements from one class.
        # Use a smooth surrogate: minimize the "regret" of removing a class,
        # which is related to how many future constraints might be affected.
        # For now, use the min-count heuristic but with a tie-breaking
        # that considers the spread of remaining counts (smoothness).
        
        min_count = counts.min()
        min_indices = np.where(counts == min_count)[0]
        
        if len(min_indices) == 1:
            target_residue = min_indices[0]
        else:
            # Tie-breaking: choose the residue class whose removal
            # leaves the most balanced distribution for future primes.
            # Approximate by choosing the residue that minimizes the
            # variance of remaining counts (smooth surrogate).
            best_residue = min_indices[0]
            best_var = float('inf')
            for r in min_indices:
                # If we remove residue r, the new counts are counts with r set to 0
                new_counts = counts.copy()
                new_counts[r] = 0
                # Variance of non-zero counts as a proxy for balance
                nonzero = new_counts[new_counts > 0]
                if len(nonzero) > 0:
                    var = float(np.var(nonzero))
                else:
                    var = 0
                if var < best_var:
                    best_var = var
                    best_residue = r
            target_residue = best_residue
        
        # Remove all x in S such that x % p == target_residue
        mask = residues != target_residue
        S_arr = S_arr[mask]
    
    return list(S_arr)
# EVOLVE-BLOCK-END
