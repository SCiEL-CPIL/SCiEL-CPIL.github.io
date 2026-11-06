"""Structural rewrite: treat the covering problem as a constraint satisfaction / local search
over residue classes, with a fast NumPy-based evaluation of the doubly-covered prefix length.
We use simulated annealing / hill climbing to maximize L."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE = 150.0  # wall-clock budget (seconds)
_START_TIME = time.time()

def _primes_up_to(n):
    """Return list of primes <= n."""
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return np.array(np.where(sieve)[0], dtype=np.int64)

def _compute_L(residues, primes, max_L=100000):
    """
    Given residues (array of same length as primes) and primes array,
    compute the largest L such that every integer 1..L satisfies at least
    two congruences x == r_p (mod p).
    Returns L (integer).
    """
    n = len(primes)
    # For each x in 1..max_L, count how many primes p have x % p == residues[i]
    # We do this efficiently: for each prime p_i with residue r_i, the covered
    # numbers in 1..max_L are those x such that x ≡ r_i (mod p_i).
    # Instead of iterating over all x, we can use a counting array.
    
    # Create a count array for 1..max_L
    counts = np.zeros(max_L + 1, dtype=np.int32)
    
    for i in range(n):
        p = primes[i]
        r = residues[i]
        # Numbers x in 1..max_L with x % p == r
        # First such number >= 1: 
        # If r == 0, numbers are p, 2p, ...
        # If r > 0, numbers are r, r+p, r+2p, ... but must be >= 1
        if r == 0:
            start = p
        else:
            start = r
        # Ensure start >= 1
        if start < 1:
            start += p * ((1 - start + p - 1) // p)
        # Iterate
        for x in range(start, max_L + 1, p):
            counts[x] += 1
    
    # Find largest L such that counts[1..L] >= 2
    # We need the first index where counts < 2
    # Use cumsum trick or just find first violation
    # Find all indices where counts < 2
    violation_mask = counts < 2
    # The first violation in 1..max_L
    violations = np.where(violation_mask[1:])[0] + 1  # +1 to convert 0-indexed to 1-indexed
    if len(violations) == 0:
        return max_L
    else:
        # The largest L is violations[0] - 1
        return int(violations[0] - 1)

def _score(residues, primes, max_L=100000):
    """Wrapper for scoring."""
    return _compute_L(residues, primes, max_L)

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    primes = _primes_up_to(1000)
    n = len(primes)
    max_L = 100000
    
    # Initial solution: use the seed strategy as a starting point
    # The seed: r_i = ceil((i+2)/2) % p_i = ((i+2)//2) % p_i
    residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int64)
    
    current_score = _score(residues, primes, max_L)
    best_residues = residues.copy()
    best_score = current_score
    
    # Simulated annealing / hill climbing
    # We'll try random perturbations and accept improvements or occasional worse solutions
    
    # For efficiency, we'll use a smaller max_L for intermediate evaluations
    # and only do full evaluation for candidates that look promising
    eval_max_L = 2000  # smaller for speed during search
    
    temp = 50.0
    cooling = 0.999
    
    # Precompute for fast evaluation
    # We'll use a vectorized approach for the small eval
    
    def fast_score(residues, primes, max_L=eval_max_L):
        """Fast score for smaller max_L."""
        n = len(primes)
        counts = np.zeros(max_L + 1, dtype=np.int32)
        for i in range(n):
            p = primes[i]
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            if start < 1:
                start += p
            for x in range(start, max_L + 1, p):
                counts[x] += 1
        violation_mask = counts < 2
        violations = np.where(violation_mask[1:])[0] + 1
        if len(violations) == 0:
            return max_L
        else:
            return int(violations[0] - 1)
    
    # Better fast score using vectorized operations
    def fast_score_vec(residues, primes, max_L=eval_max_L):
        """Vectorized fast score."""
        n = len(primes)
        counts = np.zeros(max_L + 1, dtype=np.int32)
        for i in range(n):
            p = primes[i]
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            if start < 1:
                start += p
            indices = np.arange(start, max_L + 1, p)
            counts[indices] += 1
        violation_mask = counts < 2
        violations = np.where(violation_mask[1:])[0] + 1
        if len(violations) == 0:
            return max_L
        else:
            return int(violations[0] - 1)
    
    # Even faster: precompute the step indices for each prime
    # For each prime p, the set of indices in 1..max_L that are ≡ r (mod p)
    # depends on r, so we can't fully precompute, but we can precompute the base patterns
    
    # Let's use a different approach: for each x in 1..max_L, we need to count
    # how many primes p have x % p == r_p. This is inherently hard to vectorize
    # across all p at once because each p has a different modulus.
    
    # Alternative: iterate over x and for each x, check which primes cover it.
    # For a given x, prime p covers x if x % p == r_p.
    # We can vectorize over p for a fixed x.
    
    def fast_score_xloop(residues, primes, max_L=eval_max_L):
        """Iterate over x, vectorize over primes."""
        n = len(primes)
        for x in range(1, max_L + 1):
            # Count how many primes p have x % p == residues[i]
            # Compute x % p for all p
            mods = x % primes
            count = np.sum(mods == residues)
            if count < 2:
                return x - 1
        return max_L
    
    # Let's benchmark which is faster
    # For max_L=2000 and n=168, the x-loop with vectorized mod should be fast
    
    # Use the x-loop approach for fast evaluation
    def fast_eval(residues, primes, max_L=eval_max_L):
        return fast_score_xloop(residues, primes, max_L)
    
    # Main search loop
    rng = np.random.default_rng(42)
    
    while time.time() - _START_TIME < _DEADLINE - 5:  # 5 second safety margin
        # Generate a perturbation
        # Randomly change one or a few residues
        num_changes = rng.integers(1, 4)  # 1-3 changes
        indices = rng.choice(n, size=num_changes, replace=False)
        
        new_residues = residues.copy()
        for idx in indices:
            p = primes[idx]
            new_residues[idx] = rng.integers(0, p)
        
        new_score = fast_eval(new_residues, primes, eval_max_L)
        
        # Acceptance criterion
        if new_score >= current_score:
            # Always accept improvement or equal
            residues = new_residues
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_residues = residues.copy()
        else:
            # Accept worse with some probability (simulated annealing)
            if temp > 1:
                prob = np.exp((new_score - current_score) / temp)
                if rng.random() < prob:
                    residues = new_residues
                    current_score = new_score
        
        temp *= cooling
        
        # Periodically do a full evaluation to check if we've actually improved
        # and to update best
        if current_score > best_score:
            # Do a full evaluation to confirm
            full_score = _score(residues, primes, max_L)
            if full_score > best_score:
                best_score = full_score
                best_residues = residues.copy()
            else:
                # The fast eval was optimistic; revert to best
                residues = best_residues.copy()
                current_score = best_score
    
    # Final full evaluation of best
    final_score = _score(best_residues, primes, max_L)
    
    return list(best_residues)
# EVOLVE-BLOCK-END
