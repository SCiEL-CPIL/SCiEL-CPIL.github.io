"""
Strategy: Simulated Annealing with Vectorized Evaluation.
Instead of greedily fixing the first failure point (which gets stuck in local minima),
we treat the problem as an optimization landscape where we can accept worse moves.
We use a vectorized method to compute the score (longest prefix with count >= 2)
to allow many iterations within the time limit.
"""

import time
import numpy as np
import random

def _get_primes(limit):
    """Sieve of Eratosthenes to get primes up to limit."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    
    # We check up to 20000 to allow for potential jumps, 
    # but the score is capped by the first failure.
    # If we reach 20000, we assume it's very good.
    max_check = 20000
    
    # Initialize residues randomly to avoid symmetric traps of zeros
    residues = np.random.randint(0, [p for p in primes], size=n_primes, dtype=np.int64)
    
    best_score = 0
    best_residues = residues.copy()
    
    # Precompute indices for fast vectorized count updates?
    # Actually, updating counts for one prime change is O(max_check/p).
    # Vectorized score calculation is O(max_check).
    
    def get_score(res):
        """
        Vectorized calculation of the score:
        1. Compute counts for all x in 1..max_check.
        2. Find the first x where counts[x] < 2.
        """
        # Efficient way to compute counts:
        # counts = np.zeros(max_check + 1, dtype=np.int32)
        # for i, p in enumerate(primes):
        #     r = res[i]
        #     start = r if r > 0 else p
        #     if start <= max_check:
        #         counts[start::p] += 1
        
        # This loop is 168 iterations. Each is a numpy slice operation.
        # 168 * (20000/p) operations.
        # Sum(20000/p) for p<=1000 is approx 20000 * sum(1/p) ~ 20000 * 3 = 60000 operations.
        # This is very fast in NumPy.
        
        counts = np.zeros(max_check + 1, dtype=np.int32)
        for i in range(n_primes):
            p = primes[i]
            r = res[i]
            start = r if r > 0 else p
            if start <= max_check:
                counts[start::p] += 1
        
        # Find first index in 1..max_check where count < 2
        # We want the largest L such that 1..L have count >= 2.
        # So we look for the first index where count < 2.
        # If none found in 1..max_check, score is max_check.
        
        # np.where(counts[1:max_check+1] < 2)[0] gives indices relative to 1..max_check
        # If empty, score is max_check.
        # If not empty, first index is idx, so L = idx (since 1-based indexing in the slice corresponds to value idx+1? 
        # No. counts[1] is index 0 of the slice. If counts[1] < 2, idx=0, L=0.
        # If counts[1]>=2, counts[2]<2, idx=1, L=1.
        # So L = idx.
        
        failures = np.where(counts[1:max_check+1] < 2)[0]
        if len(failures) == 0:
            return max_check
        return int(failures[0])

    # Initial score
    current_score = get_score(residues)
    if current_score > best_score:
        best_score = current_score
        best_residues = residues.copy()
    
    # Simulated Annealing Parameters
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # We will perform iterations until time runs out.
    # Each iteration:
    # 1. Pick a random prime index i.
    # 2. Pick a new random residue r_new for that prime.
    # 3. Calculate new score.
    # 4. Accept if better, or with probability exp((new-old)/T).
    
    # Optimization: 
    # Calculating full score every time is O(60000) ops.
    # 10,000 iterations * 60,000 ops = 600M ops. 
    # In NumPy, 600M simple ops might take ~1-2 seconds? 
    # Actually, the loop overhead in Python for 168 primes is the bottleneck.
    # Let's estimate: 168 numpy slice adds.
    # If we can do ~1000 iterations/sec, that's 145,000 iterations.
    # That should be enough to find a good solution.
    
    # To speed up score calculation, we can maintain the counts array 
    # and only update the differences?
    # But SA requires accepting worse moves, so we need to revert.
    # Reverting requires knowing the old state.
    # It is easier to just recalculate from scratch if it's fast enough.
    
    # Let's try to optimize the score calculation by using a precomputed list of indices?
    # No, the residues change.
    
    # Alternative: Use a "delta" approach for score?
    # Hard because score is a global property (prefix length).
    
    # Let's stick to full recalculation but optimize the loop.
    # We can vectorize the update of counts if we represent residues differently?
    # No, residues are sparse.
    
    # Let's just run the loop and see.
    
    iter_count = 0
    while time.time() < deadline:
        iter_count += 1
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Pick a random prime to modify
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        
        # Pick a new random residue
        old_r = residues[i]
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
            
        # Apply change
        residues[i] = new_r
        
        # Evaluate
        new_score = get_score(residues)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta >= 0 or random.random() < np.exp(delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_residues = residues.copy()
        else:
            # Revert
            residues[i] = old_r
            
        # Periodically, we might want to do a "restart" or "kick" if stuck?
        # If T is very low and we haven't improved in a long time, we could increase T?
        # For now, simple SA.
        
    return [int(r) for r in best_residues]
