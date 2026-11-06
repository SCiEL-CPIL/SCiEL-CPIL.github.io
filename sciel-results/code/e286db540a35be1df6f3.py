"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _sieve_primes(limit):
    """Return list of primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _score(residues, primes, max_check=100000):
    """
    Compute the largest L such that every integer 1..L satisfies at least two congruences.
    Uses a counting array for efficiency.
    """
    n = len(primes)
    if n == 0:
        return 0
    
    # We check up to max_check, but we can stop early if we find a gap
    # The score is the first x where count[x] < 2, minus 1.
    # If no such x in 1..max_check, score is max_check (capped).
    
    # Optimization: 
    # counts[x] = sum(1 for p, r in zip(primes, residues) if x % p == r)
    # We can update counts incrementally? No, changing one residue changes p entries.
    # For a full score calculation from scratch:
    # counts = np.zeros(max_check + 1, dtype=np.int16)
    # for p, r in zip(primes, residues):
    #     # x such that x % p == r are r, r+p, r+2p...
    #     # Note: if r == 0, x are p, 2p... (since we check 1..L, x=0 is not in range)
    #     # Actually, residues are 0 <= r < p.
    #     # If r == 0, the first positive x is p.
    #     # If r > 0, the first positive x is r.
    #     start = r if r > 0 else p
    #     if start > max_check: continue
    #     counts[start::p] += 1
    
    # However, doing this for every local search step is O(N * L) which might be too slow if N=168 and L=10000.
    # 168 * 10000 = 1.68M operations per score.
    # If we do 1000 local search steps, that's 1.68B operations. Too slow in Python loops.
    # We need a faster scoring or a smarter search.
    
    # Alternative: Maintain the counts array incrementally.
    # But this function is called independently.
    
    # Let's optimize the counting loop using NumPy vectorization where possible, 
    # but the strides are different for each prime.
    
    # Actually, for L around 1000-10000, the loop is fast enough if we limit the search space.
    # Let's try a hybrid: 
    # 1. Fast check for small L (e.g., up to 5000) using a simple loop or optimized numpy.
    # 2. If L > 5000, do a more detailed check.
    
    # Let's use a list of counts for speed in the inner loop if max_check is small, 
    # or numpy if large.
    # Given the budget, let's assume we care about L up to ~10000.
    
    counts = np.zeros(max_check + 1, dtype=np.int16)
    
    for i in range(n):
        p = primes[i]
        r = residues[i]
        
        # Indices: r, r+p, r+2p, ...
        # If r == 0, indices are p, 2p, ... (since x>=1)
        # If r > 0, indices are r, r+p, ...
        
        if r == 0:
            start = p
        else:
            start = r
            
        if start > max_check:
            continue
            
        # counts[start::p] += 1
        counts[start::p] += 1
        
    # Find the first index k in 1..max_check where counts[k] < 2
    # We want the largest L such that counts[1..L] >= 2.
    # This is equivalent to finding the first k where counts[k] < 2.
    # If such k exists, L = k - 1.
    # If no such k, L = max_check.
    
    # np.where(counts[1:] < 2)[0] gives indices relative to 1.
    # Let's be careful with memory.
    
    # We only need to check up to a reasonable bound first to save time?
    # No, we need the exact L.
    
    # Optimization: Check in chunks?
    # For now, assume max_check is 10000 for speed during search, 
    # and do a full 100000 check only at the end or if we suspect L > 10000.
    
    # However, the task says "capped at 10^5". 
    # To be safe and fast, let's check up to 10000 first.
    # If L >= 10000, we might need to check further.
    
    # Let's re-implement _score to take a limit parameter.
    
    pass

def _score_fast(residues, primes, limit):
    """
    Compute L for x in 1..limit.
    Returns the largest L <= limit such that all x in 1..L have count >= 2.
    """
    n = len(primes)
    counts = np.zeros(limit + 1, dtype=np.int16)
    
    for i in range(n):
        p = primes[i]
        r = residues[i]
        
        if r == 0:
            start = p
        else:
            start = r
            
        if start > limit:
            continue
            
        counts[start::p] += 1
        
    # Find first index k in 1..limit where counts[k] < 2
    # counts[0] is unused.
    # We look at counts[1:limit+1]
    
    # np.argwhere(counts[1:] < 2) might be slow if the array is large and the gap is late.
    # But limit is small (e.g. 20000).
    
    # Let's just iterate? No, numpy is faster.
    # Create a boolean mask
    mask = counts[1:limit+1] < 2
    if np.any(mask):
        first_idx = np.argmax(mask) # 0-indexed relative to 1
        return first_idx # This is k-1. If first gap is at k=5, mask[4] is True, argmax=4, L=4.
    else:
        return limit

def _full_score(residues, primes, max_check=100000):
    """
    Full score calculation.
    """
    # First check a smaller limit to see if we can stop early
    # If L < 5000, we can return that.
    # If L >= 5000, we need to check up to 100000.
    
    # Strategy:
    # 1. Check up to 10000.
    # 2. If result is 10000, check up to 100000.
    
    L1 = _score_fast(residues, primes, 10000)
    if L1 < 10000:
        return L1
    else:
        return _score_fast(residues, primes, 100000)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)
    
    # Initial heuristic: 
    # The seed used (i+2)//2 % p.
    # A better initial guess might be to try to cover small integers.
    # For small x, we want x % p == r_p.
    # This is a system of congruences.
    # We can't satisfy all, but we can try to maximize the prefix.
    
    # Let's start with the seed's strategy as a baseline, 
    # or a random one? 
    # The seed got 84.
    # Let's try a greedy assignment.
    # For each prime p, choose r_p to maximize the number of x in [1, Current_L + p] 
    # that are covered by p AND already covered by others? 
    # This is complex.
    
    # Simpler: Start with random residues or the seed's, then local search.
    # Given the "local search" lesson, let's implement a robust local search.
    
    # Initial residues: 
    # Let's try to set r_p such that it covers a specific small number?
    # Or just use the seed's init.
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    current_L = _full_score(residues, primes)
    
    # Local Search
    # We want to increase L.
    # The bottleneck is the first x where count[x] < 2.
    # Let's call this x_gap.
    # If we can increase the count of x_gap by changing some r_p, 
    # or move the gap to a larger x, we improve.
    
    # Algorithm:
    # 1. Find x_gap (first x with count < 2).
    # 2. Identify primes p that DO NOT cover x_gap (i.e., x_gap % p != r_p).
    # 3. For each such prime p, calculate if changing r_p to x_gap % p 
    #    would help.
    #    - It increases count[x_gap] by 1.
    #    - It decreases count[x] by 1 for all other x covered by the old r_p.
    #    - This might create a new gap at some y < x_gap? Unlikely if y was covered >= 2.
    #    - It might create a gap at some y > x_gap? Possible.
    # 4. We need to evaluate the new L.
    
    # To make this efficient:
    # - Maintain the counts array.
    # - When we change r_p from old_r to new_r:
    #   - Subtract 1 from counts at indices old_r, old_r+p, ...
    #   - Add 1 to counts at indices new_r, new_r+p, ...
    #   - Find the new first gap.
    
    # This allows O(L/p) update per move, which is fast.
    
    # Let's implement this incremental local search.
    
    limit_check = 20000 # Check up to 20000 for speed during search
    # If L grows beyond 20000, we might need to extend, but let's see.
    
    counts = np.zeros(limit_check + 1, dtype=np.int16)
    
    def update_counts(p, old_r, new_r, delta):
        """Update counts array for a change in residue for prime p."""
        # Remove old
        if old_r == 0:
            start = p
        else:
            start = old_r
        if start <= limit_check:
            counts[start::p] -= delta
            
        # Add new
        if new_r == 0:
            start = p
        else:
            start = new_r
        if start <= limit_check:
            counts[start::p] += delta
            
    # Initialize counts
    for i, p in enumerate(primes):
        r = residues[i]
        if r == 0:
            start = p
        else:
            start = r
        if start <= limit_check:
            counts[start::p] += 1
            
    # Function to get current L from counts
    def get_L_from_counts(counts, limit):
        # Find first index k in 1..limit where counts[k] < 2
        mask = counts[1:limit+1] < 2
        if np.any(mask):
            return np.argmax(mask)
        else:
            return limit
            
    current_L = get_L_from_counts(counts, limit_check)
    
    # Local Search Loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Find the current gap
        if current_L >= limit_check:
            # If we hit the limit, we might need to expand or just stop
            # For now, assume we are good if we hit the limit.
            # But we should try to push further?
            # Let's break if we hit the limit to avoid complexity, 
            # or increase limit_check.
            # Let's increase limit_check if we hit it.
            limit_check = min(limit_check * 2, 100000)
            # Re-initialize counts for the new limit? 
            # Or just extend the array?
            # Extending is messy. Let's just stop if we hit 100000.
            if limit_check >= 100000:
                break
            # Re-init counts for new limit
            counts = np.zeros(limit_check + 1, dtype=np.int16)
            for i, p in enumerate(primes):
                r = residues[i]
                if r == 0:
                    start = p
                else:
                    start = r
                if start <= limit_check:
                    counts[start::p] += 1
            current_L = get_L_from_counts(counts, limit_check)
            continue
            
        x_gap = current_L + 1
        
        # Find primes that do NOT cover x_gap
        # x_gap % p != residues[i]
        candidates = []
        for i, p in enumerate(primes):
            if x_gap % p != residues[i]:
                candidates.append(i)
                
        # Try each candidate
        # Shuffle? Or try all?
        # Trying all is O(N * (L/p)) which is manageable.
        
        best_delta = 0
        best_i = -1
        best_new_r = -1
        
        for i in candidates:
            p = primes[i]
            old_r = residues[i]
            new_r = x_gap % p
            
            # Apply change
            update_counts(p, old_r, new_r, 1)
            
            new_L = get_L_from_counts(counts, limit_check)
            
            # Revert change
            update_counts(p, new_r, old_r, 1)
            
            if new_L > current_L:
                best_delta = new_L - current_L
                best_i = i
                best_new_r = new_r
                
        if best_i != -1:
            # Apply the best move
            p = primes[best_i]
            old_r = residues[best_i]
            residues[best_i] = best_new_r
            update_counts(p, old_r, best_new_r, 1)
            current_L = get_L_from_counts(counts, limit_check)
            improved = True
            
    # Final score with full limit
    final_L = _full_score(residues, primes)
    
    return residues

# EVOLVE-BLOCK-END
