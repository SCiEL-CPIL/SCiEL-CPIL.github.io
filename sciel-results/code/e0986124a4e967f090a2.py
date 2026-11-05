import numpy as np
import time
import random

def construct():
    start_time = time.time()
    # Safety margin: 120s limit, keep 10s buffer
    deadline = start_time + 110.0

    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                sieve[i * i:limit + 1:i] = [False] * len(sieve[i * i:limit + 1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)
    MAX_L = 100000

    # Helper to calculate counts for a given residue set
    def get_counts(residues):
        counts = np.zeros(MAX_L + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            start = r if r > 0 else p
            if start <= MAX_L:
                counts[start::p] += 1
        return counts

    def find_L(counts):
        # Find the first index where count < 2
        # We only care up to MAX_L
        # np.where returns indices in 0-based for the slice, so add 1
        bad_idx = np.where(counts[1:MAX_L+1] < 2)[0]
        if len(bad_idx) == 0:
            return MAX_L
        return bad_idx[0]

    # --- Phase 1: Fast Greedy Initialization ---
    # We want to cover early numbers heavily.
    # For each prime, pick the residue that covers the most "currently uncovered" spots in 1..5000.
    residues = [0] * n_primes
    counts = np.zeros(MAX_L + 1, dtype=np.int32)
    
    # Process primes in increasing order? Or decreasing?
    # Smaller primes cover more numbers. Let's process in increasing order to build a base.
    for idx in range(n_primes):
        p = primes[idx]
        best_r = 0
        best_score = -1
        
        # Evaluate all residues for small primes, sample for large
        if p <= 100:
            candidates = range(p)
        else:
            # Focus on residues that hit 1..2000
            candidates = set()
            for target in range(1, 2000):
                candidates.add(target % p)
            candidates = list(candidates)
            
        for r in candidates:
            start = r if r > 0 else p
            if start > MAX_L:
                continue
            pos = np.arange(start, MAX_L + 1, p)
            current_counts = counts[pos]
            # We want to increase counts where they are < 2
            # Specifically, prioritize positions that are currently 0 or 1
            # Score: number of positions that go from <2 to >=2
            # Actually, simpler: sum of (1 if count < 2 else 0)
            gained = np.sum(current_counts < 2)
            # Penalize if we are adding to already high counts? 
            # For now, just maximize gain.
            if gained > best_score:
                best_score = gained
                best_r = r
                
        residues[idx] = best_r
        r = best_r
        start = r if r > 0 else p
        if start <= MAX_L:
            counts[start::p] += 1
            
    current_L = find_L(counts)
    best_L = current_L
    best_residues = residues.copy()
    best_counts = counts.copy()

    # --- Phase 2: Iterative Frontier Repair (Polish) ---
    # Strategy:
    # 1. Find the first point `x` where count[x] < 2.
    # 2. If no such point in 1..MAX_L, we are done (or capped).
    # 3. For this point `x`, try changing the residue of every prime `p` to `x % p`.
    # 4. Calculate the "delta" in the total number of "bad" points (count < 2) in the range 1..MAX_L.
    #    Or better, calculate the new L.
    # 5. Choose the prime/residue change that maximizes the new L (or minimizes bad points if L doesn't improve).
    
    # Optimization: We can't try all 168 primes for every single uncovered point if there are many.
    # But we focus on the FRONTIER. The frontier is the first bad point.
    # Changing a residue to cover the frontier point might also cover subsequent points.
    
    # We will run this loop until time runs out or we hit MAX_L.
    # To avoid infinite loops on non-improving steps, we track the best L.
    
    # Precompute prime indices
    prime_indices = np.arange(n_primes)
    
    # For speed, we can limit the search range for "bad points" to near the frontier.
    # But the score depends on the GLOBAL first failure.
    
    # Let's use a "Lookahead" greedy.
    # At each step, identify the current frontier `F`.
    # For each prime `p`, the only residue that covers `F` is `r = F % p`.
    # So we have exactly 1 candidate move per prime.
    # Total candidates: 168.
    # For each candidate, we can quickly estimate the gain.
    
    # To make this faster, we can maintain a list of "bad" points.
    # However, updating the full counts array is O(MAX_L/p).
    # 168 * (MAX_L/p) operations per step might be too slow if done naively in Python.
    # But we use NumPy.
    
    # Let's refine the loop:
    # 1. Get current counts.
    # 2. Find first bad index `F`.
    # 3. If F > MAX_L, break.
    # 4. For each prime i, p = primes[i]:
    #    - New residue r_new = F % p
    #    - If r_new == residues[i], continue (no change)
    #    - Temporarily apply change? No, too slow.
    #    - Calculate potential gain:
    #      We need to know how many points in `F`'s orbit (F, F+p, F+2p...) are currently < 2.
    #      And how many points in the OLD orbit are currently < 2 (and will become uncovered if we remove old residue).
    #      
    #      Let's just do the full update for the top few candidates.
    #      Actually, let's just do the full update for all 168 candidates?
    #      168 updates * 168 primes? No, 168 candidates.
    #      Each update is O(MAX_L/p). Sum of MAX_L/p is MAX_L * sum(1/p) ~ MAX_L * log(log(1000)) ~ 100000 * 2 = 200k ops.
    #      168 * 200k = 33 million ops per step. In NumPy, this is fast.
    #      But we have to do it in a loop.
    
    # Better approach:
    # 1. Find F.
    # 2. For each prime i:
    #    - Calculate score of changing i to cover F.
    #    - Score = (Number of points in new orbit with count < 2) - (Number of points in old orbit with count < 2)
    #    - This is an approximation of the change in "deficit".
    # 3. Pick the best score.
    # 4. Apply the change.
    # 5. Update counts.
    
    # Let's implement this.
    
    # To speed up "count < 2" checks, we can precompute masks? No, counts change.
    
    # We will run this for a fixed number of iterations or until time limit.
    # Each iteration involves 168 "evaluations" (which are just array sums) and 1 "update".
    
    max_iterations = 50000 # Safety, but time limit is the real constraint
    
    for _ in range(max_iterations):
        if time.time() > deadline - 2.0:
            break
            
        # 1. Find Frontier
        # We only need to check up to MAX_L
        # Optimization: If we know the previous frontier, we can start checking there?
        # No, changing a residue earlier might uncover something before F.
        # So we must find the global first bad point.
        
        # Using np.where is O(MAX_L). 100k is fine.
        bad_idx = np.where(counts[1:MAX_L+1] < 2)[0]
        if len(bad_idx) == 0:
            current_L = MAX_L
            break
        F = bad_idx[0] + 1 # 1-based index
        
        # 2. Evaluate candidates
        best_i = -1
        best_r = -1
        best_gain = -10**9
        
        # We want to maximize the number of "fixed" points (from <2 to >=2)
        # and minimize the number of "broken" points (from >=2 to <2)
        
        for i in range(n_primes):
            p = primes[i]
            r_new = F % p
            r_old = residues[i]
            
            if r_new == r_old:
                continue
                
            # Calculate gain for new orbit
            start_new = r_new if r_new > 0 else p
            if start_new > MAX_L:
                new_gain = 0
            else:
                pos_new = np.arange(start_new, MAX_L + 1, p)
                # How many of these are currently < 2?
                new_gain = np.sum(counts[pos_new] < 2)
            
            # Calculate loss for old orbit
            start_old = r_old if r_old > 0 else p
            if start_old > MAX_L:
                old_loss = 0
            else:
                pos_old = np.arange(start_old, MAX_L + 1, p)
                # How many of these are currently >= 2? (These will drop by 1)
                # If they were 2, they become 1 (bad). If they were >2, they become >=2 (still good).
                # So loss is count of positions where counts == 2.
                old_loss = np.sum(counts[pos_old] == 2)
                
            # Net gain in "good" points (points with count >= 2)
            # Actually, we want to minimize the number of bad points.
            # Bad points decrease by (new_gain - old_loss)?
            # Let's check:
            # New orbit: `new_gain` points were bad (<2), now they are good (count increases by 1).
            # Old orbit: `old_loss` points were good (count==2), now they are bad (count decreases to 1).
            # Net change in #bad points = -new_gain + old_loss.
            # We want to minimize #bad points, so maximize (new_gain - old_loss).
            
            gain = new_gain - old_loss
            
            if gain > best_gain:
                best_gain = gain
                best_i = i
                best_r = r_new
        
        # 3. Apply best change
        if best_i != -1:
            p = primes[best_i]
            r_old = residues[best_i]
            r_new = best_r
            
            # Update counts
            start_old = r_old if r_old > 0 else p
            if start_old <= MAX_L:
                counts[start_old::p] -= 1
                
            start_new = r_new if r_new > 0 else p
            if start_new <= MAX_L:
                counts[start_new::p] += 1
                
            residues[best_i] = r_new
            
            # Update best
            # We don't need to recalculate L every step if we trust the gain metric?
            # No, the gain metric is local. The global L might not improve even if local deficit decreases.
            # But usually it does. Let's check L occasionally or if gain is high.
            # For safety, let's just track best_residues based on actual L.
            
            # Recalculating L is O(MAX_L).
            # If we do it every step, 50000 * 100k = 5e9 ops. Too slow.
            # We should only check L if we think we improved the frontier.
            # Or, we can check L every 10 steps.
            
            if _ % 10 == 0:
                new_L = find_L(counts)
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues.copy()
                    best_counts = counts.copy()
        
        # If no improvement found in a while, we might be stuck.
        # But the "Frontier Repair" is quite powerful.
        
    # Final check
    final_L = find_L(counts)
    if final_L > best_L:
        best_L = final_L
        best_residues = residues.copy()

    return best_residues
