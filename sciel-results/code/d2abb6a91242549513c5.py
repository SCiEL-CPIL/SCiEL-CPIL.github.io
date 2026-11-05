import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # 1. Precompute primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    
    # 2. Generate candidate sets of primes
    # We want sets where sum(1/p) <= 1.
    # Strategy: Start with a base set of small primes and try variations.
    # The key is to include small primes (2, 3, 5, 7) as they provide high density coverage.
    
    candidates = []
    
    # Candidate 1: Greedy smallest primes
    def generate_greedy_set(exclude=set()):
        sorted_primes = sorted(primes)
        current_sum = 0
        selected = []
        for p in sorted_primes:
            if p in exclude:
                continue
            if current_sum + 1.0/p <= 1.0:
                selected.append(p)
                current_sum += 1.0/p
        return selected

    # Try including 2
    candidates.append(generate_greedy_set())
    
    # Try excluding 2 but including 3, 5, 7
    candidates.append(generate_greedy_set(exclude={2}))
    
    # Try excluding 2, 3
    candidates.append(generate_greedy_set(exclude={2, 3}))
    
    # Specific curated sets known to be good for covering
    # Set A: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
    # Sum check:
    # 1/2+1/3+1/5+1/7+1/11+1/13+1/17+1/19+1/23+1/29+1/31+1/37+1/41+1/43+1/47+1/53+1/59+1/61+1/67+1/71+1/73+1/79+1/83+1/89+1/97
    # This sum is approx 1.0003... so it might exceed 1.
    # Let's check precisely with Fraction or just float for candidate generation.
    # If it exceeds 1, the greedy will drop the last one.
    
    # Let's add a few manual variations if time permits, but the greedy sets above are the main drivers.
    
    # 3. Evaluate each candidate using Local Search with Counts
    max_L_target = 100000
    best_L = 0
    best_classes = []
    
    for P in candidates:
        if time.time() > deadline - 10:
            break
            
        P_sorted = sorted(P)
        residues = {p: 0 for p in P_sorted}
        
        # Initialize counts array to track how many classes cover each number
        # counts[i] = number of primes p in P such that i % p == residues[p]
        # We only care about 1..max_L_target
        counts = np.zeros(max_L_target + 1, dtype=np.int32)
        
        # Helper to update counts for a prime p changing from r_old to r_new
        def update_counts(p, r_old, r_new, delta):
            # delta is +1 for adding, -1 for removing
            if r_old != r_new:
                # Remove old
                start_old = r_old if r_old > 0 else p
                if start_old <= max_L_target:
                    idx_old = np.arange(start_old, max_L_target + 1, p)
                    counts[idx_old] -= delta
                # Add new
                start_new = r_new if r_new > 0 else p
                if start_new <= max_L_target:
                    idx_new = np.arange(start_new, max_L_target + 1, p)
                    counts[idx_new] += delta

        # Initial Greedy Assignment
        # Order primes by ascending size to lock in small moduli first?
        # Or descending? Small moduli are more restrictive.
        # Let's try ascending.
        for p in P_sorted:
            best_r = 0
            best_gain = -1
            
            # To find best r, we need to know how many new numbers it covers.
            # New numbers are those where counts[i] == 0.
            # So for a given r, gain = sum(counts[i] == 0 for i in class(p, r))
            
            for r in range(p):
                start = r if r > 0 else p
                if start > max_L_target:
                    continue
                idx = np.arange(start, max_L_target + 1, p)
                # Count how many of these are currently uncovered (counts == 0)
                gain = np.sum(counts[idx] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            residues[p] = best_r
            # Update counts with the chosen residue
            update_counts(p, 0, best_r, 1)
            
        # Calculate initial L
        def get_L_from_counts(counts):
            # Find first index i >= 1 where counts[i] == 0
            # We can use np.where
            uncovered = np.where(counts[1:] == 0)[0]
            if len(uncovered) == 0:
                return max_L_target
            return uncovered[0] # Since uncovered[0] is index in 1.., L = index
            
        current_L = get_L_from_counts(counts)
        
        # Local Search: Improve by re-optimizing residues
        # We can do a few passes of "best improvement"
        # For each prime, try all residues and pick the one that maximizes coverage.
        
        max_iter = 20
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            improved = False
            # Iterate through primes in random order? Or fixed?
            # Fixed order is deterministic and faster to debug.
            for p in P_sorted:
                old_r = residues[p]
                
                # Find best r
                best_r = old_r
                best_gain = -1
                
                # We need to calculate gain relative to current coverage.
                # Current coverage includes p's contribution.
                # To find gain of changing r, we should:
                # 1. Remove p's current contribution (counts -= 1 on old class)
                # 2. Try all r, calculate gain (sum(counts[idx] == 0))
                # 3. Keep the best
                # 4. If best is different, update counts
                
                # Remove current contribution of p
                # Note: We must be careful. If we remove it, counts change.
                # We can simulate removal.
                # Let's create a temporary view or just do the math.
                # Since we are doing this for one p at a time, we can:
                # 1. Decrement counts for old_r
                # 2. Evaluate all r
                # 3. Increment counts for best_r
                # 4. If best_r == old_r, we effectively did nothing (decrement then increment)
                
                # Optimization: We can calculate the gain without modifying the global counts array
                # by noting that:
                # gain(r) = (number of i in class(p,r) where counts[i] == 0)
                # But if we remove old_r first, counts[i] for i in class(p, old_r) decreases by 1.
                # So for i in class(p, old_r), the condition becomes (counts[i] - 1 == 0) i.e. counts[i] == 1.
                # For i NOT in class(p, old_r), the condition remains (counts[i] == 0).
                
                # Let's just do the explicit update/remove/add. It's O(N/p) per r, and we do it for p residues.
                # Total O(N) per prime per iteration. With ~25 primes and 20 iterations, that's 500 * N operations.
                # N=100,000. 50 million operations. Should be fast in NumPy.
                
                # Remove old
                start_old = old_r if old_r > 0 else p
                if start_old <= max_L_target:
                    idx_old = np.arange(start_old, max_L_target + 1, p)
                    counts[idx_old] -= 1
                    
                for r in range(p):
                    start = r if r > 0 else p
                    if start > max_L_target:
                        gain = 0
                    else:
                        idx = np.arange(start, max_L_target + 1, p)
                        # Now counts reflects the state WITHOUT p
                        gain = np.sum(counts[idx] == 0)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                
                # Add best
                start_best = best_r if best_r > 0 else p
                if start_best <= max_L_target:
                    idx_best = np.arange(start_best, max_L_target + 1, p)
                    counts[idx_best] += 1
                
                if best_r != old_r:
                    residues[p] = best_r
                    improved = True
                    # Recalculate L to see if we improved?
                    # We can just do it at the end of the pass, or every step.
                    # Let's do it at the end of the pass to save time.
                    # But we need to break if we hit deadline.
                    
            # End of pass for all primes
            # Check L
            new_L = get_L_from_counts(counts)
            if new_L > current_L:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_classes = [[p, residues[p]] for p in P_sorted]
            else:
                # If no improvement in L, and no residues changed, we can stop early?
                # If improved is False, no residues changed, so L is same.
                if not improved:
                    break
        
    # Fallback if nothing found (should not happen)
    if not best_classes:
        # Use the first candidate set with default residues
        P = candidates[0]
        best_classes = [[p, 0] for p in P]
        
    # Final Verification with Fractions
    s = 0
    for p, r in best_classes:
        s += Fraction(1, p)
    if s > 1:
        # Fallback to a known valid small set
        return [[p, 0] for p in [13, 17, 19]]
        
    return best_classes
