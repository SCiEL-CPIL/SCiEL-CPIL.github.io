import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin for 60s limit? Task says 60s finish, 90s kill.
    
    # Precompute Smallest Prime Factor (SPF) up to 20000
    MAX_SUM = 20000
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
                    
    def factorize(n):
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors

    # Initial Set: 12 distinct even numbers
    # Using a pool of even numbers
    pool = list(range(2, 10001, 2))
    random.shuffle(pool)
    current_set = pool[:12]
    
    # Calculate initial prime counts
    # prime_counts: dict prime -> count
    prime_counts = {}
    
    def get_prime_counts(s):
        counts = {}
        n = len(s)
        for i in range(n):
            for j in range(i+1, n):
                sum_val = s[i] + s[j]
                factors = factorize(sum_val)
                for p in factors:
                    counts[p] = counts.get(p, 0) + 1
        return counts

    current_counts = get_prime_counts(current_set)
    current_score = len(current_counts)
    
    # Local Search
    # We try to swap an element with another from the pool
    # To be efficient, we pre-compute factors for pool elements? 
    # No, just factorize on the fly. 66 pairs * factorization cost is small.
    
    # We will iterate. 
    # To avoid getting stuck, we can use a simple hill climbing.
    # If we get stuck, we can do a random restart or perturb.
    
    best_set = current_set[:]
    best_score = current_score
    
    # Pool for replacement: use a larger random sample or the whole pool
    # Let's use a subset of the pool to speed up, or just the whole pool.
    # 5000 candidates * 12 swaps * 11 factorizations = ~660k factorizations.
    # Factorization is very fast with SPF.
    
    while time.time() < deadline:
        improved = False
        # Try to improve by swapping
        # Randomly pick an index to swap
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        
        # We want to find a new_val that reduces the score
        # We can iterate through a sample of the pool
        # Or just pick random candidates
        
        # Let's try a batch of random candidates
        for _ in range(100):
            new_val = random.choice(pool)
            if new_val == old_val:
                continue
            # Check distinctness
            if new_val in current_set:
                continue
            
            # Calculate new counts
            # We need to update counts based on old_val and new_val
            # Old contributions: pairs (old_val, current_set[j]) for j != idx
            # New contributions: pairs (new_val, current_set[j]) for j != idx
            
            # To avoid modifying current_counts in place and reverting, 
            # we can calculate the delta.
            
            # 1. Decrement old primes
            temp_counts = current_counts.copy() # This is expensive. 
            # Better: just calculate the score change directly.
            
            # Let's just calculate the full set of primes for the new set?
            # No, that's 66 factorizations. 
            # Updating counts:
            # Remove old_val's partners
            for j in range(12):
                if j == idx: continue
                partner = current_set[j]
                sum_old = old_val + partner
                factors_old = factorize(sum_old)
                for p in factors_old:
                    temp_counts[p] -= 1
                    if temp_counts[p] == 0:
                        del temp_counts[p]
                        
            # Add new_val's partners
            for j in range(12):
                if j == idx: continue
                partner = current_set[j]
                sum_new = new_val + partner
                factors_new = factorize(sum_new)
                for p in factors_new:
                    if p not in temp_counts:
                        temp_counts[p] = 0
                    temp_counts[p] += 1
            
            new_score = len(temp_counts)
            
            if new_score < current_score:
                # Accept
                current_set[idx] = new_val
                current_counts = temp_counts
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_set = current_set[:]
                break # Restart search from new state
                
            # If new_score == current_score, we might accept to escape local minima?
            # For now, strict improvement.
            
        if not improved:
            # No improvement found in this batch, maybe perturb or restart?
            # Let's just continue with different random candidates.
            # If we are stuck for a long time, we could restart.
            pass

    return best_set

# EVOLVE-BLOCK-END
