import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s deadline to be safe
    
    # Candidate prime sets to try, in order of size
    # We want to minimize the number of distinct primes.
    # Start with smallest sets.
    candidate_sets = [
        [2],
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11, 13, 17],
    ]
    
    def is_smooth(n, primes):
        # Check if all prime factors of n are in primes
        # n is sum of two distinct integers <= 10000, so n <= 19999
        for p in primes:
            while n % p == 0:
                n //= p
        return n == 1

    def find_clique_for_primes(primes, max_restarts=1000):
        # Greedy randomized clique search
        # Nodes are 1..10000
        # We want a clique of size 12
        
        # Optimization: Pre-filter nodes?
        # If we pick u, v must be such that u+v is smooth.
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Start with a random node
            # To improve chances, maybe start with small numbers?
            # Sums of small numbers are small, more likely to be smooth?
            # Let's mix: 50% small, 50% random
            if random.random() < 0.5:
                u1 = random.randint(1, 100)
            else:
                u1 = random.randint(1, 10000)
            
            clique = [u1]
            candidates = [v for v in range(1, 10001) if v != u1]
            
            # Filter candidates for u1
            candidates = [v for v in candidates if is_smooth(u1 + v, primes)]
            
            if len(candidates) < 11:
                continue
                
            # Greedily pick next nodes
            # To speed up, we can shuffle and pick the first valid one, or pick the one with highest connectivity
            # For simplicity and speed, let's just pick randomly from valid candidates
            # and filter the rest.
            
            current_clique = [u1]
            valid_candidates = candidates
            
            for step in range(1, 12):
                if time.time() > deadline:
                    return None
                
                if not valid_candidates:
                    break
                
                # Pick a random candidate
                # To increase success rate, we might want to pick one that connects to many others
                # But calculating degree is O(N).
                # Let's just pick randomly.
                idx = random.randint(0, len(valid_candidates) - 1)
                u_next = valid_candidates[idx]
                current_clique.append(u_next)
                
                # Filter valid_candidates
                # Remove u_next and any v where u_next + v is not smooth
                new_valid = []
                for v in valid_candidates:
                    if v == u_next:
                        continue
                    if is_smooth(u_next + v, primes):
                        new_valid.append(v)
                valid_candidates = new_valid
                
                if len(valid_candidates) < 11 - step:
                    break
            
            if len(current_clique) == 12:
                # Verify distinctness (already ensured by v != u and filtering)
                # Verify bounds (1..10000)
                return current_clique
        
        return None

    best_result = None
    
    for primes in candidate_sets:
        if time.time() > deadline:
            break
        
        # Try to find a clique for this prime set
        # We can try a few restarts
        result = find_clique_for_primes(primes, max_restarts=200)
        
        if result is not None:
            best_result = result
            # Since we iterate by size of prime set, the first one found is optimal
            # (assuming we check sets in increasing order of size)
            break

    if best_result is not None:
        return best_result
    
    # Fallback: Return the seed solution if no better found
    # Or try a hardcoded good solution if known
    # For now, return seed
    return list(range(1, 13))

# EVOLVE-BLOCK-END
