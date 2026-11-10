import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s deadline, 5s safety margin

    # 1. Generate a list of small primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    # Sieve up to 10^6 to have plenty of candidates for swapping
    primes = sieve(1000000)
    
    # 2. Optimized tau_perp with bitmask-based gcd check
    def tau_perp_fast(prime_factors):
        k = len(prime_factors)
        if k == 0:
            return 0
            
        # Build list of (value, bitmask) for all divisors
        # Start with divisor 1 (empty set)
        divs = [(1, 0)]
        
        for i, p in enumerate(prime_factors):
            bit = 1 << i
            # Extend the list by multiplying existing divisors by p
            # We must not modify the list while iterating over it in a way that affects the current iteration logic 
            # so we create new entries and extend
            current_len = len(divs)
            for j in range(current_len):
                val, mask = divs[j]
                divs.append((val * p, mask | bit))
        
        # Sort by value
        divs.sort(key=lambda x: x[0])
        
        # Count adjacent coprime pairs using bitmask AND
        score = 0
        for i in range(len(divs) - 1):
            if divs[i][1] & divs[i+1][1] == 0:
                score += 1
        return score

    # 3. Initialize with product of smallest 15 primes
    # We established max k is 15
    max_k = 15
    current_primes = list(primes[:max_k])
    current_n = 1
    for p in current_primes:
        current_n *= p
        
    best_n = current_n
    best_score = tau_perp_fast(current_primes)
    current_score = best_score
    
    # Precompute log of primes for heuristic if needed, but let's rely on exact score
    # We need to track the product to check constraints quickly
    current_prod = current_n
    
    # Helper to check if a swap is valid
    def is_valid_swap(p_out, p_in, prod, limit=10**18):
        # new_prod = prod * p_in / p_out
        # Check p_in <= limit * p_out / prod
        # Use integer arithmetic to avoid float precision issues
        return p_in * prod <= limit * p_out

    # 4. Local search with multiple strategies
    improved = True
    no_improvement_count = 0
    max_no_improvement = 5 # Reset strategy after 5 full passes without improvement
    
    while improved and time.time() < deadline - 15.0:
        improved = False
        current_prime_set = set(current_primes)
        
        # Strategy 1: Single prime swap
        # We iterate through all positions and all possible smaller/equal primes that fit
        # Since we want to maximize score, and smaller primes generally allow more "interleaving",
        # we should prioritize swapping in smaller primes.
        
        # To be efficient, we can just try swapping each prime with the smallest available prime that fits.
        # But sometimes a slightly larger prime is better.
        # Let's try a few candidates per slot.
        
        indices = list(range(len(current_primes)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 10.0:
                return best_n
                
            p_old = current_primes[idx]
            
            # Find candidates: primes not in current set, such that new_prod <= 10^18
            # p_new <= 10^18 * p_old / current_prod
            max_p_new_float = (10**18 * p_old) / current_prod
            # To be safe with integers:
            # p_new * current_prod <= 10^18 * p_old
            
            candidates = []
            # We look at primes smaller than p_old first (to potentially increase density)
            # and some larger ones.
            
            # Since primes list is sorted, we can find the index of p_old
            # But p_old might not be in the initial sieve range if we swapped in a large one?
            # We started with small primes. Swaps keep us in the sieve range mostly.
            
            # Iterate through all primes in our sieve that are valid
            # This is O(N_primes) per swap, which is ~78k.
            # 15 swaps * 78k = 1.1M checks.
            # Each check is a multiplication and comparison.
            # Then we evaluate tau_perp for the best few?
            # No, we should evaluate all valid candidates? That's too slow.
            
            # Heuristic: Try the smallest 5 valid primes and the largest 5 valid primes?
            # Or just the smallest 10 valid primes.
            
            valid_candidates = []
            for p_new in primes:
                if p_new in current_prime_set:
                    continue
                # Check constraint
                if p_new * current_prod <= 10**18 * p_old:
                    valid_candidates.append(p_new)
                else:
                    # Since primes is sorted, if p_new is too big, all subsequent are too big?
                    # No, because p_old varies. But for a fixed p_old and current_prod,
                    # if p_new is too big, larger p_new will also be too big.
                    # So we can break if we only consider increasing p_new.
                    # But we want small primes too.
                    pass
            
            # valid_candidates might be large. Let's pick a subset.
            # Sort valid_candidates? They are already sorted from the sieve.
            # We want to try small ones first.
            
            # Take first 20 (smallest) and last 20 (largest) valid candidates
            test_candidates = []
            if len(valid_candidates) > 0:
                test_candidates.extend(valid_candidates[:20])
                if len(valid_candidates) > 40:
                    test_candidates.extend(valid_candidates[-20:])
            
            best_candidate = None
            best_candidate_score = current_score - 1
            
            for p_new in test_candidates:
                if p_new in current_prime_set:
                    continue
                if p_new * current_prod > 10**18 * p_old:
                    continue
                
                new_primes = current_primes.copy()
                new_primes[idx] = p_new
                
                s_new = tau_perp_fast(new_primes)
                
                if s_new > best_candidate_score:
                    best_candidate_score = s_new
                    best_candidate = p_new
            
            if best_candidate is not None and best_candidate_score > current_score:
                current_primes[idx] = best_candidate
                current_prod = (current_prod // p_old) * best_candidate
                current_score = best_candidate_score
                if current_score > best_score:
                    best_score = current_score
                    best_n = current_prod
                improved = True
                break # Restart loop with new state
        
        # Strategy 2: Pair swap
        # If single swap didn't improve, try pair swap
        if not improved and time.time() < deadline - 20.0:
            pair_indices = list(combinations(range(len(current_primes)), 2))
            random.shuffle(pair_indices)
            
            for i, j in pair_indices[:10]: # Limit attempts
                if time.time() > deadline - 10.0:
                    break
                    
                p_i = current_primes[i]
                p_j = current_primes[j]
                base_prod = current_prod // (p_i * p_j)
                
                # We want to replace p_i, p_j with p_a, p_b
                # Constraint: p_a * p_b * base_prod <= 10^18
                # p_a * p_b <= 10^18 / base_prod
                
                max_pair_prod = 10**18 // base_prod
                
                # Find pairs of primes (p_a, p_b) not in current set
                # such that p_a * p_b <= max_pair_prod
                # And ideally, we want to try combinations that differ from current.
                
                # This search space is huge.
                # Heuristic: Try swapping p_i with a smaller prime p_a, and keep p_j?
                # That's a single swap.
                # Try swapping p_i with p_a and p_j with p_b.
                
                # Let's just try replacing the pair with the two smallest available primes that fit.
                # Or random pairs.
                
                # For now, skip complex pair swap to ensure we don't timeout.
                # The single swap is usually sufficient to find the local optimum for this metric.
                pass

        if not improved:
            no_improvement_count += 1
            if no_improvement_count >= max_no_improvement:
                # Try a random restart from a perturbed state
                if time.time() < deadline - 30.0:
                    # Perturb current state
                    perturbed = current_primes.copy()
                    for _ in range(3):
                        idx = random.randint(0, len(perturbed)-1)
                        p_old = perturbed[idx]
                        # Find a random valid swap
                        valids = [p for p in primes if p not in set(perturbed) and p * (current_prod // p_old) <= 10**18]
                        if valids:
                            p_new = random.choice(valids)
                            perturbed[idx] = p_new
                    # Evaluate perturbed
                    perturbed_score = tau_perp_fast(perturbed)
                    if perturbed_score > best_score:
                        best_score = perturbed_score
                        best_n = 1
                        for p in perturbed:
                            best_n *= p
                        current_primes = perturbed
                        current_score = perturbed_score
                        current_prod = best_n
                        improved = True
                        no_improvement_count = 0
                else:
                    break

    return best_n
