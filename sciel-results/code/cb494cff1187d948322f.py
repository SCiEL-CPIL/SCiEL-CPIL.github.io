import time
import random
from math import isqrt

def _factorize(n):
    """Return the set of prime factors of n."""
    if n <= 1:
        return set()
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Strategy: Use a base K such that K has small prime factors.
    # Let a_i = K * m_i.
    # Sums are K * (m_i + m_j).
    # Prime factors = primes(K) U primes(m_i + m_j).
    
    # We'll try several K values and use bitmask-based factor tracking for speed.
    
    def solve_for_K(K, time_budget):
        """Find best 12 multipliers for given K using bitmask-based search."""
        max_m = 10000 // K
        if max_m < 12:
            return None, None
            
        # Collect all primes that could appear: primes(K) union primes(2..2*max_m)
        all_primes = set(_factorize(K))
        for s in range(2, 2 * max_m + 1):
            all_primes.update(_factorize(s))
        
        # Map primes to bit indices
        prime_list = sorted(all_primes)
        prime_to_idx = {p: i for i, p in enumerate(prime_list)}
        n_primes = len(prime_list)
        
        # Precompute bitmask for each sum value
        sum_mask = {}
        for s in range(2, 2 * max_m + 1):
            mask = 0
            for p in _factorize(s):
                mask |= (1 << prime_to_idx[p])
            sum_mask[s] = mask
        
        base_mask = 0
        for p in _factorize(K):
            base_mask |= (1 << prime_to_idx[p])
        
        candidates = list(range(1, max_m + 1))
        
        best_score = float('inf')
        best_set = None
        
        # Multiple greedy restarts with random shuffles
        max_restarts = 200
        for restart in range(max_restarts):
            if time.time() > time_budget:
                break
            
            shuffled = list(candidates)
            random.shuffle(shuffled)
            
            current_set = []
            current_mask = base_mask
            
            for _ in range(12):
                best_candidate = None
                best_new_mask = None
                best_count = float('inf')
                
                for m in shuffled:
                    if m in current_set:
                        continue
                    new_mask = current_mask
                    for x in current_set:
                        s = m + x
                        new_mask |= sum_mask[s]
                    count = bin(new_mask).count('1')
                    if count < best_count:
                        best_count = count
                        best_candidate = m
                        best_new_mask = new_mask
                
                if best_candidate is None:
                    break
                
                current_set.append(best_candidate)
                current_mask = best_new_mask
            
            if len(current_set) == 12:
                score = bin(current_mask).count('1')
                if score < best_score:
                    best_score = score
                    best_set = list(current_set)
                    
                    # Local search: try swapping elements to reduce prime count
                    improved = True
                    while improved:
                        if time.time() > time_budget:
                            break
                        improved = False
                        current_score = bin(current_mask).count('1')
                        
                        for i in range(12):
                            if time.time() > time_budget:
                                break
                            for m in candidates:
                                if m in current_set:
                                    continue
                                # Try swapping current_set[i] with m
                                new_set = list(current_set)
                                new_set[i] = m
                                new_mask = base_mask
                                for a in range(12):
                                    for b in range(a+1, 12):
                                        s = new_set[a] + new_set[b]
                                        new_mask |= sum_mask[s]
                                new_score = bin(new_mask).count('1')
                                if new_score < current_score:
                                    current_set = new_set
                                    current_mask = new_mask
                                    current_score = new_score
                                    best_score = min(best_score, new_score)
                                    best_set = list(current_set)
                                    improved = True
                                    break
                            if improved:
                                break
            
            # Early stop if optimal found
            if best_score == bin(base_mask).count('1'):
                break
        
        if best_set is None:
            return None, None
        return best_set, best_score
    
    # Try several K values
    K_candidates = [210, 60, 30, 42, 105, 140, 120, 180, 240, 336, 420, 840]
    
    overall_best_set = None
    overall_best_score = float('inf')
    overall_best_K = None
    
    # Allocate time budget across K candidates
    num_K = len(K_candidates)
    time_per_K = (deadline - start_time) / num_K
    
    for K in K_candidates:
        if time.time() > deadline - 2.0:
            break
        
        remaining = deadline - time.time() - 2.0
        if remaining < 1.0:
            break
        
        t_budget = min(time_per_K, remaining)
        m_set, score = solve_for_K(K, t_budget)
        
        if m_set is not None and score < overall_best_score:
            overall_best_score = score
            overall_best_set = m_set
            overall_best_K = K
    
    if overall_best_set is None:
        # Fallback
        return list(range(1, 13))
    
    # Construct final numbers
    result = [overall_best_K * m for m in overall_best_set]
    
    # Verify
    if len(set(result)) != 12 or any(x > 10000 for x in result) or any(x <= 0 for x in result):
        return list(range(1, 13))
    
    return result
