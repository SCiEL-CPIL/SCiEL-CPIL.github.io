import time
import math
import random
from fractions import Fraction

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    primes = _sieve(1000)
    all_primes = primes
    n_primes = len(all_primes)
    
    # Precompute LCM of all primes for integer budget tracking
    # LCM of primes 2..997 is enormous, so we use a different approach:
    # Track budget as a Fraction but precompute cost_fractions for speed
    cost_fractions = [Fraction(1, p) for p in all_primes]
    
    # Precompute prime index for fast lookup
    prime_to_idx = {p: i for i, p in enumerate(all_primes)}
    
    best_L = 0
    best_classes = []
    
    def run_greedy(window_size, prefer_small_on_tie, use_efficiency, deadline, noise_seed=None):
        """
        Run a greedy covering algorithm with optional score noise.
        """
        nonlocal best_L, best_classes
        
        covered = bytearray(100001)
        used = bytearray(n_primes)  # 0/1 for each prime
        budget = Fraction(1)
        classes = []
        
        # Optional noise generator
        rng = random.Random(noise_seed) if noise_seed is not None else None
        
        for n in range(1, 100001):
            if time.time() > deadline:
                return
            
            if covered[n]:
                continue
            
            # n is uncovered, need to pick a prime
            best_idx = -1
            best_p = None
            best_r = 0
            best_score = -1
            
            for pi in range(n_primes):
                if used[pi]:
                    continue
                p = all_primes[pi]
                cost = cost_fractions[pi]
                if budget < cost:
                    continue
                
                r = n % p
                # Count new coverage in [n, n+window_size]
                cnt = 0
                
                # Fast path for large p
                if p > window_size:
                    if not covered[n]:
                        cnt = 1
                else:
                    max_k = n + window_size
                    if max_k > 100000:
                        max_k = 100000
                    for k in range(n, max_k + 1, p):
                        if not covered[k]:
                            cnt += 1
                
                # Calculate Score
                if use_efficiency:
                    score = cnt * p
                else:
                    score = cnt
                    if prefer_small_on_tie:
                        score += (1000 - p)
                    else:
                        score += p
                
                # Add small noise for diversity
                if rng is not None:
                    score += rng.random() * 0.1
                
                if score > best_score:
                    best_score = score
                    best_idx = pi
                    best_p = p
                    best_r = r
            
            if best_p is None:
                break
            
            # Commit
            used[best_idx] = 1
            budget -= cost_fractions[best_idx]
            classes.append([best_p, best_r])
            
            # Mark covered
            start_k = best_r
            if start_k == 0:
                start_k = best_p
            for k in range(start_k, 100001, best_p):
                covered[k] = 1
        
        # Calculate L
        L = 0
        for i in range(1, 100001):
            if covered[i]:
                L = i
            else:
                break
        
        if L > best_L:
            best_L = L
            best_classes = [list(c) for c in classes]
        
        return L

    def try_local_search(classes, deadline):
        """
        Try to improve coverage by swapping primes.
        Returns improved classes or original if no improvement.
        """
        nonlocal best_L, best_classes
        
        if not classes:
            return classes
        
        # Build coverage from current classes
        covered = bytearray(100001)
        for p, r in classes:
            start_k = r
            if start_k == 0:
                start_k = p
            for k in range(start_k, 100001, p):
                covered[k] = 1
        
        # Compute current L
        cur_L = 0
        for i in range(1, 100001):
            if covered[i]:
                cur_L = i
            else:
                break
        
        # Find first gap
        if cur_L >= 100000:
            return classes
        
        gap_pos = cur_L + 1  # First uncovered position
        
        # Try to fix the gap by modifying one class
        used_primes = set(p for p, r in classes)
        available_primes = [p for p in all_primes if p not in used_primes]
        
        # Compute current budget usage
        budget_used = Fraction(0)
        for p, r in classes:
            budget_used += Fraction(1, p)
        remaining_budget = Fraction(1) - budget_used
        
        # Try adding a new prime to cover gap_pos
        best_new_p = None
        best_new_r = gap_pos % all_primes[0] if all_primes else 0
        best_gain = 0
        
        for p in available_primes:
            cost = Fraction(1, p)
            if cost > remaining_budget:
                continue
            r = gap_pos % p
            # Count how many new positions this would cover
            start_k = r
            if start_k == 0:
                start_k = p
            gain = 0
            for k in range(start_k, min(gap_pos + 1000, 100001), p):
                if not covered[k]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_new_p = p
                best_new_r = r
        
        if best_new_p is not None and best_gain > 0:
            new_classes = classes + [[best_new_p, best_new_r]]
            # Verify
            covered2 = bytearray(100001)
            for p, r in new_classes:
                start_k = r
                if start_k == 0:
                    start_k = p
                for k in range(start_k, 100001, p):
                    covered2[k] = 1
            new_L = 0
            for i in range(1, 100001):
                if covered2[i]:
                    new_L = i
                else:
                    break
            if new_L > cur_L:
                if new_L > best_L:
                    best_L = new_L
                    best_classes = [list(c) for c in new_classes]
                return new_classes
        
        return classes

    # Define strategies
    strategies = [
        (1000, True, False),
        (1000, False, False),
        (5000, True, False),
        (5000, False, False),
        (10000, True, False),
        (10000, False, False),
        (1000, True, True),
        (1000, False, True),
        (5000, True, True),
        (5000, False, True),
        (10000, True, True),
        (10000, False, True),
        (2000, True, False),
        (2000, False, False),
        (2000, True, True),
        (2000, False, True),
        (3000, True, False),
        (3000, False, False),
        (3000, True, True),
        (3000, False, True),
        (7500, True, False),
        (7500, False, False),
        (7500, True, True),
        (7500, False, True),
    ]
    
    # Run all strategies
    for w, ps, ue in strategies:
        if time.time() > deadline - 5:
            break
        run_greedy(w, ps, ue, deadline)
    
    # Try local search on best solution
    if best_classes and time.time() < deadline - 5:
        improved = try_local_search(best_classes, deadline)
        if improved != best_classes:
            best_classes = improved
    
    # Random restarts with noise if time remains
    if time.time() < deadline - 10:
        for seed in range(200):
            if time.time() > deadline - 5:
                break
            w = random.choice([500, 1000, 2000, 3000, 5000, 7500])
            ps = random.choice([True, False])
            ue = random.choice([True, False])
            run_greedy(w, ps, ue, deadline, noise_seed=seed)
    
    # Fallback
    if not best_classes:
        primes_13_plus = [p for p in all_primes if p >= 13]
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(primes_13_plus)]
        
    return best_classes
