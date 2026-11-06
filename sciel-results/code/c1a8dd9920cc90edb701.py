import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    N = 512
    
    # Precompute values
    values = [0.0] * N
    for a in range(1, N):
        values[a] = 1.0 / (N - a)
        
    # Precompute factors
    all_factors = [set() for _ in range(N)]
    for i in range(2, N):
        n = i
        d = 2
        while d * d <= n:
            while n % d == 0:
                all_factors[i].add(d)
                n //= d
            d += 1
        if n > 1:
            all_factors[i].add(n)
            
    # Sort indices by value descending
    sorted_indices = sorted(range(1, N), key=lambda x: values[x], reverse=True)
    
    def greedy_fill(chosen_set, forbidden_primes, score):
        """
        Greedily add numbers to chosen_set that are coprime to forbidden_primes.
        Modifies chosen_set and forbidden_primes.
        Returns new score.
        """
        for a in sorted_indices:
            if a in chosen_set:
                continue
            if all_factors[a] & forbidden_primes:
                continue
            chosen_set.add(a)
            forbidden_primes.update(all_factors[a])
            score += values[a]
        return score

    def local_search(chosen_set, forbidden_primes, score):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add any single number
            # Since we do greedy fill in other moves, this might be redundant,
            # but it's fast.
            added = False
            for a in sorted_indices:
                if a in chosen_set:
                    continue
                if all_factors[a] & forbidden_primes:
                    continue
                chosen_set.add(a)
                forbidden_primes.update(all_factors[a])
                score += values[a]
                improved = True
                added = True
                break # One add per pass
            
            if time.time() > deadline:
                break
                
            # 2. Try Remove 1 and Refill
            for a in list(chosen_set):
                if time.time() > deadline:
                    return score, chosen_set, forbidden_primes
                
                # Save state
                old_score = score
                old_chosen = chosen_set.copy()
                old_forbidden = forbidden_primes.copy()
                
                # Remove a
                chosen_set.remove(a)
                forbidden_primes.difference_update(all_factors[a])
                score -= values[a]
                
                # Refill
                new_score = greedy_fill(chosen_set, forbidden_primes, score)
                
                if new_score > old_score + 1e-12:
                    score = new_score
                    improved = True
                else:
                    # Revert
                    chosen_set = old_chosen
                    forbidden_primes = old_forbidden
                    score = old_score
                    
            if time.time() > deadline:
                break
                
            # 3. Try Remove 2 and Refill
            chosen_list = list(chosen_set)
            for i in range(len(chosen_list)):
                for j in range(i + 1, len(chosen_list)):
                    if time.time() > deadline:
                        return score, chosen_set, forbidden_primes
                        
                    a = chosen_list[i]
                    b = chosen_list[j]
                    
                    old_score = score
                    old_chosen = chosen_set.copy()
                    old_forbidden = forbidden_primes.copy()
                    
                    # Remove a, b
                    chosen_set.remove(a)
                    chosen_set.remove(b)
                    forbidden_primes.difference_update(all_factors[a])
                    forbidden_primes.difference_update(all_factors[b])
                    score -= values[a] + values[b]
                    
                    # Refill
                    new_score = greedy_fill(chosen_set, forbidden_primes, score)
                    
                    if new_score > old_score + 1e-12:
                        score = new_score
                        improved = True
                    else:
                        chosen_set = old_chosen
                        forbidden_primes = old_forbidden
                        score = old_score
                        
        return score, chosen_set, forbidden_primes

    best_score = 0.0
    best_chosen = set()
    
    # Multiple restarts
    import random
    random.seed(42)
    
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        # Randomized Greedy Init
        chosen_set = set()
        forbidden_primes = set()
        score = 0.0
        
        # Shuffle sorted_indices slightly? Or just use it.
        # To get diversity, we can skip some top candidates randomly.
        # Or just use the same greedy, it's deterministic.
        # Let's add a bit of randomness to the order.
        indices = sorted_indices[:]
        if restart > 0:
            # Perturb the order
            for k in range(len(indices) - 1):
                if random.random() < 0.1:
                    indices[k], indices[k+1] = indices[k+1], indices[k]
                    
        for a in indices:
            if all_factors[a] & forbidden_primes:
                continue
            chosen_set.add(a)
            forbidden_primes.update(all_factors[a])
            score += values[a]
            
        # Local Search
        score, chosen_set, forbidden_primes = local_search(chosen_set, forbidden_primes, score)
        
        if score > best_score:
            best_score = score
            best_chosen = chosen_set.copy()
            
    return list(best_chosen)
