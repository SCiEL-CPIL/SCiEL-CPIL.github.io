import time
import math
import random

def construct():
    start = time.time()
    deadline = start + 140.0  # 10s safety margin
    
    N = 512
    # Precompute primes below 512
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n) if is_prime[i]]
    
    primes = sieve(512)
    # Precompute prime factors for each number 1..511
    prime_factors = [set() for _ in range(512)]
    for p in primes:
        for multiple in range(p, 512, p):
            prime_factors[multiple].add(p)
    
    # Score function
    def score_set(chosen):
        return sum(1.0 / (512 - a) for a in chosen)
    
    # Greedy: pick numbers in order of highest score (lowest a), then check coprimality
    # But we want to maximize sum of 1/(512-a), so we prefer small a.
    # However, small a may conflict with other small a's.
    # Strategy: Try to build a set by greedily adding the best available number.
    
    def greedy_build():
        chosen = []
        chosen_set = set()
        # Sort candidates by score descending (i.e., by a ascending)
        candidates = list(range(1, 512))
        # But we can be smarter: try to add numbers that don't conflict
        # Let's iterate from 1 to 511, and add if coprime to all chosen
        for a in range(1, 512):
            if all(math.gcd(a, b) == 1 for b in chosen):
                chosen.append(a)
        return chosen
    
    best = greedy_build()
    best_score = score_set(best)
    
    # Local search: try to improve by swapping
    def local_search(initial, max_iter=10000):
        current = list(initial)
        current_score = score_set(current)
        chosen_set = set(current)
        
        # Precompute gcd for all pairs
        # Actually, let's just use math.gcd
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            # Try to add a new number
            for a in range(1, 512):
                if a in chosen_set:
                    continue
                if all(math.gcd(a, b) == 1 for b in current):
                    new_score = current_score + 1.0 / (512 - a)
                    if new_score > current_score + 1e-12:
                        current.append(a)
                        chosen_set.add(a)
                        current_score = new_score
                        improved = True
                        break
            if improved:
                continue
            
            # Try to swap: remove one, add one or more
            for i, a in enumerate(current):
                for b in range(1, 512):
                    if b in chosen_set:
                        continue
                    # Check if b is coprime to all except a
                    rest = [x for j, x in enumerate(current) if j != i]
                    if all(math.gcd(b, x) == 1 for x in rest):
                        new_score = current_score - 1.0 / (512 - a) + 1.0 / (512 - b)
                        if new_score > current_score + 1e-12:
                            current[i] = b
                            chosen_set.remove(a)
                            chosen_set.add(b)
                            current_score = new_score
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        
        return current, current_score
    
    # Run local search from greedy
    best, best_score = local_search(best)
    
    # Try random restarts
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        # Random perturbation
        perturbed = list(best)
        # Remove 1-3 random elements
        num_remove = random.randint(1, 3)
        for _ in range(num_remove):
            if perturbed:
                idx = random.randint(0, len(perturbed) - 1)
                perturbed.pop(idx)
        
        # Greedily add back
        perturbed_set = set(perturbed)
        for a in range(1, 512):
            if a in perturbed_set:
                continue
            if all(math.gcd(a, b) == 1 for b in perturbed):
                perturbed.append(a)
                perturbed_set.add(a)
        
        # Local search
        candidate, cand_score = local_search(perturbed, max_iter=5000)
        if cand_score > best_score + 1e-12:
            best = candidate
            best_score = cand_score
    
    # Final check
    if time.time() > deadline - 2:
        pass  # Just return
    
    # Verify pairwise coprime
    for i in range(len(best)):
        for j in range(i + 1, len(best)):
            if math.gcd(best[i], best[j]) != 1:
                # Fallback to greedy
                best = greedy_build()
                break
    
    return best
