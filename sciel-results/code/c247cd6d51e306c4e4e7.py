import time
import math
import random
from math import gcd

_DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _elapsed(t0):
    return time.time() - t0

def _time_exhausted(t0, margin=3.0):
    return _elapsed(t0) > (_DEADLINE - margin)

def _score(chosen):
    """Compute sum 1/(512 - a) as float."""
    s = 0.0
    for a in chosen:
        s += 1.0 / (512.0 - a)
    return s

def _get_prime_factors(n, primes):
    """Get unique prime factors of n."""
    factors = set()
    temp = n
    for p in primes:
        if p * p > temp:
            break
        if temp % p == 0:
            factors.add(p)
            while temp % p == 0:
                temp //= p
    if temp > 1:
        factors.add(temp)
    return factors

def _greedy_init(prime_factors, order=None):
    """Greedy initialization."""
    if order is None:
        order = list(range(511, 0, -1))
    chosen = []
    chosen_prime_factors = set()
    for a in order:
        if a == 1:
            chosen.append(a)
            continue
        if not (prime_factors[a] & chosen_prime_factors):
            chosen.append(a)
            chosen_prime_factors |= prime_factors[a]
    return chosen

def _compute_factors(chosen, prime_factors):
    """Compute the union of prime factors of chosen elements."""
    all_factors = set()
    for a in chosen:
        all_factors |= prime_factors[a]
    return all_factors

def _local_search(chosen, prime_factors, t0, max_iter=5000):
    """Local search with 1-swap, add, and 2-swap moves."""
    chosen_set = set(chosen)
    current_score = _score(chosen)
    
    for it in range(max_iter):
        if _time_exhausted(t0):
            break
        
        improved = False
        n = len(chosen)
        
        # Try 1-swap: replace one element with another
        for i in range(n):
            if _time_exhausted(t0):
                return chosen, current_score
            
            current = chosen[i]
            current_score_i = 1.0 / (512.0 - current)
            
            # Required prime factors from other chosen elements
            req_factors = set()
            for j in range(n):
                if j != i:
                    req_factors |= prime_factors[chosen[j]]
            
            # Find best replacement
            best_new = -1
            best_gain = 0.0
            
            for a in range(511, 0, -1):
                if a in chosen_set or a == current:
                    continue
                if a == 1:
                    new_score = 1.0 / (512.0 - 1)
                else:
                    if prime_factors[a] & req_factors:
                        continue
                    new_score = 1.0 / (512.0 - a)
                
                gain = new_score - current_score_i
                if gain > best_gain:
                    best_gain = gain
                    best_new = a
                    break  # Largest a gives best score
            
            if best_new != -1 and best_gain > 0:
                chosen[i] = best_new
                chosen_set.remove(current)
                chosen_set.add(best_new)
                current_score += best_gain
                improved = True
                break  # Restart after swap
        
        # Try add move: add a new element if it doesn't conflict
        if not improved:
            all_factors = _compute_factors(chosen, prime_factors)
            
            for a in range(511, 0, -1):
                if a in chosen_set:
                    continue
                if a == 1:
                    chosen.append(a)
                    chosen_set.add(a)
                    current_score += 1.0 / (512.0 - 1)
                    improved = True
                    break
                else:
                    if not (prime_factors[a] & all_factors):
                        chosen.append(a)
                        chosen_set.add(a)
                        current_score += 1.0 / (512.0 - a)
                        improved = True
                        break
        
        # Try 2-swap: replace two elements with two new ones
        if not improved and n >= 2:
            # Try all pairs of elements to remove
            for i in range(n):
                if _time_exhausted(t0):
                    return chosen, current_score
                for j in range(i + 1, n):
                    # Factors required by remaining elements
                    req_factors = set()
                    for k in range(n):
                        if k != i and k != j:
                            req_factors |= prime_factors[chosen[k]]
                    
                    removed_score = 1.0 / (512.0 - chosen[i]) + 1.0 / (512.0 - chosen[j])
                    old_i = chosen[i]
                    old_j = chosen[j]
                    
                    # Find best pair of replacements
                    best_a = -1
                    best_b = -1
                    best_pair_score = 0.0
                    
                    # Build list of candidates coprime to req_factors
                    candidates = []
                    for a in range(511, 0, -1):
                        if a in chosen_set:
                            continue
                        if a == 1:
                            candidates.append(a)
                        else:
                            if not (prime_factors[a] & req_factors):
                                candidates.append(a)
                    
                    # Find best pair from candidates that are coprime to each other
                    for idx_a in range(len(candidates)):
                        a = candidates[idx_a]
                        if a == 1:
                            score_a = 1.0 / (512.0 - 1)
                            factors_a = set()
                        else:
                            score_a = 1.0 / (512.0 - a)
                            factors_a = prime_factors[a]
                        
                        for idx_b in range(idx_a + 1, len(candidates)):
                            b = candidates[idx_b]
                            if b == 1:
                                score_b = 1.0 / (512.0 - 1)
                                factors_b = set()
                            else:
                                score_b = 1.0 / (512.0 - b)
                                factors_b = prime_factors[b]
                            
                            # Check if a and b are coprime
                            if a != 1 and b != 1 and (factors_a & factors_b):
                                continue
                            
                            pair_score = score_a + score_b
                            if pair_score > best_pair_score:
                                best_pair_score = pair_score
                                best_a = a
                                best_b = b
                    
                    if best_a != -1 and best_pair_score > removed_score:
                        # Perform 2-swap
                        chosen[i] = best_a
                        chosen[j] = best_b
                        chosen_set.remove(old_i)
                        chosen_set.remove(old_j)
                        chosen_set.add(best_a)
                        chosen_set.add(best_b)
                        current_score += best_pair_score - removed_score
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return chosen, current_score

def _perturb(chosen, prime_factors, t0, num_swaps=2):
    """Perturb the solution by doing random swaps."""
    if len(chosen) < 2:
        return list(chosen)
    
    chosen = list(chosen)
    chosen_set = set(chosen)
    
    for _ in range(num_swaps):
        if _time_exhausted(t0):
            break
        
        # Pick a random element to remove
        idx = random.randint(0, len(chosen) - 1)
        removed = chosen[idx]
        chosen.pop(idx)
        chosen_set.remove(removed)
        
        # Compute factors of remaining
        req_factors = _compute_factors(chosen, prime_factors)
        
        # Pick a random valid element to add
        candidates = []
        for a in range(511, 0, -1):
            if a in chosen_set:
                continue
            if a == 1:
                candidates.append(a)
            else:
                if not (prime_factors[a] & req_factors):
                    candidates.append(a)
        
        if candidates:
            # Weighted random selection favoring larger a
            weights = [1.0 / (512.0 - a) for a in candidates]
            total_w = sum(weights)
            r = random.random() * total_w
            cum = 0.0
            chosen_a = candidates[0]
            for a, w in zip(candidates, weights):
                cum += w
                if r <= cum:
                    chosen_a = a
                    break
            chosen.append(chosen_a)
            chosen_set.add(chosen_a)
    
    return chosen

def construct():
    t0 = time.time()
    random.seed(42)
    
    # Sieve of Eratosthenes for primes up to 512
    limit = 512
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Pre-compute prime factors for all numbers 1..511
    prime_factors = [set() for _ in range(limit)]
    for i in range(1, limit):
        prime_factors[i] = _get_prime_factors(i, primes)
    
    # Best solution found
    best_chosen = []
    best_score = 0.0
    
    # Multiple restarts with different initializations
    num_restarts = 30
    
    for restart in range(num_restarts):
        if _time_exhausted(t0):
            break
        
        if restart == 0:
            # Standard greedy: descending a
            chosen = _greedy_init(prime_factors, order=list(range(511, 0, -1)))
        elif restart == 1:
            # Greedy with randomized order (biased toward larger a)
            order = list(range(511, 0, -1))
            half = len(order) // 2
            random.shuffle(order[:half])
            chosen = _greedy_init(prime_factors, order=order)
        else:
            # Random perturbation of best solution
            if best_chosen:
                chosen = _perturb(best_chosen, prime_factors, t0, num_swaps=random.randint(1, 4))
            else:
                chosen = _greedy_init(prime_factors)
        
        # Local search
        chosen, score = _local_search(chosen, prime_factors, t0, max_iter=3000)
        
        if score > best_score:
            best_score = score
            best_chosen = list(chosen)
    
    # Final local search on best solution with more iterations
    if best_chosen and not _time_exhausted(t0):
        best_chosen, best_score = _local_search(best_chosen, prime_factors, t0, max_iter=20000)
    
    return best_chosen
