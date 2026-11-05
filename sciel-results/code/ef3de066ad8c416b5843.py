"""Combined program: conflict-aware greedy + improved local search for pairwise coprime set."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    N = 512
    
    # Sieve for primes up to 511
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return is_prime
    
    is_prime = sieve(511)
    
    # For each number 1..511, compute its set of prime factors
    prime_factors = [set() for _ in range(512)]
    for p in range(2, 512):
        if is_prime[p]:
            for multiple in range(p, 512, p):
                prime_factors[multiple].add(p)
    
    # Score for each a: 1/(512-a)
    scores = [0.0] * 512
    for a in range(1, 512):
        scores[a] = 1.0 / (512 - a)
    
    # Conflict cost: number of candidates excluded by picking a
    # (approximate: count of numbers sharing at least one prime factor with a)
    conflict_cost = [0] * 512
    for a in range(1, 512):
        if a == 1:
            conflict_cost[a] = 0
            continue
        cost = 0
        for b in range(1, 512):
            if b != a and prime_factors[a] & prime_factors[b]:
                cost += 1
        conflict_cost[a] = cost
    
    # Greedy baseline: pick in order of highest score that is coprime to all chosen
    def greedy():
        chosen = []
        used_primes = set()
        for a in range(511, 0, -1):
            if not (prime_factors[a] & used_primes):
                chosen.append(a)
                used_primes |= prime_factors[a]
        return chosen
    
    # Conflict-aware greedy: balance value vs conflict cost
    def greedy_conflict_aware(alpha=0.0):
        chosen = []
        used_primes = set()
        # Score adjusted by conflict cost
        adj_scores = [0.0] * 512
        for a in range(1, 512):
            adj_scores[a] = scores[a] - alpha * conflict_cost[a] * 0.01
        for a in range(511, 0, -1):
            if not (prime_factors[a] & used_primes):
                chosen.append(a)
                used_primes |= prime_factors[a]
        return chosen
    
    def get_score(s):
        return sum(scores[a] for a in s)
    
    def valid(s):
        used = set()
        for a in s:
            if prime_factors[a] & used:
                return False
            used |= prime_factors[a]
        return True
    
    # Improved hill climbing with 1-for-1 and 1-for-2 swaps
    def hill_climb(initial_set, max_iters=5000):
        current = list(initial_set)
        current_score = get_score(current)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            used_primes = set()
            for a in current:
                used_primes |= prime_factors[a]
            
            # Try adding any a not in current that is coprime
            current_set = set(current)
            for a in range(511, 0, -1):
                if a not in current_set and not (prime_factors[a] & used_primes):
                    new_score = current_score + scores[a]
                    if new_score > current_score + 1e-15:
                        current = current + [a]
                        current_score = new_score
                        current_set.add(a)
                        used_primes |= prime_factors[a]
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try 1-for-1 swap: remove one, add one
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                remaining = current[:i] + current[i+1:]
                rem_score = current_score - scores[removed]
                
                rem_used = set()
                for a in remaining:
                    rem_used |= prime_factors[a]
                
                best_add = None
                best_add_score = 0
                for a in range(511, 0, -1):
                    if a not in remaining and not (prime_factors[a] & rem_used):
                        s = rem_score + scores[a]
                        if s > best_add_score:
                            best_add_score = s
                            best_add = a
                
                if best_add is not None and best_add_score > current_score + 1e-15:
                    current = remaining + [best_add]
                    current_score = best_add_score
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-for-2 swap: remove one, add two
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                remaining = current[:i] + current[i+1:]
                rem_score = current_score - scores[removed]
                
                rem_used = set()
                for a in remaining:
                    rem_used |= prime_factors[a]
                
                # Find best pair to add
                best_pair = None
                best_pair_score = 0
                rem_set = set(remaining)
                
                for a in range(511, 0, -1):
                    if a in rem_set or (prime_factors[a] & rem_used):
                        continue
                    for b in range(a-1, 0, -1):
                        if b in rem_set or (prime_factors[b] & rem_used) or (prime_factors[b] & prime_factors[a]):
                            continue
                        s = rem_score + scores[a] + scores[b]
                        if s > best_pair_score:
                            best_pair_score = s
                            best_pair = (a, b)
                
                if best_pair is not None and best_pair_score > current_score + 1e-15:
                    current = remaining + list(best_pair)
                    current_score = best_pair_score
                    improved = True
                    break
            
            if not improved:
                break
        
        return current, current_score
    
    # Multiple strategies
    rng = random.Random(42)
    
    # Start with standard greedy
    best_set = greedy()
    best_score = get_score(best_set)
    
    # Try conflict-aware greedy with different alphas
    for alpha in [0.1, 0.5, 1.0, 2.0]:
        if time.time() > deadline - 30:
            break
        ca_set = greedy_conflict_aware(alpha)
        ca_score = get_score(ca_set)
        if ca_score > best_score + 1e-15:
            best_set = ca_set
            best_score = ca_score
    
    # Hill climb the best
    result, score = hill_climb(best_set, max_iters=3000)
    if score > best_score + 1e-15:
        best_set = result
        best_score = score
    
    # Random restarts with perturbation
    for restart in range(150):
        if time.time() > deadline - 10:
            break
        
        # Create a perturbed version
        perturbed = best_set[:]
        k = rng.randint(1, min(3, len(perturbed)))
        for _ in range(k):
            if len(perturbed) > 1:
                idx = rng.randrange(len(perturbed))
                perturbed.pop(idx)
        
        # Alternate between perturbation and randomized greedy
        if restart % 2 == 0:
            # Randomized greedy with noise
            candidates = list(range(511, 0, -1))
            noisy_scores = []
            for a in candidates:
                noise = rng.uniform(0, 0.3 * scores[a])
                noisy_scores.append((scores[a] + noise, a))
            noisy_scores.sort(reverse=True)
            
            perturbed = []
            used = set()
            for _, a in noisy_scores:
                if not (prime_factors[a] & used):
                    perturbed.append(a)
                    used |= prime_factors[a]
        
        result, score = hill_climb(perturbed, max_iters=2000)
        
        if score > best_score + 1e-15:
            best_set = result
            best_score = score
    
    # Final polish: more iterations on the best
    if time.time() < deadline - 5:
        result, score = hill_climb(best_set, max_iters=10000)
        if score > best_score + 1e-15:
            best_set = result
            best_score = score
    
    return best_set
# EVOLVE-BLOCK-END
