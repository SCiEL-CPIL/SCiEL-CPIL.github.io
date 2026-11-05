import math
import time
import random
from fractions import Fraction

def construct():
    N = 511
    deadline = time.time() + 130.0  # Safety margin before 150s kill
    
    # Precompute primes < 512
    primes = []
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    
    # Map prime to index for bitmask
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Precompute prime factor mask and score for each number 1..511
    masks = [0] * (N + 1)
    scores = [0.0] * (N + 1)
    for a in range(1, N + 1):
        scores[a] = 1.0 / (512 - a)
        x = a
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                masks[a] |= (1 << prime_to_idx[p])
                while x % p == 0:
                    x //= p
        if x > 1:
            masks[a] |= (1 << prime_to_idx[x])
    
    # Sort candidates by score descending (larger a = higher score)
    candidates = list(range(1, N + 1))
    candidates.sort(key=lambda a: -scores[a])
    
    # Greedy construction: add highest-score candidate that's coprime to all selected
    def greedy_construct():
        selected = []
        selected_mask = 0
        for a in candidates:
            if masks[a] & selected_mask == 0:
                selected.append(a)
                selected_mask |= masks[a]
        return selected
    
    best_set = greedy_construct()
    best_score = sum(scores[a] for a in best_set)
    
    # Local search with multiple move types
    def try_improve(current_set, current_score, time_limit):
        """Hill climbing with swap, merge, and split moves."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            n = len(current_set)
            
            # Try swaps: replace one element with another
            for i in range(n):
                a_old = current_set[i]
                for a_new in candidates:
                    if a_new == a_old or a_new in current_set:
                        continue
                    # Check if a_new is coprime to all others
                    valid = True
                    for j in range(n):
                        if j == i:
                            continue
                        if masks[a_new] & masks[current_set[j]]:
                            valid = False
                            break
                    if not valid:
                        continue
                    new_score = current_score - scores[a_old] + scores[a_new]
                    if new_score > current_score + 1e-15:
                        current_set[i] = a_new
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_set = list(current_set)
                        break  # Restart after a successful swap
            
            # Try merges: replace two coprime elements a, b with c = k*lcm(a,b)
            n = len(current_set)
            merged = False
            for i in range(n):
                for j in range(i + 1, n):
                    a = current_set[i]
                    b = current_set[j]
                    base = a * b  # lcm since coprime
                    if base > N:
                        continue
                    # Try multiples of base
                    for mult in range(1, N // base + 1):
                        c = base * mult
                        if c == a or c == b:
                            continue
                        # Check c is coprime to all other elements
                        valid = True
                        for k in range(n):
                            if k == i or k == j:
                                continue
                            if masks[c] & masks[current_set[k]]:
                                valid = False
                                break
                        if not valid:
                            continue
                        new_score = current_score - scores[a] - scores[b] + scores[c]
                        if new_score > current_score + 1e-15:
                            # Replace a and b with c
                            current_set[i] = c
                            current_set.pop(j)
                            current_score = new_score
                            improved = True
                            merged = True
                            if current_score > best_score:
                                best_score = current_score
                                best_set = list(current_set)
                            break
                    if merged:
                        break
                if merged:
                    break
            
            # Try splits: replace composite a with its prime factors
            n = len(current_set)
            split = False
            for i in range(n):
                a = current_set[i]
                if is_prime[a]:
                    continue
                # Get distinct prime factors
                factors = []
                x = a
                for p in primes:
                    if p * p > x:
                        break
                    if x % p == 0:
                        factors.append(p)
                        while x % p == 0:
                            x //= p
                if x > 1:
                    factors.append(x)
                
                # Check if factors are coprime to all other elements
                valid = True
                for k in range(n):
                    if k == i:
                        continue
                    for f in factors:
                        if masks[f] & masks[current_set[k]]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                new_score = current_score - scores[a] + sum(scores[f] for f in factors)
                if new_score > current_score + 1e-15:
                    current_set.pop(i)
                    current_set.extend(factors)
                    current_score = new_score
                    improved = True
                    split = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current_set)
                    break
            if split:
                continue
        
        return current_score, best_score, best_set
    
    # Run local search from greedy start
    current_set = list(best_set)
    current_score = best_score
    current_score, best_score, best_set = try_improve(current_set, current_score, deadline - 5.0)
    
    # Multiple restarts with different greedy orderings
    random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 5.0:
            break
        
        # Perturb the candidate order
        perturbed_candidates = list(candidates)
        # Randomly shuffle a portion
        k = min(50, len(perturbed_candidates))
        idx = random.sample(range(len(perturbed_candidates)), k)
        for i in idx:
            j = random.randint(0, len(perturbed_candidates) - 1)
            perturbed_candidates[i], perturbed_candidates[j] = perturbed_candidates[j], perturbed_candidates[i]
        
        # Greedy with perturbed order
        selected = []
        selected_mask = 0
        for a in perturbed_candidates:
            if masks[a] & selected_mask == 0:
                selected.append(a)
                selected_mask |= masks[a]
        
        if sum(scores[a] for a in selected) > best_score:
            best_set = list(selected)
            best_score = sum(scores[a] for a in selected)
        
        # Local search from this start
        current_set = list(selected)
        current_score = sum(scores[a] for a in current_set)
        current_score, best_score, best_set = try_improve(current_set, current_score, deadline - 5.0)
    
    return best_set
