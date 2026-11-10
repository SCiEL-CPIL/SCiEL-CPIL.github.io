import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Precompute primes and masks
    N = 511
    # Sieve of Eratosthenes to find primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Calculate masks and weights for each number
    masks = [0] * (N + 1)
    weights = [0.0] * (N + 1)
    
    for a in range(1, N + 1):
        weights[a] = 1.0 / (512 - a)
        mask = 0
        x = a
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                mask |= (1 << prime_to_idx[p])
                while x % p == 0:
                    x //= p
        if x > 1:
            # x is a prime
            mask |= (1 << prime_to_idx[x])
        masks[a] = mask
    
    # 2. Greedy Initialization
    # Sort indices by weight descending
    indices = list(range(1, N + 1))
    indices.sort(key=lambda x: weights[x], reverse=True)
    
    def greedy_init():
        chosen = []
        chosen_mask = 0
        for a in indices:
            if masks[a] & chosen_mask == 0:
                chosen.append(a)
                chosen_mask |= masks[a]
        return chosen, chosen_mask
    
    chosen, chosen_mask = greedy_init()
    chosen_set = set(chosen)
    
    def get_score():
        return sum(weights[a] for a in chosen)
    
    best_score = get_score()
    best_chosen = list(chosen)
    best_chosen_mask = chosen_mask
    
    # 3. Local Search with Simulated Annealing
    # Temperature schedule
    T = 0.01
    T_min = 1e-6
    cooling = 0.999
    
    def local_search(chosen, chosen_mask, chosen_set, T):
        nonlocal best_score, best_chosen, best_chosen_mask
        improved = True
        no_improve_count = 0
        
        while improved and time.time() < deadline:
            improved = False
            no_improve_count = 0
            
            # 1-for-1 swaps with simulated annealing
            for a in list(chosen):
                base_mask = chosen_mask ^ masks[a]
                w_a = weights[a]
                
                # Collect all valid swaps
                valid_swaps = []
                for b in indices:
                    if weights[b] <= w_a * 0.5:  # Early break optimization
                        break
                    if b in chosen_set:
                        continue
                    if masks[b] & base_mask:
                        continue
                    delta_w = weights[b] - w_a
                    if delta_w > 0:
                        valid_swaps.append((b, delta_w))
                    elif delta_w > -T:  # Accept worse moves with probability
                        valid_swaps.append((b, delta_w))
                
                if not valid_swaps:
                    continue
                
                # Pick best improving or random worsening move
                improving = [s for s in valid_swaps if s[1] > 0]
                if improving:
                    b, delta_w = max(improving, key=lambda x: x[1])
                    accept = True
                else:
                    b, delta_w = random.choice(valid_swaps)
                    # Metropolis criterion
                    if delta_w < 0:
                        import math
                        accept = random.random() < math.exp(delta_w / T)
                    else:
                        accept = True
                
                if accept:
                    chosen.remove(a)
                    chosen_set.remove(a)
                    chosen.append(b)
                    chosen_set.add(b)
                    chosen_mask = base_mask | masks[b]
                    improved = True
                    no_improve_count = 0
                    
                    # Update best
                    current_score = get_score()
                    if current_score > best_score:
                        best_score = current_score
                        best_chosen = list(chosen)
                        best_chosen_mask = chosen_mask
                    break  # Restart after any move to maintain consistency
            
            # 1-for-2 swaps (only if we have time)
            if not improved and time.time() < deadline - 5:
                for a in list(chosen):
                    base_mask = chosen_mask ^ masks[a]
                    w_a = weights[a]
                    
                    found_swap = False
                    for b in indices:
                        if weights[b] * 2 <= w_a:  # Even max c won't help
                            break
                        if b in chosen_set:
                            continue
                        if masks[b] & base_mask:
                            continue
                        
                        w_b = weights[b]
                        for c in indices:
                            if weights[c] < w_a - w_b:
                                break
                            if c in chosen_set or c == b:
                                continue
                            if masks[c] & base_mask:
                                continue
                            if masks[b] & masks[c]:
                                continue
                            
                            # Valid swap
                            chosen.remove(a)
                            chosen_set.remove(a)
                            chosen.append(b)
                            chosen.append(c)
                            chosen_set.add(b)
                            chosen_set.add(c)
                            chosen_mask = base_mask | masks[b] | masks[c]
                            improved = True
                            found_swap = True
                            
                            # Update best
                            current_score = get_score()
                            if current_score > best_score:
                                best_score = current_score
                                best_chosen = list(chosen)
                                best_chosen_mask = chosen_mask
                            break
                        if found_swap:
                            break
                    if found_swap:
                        break
            
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min
        
        return chosen, chosen_mask, chosen_set
    
    # Run local search with multiple restarts
    restarts = 0
    max_restarts = 5
    
    while time.time() < deadline - 10:
        # Run local search
        chosen, chosen_mask, chosen_set = local_search(chosen, chosen_mask, chosen_set, T)
        restarts += 1
        
        # If we've done enough restarts or are near deadline, stop
        if restarts >= max_restarts or time.time() > deadline - 15:
            break
        
        # Random restart: perturb the current best
        if random.random() < 0.3:
            # Small perturbation
            if len(best_chosen) > 2:
                idx = random.randint(0, len(best_chosen) - 1)
                a_to_remove = best_chosen[idx]
                chosen = list(best_chosen)
                chosen.remove(a_to_remove)
                chosen_set = set(chosen)
                chosen_mask = 0
                for a in chosen:
                    chosen_mask |= masks[a]
                
                # Try to add a random compatible element
                for b in indices:
                    if b not in chosen_set and masks[b] & chosen_mask == 0:
                        chosen.append(b)
                        chosen_set.add(b)
                        chosen_mask |= masks[b]
                        break
            else:
                # Full restart
                chosen, chosen_mask = greedy_init()
                chosen_set = set(chosen)
        
        # Reset temperature for next phase
        T = 0.01
    
    # Final polish: run one more local search on the best solution
    if time.time() < deadline - 5:
        chosen = list(best_chosen)
        chosen_mask = best_chosen_mask
        chosen_set = set(chosen)
        chosen, chosen_mask, chosen_set = local_search(chosen, chosen_mask, chosen_set, 0.001)
    
    # Return the best solution found
    return best_chosen

# EVOLVE-BLOCK-END
