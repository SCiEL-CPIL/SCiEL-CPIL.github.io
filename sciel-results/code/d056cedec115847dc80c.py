import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    N = 512
    max_a = N - 1
    
    # Precompute prime factors for each a in 1..511
    # factor_masks[a] is a bitmask of prime indices
    # We need to map primes to indices
    primes = []
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
        
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    factor_masks = [0] * (N + 1)
    factor_lists = [[] for _ in range(N + 1)]
    for a in range(1, N):
        temp = a
        mask = 0
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                mask |= (1 << prime_to_idx[p])
                factor_lists[a].append(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            mask |= (1 << prime_to_idx[temp])
            factor_lists[a].append(temp)
        factor_masks[a] = mask
        
    # Weights
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)
        
    # Greedy Initialization
    chosen = []
    mask = 0
    score = 0.0
    for a in range(max_a, 0, -1):
        if (mask & factor_masks[a]) == 0:
            chosen.append(a)
            mask |= factor_masks[a]
            score += weights[a]
            
    # Data structures for local search
    # prime_count[p_idx] = count of chosen numbers having prime p
    num_primes = len(primes)
    prime_count = [0] * num_primes
    
    def update_counts(a, delta):
        for p_idx in [prime_to_idx[p] for p in factor_lists[a]]:
            prime_count[p_idx] += delta
            
    # Initialize counts
    for a in chosen:
        update_counts(a, 1)
        
    in_set = [False] * N
    for a in chosen:
        in_set[a] = True
        
    def is_coprime_to_set(a, exclude=None):
        # Check if a is coprime to all elements in chosen, except 'exclude'
        # Using prime_count
        for p in factor_lists[a]:
            p_idx = prime_to_idx[p]
            count = prime_count[p_idx]
            if exclude is not None and p in factor_lists[exclude]:
                # If exclude has this prime, the effective count is count - 1
                if count > 1:
                    return False
            else:
                if count > 0:
                    return False
        return True

    def greedy_fill(current_chosen, current_mask, current_score, current_counts):
        # Returns (new_chosen, new_mask, new_score, new_counts)
        # We simulate adding numbers greedily to the current set
        new_chosen = list(current_chosen)
        new_mask = current_mask
        new_score = current_score
        new_counts = list(current_counts)
        
        for a in range(max_a, 0, -1):
            if a in new_chosen:
                continue
            # Check coprimality using new_counts
            valid = True
            for p in factor_lists[a]:
                p_idx = prime_to_idx[p]
                if new_counts[p_idx] > 0:
                    valid = False
                    break
            if valid:
                new_chosen.append(a)
                new_mask |= factor_masks[a]
                new_score += weights[a]
                for p in factor_lists[a]:
                    new_counts[prime_to_idx[p]] += 1
        return new_chosen, new_mask, new_score, new_counts

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try Single Swaps
        # For each x in chosen, try to replace with y not in chosen
        for i, x in enumerate(chosen):
            # Try replacing x with y
            # We iterate y from high to low to find the best replacement quickly?
            # Or just find any improvement.
            best_y = -1
            best_gain = 0.0
            for y in range(max_a, 0, -1):
                if in_set[y]:
                    continue
                # Check if y is coprime to chosen \ {x}
                # Use prime_count logic
                valid = True
                for p in factor_lists[y]:
                    p_idx = prime_to_idx[p]
                    cnt = prime_count[p_idx]
                    if p in factor_lists[x]:
                        if cnt > 1:
                            valid = False
                            break
                    else:
                        if cnt > 0:
                            valid = False
                            break
                if valid:
                    gain = weights[y] - weights[x]
                    if gain > best_gain:
                        best_gain = gain
                        best_y = y
            
            if best_y != -1:
                # Apply swap
                # Remove x
                update_counts(x, -1)
                in_set[x] = False
                chosen[i] = best_y
                # Add best_y
                update_counts(best_y, 1)
                in_set[best_y] = True
                score += best_gain
                improved = True
                # Break to restart with new state
                break
        
        if improved:
            continue
            
        # Try Additions
        for y in range(max_a, 0, -1):
            if in_set[y]:
                continue
            if is_coprime_to_set(y):
                # Add y
                chosen.append(y)
                in_set[y] = True
                update_counts(y, 1)
                score += weights[y]
                improved = True
                break
        
        if improved:
            continue
            
        # Try Remove-and-Greedy-Fill
        # For each x in chosen, remove it, then greedily fill
        # This is more expensive, so we do it less frequently or only if stuck
        # To save time, we only try this for a few random x or the smallest x
        # Let's try removing the smallest x (lowest weight)
        if chosen:
            min_idx = 0
            min_w = weights[chosen[0]]
            for i in range(1, len(chosen)):
                if weights[chosen[i]] < min_w:
                    min_w = weights[chosen[i]]
                    min_idx = i
            
            x = chosen[min_idx]
            # Remove x
            temp_chosen = [a for i, a in enumerate(chosen) if i != min_idx]
            temp_mask = 0
            temp_score = 0.0
            temp_counts = [0] * num_primes
            for a in temp_chosen:
                temp_mask |= factor_masks[a]
                temp_score += weights[a]
                for p in factor_lists[a]:
                    temp_counts[prime_to_idx[p]] += 1
            
            new_chosen, new_mask, new_score, new_counts = greedy_fill(temp_chosen, temp_mask, temp_score, temp_counts)
            
            if new_score > score + 1e-9:
                # Accept
                chosen = new_chosen
                mask = new_mask
                score = new_score
                prime_count = new_counts
                in_set = [False] * N
                for a in chosen:
                    in_set[a] = True
                improved = True
                # Restart loop
                continue

    return chosen
