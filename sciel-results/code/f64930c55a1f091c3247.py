import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 10.0
    
    N = 512
    max_a = N - 1
    
    # Pre-compute weights
    weights = [0.0] * (max_a + 1)
    for a in range(1, max_a + 1):
        weights[a] = 1.0 / (N - a)
        
    # Precompute conflict masks
    conflict_mask = [0] * (max_a + 1)
    for a in range(1, max_a + 1):
        mask = 0
        for b in range(1, max_a + 1):
            if gcd(a, b) > 1:
                mask |= (1 << b)
        conflict_mask[a] = mask

    # 1. Construct Prime Power Set (PPS) + 1
    # This is a strong baseline.
    best_mask = 1 << 1 # Include 1
    best_score = weights[1]
    
    # Find primes and their max powers
    # Sieve of Eratosthenes for primes < 512
    is_prime = [True] * (max_a + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_a**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_a + 1, i):
                is_prime[j] = False
                
    for p in range(2, max_a + 1):
        if is_prime[p]:
            # Find max power p^k < 512
            power = p
            while power * p < N:
                power *= p
            # Check if power is already in mask (shouldn't be, distinct primes)
            # But 1 is in mask. 1 is coprime to all.
            if not (best_mask >> power) & 1:
                # Check coprime to current best_mask
                # Since best_mask contains powers of other primes and 1, it should be coprime.
                if (best_mask & conflict_mask[power]) == 0:
                    best_mask |= (1 << power)
                    best_score += weights[power]

    # Local Search
    improved = True
    while improved:
        improved = False
        
        # 1-Opt: Swap one element
        current_elements = [a for a in range(1, max_a + 1) if (best_mask >> a) & 1]
        
        for a in current_elements:
            others_mask = best_mask ^ (1 << a)
            # Find best b > a (higher weight) not in mask, coprime to others
            # Iterate b from max_a down to 1
            for b in range(max_a, 0, -1):
                if (best_mask >> b) & 1:
                    continue
                if (others_mask & conflict_mask[b]) == 0:
                    if weights[b] > weights[a]:
                        best_mask = (best_mask ^ (1 << a)) | (1 << b)
                        best_score += weights[b] - weights[a]
                        improved = True
                        break
            if improved:
                break
        if improved:
            if time.time() - start_time > deadline - safety_margin:
                break
            continue

        # 2-Opt: Swap two elements
        # Only run if time permits
        if time.time() - start_time < deadline - safety_margin - 20:
            current_elements = [a for a in range(1, max_a + 1) if (best_mask >> a) & 1]
            # Sort by weight ascending to prioritize removing low weight elements?
            # Or just iterate all pairs.
            # To save time, iterate a, b and look for c, d.
            # We'll do a limited 2-Opt: try to replace the two lowest weight elements with two higher weight ones.
            # This is a heuristic.
            
            # Get sorted elements by weight
            sorted_els = sorted(current_elements, key=lambda x: weights[x])
            # Try to replace the last 2 (lowest weight)
            if len(sorted_els) >= 2:
                a, b = sorted_els[-2], sorted_els[-1]
                others_mask = best_mask ^ (1 << a) ^ (1 << b)
                
                # Find best c, d
                best_c = -1
                best_d = -1
                best_gain = 0
                target_gain = weights[a] + weights[b]
                
                # Iterate c, d from high weight
                # This is O(N^2)
                for c in range(max_a, 0, -1):
                    if (best_mask >> c) & 1:
                        continue
                    if (others_mask & conflict_mask[c]) != 0:
                        continue
                    for d in range(c - 1, 0, -1):
                        if (best_mask >> d) & 1:
                            continue
                        if (others_mask & conflict_mask[d]) != 0:
                            continue
                        if (conflict_mask[c] >> d) & 1:
                            continue
                        gain = weights[c] + weights[d] - target_gain
                        if gain > best_gain:
                            best_gain = gain
                            best_c = c
                            best_d = d
                            
                if best_gain > 1e-12:
                    best_mask = (best_mask ^ (1 << a) ^ (1 << b)) | (1 << best_c) | (1 << best_d)
                    best_score += best_gain
                    improved = True

        if improved:
            continue
        if time.time() - start_time > deadline - safety_margin:
            break

    # Drop-and-Refill
    improved = True
    while improved:
        improved = False
        current_elements = [a for a in range(1, max_a + 1) if (best_mask >> a) & 1]
        for a in current_elements:
            temp_mask = best_mask ^ (1 << a)
            temp_score = best_score - weights[a]
            
            for b in range(max_a, 0, -1):
                if (temp_mask >> b) & 1:
                    continue
                if (temp_mask & conflict_mask[b]) == 0:
                    temp_mask |= (1 << b)
                    temp_score += weights[b]
                    
            if temp_score > best_score + 1e-12:
                best_mask = temp_mask
                best_score = temp_score
                improved = True
                break
            if time.time() - start_time > deadline - safety_margin:
                break
        if improved:
            continue
        if time.time() - start_time > deadline - safety_margin:
            break

    best_chosen = [a for a in range(1, max_a + 1) if (best_mask >> a) & 1]
    return best_chosen
