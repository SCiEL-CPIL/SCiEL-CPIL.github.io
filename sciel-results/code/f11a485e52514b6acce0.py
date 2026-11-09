import time
from math import gcd
from fractions import Fraction
from itertools import combinations

# Precomputation
def _precompute():
    N = 512
    primes = []
    for i in range(2, N):
        is_prime = True
        for p in primes:
            if p * p > i:
                break
            if i % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
    
    prime_index = {p: i for i, p in enumerate(primes)}
    
    masks = [0] * N
    for a in range(1, N):
        m = 0
        temp = a
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                m |= (1 << prime_index[p])
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            m |= (1 << prime_index[temp])
        masks[a] = m
        
    values = [Fraction(0)] * N
    for a in range(1, N):
        values[a] = Fraction(1, 512 - a)
        
    return masks, values, primes

def _score(S, values):
    total = Fraction(0)
    for a in S:
        total += values[a]
    return total

def _mask_of_set(S, masks):
    m = 0
    for a in S:
        m |= masks[a]
    return m

def _greedy_fill(S, masks, values):
    """
    Greedily adds best possible a to S (modifies S and returns new mask).
    S is a list/set.
    """
    S_set = set(S)
    current_mask = _mask_of_set(S_set, masks)
    changed = True
    while changed:
        changed = False
        best_a = -1
        best_val = Fraction(0)
        for a in range(511, 0, -1):
            if a in S_set:
                continue
            if masks[a] & current_mask == 0:
                if values[a] > best_val:
                    best_val = values[a]
                    best_a = a
        if best_a != -1:
            S_set.add(best_a)
            current_mask |= masks[best_a]
            changed = True
    return list(S_set), current_mask

def _greedy_with_order(order, masks, values):
    """
    Greedy fill using a specific order of candidates.
    Returns the set of selected elements.
    """
    S_set = set()
    current_mask = 0
    for a in order:
        if masks[a] & current_mask == 0:
            S_set.add(a)
            current_mask |= masks[a]
    return S_set

def _swap_local_search(S, masks, values, deadline):
    """
    Local search using swap operations.
    Try replacing one element with another if it improves the score.
    """
    S_list = list(S)
    current_score = _score(S_list, values)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        for i, a in enumerate(S_list):
            if time.time() > deadline:
                return S_list
            # Try removing a and adding a different b
            S_without_a = [x for x in S_list if x != a]
            mask_without_a = _mask_of_set(S_without_a, masks)
            
            # Find best b to add
            best_b = -1
            best_gain = Fraction(0)
            for b in range(511, 0, -1):
                if b in S_list:
                    continue
                if masks[b] & mask_without_a == 0:
                    gain = values[b] - values[a]
                    if gain > best_gain:
                        best_gain = gain
                        best_b = b
            
            if best_b != -1 and best_gain > 0:
                S_list[i] = best_b
                current_score += best_gain
                improved = True
    
    return S_list

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    masks, values, primes = _precompute()
    
    # Generate multiple candidate orderings
    # 1. Descending by value (natural greedy)
    order1 = list(range(511, 0, -1))
    
    # 2. Descending by value, but with some randomization
    import random
    random.seed(42)
    order2 = list(range(511, 0, -1))
    # Shuffle the lower half to create diversity
    half = len(order2) // 2
    lower_half = order2[half:]
    random.shuffle(lower_half)
    order2 = order2[:half] + lower_half
    
    # 3. Random order
    order3 = list(range(1, 512))
    random.shuffle(order3)
    
    # 4. Ascending (to see if it helps)
    order4 = list(range(1, 512))
    
    # Try each ordering
    best_S = None
    best_score = Fraction(0)
    
    for order in [order1, order2, order3, order4]:
        S_set = _greedy_with_order(order, masks, values)
        S_list = list(S_set)
        score = _score(S_list, values)
        
        if best_S is None or score > best_score:
            best_S = S_list
            best_score = score
    
    # Local search with swaps
    if time.time() < deadline:
        best_S = _swap_local_search(best_S, masks, values, deadline)
        best_score = _score(best_S, values)
    
    # Additional local search: remove and refill
    while time.time() < deadline:
        S = list(best_S)
        improved = False
        
        # Try removing each element and greedily refilling
        S_sorted = sorted(S, key=lambda x: values[x])
        
        for b in S_sorted:
            if time.time() > deadline:
                break
                
            S_temp = [x for x in S if x != b]
            S_filled, _ = _greedy_fill(S_temp, masks, values)
            score_temp = _score(S_filled, values)
            
            if score_temp > best_score:
                best_S = S_filled
                best_score = score_temp
                improved = True
                break
        
        if not improved:
            # Try a random perturbation
            if best_S and len(best_S) > 1:
                idx = int(time.time() * 1000) % len(best_S)
                b = best_S[idx]
                S_temp = [x for x in best_S if x != b]
                S_new, _ = _greedy_fill(S_temp, masks, values)
                score_new = _score(S_new, values)
                if score_new > best_score:
                    best_S = S_new
                    best_score = score_new
                else:
                    # Try swap perturbation
                    if len(best_S) >= 2:
                        i = int(time.time() * 1000) % len(best_S)
                        a = best_S[i]
                        S_without_a = [x for x in best_S if x != a]
                        mask_without_a = _mask_of_set(S_without_a, masks)
                        
                        # Find a random b to swap with
                        candidates = [b for b in range(511, 0, -1) 
                                     if b not in best_S and (masks[b] & mask_without_a) == 0]
                        if candidates:
                            b = candidates[int(time.time() * 1000) % len(candidates)]
                            S_new = S_without_a + [b]
                            score_new = _score(S_new, values)
                            if score_new > best_score:
                                best_S = S_new
                                best_score = score_new
                    break
    
    return best_S
