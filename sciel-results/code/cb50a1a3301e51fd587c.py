import time
import math
import random
import numpy as np

def _is_dissociated(S):
    """
    Check if set S has distinct reciprocal subset sums using NumPy.
    """
    k = len(S)
    if k <= 1:
        return True
    
    # Calculate LCM
    L = 1
    for x in S:
        L = (L * x) // math.gcd(L, x)
        
    # If L is too large for int64, fall back to Python ints (unlikely for [1,22])
    # Max LCM for subset of [1,22] is ~2.3e8, fits in int64.
    
    # Calculate integer weights
    weights = np.array([L // x for x in S], dtype=np.int64)
    
    # Generate all subset sums using doubling technique
    # Start with sums = [0]
    sums = np.array([0], dtype=np.int64)
    
    for w in weights:
        # New sums are old sums and old sums + w
        # Concatenate
        sums = np.concatenate((sums, sums + w))
        
    # Check if all are distinct
    # For small k, np.unique is fast
    # Alternative: sort and check diffs, but unique is fine for 2^k < 10000
    unique_sums = np.unique(sums)
    
    return len(unique_sums) == len(sums)

def _greedy_construct(order, time_limit=1.0):
    """
    Construct a dissociated set by greedily adding elements in the given order.
    """
    S = []
    start = time.time()
    for x in order:
        if time.time() - start > time_limit:
            break
        if _is_dissociated(S + [x]):
            S.append(x)
    return S

def construct():
    """
    Find a dissociated reciprocal set from [1, 22] with maximum cardinality.
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    all_elements = list(range(1, 23))
    best_set = []
    
    # Phase 1: Greedy with different orderings to find a good starting point
    # Increasing
    S_inc = _greedy_construct(all_elements)
    if len(S_inc) > len(best_set):
        best_set = S_inc
        
    # Decreasing
    S_dec = _greedy_construct(sorted(all_elements, reverse=True))
    if len(S_dec) > len(best_set):
        best_set = S_dec
        
    # Primes first
    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    non_primes = [x for x in all_elements if x not in primes]
    S_prime = _greedy_construct(primes + non_primes)
    if len(S_prime) > len(best_set):
        best_set = S_prime
        
    # Random orders
    random.seed(42)
    for _ in range(50):
        order = all_elements[:]
        random.shuffle(order)
        S_rand = _greedy_construct(order, time_limit=0.05)
        if len(S_rand) > len(best_set):
            best_set = S_rand
            
    best_size = len(best_set)
    if best_size == 0:
        best_set = [1]
        best_size = 1

    current_set = best_set[:]
    in_set = set(current_set)
    
    # Phase 2: Local Search
    # We try to add elements, and if stuck, we try swaps to rearrange.
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        # Try to add an element
        added = False
        remaining = [x for x in all_elements if x not in in_set]
        if remaining:
            # Shuffle to randomize search
            random.shuffle(remaining)
            for cand in remaining:
                if time.time() > deadline:
                    break
                test_set = current_set + [cand]
                if _is_dissociated(test_set):
                    current_set = test_set
                    in_set.add(cand)
                    added = True
                    break
        
        if added:
            # If we added, update best
            if len(current_set) > best_size:
                best_set = current_set[:]
                best_size = len(best_set)
                no_improve_count = 0
            # Continue loop to try adding more
            continue
        
        # If no add possible, try a swap
        swapped = False
        if current_set and remaining:
            # Pick a random element to remove and a random element to add
            # To be more systematic, we can iterate through some candidates
            # But random is good for exploration
            
            # Try a few random swaps
            for _ in range(10):
                if time.time() > deadline:
                    break
                out_cand = random.choice(remaining)
                in_cand = random.choice(current_set)
                
                if out_cand == in_cand:
                    continue
                
                new_set = [x for x in current_set if x != in_cand] + [out_cand]
                
                if _is_dissociated(new_set):
                    # Accept the swap
                    current_set = new_set
                    in_set = set(current_set)
                    remaining = [x for x in all_elements if x not in in_set]
                    swapped = True
                    break
            
        if not swapped and not added:
            no_improve_count += 1
            
            # If stuck for a while, try a more aggressive restart or reset
            if no_improve_count > max_no_improve:
                # Reset to best
                current_set = best_set[:]
                in_set = set(current_set)
                remaining = [x for x in all_elements if x not in in_set]
                no_improve_count = 0
                
                # Occasionally, try to build a new random set to escape local optimum
                if random.random() < 0.05:
                    # Try a random greedy construction with a different seed
                    random.seed(int(time.time() * 1000) % 10000)
                    order = all_elements[:]
                    random.shuffle(order)
                    S_new = _greedy_construct(order, time_limit=0.1)
                    if len(S_new) > len(current_set):
                        current_set = S_new
                        in_set = set(current_set)
                        remaining = [x for x in all_elements if x not in in_set]
                        if len(current_set) > best_size:
                            best_set = current_set[:]
                            best_size = len(best_set)

    return sorted(best_set)

if __name__ == "__main__":
    result = construct()
    print(f"Found set of size {len(result)}: {result}")
