import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s
    
    # Precompute weights
    # L = lcm(1, ..., 22)
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def lcm(a, b):
        return a * b // gcd(a, b)

    L = 1
    for i in range(1, 23):
        L = lcm(L, i)
        
    weights = [0] * 23
    for i in range(1, 23):
        weights[i] = L // i
        
    # Faster incremental check
    def check_addable(current_sums, element_idx):
        w = weights[element_idx]
        new_sums = set()
        for s in current_sums:
            ns = s + w
            if ns in current_sums:
                return None
            new_sums.add(ns)
        return new_sums

    all_elements = list(range(1, 23))
    best_set = []
    best_sums = None
    
    # Deterministic initial attempts with good heuristics
    # 1. Ascending weights (largest indices first) - usually allows more elements
    # 2. Descending weights (smallest indices first) - restrictive but might find specific structures
    # 3. Random restarts
    
    orders = [
        sorted(all_elements, key=lambda x: weights[x]),      # Ascending weights
        sorted(all_elements, key=lambda x: -weights[x]),     # Descending weights
        all_elements[::-1], 
        all_elements
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        current = []
        current_sums = {0}
        
        for e in order:
            if time.time() > deadline:
                break
            new_sums = check_addable(current_sums, e)
            if new_sums is not None:
                current.append(e)
                current_sums.update(new_sums)
        
        if len(current) > len(best_set):
            best_set = current[:]
            best_sums = set(current_sums)
            
    # Local search: try to add remaining elements to best_set
    if time.time() < deadline and best_set:
        remaining = [e for e in all_elements if e not in best_set]
        improved = True
        while improved:
            improved = False
            for e in remaining[:]:
                if time.time() > deadline:
                    break
                new_sums = check_addable(best_sums, e)
                if new_sums is not None:
                    best_set.append(e)
                    best_sums.update(new_sums)
                    remaining.remove(e)
                    improved = True

    # Randomized restarts
    # The greedy choice is order-dependent. Randomizing the order helps find larger sets.
    while time.time() < deadline - 2: # Leave 2s for final polish
        # Shuffle elements
        order = all_elements[:]
        random.shuffle(order)
        
        current = []
        current_sums = {0}
        
        for e in order:
            if time.time() > deadline - 1:
                break
            new_sums = check_addable(current_sums, e)
            if new_sums is not None:
                current.append(e)
                current_sums.update(new_sums)
        
        # Try to add remaining elements to this new candidate
        if len(current) >= len(best_set): # Only worth polishing if it's competitive
            remaining = [e for e in all_elements if e not in current]
            improved = True
            while improved:
                improved = False
                for e in remaining[:]:
                    if time.time() > deadline - 1:
                        break
                    new_sums = check_addable(current_sums, e)
                    if new_sums is not None:
                        current.append(e)
                        current_sums.update(new_sums)
                        remaining.remove(e)
                        improved = True
            
            if len(current) > len(best_set):
                best_set = current[:]
                best_sums = set(current_sums)
                
        # Early exit if we found a very large set (unlikely to beat 20+ easily)
        if len(best_set) >= 19:
            # We can reduce the effort or break? 
            # Let's keep searching a bit more to be sure, but maybe skip some shuffles?
            # No, just let it run.
            pass

    # Final "add remaining" pass on the potentially improved best_set
    if time.time() < deadline and best_set:
        remaining = [e for e in all_elements if e not in best_set]
        improved = True
        while improved:
            improved = False
            for e in remaining[:]:
                if time.time() > deadline:
                    break
                new_sums = check_addable(best_sums, e)
                if new_sums is not None:
                    best_set.append(e)
                    best_sums.update(new_sums)
                    remaining.remove(e)
                    improved = True

    # Fallback
    if not best_set:
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
