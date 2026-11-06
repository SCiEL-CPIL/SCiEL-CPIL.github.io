"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def max_sum_free(elements, max_time=30.0):
        n = len(elements)
        if n <= 3:
            return list(elements)
        
        # 1. Greedy initialization: sort by absolute value descending
        #    Large numbers are less likely to be sums of two others in the set
        #    (since sums tend to be larger than summands, but here we have mixed signs).
        #    Actually, for mixed signs, it's complex. 
        #    Let's use a heuristic: elements with high absolute value are "safer" 
        #    because it's harder to form them as a sum of two smaller magnitude numbers 
        #    if the set is sparse, but actually the opposite is true for sums.
        #    Wait: a+b=c. If c is large, a and b must be large. 
        #    If we pick large numbers, their sums are even larger (or negative large).
        #    If the range is bounded, large numbers have fewer partners that sum to another in-set number.
        
        sorted_elems = sorted(elements, key=lambda x: abs(x), reverse=True)
        
        S = []
        S_set = set()
        
        # Greedy add
        for x in sorted_elems:
            # Check if adding x violates sum-free property
            # x = a + b  (x is sum) -> need a, b in S
            # a + b = x  (x is summand) -> need a in S such that x-a in S, OR b in S such that x-b in S
            # Since a=b allowed, we check:
            # 1. Is x a sum of two elements in S?
            #    For a in S: if (x - a) in S: conflict
            # 2. Is x part of a sum with another element in S?
            #    For a in S: if (x + a) in S: conflict (x + a = c, c in S)
            #    Note: if x+a = x, then a=0, but 0 is not in set.
            
            conflict = False
            for a in S:
                # Case 1: a + b = x => b = x - a
                if (x - a) in S_set:
                    conflict = True
                    break
                # Case 2: a + x = c => c = a + x
                # If c is in S, conflict.
                if (a + x) in S_set:
                    conflict = True
                    break
                # Case 3: x + x = c => c = 2x
                if (2 * x) in S_set:
                    conflict = True
                    break
            
            if not conflict:
                S.append(x)
                S_set.add(x)
        
        # 2. Local Search / Improvement
        # Try to add elements we skipped, or swap elements to allow adding more.
        # Simpler: Try to add any remaining element if it fits.
        remaining = [x for x in elements if x not in S_set]
        
        # Shuffle remaining for variety in local search
        rng_local = Random(42)
        rng_local.shuffle(remaining)
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for x in remaining:
                if time.time() > max_time:
                    break
                # Check if x can be added to current S
                conflict = False
                for a in S:
                    if (x - a) in S_set or (a + x) in S_set:
                        conflict = True
                        break
                if (2 * x) in S_set:
                    conflict = True
                    break
                
                if not conflict:
                    S.append(x)
                    S_set.add(x)
                    remaining.remove(x)
                    improved = True
        
        # 3. Advanced Local Search: Swap
        # Try swapping an element in S with an element not in S to potentially allow more additions later?
        # This is complex. Given time constraints, the greedy + fill is likely strong.
        # Let's add a simple "try to remove one element to add two" heuristic if time permits.
        
        if time.time() < max_time - 2.0:
            # Try to improve by removing 1 and adding 2
            # This is O(N^3) roughly, might be slow.
            # Let's stick to the greedy fill which is O(N^2).
            pass

        return S

    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    for size, seed in instances:
        if time.time() > deadline:
            # Return best effort so far, or empty lists if not started
            results.append([])
            continue
            
        elems = instance(size, seed)
        # Allocate time based on size, but cap it
        # 60: 5s, 120: 10s, 240: 20s?
        # Total available ~110s.
        # Let's give proportional time.
        # 60/420 * 100 = 14s
        # 120/420 * 100 = 28s
        # 240/420 * 100 = 57s
        # This fits in 110s.
        
        time_budget = (size / 420.0) * 100.0
        
        # Ensure we don't exceed deadline
        remaining_time = deadline - time.time()
        if time_budget > remaining_time:
            time_budget = max(1.0, remaining_time - 1.0)
            
        subset = max_sum_free(elems, max_time=time_budget)
        results.append(subset)

    return results
# EVOLVE-BLOCK-END
