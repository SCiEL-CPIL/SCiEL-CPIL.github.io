"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill
    
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)
    
    def greedy_sum_free(elements, order):
        """
        Greedily construct a sum-free set from 'elements' following the order in 'order'.
        """
        S = []
        S_set = set()
        
        # Optimization: Pre-compute nothing, just iterate.
        for x in order:
            if x in S_set:
                continue
            
            # Check if adding x creates a violation
            # Violation if:
            # 1. There exists a in S such that a + x in S
            # 2. There exists b in S such that x - b in S  (which is b + (x-b) = x, so x-b, b, x are in S)
            #    Note: If a=b, then 2a=x.
            # 3. 2x in S (which is x+x=2x, so x, x, 2x are in S)
            
            # The check "a+x in S" covers case 1.
            # The check "x-a in S" covers case 2 (where b=a).
            # The check "2x in S" covers case 3.
            
            conflict = False
            # We can potentially break early
            for a in S:
                if (a + x) in S_set:
                    conflict = True
                    break
                if (x - a) in S_set:
                    conflict = True
                    break
            
            if not conflict:
                if (2 * x) in S_set:
                    conflict = True
            
            if not conflict:
                S.append(x)
                S_set.add(x)
                
        return S

    def solve_instance(elements, seed_offset):
        rng = random.Random(seed_offset)
        n = len(elements)
        
        # Generate diverse orders
        orders = []
        
        # 1. Sorted Ascending
        orders.append(sorted(elements))
        # 2. Sorted Descending
        orders.append(sorted(elements, reverse=True))
        # 3. Sorted by Absolute Value
        orders.append(sorted(elements, key=abs))
        # 4. Sorted by Absolute Value Descending
        orders.append(sorted(elements, key=abs, reverse=True))
        
        # 5. Odds first, then evens (randomized within groups)
        odds = [x for x in elements if x % 2 != 0]
        evens = [x for x in elements if x % 2 == 0]
        rng.shuffle(odds)
        rng.shuffle(evens)
        orders.append(odds + evens)
        
        # 6. Positives first, then negatives
        pos = [x for x in elements if x > 0]
        neg = [x for x in elements if x < 0]
        rng.shuffle(pos)
        rng.shuffle(neg)
        orders.append(pos + neg)
        
        best_S = []
        best_size = 0
        
        # Run fixed orders first
        for order in orders:
            S = greedy_sum_free(elements, order)
            if len(S) > best_size:
                best_size = len(S)
                best_S = S
        
        # Run randomized orders
        # Estimate time per run to determine budget
        # For n=240, a run is fast. Let's just run until deadline.
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > DEADLINE:
                break
            
            # Create a random permutation
            order = elements[:]
            rng.shuffle(order)
            
            S = greedy_sum_free(elements, order)
            if len(S) > best_size:
                best_size = len(S)
                best_S = S
            
            # Occasionally try a "biased" random order
            # e.g., sort by random weight
            if rng.random() < 0.1:
                weights = [rng.random() for _ in elements]
                order = [x for _, x in sorted(zip(weights, elements))]
                S = greedy_sum_free(elements, order)
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = S
        
        return best_S

    # Generate instances
    instances = []
    for size, seed in [(60, 801), (120, 802), (240, 803)]:
        instances.append(instance(size, seed))
    
    results = []
    for i, elems in enumerate(instances):
        if time.time() - start_time > DEADLINE:
            # Fallback: odd numbers
            results.append([x for x in elems if x % 2 != 0])
            continue
        
        result = solve_instance(elems, seed_offset=801 + i)
        results.append(result)
        
        # Check time again
        if time.time() - start_time > DEADLINE:
             # Ensure we return something valid for remaining instances if we run out of time
             # though the loop above handles the break, we need to ensure results list is full
             pass

    # If we ran out of time inside the loop, some results might be missing
    # The loop above breaks out of the inner while, but the outer for continues?
    # No, `break` only exits the `while` loop in `solve_instance`.
    # `solve_instance` returns `best_S`.
    # The outer `for` loop continues.
    # We should check time in the outer loop too.
    
    # Let's refine the outer loop to handle time strictly
    results = []
    for i, elems in enumerate(instances):
        if time.time() - start_time > DEADLINE:
            results.append([x for x in elems if x % 2 != 0])
            continue
        result = solve_instance(elems, seed_offset=801 + i)
        results.append(result)

    return results
# EVOLVE-BLOCK-END
