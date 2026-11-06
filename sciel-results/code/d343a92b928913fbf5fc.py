import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    best_set = []
    best_size = 0
    
    max_n = 1000
    sum_size = 2 * max_n + 1
    
    def run_greedy(order):
        chosen = []
        counts = np.zeros(sum_size, dtype=np.int8)
        
        for c in order:
            valid = True
            
            # Check 2*c
            if counts[2 * c] >= 2:
                valid = False
            
            if valid:
                # Check c + x for x in chosen
                # Vectorized check: counts[c + chosen]
                if chosen:
                    sums = c + np.array(chosen)
                    if np.any(counts[sums] >= 2):
                        valid = False
            
            if valid:
                # Update counts
                counts[2 * c] += 1
                chosen.append(c)
                if chosen:
                    # Update counts for c + x
                    # Note: chosen now includes c, so we need c + (chosen without c)
                    # Or just c + np.array(chosen[:-1])
                    if len(chosen) > 1:
                        sums = c + np.array(chosen[:-1])
                        counts[sums] += 1
                
        return chosen

    # 1. Deterministic Ascending
    asc_order = list(range(1, max_n + 1))
    current_set = run_greedy(asc_order)
    best_set = current_set
    best_size = len(current_set)

    # 2. Random Restarts
    while time.time() - start_time < deadline - 10.0:  # Leave 10s for local search
        order = list(range(1, max_n + 1))
        random.shuffle(order)
        current_set = run_greedy(order)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set

    # 3. Local Search on Best Set
    if time.time() - start_time < deadline - 5.0:
        best_set = local_search(best_set, start_time, deadline - 5.0)
        best_size = len(best_set)

    return best_set

def local_search(A, start_time, deadline):
    A = set(A)
    max_n = 1000
    sum_size = 2 * max_n + 1
    counts = np.zeros(sum_size, dtype=np.int8)
    
    # Build initial counts
    A_list = list(A)
    for i, a in enumerate(A_list):
        counts[2*a] += 1
        for b in A_list[i+1:]:
            counts[a+b] += 1
        # Wait, the greedy logic adds c+x for x in chosen.
        # This counts each pair (c, x) once.
        # Since a <= b is required, and we iterate i, j with i<j, we cover a<=b.
        # But 2a is handled separately.
        # My greedy logic:
        # counts[2c] += 1
        # counts[c+x] += 1 for x in chosen (x < c? No, x in chosen, c is new).
        # If we add 1, then 2:
        # Add 1: counts[2]+=1.
        # Add 2: counts[4]+=1. sums = 2 + [1] = [3]. counts[3]+=1.
        # Correct.
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try Insert
        candidates = [x for x in range(1, max_n + 1) if x not in A]
        random.shuffle(candidates)
        
        for b in candidates:
            if time.time() > deadline: break
            # Check if b is valid
            if counts[2*b] >= 2: continue
            if A:
                sums = b + np.array(list(A))
                if np.any(counts[sums] >= 2): continue
            
            # Valid, insert
            A.add(b)
            counts[2*b] += 1
            if A:
                # Update counts for b + x
                # We need to add 1 to counts[b+x] for all x in A, x != b
                # Since b is in A now, we can do:
                other_A = [x for x in A if x != b]
                if other_A:
                    sums = b + np.array(other_A)
                    counts[sums] += 1
            improved = True
            break
        
        if improved:
            continue
            
        # Try Swap to enable Insert
        # This is expensive, so we limit it
        A_list = list(A)
        random.shuffle(A_list)
        
        for a in A_list:
            if time.time() > deadline: break
            # Try to remove a
            # We need to check if any b can replace a
            # And if that swap allows an insert
            
            # To speed up, we only try a few b's
            candidates = [x for x in range(1, max_n + 1) if x not in A]
            random.shuffle(candidates)
            candidates = candidates[:20] # Limit
            
            for b in candidates:
                if time.time() > deadline: break
                
                # Check if A - {a} + {b} is valid
                # Simulate counts
                # counts_minus_a = counts - contrib(a)
                # contrib(a) has 1 at 2a, and 1 at a+x for x in A, x != a
                
                # Check if b is valid in A - {a}
                # This is equivalent to:
                # 1. counts[2b] - (1 if 2b == 2a else 0) - (1 if 2b == a+x for some x) < 2
                # This is getting complex.
                
                # Simpler: Just check validity of b in A - {a} using the counts array
                # We can temporarily subtract a's contribution
                # But doing this for every b is slow.
                
                # Alternative: Just check if b is "almost" valid in A.
                # If b is valid in A, we would have inserted it in the first phase.
                # So b is invalid in A.
                # It is invalid because of some conflicts.
                # If removing a resolves all conflicts for b, then swap is valid.
                
                # Conflicts for b in A:
                # 1. counts[2b] >= 2
                # 2. counts[b+x] >= 2 for some x in A
                
                # If we remove a, counts decrease.
                # Decrease at 2a and a+x.
                
                # This is too slow to implement correctly in the time remaining.
                # I will skip the Swap-to-Insert for now and rely on the faster greedy + simple insert local search.
                pass

    return list(A)
