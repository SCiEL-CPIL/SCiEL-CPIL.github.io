# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 4096
    # We want to find a set S of distinct positive integers <= 4096
    # such that all unordered triples (with repetition) have unique sums.
    # This is a B_3[1] sequence (Sidon set of order 3).
    
    # The greedy approach in the seed picks small numbers sequentially.
    # A better strategy often involves using larger gaps or specific constructions.
    # However, a randomized greedy or backtracking with heuristics can be better.
    
    # Let's try a randomized greedy approach with multiple restarts.
    # We'll pick elements in a random order (or with some bias) and check validity.
    # To speed up checking, we can maintain a set of existing triple sums.
    
    # Number of triples for k elements is C(k+2, 3) = (k+2)(k+1)k/6.
    # For k=50, that's 22100 triples. Checking each new element involves
    # checking C(k+2, 3) - C(k+1, 3) = (k+1)k/2 new triples.
    # For k=50, that's 1275 new sums to check.
    # With 4096 candidates, a single greedy pass is fast.
    # We can do many random restarts.
    
    best_set = []
    
    # Pre-compute nothing, just use sets.
    
    # Strategy 1: Randomized Greedy
    # Try several random orderings of [1..4096] and greedily add.
    # Or, better: try to add elements from a random subset.
    
    # Actually, a known good construction for B_h sequences uses powers of h.
    # For B_3, powers of 3: 1, 3, 9, 27, ... 3^7 = 2187, 3^8 = 6561 > 4096.
    # So {1, 3, 9, 27, 81, 243, 729, 2187} is a valid B_3 set of size 8.
    # This is very small. We want to maximize cardinality.
    
    # The greedy method in the seed found a set of size 512? No, the seed stops when it can't add more.
    # Let's check the size of the seed's output.
    # The seed iterates x from 1 to 512. It might return up to 512 elements if they all fit.
    # But triple sums grow fast. Max sum is 3*4096 = 12288.
    # Number of possible sums is 12288.
    # Number of triples for k elements is ~k^3/6.
    # k^3/6 <= 12288 => k^3 <= 73728 => k <= 42.
    # So the maximum size is bounded by ~42-50.
    
    # Let's verify the upper bound.
    # Sum range: min 3*1=3, max 3*4096=12288.
    # Number of distinct sums = 12286.
    # For a B_3 set of size k, the number of distinct triple sums is exactly C(k+2, 3).
    # C(k+2, 3) = (k+2)(k+1)k/6.
    # We need (k+2)(k+1)k/6 <= 12286.
    # k=40: 42*41*40/6 = 11480.
    # k=41: 43*42*41/6 = 12341 > 12286.
    # So max k is 40.
    
    # The seed program likely returns a set of size ~40.
    # Let's try to find a set of size 40.
    
    # A simple greedy approach that picks the smallest available number works well.
    # But maybe we can do better with randomization or backtracking.
    
    # Let's implement a fast greedy with a set of sums.
    
    def greedy_build(order):
        selected = []
        sums = set()
        for x in order:
            # Check if adding x creates any duplicate triple sums
            # New triples involving x:
            # (x, x, x), (x, x, y), (x, y, y), (x, y, z) for y, z in selected
            # Actually, we just need to check all triples in the new set against existing sums.
            # But we only need to check the NEW triples formed by x.
            
            # Triples with x:
            # (x, x, x)
            # (x, x, y) for y in selected
            # (x, y, y) for y in selected
            # (x, y, z) for y < z in selected
            
            # Let's generate these and check against sums.
            # If any is in sums, x cannot be added.
            
            # To be efficient, we can check:
            # 1. 3*x not in sums
            # 2. 2*x + y not in sums for all y in selected
            # 3. x + 2*y not in sums for all y in selected
            # 4. x + y + z not in sums for all y < z in selected
            
            # This is O(k^2) per candidate.
            # With k <= 40, this is fast.
            
            valid = True
            new_sums = set()
            
            # (x, x, x)
            s = 3*x
            if s in sums:
                valid = False
            else:
                new_sums.add(s)
            
            if valid:
                for y in selected:
                    s = 2*x + y
                    if s in sums or s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
                    s = x + 2*y
                    if s in sums or s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
            
            if valid:
                for i in range(len(selected)):
                    y = selected[i]
                    for j in range(i+1, len(selected)):
                        z = selected[j]
                        s = x + y + z
                        if s in sums or s in new_sums:
                            valid = False
                            break
                        new_sums.add(s)
                    if not valid:
                        break
            
            if valid:
                selected.append(x)
                sums.update(new_sums)
        
        return selected
    
    # Try the standard greedy (smallest first)
    standard_order = list(range(1, n+1))
    best_set = greedy_build(standard_order)
    best_size = len(best_set)
    
    # Try randomized orders
    np.random.seed(42)
    for i in range(1000):
        if time.time() > deadline - 5:
            break
        
        # Random permutation is expensive.
        # Instead, let's try a "randomized greedy" where we pick from a random subset of candidates.
        # Or, shuffle the order.
        
        # Shuffling 4096 elements is fast.
        order = np.random.permutation(n) + 1
        current_set = greedy_build(order)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
    
    # Try a local search: start with a good set, try to swap elements.
    # This is complex. Let's stick to randomized greedy.
    
    # Another idea: use a specific construction.
    # For B_3, one can use polynomials or other structures.
    # But randomized greedy often finds near-optimal solutions for small k.
    
    # Let's also try a greedy that picks from a random subset of size 100, then adds the rest greedily.
    for i in range(200):
        if time.time() > deadline - 5:
            break
        
        # Pick 50 random numbers
        random_subset = np.random.choice(n, 50, replace=False)
        # Sort them
        random_subset = sorted(random_subset)
        # Greedily add from this subset, then from the rest
        order = list(random_subset) + [x for x in range(1, n+1) if x not in random_subset]
        current_set = greedy_build(order)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
    
    return best_set

# EVOLVE-BLOCK-END
