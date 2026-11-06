"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 115.0
    best_set = None
    best_size = 0
    
    def validate_and_size(arr):
        """Check B2[2] property and return size if valid, else 0."""
        n = len(arr)
        if n == 0:
            return 0
        # Use a dictionary or array to count sums
        # Max sum is 2000. Min sum is 2.
        counts = [0] * 2001
        valid = True
        for i in range(n):
            ai = arr[i]
            for j in range(i, n):
                s = ai + arr[j]
                counts[s] += 1
                if counts[s] > 2:
                    return 0
        return n

    def greedy_start(start=1, step=1):
        chosen = []
        counts = {}
        for x in range(start, 1001, step):
            new_sums = [x + y for y in chosen]
            new_sums.append(2 * x)
            # Check if any new sum exceeds 2
            ok = True
            for s in new_sums:
                if counts.get(s, 0) + 1 > 2:
                    ok = False
                    break
            if ok:
                chosen.append(x)
                for s in new_sums:
                    counts[s] = counts.get(s, 0) + 1
        return chosen

    # 1. Try simple greedy
    g1 = greedy_start(1, 1)
    if validate_and_size(g1) > best_size:
        best_set = g1
        best_size = len(g1)
        
    # 2. Try randomized greedy with different seeds
    # The greedy choice is deterministic given the set. 
    # To improve, we can try "local search" or "simulated annealing" style swaps.
    # Or simply try different starting points or skipping some numbers early on.
    
    # Let's try a more robust local search:
    # Start with a valid set, then try to add more numbers or swap numbers to allow denser packing.
    
    # A common improvement for B2 sequences is to use a greedy algorithm that looks ahead or
    # uses a different ordering.
    
    # Let's try a heuristic: 
    # 1. Generate a base set using standard greedy.
    # 2. Try to insert missing numbers from 1..1000 into the set, checking validity.
    # 3. If insertion fails, try to remove a conflicting element and insert the new one (swap).
    
    def try_improve(initial_set):
        current_set = set(initial_set)
        current_arr = sorted(current_set)
        current_size = len(current_arr)
        
        # Precompute counts for current set
        counts = {}
        for i in range(len(current_arr)):
            for j in range(i, len(current_arr)):
                s = current_arr[i] + current_arr[j]
                counts[s] = counts.get(s, 0) + 1
        
        improved = False
        # Try adding numbers not in the set
        candidates = [x for x in range(1, 1001) if x not in current_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            # Check if x can be added
            new_sums = [x + y for y in current_arr] + [2 * x]
            can_add = True
            for s in new_sums:
                if counts.get(s, 0) + 1 > 2:
                    can_add = False
                    break
            
            if can_add:
                # Add x
                current_set.add(x)
                current_arr.append(x)
                current_arr.sort()
                for s in new_sums:
                    counts[s] = counts.get(s, 0) + 1
                current_size += 1
                improved = True
            else:
                # Try to swap: remove one element y from current_set such that x can be added
                # This is expensive. Let's just try adding for now.
                # If we have time, we can implement swaps.
                pass
        
        if improved:
            # Re-validate to be safe
            sz = validate_and_size(current_arr)
            if sz > 0:
                return current_arr, sz
        return current_arr, current_size

    # Run improvement from the greedy set
    if best_set:
        improved_set, improved_size = try_improve(best_set)
        if improved_size > best_size:
            best_set = improved_set
            best_size = improved_size

    # Try different random starting points for greedy
    # e.g., start with a random subset of small numbers
    for _ in range(20):
        if time.time() > deadline:
            break
        # Random subset of 1..50 as seed
        seed_size = random.randint(5, 20)
        seed = random.sample(range(1, 51), seed_size)
        seed.sort()
        
        # Greedy completion from this seed
        chosen = list(seed)
        counts = {}
        for i in range(len(chosen)):
            for j in range(i, len(chosen)):
                s = chosen[i] + chosen[j]
                counts[s] = counts.get(s, 0) + 1
        
        # Check if seed is valid
        if any(c > 2 for c in counts.values()):
            continue
            
        # Complete greedily
        for x in range(1, 1001):
            if x in chosen:
                continue
            new_sums = [x + y for y in chosen] + [2 * x]
            ok = True
            for s in new_sums:
                if counts.get(s, 0) + 1 > 2:
                    ok = False
                    break
            if ok:
                chosen.append(x)
                for s in new_sums:
                    counts[s] = counts.get(s, 0) + 1
        
        sz = validate_and_size(chosen)
        if sz > best_size:
            best_set = chosen
            best_size = sz
            
        # Try improvement on this new set
        if sz > 0:
            improved_set, improved_size = try_improve(chosen)
            if improved_size > best_size:
                best_set = improved_set
                best_size = improved_size

    if best_set is None:
        return [1]
    return best_set

# EVOLVE-BLOCK-END
