import time
import random
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    # Numbers of interest: 2 to 256
    N = 256
    numbers = list(range(2, N + 1))
    num_to_idx = {x: i for i, x in enumerate(numbers)}
    n = len(numbers)
    
    # Precompute conflicts
    # conflict[i] is a set of indices j such that lcm(numbers[i], numbers[j]) <= 256
    conflicts = [set() for _ in range(n)]
    for i in range(n):
        a = numbers[i]
        for j in range(i + 1, n):
            b = numbers[j]
            if lcm(a, b) <= 256:
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    def is_compatible(idx, current_set):
        """Check if numbers[idx] is compatible with all indices in current_set."""
        for s in current_set:
            if s in conflicts[idx]:
                return False
        return True
        
    def get_score(current_set):
        return sum(1.0 / numbers[i] for i in current_set)
        
    def improve(initial_set):
        """Local search to improve the set."""
        current_set = set(initial_set)
        current_score = get_score(current_set)
        
        # Pre-compute weights for sorting
        # We want to try adding small numbers first, or swapping to smaller numbers
        
        improved = True
        while improved and (time.time() < deadline):
            improved = False
            
            # 1. Try to add a number
            # Sort candidates not in set by value (ascending)
            candidates = [i for i in range(n) if i not in current_set]
            candidates.sort(key=lambda i: numbers[i])
            
            added = False
            for idx in candidates:
                if is_compatible(idx, current_set):
                    current_set.add(idx)
                    current_score += 1.0 / numbers[idx]
                    improved = True
                    added = True
                    break # Restart loop after addition
            
            if added:
                continue
                
            # 2. Try to swap
            # For each element in current set, try to replace it with a better compatible one
            # Sort current set by value? Or just iterate.
            # We want to replace x with y if 1/y > 1/x (i.e., y < x)
            # So we should look for y < x.
            
            swapped = False
            # Iterate over a copy of the set to avoid modification during iteration
            for x_idx in list(current_set):
                x_val = numbers[x_idx]
                # Candidates for replacement: not in current set, value < x_val
                # Also must be compatible with current_set - {x_idx}
                
                # Optimization: Only consider y < x_val
                # We can iterate y from 2 to x_val-1
                # But we need to check if y is in current_set (skip)
                
                # Let's iterate through all potential y < x_val not in current_set
                # This is O(N) for each x, so O(N^2) per pass.
                # N=255, N^2 ~ 65k. 65k checks is fast.
                
                best_y = -1
                best_y_score = 0.0
                
                # To make it faster, we can just find ANY y that improves score.
                # Since we want to maximize sum, any y < x is an improvement.
                # We just need one valid y.
                
                # Iterate y from 2 to x_val - 1
                for y_val in range(2, x_val):
                    if y_val not in num_to_idx:
                        continue
                    y_idx = num_to_idx[y_val]
                    if y_idx in current_set:
                        continue
                    
                    # Check compatibility with current_set - {x_idx}
                    # We can check is_compatible(y_idx, current_set) but we must ignore x_idx
                    # is_compatible checks against the set passed.
                    # So we need a check that ignores x_idx.
                    
                    compatible = True
                    for s in current_set:
                        if s == x_idx:
                            continue
                        if s in conflicts[y_idx]:
                            compatible = False
                            break
                            
                    if compatible:
                        # Swap
                        current_set.remove(x_idx)
                        current_set.add(y_idx)
                        current_score += (1.0 / y_val) - (1.0 / x_val)
                        improved = True
                        swapped = True
                        break # Restart outer loops
                
                if swapped:
                    break # Restart while loop
            
        return current_set, current_score

    # Generate initial solutions
    initial_sets = []
    
    # 1. Seed solution (Greedy from 2)
    seed_set = []
    for x in range(2, 257):
        idx = num_to_idx[x]
        if all(idx not in conflicts[seed_idx] for seed_idx in seed_set):
            seed_set.append(idx)
    initial_sets.append(seed_set)
    
    # 2. Greedy from 3
    g3 = []
    for x in range(3, 257):
        idx = num_to_idx[x]
        if all(idx not in conflicts[seed_idx] for seed_idx in g3):
            g3.append(idx)
    initial_sets.append(g3)
    
    # 3. Greedy from 4
    g4 = []
    for x in range(4, 257):
        idx = num_to_idx[x]
        if all(idx not in conflicts[seed_idx] for seed_idx in g4):
            g4.append(idx)
    initial_sets.append(g4)
            
    # 4. Random restarts
    random.seed(42)
    for _ in range(10):
        rand_set = []
        # Pick random small numbers first to get high weight, then fill
        # Or just pick random numbers and check
        # Let's pick a random subset of small numbers
        smalls = list(range(2, 50))
        random.shuffle(smalls)
        for x in smalls:
            idx = num_to_idx[x]
            if all(idx not in conflicts[seed_idx] for seed_idx in rand_set):
                rand_set.append(idx)
        # Then fill with remaining numbers greedily
        for x in range(2, 257):
            if x in [numbers[i] for i in rand_set]:
                continue
            idx = num_to_idx[x]
            if all(idx not in conflicts[seed_idx] for seed_idx in rand_set):
                rand_set.append(idx)
        initial_sets.append(rand_set)
        
    best_set = None
    best_score = -1.0
    
    for s in initial_sets:
        if time.time() > deadline:
            break
        improved_set, score = improve(s)
        if score > best_score:
            best_score = score
            best_set = improved_set
            
    if best_set is None:
        return []
        
    # Convert indices back to values
    return [numbers[i] for i in best_set]

if __name__ == "__main__":
    # Test
    res = construct()
    print(len(res), sum(1/x for x in res))
