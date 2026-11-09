import numpy as np
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit, leaving 10s margin
    
    max_val = 4096
    candidates = list(range(1, max_val + 1))
    
    best_set = []
    best_size = 0
    
    # Use a numpy array for sum lookup for speed
    # Max sum is 3 * 4096 = 12288
    sum_size = 3 * max_val + 10
    sum_used = np.zeros(sum_size, dtype=bool)
    
    def can_add(x, current_set, sum_used):
        # Generate new sums
        # 3x
        if sum_used[3 * x]:
            return False
        
        # 2x + y
        # x + 2y
        # x + y + z
        
        # To check internal collisions, we can just collect them in a small list/set
        # Since current_set size is small (< 50), this is fast.
        
        new_sums = [3 * x]
        
        for y in current_set:
            s1 = 2 * x + y
            if sum_used[s1]:
                return False
            new_sums.append(s1)
            
            s2 = x + 2 * y
            if sum_used[s2]:
                return False
            new_sums.append(s2)
            
        for i in range(len(current_set)):
            for j in range(i + 1, len(current_set)):
                s3 = x + current_set[i] + current_set[j]
                if sum_used[s3]:
                    return False
                new_sums.append(s3)
                
        # Check internal collisions in new_sums
        # Convert to set to check uniqueness
        if len(set(new_sums)) != len(new_sums):
            return False
            
        return True

    def add_to_set(x, current_set, sum_used):
        sum_used[3 * x] = True
        for y in current_set:
            sum_used[2 * x + y] = True
            sum_used[x + 2 * y] = True
        for i in range(len(current_set)):
            for j in range(i + 1, len(current_set)):
                sum_used[x + current_set[i] + current_set[j]] = True
        current_set.append(x)

    def remove_from_set(x, current_set, sum_used):
        # Find index of x
        idx = current_set.index(x)
        current_set.pop(idx)
        
        sum_used[3 * x] = False
        for y in current_set:
            sum_used[2 * x + y] = False
            sum_used[x + 2 * y] = False
        for i in range(len(current_set)):
            for j in range(i + 1, len(current_set)):
                sum_used[x + current_set[i] + current_set[j]] = False

    # Run multiple trials
    trial = 0
    while time.time() < deadline:
        random.shuffle(candidates)
        current_set = []
        # Reset sum_used
        sum_used[:] = False
        
        # Greedy Phase
        for x in candidates:
            if can_add(x, current_set, sum_used):
                add_to_set(x, current_set, sum_used)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            if best_size >= 42:
                break

        # Local Search Phase (1-out, 2-in)
        # Only if we have time
        if time.time() > deadline - 5:
            continue
            
        # Candidates that were not added
        # We can just iterate through candidates again and see which ones fail
        # But we need to try removing one and adding two.
        
        # To speed up, we can pick a random element to remove and try to add two random candidates.
        # This is a heuristic.
        
        # Let's try a few iterations of local search
        for _ in range(10): # Limit local search steps per trial
            if time.time() > deadline - 2:
                break
            if not current_set:
                break
                
            # Pick a random element to remove
            idx_to_remove = random.randint(0, len(current_set) - 1)
            x_remove = current_set[idx_to_remove]
            
            # Temporarily remove
            remove_from_set(x_remove, current_set, sum_used)
            
            # Try to add two new elements
            added_count = 0
            added_elements = []
            
            # Shuffle candidates again for randomness in selection
            # But we need to be careful not to re-add x_remove
            random.shuffle(candidates)
            
            for x_add in candidates:
                if x_add == x_remove:
                    continue
                if can_add(x_add, current_set, sum_used):
                    add_to_set(x_add, current_set, sum_used)
                    added_count += 1
                    added_elements.append(x_add)
                    if added_count == 2:
                        break
                        
            if added_count == 2:
                # Accept the move
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
            else:
                # Revert: remove the added elements and re-add x_remove
                for x_add in added_elements:
                    remove_from_set(x_add, current_set, sum_used)
                add_to_set(x_remove, current_set, sum_used)
                
        trial += 1
        
    return best_set
