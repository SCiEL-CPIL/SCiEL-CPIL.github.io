import time
import random

N = 512
DEADLINE = 55.0

def construct():
    start_time = time.time()
    best_set = set()
    best_size = 0
    
    # Helper to check validity using a boolean list for speed
    def is_valid_add(in_set, y):
        # y is 1-indexed
        # Case 1: y is first term: y, y+d, y+2d, y+3d
        max_d = (N - y) // 3
        for d in range(1, max_d + 1):
            if in_set[y + d] and in_set[y + 2*d] and in_set[y + 3*d]:
                return False
        
        # Case 2: y is second term: y-d, y, y+d, y+2d
        max_d = min(y - 1, (N - y) // 2)
        for d in range(1, max_d + 1):
            if in_set[y - d] and in_set[y + d] and in_set[y + 2*d]:
                return False
        
        # Case 3: y is third term: y-2d, y-d, y, y+d
        max_d = min((y - 1) // 2, N - y)
        for d in range(1, max_d + 1):
            if in_set[y - 2*d] and in_set[y - d] and in_set[y + d]:
                return False
        
        # Case 4: y is fourth term: y-3d, y-2d, y-d, y
        max_d = (y - 1) // 3
        for d in range(1, max_d + 1):
            if in_set[y - 3*d] and in_set[y - 2*d] and in_set[y - d]:
                return False
                
        return True

    numbers = list(range(1, N + 1))
    
    while time.time() - start_time < DEADLINE:
        # Greedy Initialization
        random.shuffle(numbers)
        in_set = [False] * (N + 1)
        current_set = set()
        
        for x in numbers:
            if is_valid_add(in_set, x):
                in_set[x] = True
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
        
        # Local Search Phase
        # We work on a copy of the current state
        # To save time, we use the in_set list directly if it's the best one?
        # No, better to keep separate state for local search to not corrupt best_set
        # But converting set to list is O(N).
        
        # Let's just use the set for logic, but use a local in_set list for checks
        # We need to sync in_set with current_set
        local_in_set = [False] * (N + 1)
        for x in current_set:
            local_in_set[x] = True
        
        working_set = current_set
        stagnant_swaps = 0
        max_swaps_per_iteration = 50 # Prevent infinite loops
        
        while time.time() - start_time < DEADLINE:
            # 1. Try to add any element
            added = False
            # To make it faster, we can pick a random subset of candidates or just iterate
            # Iterating all 512 is ~512 * 100 ops = 50k ops. Fast.
            for y in range(1, N + 1):
                if not local_in_set[y]:
                    if is_valid_add(local_in_set, y):
                        # Add y
                        local_in_set[y] = True
                        working_set.add(y)
                        added = True
                        stagnant_swaps = 0
                        if len(working_set) > best_size:
                            best_size = len(working_set)
                            best_set = working_set.copy()
                        break # Restart loop to find more adds
            
            if added:
                continue
            
            # 2. If no adds, try swaps
            if stagnant_swaps >= max_swaps_per_iteration:
                break
            
            # Pick random x in working_set
            x = random.choice(list(working_set))
            # Pick random y not in working_set
            # Optimization: Precompute list of not-in-set? 
            # Or just random pick and check
            y = random.randint(1, N)
            if local_in_set[y]:
                # Retry a few times to find a y not in set
                for _ in range(10):
                    y = random.randint(1, N)
                    if not local_in_set[y]:
                        break
                else:
                    # If we can't find a y not in set, set is full? Unlikely
                    break
            
            # Check if swap x -> y is valid
            # Remove x
            local_in_set[x] = False
            if is_valid_add(local_in_set, y):
                # Swap is valid
                local_in_set[y] = True
                working_set.remove(x)
                working_set.add(y)
                stagnant_swaps += 1
            else:
                # Restore x
                local_in_set[x] = True
                stagnant_swaps += 1

    return sorted(best_set)
