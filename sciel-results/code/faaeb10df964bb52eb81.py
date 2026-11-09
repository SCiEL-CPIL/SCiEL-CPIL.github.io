import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 58  # 2s margin for 60s limit
    
    max_val = 4096
    max_sum = 3 * max_val
    
    best_set = []
    best_size = 0
    
    # Helper to generate all sums for a set
    def get_all_sums(S):
        sums = set()
        n = len(S)
        S_list = S
        for i in range(n):
            x = S_list[i]
            sums.add(3 * x)
            for j in range(i + 1, n):
                y = S_list[j]
                sums.add(2 * x + y)
                sums.add(x + 2 * y)
                for k in range(j + 1, n):
                    z = S_list[k]
                    sums.add(x + y + z)
        return sums

    # Helper to get sums involving a specific element x in set S
    # Assumes S is a list and x is in S
    def get_sums_involving(x, S):
        sums = []
        n = len(S)
        # 3x
        sums.append(3 * x)
        
        # 2x + y, x + 2y
        for y in S:
            if y != x:
                sums.append(2 * x + y)
                sums.append(x + 2 * y)
                
        # x + y + z
        # We need y < z, y != x, z != x
        # It's easier to iterate over pairs from S
        for i in range(n):
            y = S[i]
            if y == x: continue
            for j in range(i + 1, n):
                z = S[j]
                if z == x: continue
                if y < z: # Ensure we don't double count? 
                    # Actually, the sum x+y+z is the same regardless of order.
                    # The set S contains distinct elements.
                    # The triple is {x, y, z}.
                    # We just need to generate x+y+z once for each pair {y,z} from S\{x}.
                    # So iterating over pairs from S\{x} is correct.
                    sums.append(x + y + z)
        return sums

    def try_add(x, S, sum_set):
        """
        Check if x can be added to S.
        Returns True if valid, and updates S and sum_set in place if valid.
        """
        new_sums = get_sums_involving(x, S + [x]) # Wait, S doesn't have x yet.
        # get_sums_involving(x, S) generates sums involving x and elements of S.
        # But it also generates 3x.
        # Let's just generate them directly here for clarity and speed.
        
        potential_sums = []
        potential_sums.append(3 * x)
        
        for y in S:
            potential_sums.append(2 * x + y)
            potential_sums.append(x + 2 * y)
            
        n = len(S)
        for i in range(n):
            y = S[i]
            for j in range(i + 1, n):
                z = S[j]
                potential_sums.append(x + y + z)
                
        # Check for collisions with existing sums
        for s in potential_sums:
            if s in sum_set:
                return False
                
        # Check for internal collisions
        if len(set(potential_sums)) != len(potential_sums):
            return False
            
        # If valid, update
        sum_set.update(potential_sums)
        S.append(x)
        return True

    def try_remove(x, S, sum_set):
        """
        Remove x from S and update sum_set.
        """
        sums_to_remove = get_sums_involving(x, S)
        for s in sums_to_remove:
            sum_set.discard(s)
        S.remove(x)
        return True

    # 1. Greedy Initialization
    # Try several random greedy starts
    for attempt in range(20):
        if time.time() > deadline:
            break
            
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        
        current_set = []
        sum_set = set()
        
        for x in candidates:
            if time.time() > deadline:
                break
            if try_add(x, current_set, sum_set):
                pass
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    # 2. Simulated Annealing
    if not best_set:
        return []
        
    current_set = best_set[:]
    current_sum_set = get_all_sums(current_set)
    
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.995
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
            
        move_type = random.random()
        
        if move_type < 0.6:
            # Try to add
            x = random.randint(1, max_val)
            if x in current_set:
                continue
            # Save state? No, try_add modifies in place. 
            # We need to rollback if it fails? try_add returns False if invalid, 
            # but does it modify? 
            # In my implementation above, try_add only modifies if valid. 
            # So if it returns False, state is unchanged.
            if try_add(x, current_set, current_sum_set):
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                    # Reset temperature?
                    T = max(T, 1.0)
            else:
                # Try swap
                if current_set:
                    y = random.choice(current_set)
                    # Try to swap y with x
                    # 1. Remove y
                    # We need to check if x is valid in S\{y}
                    # To do this efficiently, we can:
                    # a. Remove y from sum_set
                    # b. Check x against new sum_set
                    # c. If valid, add x (which updates sum_set)
                    # d. If invalid, we need to restore y.
                    
                    # Let's implement a specific swap check
                    # Temporarily remove y
                    sums_y = get_sums_involving(y, current_set)
                    for s in sums_y:
                        current_sum_set.discard(s)
                    current_set.remove(y)
                    
                    # Now try to add x
                    if try_add(x, current_set, current_sum_set):
                        # Success, x is in, y is out.
                        # Size is same.
                        # We accept this move to explore.
                        # Maybe update best if we are lucky? No, size is same.
                        pass
                    else:
                        # Failed. Restore y.
                        # x was NOT added.
                        # We need to add y back.
                        # But y's sums were removed.
                        # We can just re-add y using try_add? 
                        # try_add(y, current_set, current_sum_set)
                        # This will re-calculate sums for y and add them.
                        if not try_add(y, current_set, current_sum_set):
                            # This should never happen if the original set was valid
                            # and x didn't interfere. But if x failed, it means x collided 
                            # with something in S\{y}. 
                            # So y should still be valid in S\{y}.
                            pass

        elif move_type < 0.9:
            # Try to remove (to escape local minima)
            if current_set:
                y = random.choice(current_set)
                # Accept with probability based on T
                # Since size decreases, we accept with low prob unless T is high
                if random.random() < 0.05 * T:
                    try_remove(y, current_set, current_sum_set)
                    
        else:
            # Random restart? No, too expensive.
            # Just cool down
            pass

        T *= cooling_rate
        
    return best_set
