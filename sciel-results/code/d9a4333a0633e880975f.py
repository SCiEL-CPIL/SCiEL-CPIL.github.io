import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    prescribed = {1, 2, 5, 11}
    max_val = 284
    
    def init_counts(s):
        counts = {}
        s_list = sorted(s)
        for i in range(len(s_list)):
            for j in range(i, len(s_list)):
                val = s_list[i] + s_list[j]
                counts[val] = counts.get(val, 0) + 1
        return counts

    current_set = list(prescribed)
    current_set.sort()
    sum_count = init_counts(current_set)
    
    # Greedy Add
    for x in range(1, max_val + 1):
        if x in current_set:
            continue
        # Check if x can be added
        valid = True
        if sum_count.get(2 * x, 0) > 0:
            valid = False
        else:
            for y in current_set:
                if sum_count.get(x + y, 0) > 0:
                    valid = False
                    break
        if valid:
            current_set.append(x)
            current_set.sort()
            # Update counts
            sum_count[2 * x] = sum_count.get(2 * x, 0) + 1
            for y in current_set:
                if y != x: # x is now in set, but we added x+x above. 
                           # Wait, current_set includes x now.
                           # We need to add x+y for y in OLD set.
                           # My loop iterates over NEW set.
                           # This is messy. Let's just re-init counts in greedy phase.
                           # It's only 20 elements.
                           pass
            # Simpler: re-init
            sum_count = init_counts(current_set)
            
    best_set = current_set[:]
    best_size = len(best_set)
    
    # Local Search
    # We will use the count-based check
    
    while time.time() < deadline:
        if len(current_set) >= 20:
            break
            
        added = False
        # Try to add
        for _ in range(100):
            x = random.randint(1, max_val)
            if x in current_set:
                continue
            # Check
            if sum_count.get(2 * x, 0) > 0:
                continue
            conflict = False
            for y in current_set:
                if sum_count.get(x + y, 0) > 0:
                    conflict = True
                    break
            if not conflict:
                # Add
                current_set.append(x)
                current_set.sort()
                # Update counts
                sum_count[2 * x] = sum_count.get(2 * x, 0) + 1
                for y in current_set:
                    if y != x:
                        s = x + y
                        sum_count[s] = sum_count.get(s, 0) + 1
                added = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                break
        
        if added:
            continue
            
        # Try swap
        swapped = False
        for _ in range(100):
            if len(current_set) == 0: break
            idx = random.randint(0, len(current_set) - 1)
            x = current_set[idx]
            y = random.randint(1, max_val)
            if y == x or y in current_set:
                continue
            
            # 1. Remove x from counts
            sum_count[2 * x] -= 1
            if sum_count[2 * x] == 0:
                del sum_count[2 * x]
            
            for u in current_set:
                if u != x:
                    s = x + u
                    sum_count[s] -= 1
                    if sum_count[s] == 0:
                        del sum_count[s]
            
            # 2. Check if y can be added
            valid = True
            if sum_count.get(2 * y, 0) > 0:
                valid = False
            else:
                for u in current_set:
                    if u != x: # x is removed from set logic, but still in list?
                               # current_set still contains x.
                               # We must iterate over current_set and skip x.
                        if sum_count.get(y + u, 0) > 0:
                            valid = False
                            break
            
            if valid:
                # 3. Add y to counts
                sum_count[2 * y] = sum_count.get(2 * y, 0) + 1
                for u in current_set:
                    if u != x:
                        s = y + u
                        sum_count[s] = sum_count.get(s, 0) + 1
                
                # 4. Update set
                current_set[idx] = y
                current_set.sort()
                swapped = True
                break
            else:
                # Revert counts for x
                sum_count[2 * x] = sum_count.get(2 * x, 0) + 1
                for u in current_set:
                    if u != x:
                        s = x + u
                        sum_count[s] = sum_count.get(s, 0) + 1

    return best_set
