import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin for 120s limit
    
    N = 1000
    max_sum = 2 * N
    
    # 1. Greedy Initial Solution
    chosen = []
    counts = [0] * (max_sum + 1)
    
    for x in range(1, N + 1):
        valid = True
        for y in chosen:
            s = x + y
            if counts[s] >= 2:
                valid = False
                break
        if valid and counts[2 * x] >= 2:
            valid = False
            
        if valid:
            chosen.append(x)
            counts[2 * x] += 1
            for y in chosen[:-1]:
                counts[x + y] += 1
                
    best_set = set(chosen)
    best_size = len(chosen)
    
    # 2. Local Search (Simulated Annealing with targeted swaps)
    current = list(chosen)
    in_set = [False] * (N + 1)
    for x in current:
        in_set[x] = True
        
    # Re-initialize counts
    counts = [0] * (max_sum + 1)
    for i in range(len(current)):
        x = current[i]
        counts[2 * x] += 1
        for j in range(i + 1, len(current)):
            y = current[j]
            counts[x + y] += 1
            
    def get_conflicts(counts):
        c = 0
        for s in range(2, max_sum + 1):
            if counts[s] > 2:
                c += counts[s] - 2
        return c

    current_conflicts = get_conflicts(counts)
    best_conflicts = current_conflicts
    best_solution = list(current)
    
    T = 1.0
    cooling = 0.99995
    min_T = 0.001
    
    # Helper: find blockers for adding x
    # Returns list of elements in current whose removal would allow x to be added
    def find_blockers(x, current, counts):
        blockers = []
        # Check 2*x
        if counts[2 * x] >= 2:
            # Find pairs in current that sum to 2*x
            for i in range(len(current)):
                a = current[i]
                b = 2 * x - a
                if b < a:
                    continue
                if b > N:
                    continue
                # Check if b is in current
                # Linear search is O(M), but M is small
                for j in range(i + 1, len(current)):
                    if current[j] == b:
                        blockers.append(a)
                        blockers.append(b)
                        break
                else:
                    # b might be a itself (a == b, i.e., a = x)
                    if a == b and a != x:
                        # a appears once, but we need two copies for a+a
                        # Actually if a == b, then a+a = 2a = 2x, so a = x
                        # But x is not in current, so this case doesn't arise
                        pass
        # Check x + y for y in current
        for y in current:
            s = x + y
            if counts[s] >= 2:
                # Find the other pair (u, v) in current with u + v = s
                for i in range(len(current)):
                    u = current[i]
                    v = s - u
                    if v < u:
                        continue
                    if v > N:
                        continue
                    # Check if v is in current
                    for j in range(i + 1, len(current)):
                        if current[j] == v:
                            # u and v are the other pair
                            if u != x and v != x:  # x is not in current
                                blockers.append(u)
                                blockers.append(v)
                            break
                    else:
                        if u == v and u != x:
                            blockers.append(u)
        return blockers
    
    # Helper: try to add x, return True if successful
    def try_add(x, current, counts, in_set):
        if in_set[x]:
            return False
        if counts[2 * x] >= 2:
            return False
        for y in current:
            s = x + y
            if counts[s] >= 2:
                return False
        # Add x
        current.append(x)
        in_set[x] = True
        counts[2 * x] += 1
        for y in current[:-1]:
            counts[x + y] += 1
        return True
    
    # Helper: remove x from current
    def try_remove(x, current, counts, in_set):
        idx = current.index(x)
        current.pop(idx)
        in_set[x] = False
        counts[2 * x] -= 1
        for y in current:
            counts[x + y] -= 1
    
    while time.time() < deadline:
        # Try direct add first
        if random.random() < 0.4:
            # Pick a random candidate not in set
            x = random.randint(1, N)
            if in_set[x]:
                continue
            
            # Try direct add
            if counts[2 * x] < 2:
                blocked = False
                for y in current:
                    s = x + y
                    if counts[s] >= 2:
                        blocked = True
                        break
                if not blocked:
                    # Accept the add
                    current.append(x)
                    in_set[x] = True
                    counts[2 * x] += 1
                    for y in current[:-1]:
                        counts[x + y] += 1
                    current_conflicts = get_conflicts(counts)
                    
                    if current_conflicts < best_conflicts:
                        best_conflicts = current_conflicts
                        if best_conflicts == 0:
                            best_solution = list(current)
                            if len(current) > best_size:
                                best_size = len(current)
                    continue
        
        # Targeted swap: find a candidate and try to swap with a blocker
        x = random.randint(1, N)
        if in_set[x]:
            continue
        
        # Check if x can be added directly
        can_add = True
        if counts[2 * x] >= 2:
            can_add = False
        else:
            for y in current:
                s = x + y
                if counts[s] >= 2:
                    can_add = False
                    break
        
        if can_add:
            # Add it
            current.append(x)
            in_set[x] = True
            counts[2 * x] += 1
            for y in current[:-1]:
                counts[x + y] += 1
            current_conflicts = get_conflicts(counts)
            
            if current_conflicts < best_conflicts:
                best_conflicts = current_conflicts
                if best_conflicts == 0:
                    best_solution = list(current)
                    if len(current) > best_size:
                        best_size = len(current)
            continue
        
        # Find blockers
        blockers = find_blockers(x, current, counts)
        if not blockers:
            continue
        
        # Try swapping with a random blocker
        blocker = random.choice(blockers)
        # Remove blocker, try to add x
        try_remove(blocker, current, counts, in_set)
        if try_add(x, current, counts, in_set):
            # Success
            current_conflicts = get_conflicts(counts)
            
            if current_conflicts < best_conflicts:
                best_conflicts = current_conflicts
                if best_conflicts == 0:
                    best_solution = list(current)
                    if len(current) > best_size:
                        best_size = len(current)
        else:
            # Failed, restore blocker
            try_add(blocker, current, counts, in_set)
        
        T *= cooling
        if T < min_T:
            T = min_T

    # Final check
    final_set = set(best_solution)
    final_counts = [0] * (max_sum + 1)
    for i in range(len(best_solution)):
        x = best_solution[i]
        final_counts[2 * x] += 1
        for j in range(i + 1, len(best_solution)):
            y = best_solution[j]
            final_counts[x + y] += 1
            
    valid = True
    for s in range(2, max_sum + 1):
        if final_counts[s] > 2:
            valid = False
            break
            
    if valid:
        return best_solution
    else:
        return chosen
