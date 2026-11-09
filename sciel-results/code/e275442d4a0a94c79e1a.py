import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 57  # 3s margin for 60s limit
    
    max_val = 4096
    max_size = 80
    
    best_set = []
    best_size = 0
    
    def get_all_sums(S):
        """Generate all triple sums for a sorted list S."""
        sums = set()
        n = len(S)
        for i in range(n):
            x = S[i]
            sums.add(3 * x)
            for j in range(i + 1, n):
                y = S[j]
                sums.add(2 * x + y)
                sums.add(x + 2 * y)
                for k in range(j + 1, n):
                    z = S[k]
                    sums.add(x + y + z)
        return sums
    
    def get_sums_involving(x, S):
        """Get all triple sums that involve element x, where S is a list containing x."""
        sums = set()
        n = len(S)
        idx = S.index(x)
        # 3x
        sums.add(3 * x)
        # 2x + y and x + 2y for all y != x
        for i in range(n):
            if i == idx:
                continue
            y = S[i]
            sums.add(2 * x + y)
            sums.add(x + 2 * y)
        # x + y + z for all pairs y < z, y != x, z != x
        others = [S[i] for i in range(n) if i != idx]
        m = len(others)
        for i in range(m):
            y = others[i]
            for j in range(i + 1, m):
                z = others[j]
                sums.add(x + y + z)
        return sums
    
    def check_add(x, S, sum_set):
        """Check if x can be added to S. Returns (valid, new_sums_set)."""
        new_sums = set()
        new_sums.add(3 * x)
        n = len(S)
        for i in range(n):
            y = S[i]
            new_sums.add(2 * x + y)
            new_sums.add(x + 2 * y)
        for i in range(n):
            y = S[i]
            for j in range(i + 1, n):
                z = S[j]
                new_sums.add(x + y + z)
        
        # Check collisions with existing sums
        for s in new_sums:
            if s in sum_set:
                return False, None
        
        # Check internal collisions
        if len(new_sums) != 1 + 2 * n + n * (n - 1) // 2:
            return False, None
        
        return True, new_sums
    
    def do_add(x, S, sum_set, new_sums):
        """Actually add x to S and update sum_set."""
        S.append(x)
        S.sort()
        sum_set.update(new_sums)
    
    def do_remove(x, S, sum_set):
        """Remove x from S and update sum_set."""
        idx = S.index(x)
        S.pop(idx)
        sums_to_remove = get_sums_involving(x, S + [x])
        # Need to compute sums involving x in the set BEFORE removal
        # Let's recompute: we need sums that involved x when x was in S
        # Actually, let's just recompute from scratch for correctness
        # But that's expensive. Let's compute before removing.
        pass
    
    def remove_element(x, S, sum_set):
        """Remove x from S and update sum_set correctly."""
        # First compute sums involving x while x is still in S
        full_S = S[:]
        full_S.append(x)
        full_S.sort()
        sums_x = get_sums_involving(x, full_S)
        # Now remove x from S
        idx = S.index(x)
        S.pop(idx)
        # Remove the sums
        for s in sums_x:
            sum_set.discard(s)
    
    def greedy_build(order):
        """Build a set by greedily adding elements in the given order."""
        current_set = []
        sum_set = set()
        for x in order:
            if len(current_set) >= max_size:
                break
            if x in current_set:
                continue
            valid, new_sums = check_add(x, current_set, sum_set)
            if valid:
                do_add(x, current_set, sum_set, new_sums)
        return current_set, sum_set
    
    # 1. Multiple greedy initializations
    strategies = []
    
    # Ascending
    strategies.append(list(range(1, max_val + 1)))
    # Descending
    strategies.append(list(range(max_val, 0, -1)))
    
    # Random shuffles
    for _ in range(15):
        if time.time() > deadline:
            break
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        strategies.append(candidates)
    
    # Spaced starts
    for spacing in [50, 100, 200, 400, 800]:
        if time.time() > deadline:
            break
        candidates = list(range(1, max_val + 1, spacing))
        random.shuffle(candidates)
        strategies.append(candidates)
    
    for order in strategies:
        if time.time() > deadline:
            break
        current_set, sum_set = greedy_build(order)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    if not best_set:
        return []
    
    # 2. Simulated Annealing / Local Search
    current_set = best_set[:]
    current_sum_set = get_all_sums(current_set)
    current_size = len(current_set)
    
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # Track best
    best_set = current_set[:]
    best_size = current_size
    
    step = 0
    while time.time() < deadline:
        step += 1
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to add a random element
            if current_size < max_size:
                x = random.randint(1, max_val)
                if x not in current_set:
                    valid, new_sums = check_add(x, current_set, current_sum_set)
                    if valid:
                        # Accept: size increases
                        do_add(x, current_set, current_sum_set, new_sums)
                        current_size += 1
                        if current_size > best_size:
                            best_size = current_size
                            best_set = current_set[:]
                            T = max(T, 2.0)
                        
        elif move_type < 0.8:
            # Try swap: remove one, add one
            if current_size > 0:
                y = random.choice(current_set)
                x = random.randint(1, max_val)
                if x == y or x in current_set:
                    continue
                
                # Check if we can swap y -> x
                # Remove y temporarily
                full_S = current_set[:]
                idx_y = full_S.index(y)
                full_S.pop(idx_y)
                full_S.sort()
                
                # Get sums involving y in the full set
                sums_y = get_sums_involving(y, current_set)
                
                # Check if x is valid in S\{y}
                # We need to check x against (sum_set - sums_y)
                valid, new_sums_x = check_add(x, full_S, current_sum_set)
                # But check_add checks against current_sum_set which still has sums_y
                # We need to check against sum_set minus sums_y
                # Let's do it manually
                new_sums_x = set()
                new_sums_x.add(3 * x)
                n = len(full_S)
                for i in range(n):
                    yy = full_S[i]
                    new_sums_x.add(2 * x + yy)
                    new_sums_x.add(x + 2 * yy)
                for i in range(n):
                    yy = full_S[i]
                    for j in range(i + 1, n):
                        zz = full_S[j]
                        new_sums_x.add(x + yy + zz)
                
                # Check for collisions with (sum_set - sums_y)
                collides = False
                for s in new_sums_x:
                    if s in current_sum_set and s not in sums_y:
                        collides = True
                        break
                
                if not collides:
                    # Check internal consistency of new_sums_x
                    expected_count = 1 + 2 * n + n * (n - 1) // 2
                    if len(new_sums_x) == expected_count:
                        # Accept the swap
                        # Remove y
                        remove_element(y, current_set, current_sum_set)
                        # Add x
                        current_set.append(x)
                        current_set.sort()
                        current_sum_set.update(new_sums_x)
                        # Size unchanged, but this helps exploration
                        # With some probability, accept even if it doesn't improve
                        # Since size is same, we accept to explore
                        pass
        
        else:
            # Try removal (to escape local optima)
            if current_size > 1:
                y = random.choice(current_set)
                # Accept removal with probability based on temperature
                if random.random() < 0.1 * T:
                    remove_element(y, current_set, current_sum_set)
                    current_size -= 1
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # Final polish: try to add more elements
    if time.time() < deadline - 1:
        current_set = best_set[:]
        current_sum_set = get_all_sums(current_set)
        for x in range(1, max_val + 1):
            if time.time() > deadline - 1:
                break
            if x in current_set:
                continue
            if len(current_set) >= max_size:
                break
            valid, new_sums = check_add(x, current_set, current_sum_set)
            if valid:
                do_add(x, current_set, current_sum_set, new_sums)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    return best_set
