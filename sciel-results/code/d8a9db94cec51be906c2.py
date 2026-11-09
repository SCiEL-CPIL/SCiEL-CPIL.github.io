import time
import random

def construct():
    prescribed = [1, 2, 5, 11]
    prescribed.sort()
    
    # Pre-compute initial sums
    initial_sums = set()
    for i in range(len(prescribed)):
        for j in range(i, len(prescribed)):
            initial_sums.add(prescribed[i] + prescribed[j])
            
    # 1. Greedy Initialization
    # Start with prescribed, add smallest valid numbers
    current_set = list(prescribed)
    current_sums = set(initial_sums)
    
    # Helper to check if x can be added
    def can_add(x, s_set, s_sums):
        for y in s_set:
            if x + y in s_sums:
                return False
        if 2 * x in s_sums:
            return False
        return True

    # Greedy fill
    for x in range(12, 285):
        if can_add(x, current_set, current_sums):
            current_set.append(x)
            for y in current_set[:-1]:
                current_sums.add(x + y)
            current_sums.add(2 * x)
            
    best_set = list(current_set)
    best_size = len(current_set)
    
    # 2. Local Search / Simulated Annealing
    # Goal: Maximize size.
    # We will try to improve the set.
    # State: (set, sums)
    # Moves:
    # 1. Add a random valid element (if possible).
    # 2. Swap: Remove a random non-prescribed element, add a random unused element.
    # 3. Replace: Remove 2, Add 2.
    
    # To ensure we don't timeout, we run for a fixed duration.
    start_time = time.time()
    time_limit = 110  # 10s buffer
    
    # We keep the best set found so far
    # We operate on a "working" set
    
    # For efficiency, we represent the set as a list and sums as a set.
    # We need to be careful with updates.
    
    # Let's use a simpler heuristic:
    # Try to increase size by 1.
    # If stuck, perform random swaps to explore space.
    
    # Since we already have a greedy solution, let's try to improve it.
    # We will use a "kick" strategy:
    # While time < limit:
    #   Try to add a random unused number.
    #   If success: update best, continue.
    #   If fail: 
    #      Pick a random element x from current_set (not prescribed).
    #      Remove x.
    #      Try to add 1 or 2 random unused numbers.
    #      If valid and size >= old_size: accept.
    #      If size > old_size: update best.
    #      If size == old_size: accept (explore).
    #      If size < old_size: reject (unless we are doing SA, but let's keep it simple: only accept if size doesn't decrease, or accept with low prob if it increases potential? No, let's stick to non-decreasing size for simplicity and robustness).
    #      Actually, sometimes you need to decrease size to increase it later. 
    #      Let's use SA: Accept worse states with probability exp(-delta/T).
    #      Objective: Maximize size. Penalty for invalidity is infinite (we only consider valid moves).
    #      So "worse" means smaller size.
    
    # Let's implement a simple SA.
    # Temperature starts high, cools down.
    # Move: 
    #   - With prob 0.2: Try to add a random valid element.
    #   - With prob 0.5: Swap one element (remove x, add y).
    #   - With prob 0.3: Swap two elements (remove x,z, add y,w).
    
    # To make it faster, we pre-compute the list of unused numbers.
    
    # Let's refine the loop:
    T = 100.0
    cooling = 0.99
    
    # We need to track the best set found.
    
    # To avoid deep copying large sets every step, we will modify in place and revert if rejected.
    
    def add_element(x, s_set, s_sums):
        s_set.append(x)
        for y in s_set[:-1]:
            s_sums.add(x + y)
        s_sums.add(2 * x)
        
    def remove_element(x, s_set, s_sums):
        # Remove sums involving x
        for i in range(len(s_set)):
            if s_set[i] == x:
                idx = i
                break
        # Remove x from set
        s_set.pop(idx)
        # Remove sums
        for y in s_set:
            s_sums.discard(x + y)
        s_sums.discard(2 * x)

    # Initialize working set
    work_set = list(current_set)
    work_sums = set(current_sums)
    
    unused = set(range(1, 285)) - set(work_set)
    
    while time.time() < time_limit:
        T *= cooling
        if T < 0.1:
            # Restart or break? Let's just keep going with low T
            pass
            
        move_type = random.random()
        accepted = False
        old_size = len(work_set)
        
        if move_type < 0.3:
            # Try to add a random element
            if unused:
                x = random.choice(list(unused))
                if can_add(x, work_set, work_sums):
                    add_element(x, work_set, work_sums)
                    unused.discard(x)
                    accepted = True
                    # This is always an improvement (size +1)
                    if len(work_set) > best_size:
                        best_set = list(work_set)
                        best_size = len(work_set)
                else:
                    accepted = False
        elif move_type < 0.8:
            # Swap 1 element
            # Pick a random element to remove (not prescribed)
            removable = [x for x in work_set if x not in prescribed]
            if removable:
                x = random.choice(removable)
                # Pick a random unused element to add
                if unused:
                    y = random.choice(list(unused))
                    # Remove x
                    remove_element(x, work_set, work_sums)
                    unused.add(x)
                    # Try to add y
                    if can_add(y, work_set, work_sums):
                        add_element(y, work_set, work_sums)
                        unused.discard(y)
                        accepted = True
                        # Size is same
                        # Check if this leads to a better "potential"? 
                        # For now, accept if valid. 
                        # To explore, we might accept worse states, but size is same here.
                        # So we accept to move in the space.
                    else:
                        # Revert: remove y (not added), add x back
                        # y was not added, so just add x back
                        add_element(x, work_set, work_sums)
                        unused.discard(x)
                        accepted = False
                else:
                    accepted = False
            else:
                accepted = False
        else:
            # Swap 2 elements
            removable = [x for x in work_set if x not in prescribed]
            if len(removable) >= 2 and len(unused) >= 2:
                # Pick 2 to remove
                x, z = random.sample(removable, 2)
                # Pick 2 to add
                y, w = random.sample(list(unused), 2)
                
                # Remove x, z
                remove_element(x, work_set, work_sums)
                remove_element(z, work_set, work_sums)
                unused.add(x)
                unused.add(z)
                
                # Try to add y, w
                # We need to check if y and w can be added simultaneously
                # Check y against current (reduced) set
                # Check w against current (reduced) set + y
                # And y+w sum not in sums
                
                valid = True
                # Check y
                for v in work_set:
                    if y + v in work_sums:
                        valid = False
                        break
                if valid and 2*y in work_sums:
                    valid = False
                # Check w
                if valid:
                    for v in work_set:
                        if w + v in work_sums:
                            valid = False
                            break
                    if valid and 2*w in work_sums:
                        valid = False
                    if valid and y + w in work_sums:
                        valid = False
                    # Also need to check if y+w creates conflict with existing sums? 
                    # The check "w + v in work_sums" covers w+existing.
                    # "y + v in work_sums" covers y+existing.
                    # "2*y" and "2*w" cover doubles.
                    # "y+w" covers the new pair.
                    # But we also need to ensure that adding y doesn't conflict with w's future addition?
                    # The order matters. Let's just add y, then check w.
                    
                if valid:
                    add_element(y, work_set, work_sums)
                    unused.discard(y)
                    # Now check w against the set that includes y
                    valid2 = True
                    for v in work_set:
                        if w + v in work_sums:
                            valid2 = False
                            break
                    if valid2 and 2*w in work_sums:
                        valid2 = False
                    if valid2:
                        add_element(w, work_set, work_sums)
                        unused.discard(w)
                        accepted = True
                    else:
                        # Revert y
                        remove_element(y, work_set, work_sums)
                        unused.add(y)
                        # Revert x, z
                        add_element(x, work_set, work_sums)
                        add_element(z, work_set, work_sums)
                        unused.discard(x)
                        unused.discard(z)
                        accepted = False
                else:
                    # Revert x, z
                    add_element(x, work_set, work_sums)
                    add_element(z, work_set, work_sums)
                    unused.discard(x)
                    unused.discard(z)
                    accepted = False
            else:
                accepted = False
        
        if not accepted:
            # If we couldn't make a move, we might be stuck.
            # We can try to force a "kick" by removing a random element and not replacing it immediately?
            # No, that decreases size. 
            # Let's just continue. The random choices will eventually find a path.
            pass

    return best_set
