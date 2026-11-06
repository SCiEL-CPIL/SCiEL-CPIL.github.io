import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    target_elements = [1, 2, 5, 11]
    max_val = 284
    
    def get_sums(s):
        sums = set()
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                sums.add(s[i] + s[j])
        return sums

    def is_valid_addition(s, x, current_sums):
        # Check if adding x to s creates any duplicate sums
        # New sums are {x + y for y in s} U {2x}
        # Existing sums are current_sums
        # We need to ensure new_sums are distinct from each other and from current_sums
        # Note: x is not in s
        new_sums = {x + y for y in s}
        new_sums.add(2 * x)
        
        # Check internal collisions in new_sums
        if len(new_sums) != len(s) + 1:
            return False
        
        # Check collisions with existing sums
        if new_sums & current_sums:
            return False
            
        return True

    def greedy_build(seed=None):
        if seed is not None:
            random.seed(seed)
        
        s = list(target_elements)
        s.sort()
        current_sums = get_sums(s)
        
        # Try to add elements in increasing order, but with some randomization
        # to escape deterministic traps.
        candidates = list(range(12, max_val + 1))
        random.shuffle(candidates)
        
        # First pass: try all in random order
        for x in candidates:
            if is_valid_addition(s, x, current_sums):
                s.append(x)
                s.sort()
                # Update sums
                new_sums = {x + y for y in s if y != x} # Wait, s now includes x
                # Better to just recalculate or update incrementally
                # Incremental update:
                # New sums involving x: {x+y for y in old_s} U {2x}
                # old_s is s before append
                # Let's just recalculate for simplicity if size is small, 
                # but size can be ~20, so O(400) is fine.
                current_sums = get_sums(s)
        
        return s, current_sums

    def try_improve(s, current_sums, time_limit):
        """
        Local search: try to add, swap, or double-swap to improve size.
        Returns improved set and sums.
        """
        s = list(s)
        s.sort()
        current_sums = get_sums(s) # Ensure fresh
        size = len(s)
        no_improve_count = 0
        
        while time.time() < time_limit:
            if no_improve_count > 100:
                break
                
            # 1. Try to add a random element
            found_add = False
            candidates = [x for x in range(1, max_val + 1) if x not in s]
            random.shuffle(candidates)
            for x in candidates:
                if is_valid_addition(s, x, current_sums):
                    s.append(x)
                    s.sort()
                    current_sums = get_sums(s)
                    size += 1
                    no_improve_count = 0
                    found_add = True
                    break
            
            if found_add:
                continue
                
            # 2. Try single swap
            found_swap = False
            for _ in range(50): # Limit swap attempts per iteration to avoid stalling
                if time.time() > time_limit:
                    return s, current_sums
                if not s: continue
                idx = random.randint(0, len(s) - 1)
                x = s[idx]
                # Try to replace x with y
                # y must be different from all elements in s
                # We pick a random y from 1..284
                y = random.randint(1, max_val)
                if y in s:
                    continue
                    
                # Check if s \ {x} + {y} is valid
                s_rest = [e for i, e in enumerate(s) if i != idx]
                # Get sums of s_rest
                sums_rest = get_sums(s_rest)
                
                # Check if y can be added to s_rest
                if is_valid_addition(s_rest, y, sums_rest):
                    # Perform swap
                    s[idx] = y
                    s.sort()
                    current_sums = get_sums(s)
                    no_improve_count = 0
                    found_swap = True
                    break
            
            if found_swap:
                continue
                
            # 3. Try double swap
            found_double_swap = False
            if len(s) >= 2:
                for _ in range(20):
                    if time.time() > time_limit:
                        return s, current_sums
                    idx1 = random.randint(0, len(s) - 1)
                    idx2 = random.randint(0, len(s) - 1)
                    if idx1 == idx2:
                        continue
                    x1 = s[idx1]
                    x2 = s[idx2]
                    
                    y1 = random.randint(1, max_val)
                    y2 = random.randint(1, max_val)
                    
                    # Ensure y1, y2 are distinct and not in s \ {x1, x2}
                    s_rest = [e for i, e in enumerate(s) if i != idx1 and i != idx2]
                    if y1 in s_rest or y2 in s_rest or y1 == y2:
                        continue
                        
                    # Check if s_rest + {y1, y2} is valid
                    sums_rest = get_sums(s_rest)
                    
                    # Check y1 against s_rest
                    if not is_valid_addition(s_rest, y1, sums_rest):
                        continue
                    # Update sums_rest with y1
                    new_sums_rest = sums_rest | {y1 + e for e in s_rest} | {2 * y1}
                    
                    # Check y2 against s_rest + {y1}
                    s_temp = s_rest + [y1]
                    if not is_valid_addition(s_temp, y2, new_sums_rest):
                        continue
                        
                    # Perform double swap
                    s[idx1] = y1
                    s[idx2] = y2
                    s.sort()
                    current_sums = get_sums(s)
                    no_improve_count = 0
                    found_double_swap = True
                    break
            
            if found_double_swap:
                continue
                
            no_improve_count += 1
            
        return s, current_sums

    best_set = []
    best_size = 0
    seed_counter = 0
    
    while time.time() < deadline:
        # Greedy build with randomization
        s, sums = greedy_build(seed=seed_counter)
        seed_counter += 1
        
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
            
        # If we hit the theoretical max or close to it, we can stop early
        if best_size >= 20:
            break
            
        # Try to improve the current set
        # Allocate a small time budget for improvement, e.g., 0.5s or until stuck
        improve_deadline = time.time() + 0.5
        if time.time() + 0.5 > deadline:
            improve_deadline = deadline - 1.0
            
        s, sums = try_improve(s, sums, improve_deadline)
        
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
            
        if best_size >= 20:
            break
            
    return best_set
