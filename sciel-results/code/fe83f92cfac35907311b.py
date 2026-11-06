"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85.0  # 5s safety margin
    best_set = None
    best_len = 0
    
    def check_set_validity(s):
        """Check if the entire set satisfies the constraint."""
        s_sorted = sorted(s)
        n = len(s_sorted)
        for i in range(n):
            a = s_sorted[i]
            for j in range(i+1, n):
                b = s_sorted[j]
                for k in range(j, n):
                    c = s_sorted[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def can_add(x, current_set):
        """Check if x can be added to current_set without violating constraints."""
        # x must not be in current_set
        if x in current_set:
            return False
        
        new_set = current_set | {x}
        new_sorted = sorted(new_set)
        n = len(new_sorted)
        
        for i in range(n):
            a = new_sorted[i]
            for j in range(i+1, n):
                b = new_sorted[j]
                for k in range(j, n):
                    c = new_sorted[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def greedy_extend(start_set):
        """Try to extend a set by adding smaller elements."""
        current = set(start_set)
        
        # Try adding elements from 1 to min(current) - 1
        candidates = list(range(1, min(current)))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(x, current):
                current.add(x)
        
        return current
    
    # Try multiple starting points
    for start in range(150, 257):
        if time.time() > deadline:
            break
        
        # Start with upper tail from start to 256
        start_set = list(range(start, 257))
        
        # Verify base set is valid
        if not check_set_validity(start_set):
            continue
        
        if len(start_set) > best_len:
            best_set = start_set
            best_len = len(start_set)
        
        # Try to extend
        extended = greedy_extend(start_set)
        
        if len(extended) > best_len:
            best_set = sorted(extended)
            best_len = len(best_set)
    
    # If we didn't find anything, return a known valid set
    if best_set is None:
        best_set = list(range(192, 257))
    
    return best_set
# EVOLVE-BLOCK-END
