"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 82.0  # 8s safety margin
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
        """Check if x can be added to current_set without violating constraints.
        Only need to check triples where x is involved as b or c, or where
        existing elements interact with x."""
        if x in current_set:
            return False
        
        # x could be the smallest (a), middle (b), or largest (c) in new triples
        # Case 1: x is 'a' (smallest) - need b,c from current_set with x < b <= c
        for b in current_set:
            if b <= x:
                continue
            for c in current_set:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is 'b' (middle) - need a < x <= c, a from current_set
        for a in current_set:
            if a >= x:
                continue
            for c in current_set:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is 'c' (largest) - need a < b <= x, a,b from current_set
        for a in current_set:
            for b in current_set:
                if b <= a:
                    continue
                if b > x:
                    continue
                if (b + x) % a == 0:
                    return False
        
        return True
    
    def greedy_extend(start_set, num_passes=3):
        """Try to extend a set by adding smaller elements, multiple passes."""
        current = set(start_set)
        
        for _ in range(num_passes):
            if time.time() > deadline:
                break
            
            min_curr = min(current)
            candidates = list(range(1, min_curr))
            random.shuffle(candidates)
            
            added_any = False
            for x in candidates:
                if time.time() > deadline:
                    break
                if can_add(x, current):
                    current.add(x)
                    added_any = True
            
            if not added_any:
                break
        
        return current
    
    def local_search(current_set, max_swaps=50):
        """Try to improve the set by swapping elements."""
        current = set(current_set)
        current_sorted = sorted(current)
        n = len(current_sorted)
        
        for _ in range(max_swaps):
            if time.time() > deadline:
                break
            
            made_swap = False
            # Try removing each element and replacing with a smaller one
            for i in range(n):
                if time.time() > deadline:
                    break
                to_remove = current_sorted[i]
                # Try replacing with elements not in current, smaller than to_remove
                candidates = [x for x in range(1, to_remove) if x not in current]
                random.shuffle(candidates)
                
                for x in candidates:
                    if time.time() > deadline:
                        break
                    # Check if removing to_remove and adding x gives a valid set
                    new_set = current - {to_remove} | {x}
                    if check_set_validity(new_set):
                        current = new_set
                        current_sorted = sorted(current)
                        made_swap = True
                        break
                if made_swap:
                    break
            
            if not made_swap:
                break
        
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
            best_set = list(start_set)
            best_len = len(start_set)
        
        # Try to extend
        extended = greedy_extend(start_set, num_passes=3)
        
        if len(extended) > best_len:
            best_set = sorted(extended)
            best_len = len(best_set)
        
        # Try local search on the extended set
        if time.time() < deadline - 10:  # Only if we have time
            improved = local_search(extended, max_swaps=30)
            if len(improved) > best_len:
                best_set = sorted(improved)
                best_len = len(best_set)
    
    # If we didn't find anything, return a known valid set
    if best_set is None:
        best_set = list(range(192, 257))
    
    return best_set
# EVOLVE-BLOCK-END
