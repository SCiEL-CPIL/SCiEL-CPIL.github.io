"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85.0  # 5s safety margin
    best_set = None
    best_len = 0
    n_max = 256
    
    # Precompute: for each a, which (b,c) pairs with a < b <= c <= n_max have a | (b+c)
    # conflict[a] = set of (b,c) pairs that conflict with a
    conflict = [set() for _ in range(n_max + 1)]
    for a in range(1, n_max + 1):
        for b in range(a + 1, n_max + 1):
            for c in range(b, n_max + 1):
                if (b + c) % a == 0:
                    conflict[a].add((b, c))
    
    def check_set_validity(s):
        """Check if the entire set satisfies the constraint."""
        s_sorted = sorted(s)
        s_set = set(s_sorted)
        for a in s_sorted:
            for (b, c) in conflict[a]:
                if b in s_set and c in s_set:
                    return False
        return True
    
    def can_add(x, current_set):
        """Check if x can be added to current_set without violating constraints."""
        if x in current_set:
            return False
        current_set.add(x)
        
        # Check all triples involving x
        # Case 1: x is the smallest (a = x)
        for (b, c) in conflict[x]:
            if b in current_set and c in current_set:
                current_set.remove(x)
                return False
        
        # Case 2: x is b or c in some triple (a, x, c) or (a, b, x) where a < x
        # For each a in current_set with a < x, check if (x, c) or (b, x) is in conflict[a]
        for a in current_set:
            if a >= x:
                continue
            # Check if there exists c >= x in current_set such that a | (x + c)
            # i.e., (x, c) in conflict[a]
            for c in current_set:
                if c < x:
                    continue
                if (x, c) in conflict[a]:
                    current_set.remove(x)
                    return False
            # Check if there exists b with a < b <= x in current_set such that a | (b + x)
            # i.e., (b, x) in conflict[a]
            for b in current_set:
                if b <= a or b > x:
                    continue
                if (b, x) in conflict[a]:
                    current_set.remove(x)
                    return False
        
        return True
    
    def local_search(current_set):
        """Try to improve the set via swap operations."""
        current = set(current_set)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_list = sorted(current)
            # Try swapping out one element for another
            for i, out_elem in enumerate(current_list):
                if time.time() > deadline:
                    break
                # Try adding elements not in current
                for in_elem in range(1, n_max + 1):
                    if in_elem in current:
                        continue
                    # Try swap: remove out_elem, add in_elem
                    trial = current - {out_elem} | {in_elem}
                    if check_set_validity(trial):
                        if len(trial) > len(current):
                            current = trial
                            improved = True
                            break
            if time.time() > deadline:
                break
        return current
    
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
        
        # Local search to improve
        if time.time() < deadline:
            improved_set = local_search(extended)
            if len(improved_set) > best_len:
                best_set = sorted(improved_set)
                best_len = len(best_set)
    
    # If we didn't find anything, return a known valid set
    if best_set is None:
        best_set = list(range(192, 257))
    
    return best_set
# EVOLVE-BLOCK-END
