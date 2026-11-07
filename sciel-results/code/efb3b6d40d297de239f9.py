import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    def check_divisibility(sums_set):
        """
        Check if any two distinct non-zero sums in sums_set divide each other.
        """
        non_zero_sums = [s for s in sums_set if s != 0]
        if not non_zero_sums:
            return True
        
        s_set = set(non_zero_sums)
        max_sum = max(non_zero_sums)
        
        for s in non_zero_sums:
            multiple = s * 2
            while multiple <= max_sum:
                if multiple in s_set:
                    return False
                multiple += s
        return True

    def try_add(current_sums, x):
        """
        Attempts to add x to the set defined by current_sums.
        Returns the new set of sums if valid, otherwise None.
        """
        new_sums = set(current_sums)
        for s in current_sums:
            ns = s + x
            # Check for collision with existing sums
            if ns in current_sums:
                return None
            new_sums.add(ns)
            
        # Check divisibility
        if not check_divisibility(new_sums):
            return None
            
        return new_sums

    def solve_for_size(target_size):
        """
        Backtracking search for a valid set of size target_size.
        """
        # Heuristic: Try larger numbers first
        candidates = list(range(256, 0, -1))
        current_set = []
        current_sums = {0}
        
        def dfs(depth):
            if time.time() > deadline:
                return None
                
            if depth == target_size:
                return list(current_set)
                
            for i in range(len(candidates)):
                x = candidates[i]
                if x in current_set:
                    continue
                    
                new_sums = try_add(current_sums, x)
                if new_sums is not None:
                    current_set.append(x)
                    old_sums = current_sums
                    current_sums = new_sums
                    
                    result = dfs(depth + 1)
                    if result is not None:
                        return result
                        
                    current_set.pop()
                    current_sums = old_sums
                        
            return None
            
        return dfs(0)

    # Try to find a set of size 15 down to 6
    for size in range(15, 5, -1):
        result = solve_for_size(size)
        if result:
            # Final validation
            sums = {0}
            for x in result:
                new_sums = {s + x for s in sums}
                sums.update(new_sums)
            if len(sums) == (1 << len(result)) and check_divisibility(sums):
                return [int(x) for x in result]
                
    # Fallback: Greedy construction
    elements = []
    sums = {0}
    for x in range(256, 0, -1):
        new_sums = try_add(sums, x)
        if new_sums is not None:
            elements.append(x)
            sums = new_sums
            
    if elements:
        if len(sums) == (1 << len(elements)) and check_divisibility(sums):
            return [int(x) for x in elements]
            
    return [1]
