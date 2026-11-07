def construct():
    start_time = time.time()
    best_set = []
    
    def _is_valid_addition(current_sums, new_sum):
        if new_sum in current_sums:
            return False
        
        max_existing = max(current_sums) if current_sums else 0
        
        # Check if any existing sum is a multiple of new_sum
        multiple = new_sum
        while multiple <= max_existing:
            if multiple in current_sums:
                return False
            multiple += new_sum
            
        # Check if new_sum is a multiple of any existing sum
        i = 1
        while i * i <= new_sum:
            if new_sum % i == 0:
                if i in current_sums:
                    return False
                other = new_sum // i
                if other != i and other in current_sums:
                    return False
            i += 1
            
        return True

    def _can_add_element(current_sums, x):
        new_sums = []
        for s in current_sums:
            ns = s + x
            if ns in current_sums:
                return False
            new_sums.append(ns)
            
        for ns in new_sums:
            if not _is_valid_addition(current_sums, ns):
                return False
                
        new_sums_set = set(new_sums)
        max_new = max(new_sums)
        for ns in new_sums:
            multiple = ns + ns
            while multiple <= max_new:
                if multiple in new_sums_set:
                    return False
                multiple += ns
                
        return True

    def backtrack(current_elements, current_sums, candidates, start_idx):
        nonlocal best_set
        
        if time.time() - start_time > DEADLINE_SECONDS - 2:
            return
        
        if len(current_elements) > len(best_set):
            best_set = current_elements[:]
            
        if start_idx >= len(candidates):
            return
            
        for i in range(start_idx, len(candidates)):
            x = candidates[i]
            
            if _can_add_element(current_sums, x):
                new_sums = set(current_sums)
                for s in current_sums:
                    new_sums.add(s + x)
                
                backtrack(
                    current_elements + [x], 
                    new_sums, 
                    candidates, 
                    i + 1
                )
                
                if time.time() - start_time > DEADLINE_SECONDS - 2:
                    return

    # Strategy 1: Descending order
    candidates = list(range(256, 0, -1))
    initial_sums = {0}
    backtrack([], initial_sums, candidates, 0)
    
    # Strategy 2: Randomized Greedy
    for restart in range(100):
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
            
        current_elements = []
        current_sums = {0}
        available = list(range(1, 257))
        random.shuffle(available)
        
        for x in available:
            if time.time() - start_time > DEADLINE_SECONDS - 5:
                break
            if _can_add_element(current_sums, x):
                new_sums = set(current_sums)
                for s in current_sums:
                    new_sums.add(s + x)
                current_sums = new_sums
                current_elements.append(x)
                
        if len(current_elements) > len(best_set):
            best_set = current_elements[:]
            
    if not best_set:
        fallback = []
        sums = {0}
        for x in range(256, 0, -1):
            if _can_add_element(sums, x):
                new_sums = set(sums)
                for s in sums:
                    new_sums.add(s + x)
                sums = new_sums
                fallback.append(x)
        best_set = fallback

    return [int(x) for x in best_set]
