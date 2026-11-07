    def construct():
        start_time = time.time()
        deadline = start_time + 85.0
        best_set = []
        best_size = 0
        
        candidates_base = list(range(1, MAX_VAL + 1))
        
        while time.time() < deadline:
            candidates = candidates_base[:]
            random.shuffle(candidates)
            
            # Optional: Sort by some heuristic? 
            # Let's stick to random for now.
            
            current_set = []
            sums_mask = 0
            
            def dfs(idx, depth, current_set, sums_mask):
                nonlocal best_set, best_size
                
                if time.time() > deadline:
                    return
                
                if depth > best_size:
                    best_size = depth
                    best_set = current_set[:]
                
                if depth == MAX_SIZE:
                    return
                
                # Prune if not enough candidates left
                if depth + (len(candidates) - idx) <= best_size:
                    return
                
                for i in range(idx, len(candidates)):
                    x = candidates[i]
                    
                    new_sums_mask = (sums_mask << x) | (1 << x)
                    
                    if new_sums_mask & sums_mask:
                        continue
                    
                    # Check conflict with old sums
                    union_div = 0
                    temp = new_sums_mask
                    while temp:
                        lsb = temp & -temp
                        n = lsb.bit_length() - 1
                        union_div |= div_mask[n]
                        temp ^= lsb
                        
                    if union_div & sums_mask:
                        continue
                    
                    # Check internal conflict
                    valid = True
                    temp = new_sums_mask
                    while temp and valid:
                        lsb = temp & -temp
                        n = lsb.bit_length() - 1
                        if (new_sums_mask & div_mask[n]) != lsb:
                            valid = False
                            break
                        temp ^= lsb
                    
                    if not valid:
                        continue
                    
                    current_set.append(x)
                    dfs(i + 1, depth + 1, current_set, new_sums_mask)
                    current_set.pop()
                    
                    if best_size == MAX_SIZE:
                        return

            dfs(0, 0, current_set, sums_mask)
            
            if best_size == MAX_SIZE:
                break

        return [int(x) for x in best_set]
    