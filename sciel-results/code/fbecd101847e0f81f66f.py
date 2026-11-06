    def backtrack(selected, order_list, start_idx, sums, target_size):
        nonlocal best_result, best_size
        if time.time() > deadline:
            return False
        
        current_size = len(selected)
        if current_size == target_size:
            if target_size > best_size:
                best_result = selected[:]
                best_size = target_size
            return True
        
        # Pruning
        if current_size + (len(order_list) - start_idx) < target_size:
            return False
            
        for i in range(start_idx, len(order_list)):
            idx = order_list[i]
            val = v[idx - 1]
            
            new_sums_check = {s + val for s in sums}
            if not sums.isdisjoint(new_sums_check):
                continue
                
            next_sums = set(sums)
            next_sums.update(new_sums_check)
            
            selected.append(idx)
            if backtrack(selected, order_list, i + 1, next_sums, target_size):
                selected.pop()
                return True
            selected.pop()
            
        return False
