import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit, 10s margin
    
    def try_add(idx, seq, prods, suffix_prods):
        if not seq:
            return [idx], {idx}, [idx]
        
        new_suffix = [idx]
        current_prod = idx
        for p in suffix_prods:
            current_prod *= p
            new_suffix.append(current_prod)
        
        # Check collisions
        for p in new_suffix:
            if p in prods:
                return None
        
        # Create new set
        # Optimization: If len(prods) is small, union is fast.
        new_prods = set(prods)
        new_prods.update(new_suffix)
        
        new_seq = seq + [idx]
        return new_seq, new_prods, new_suffix

    def greedy_fill(idx, seq, prods, suffix_prods):
        current_seq = seq
        current_prods = prods
        current_suffix = suffix_prods
        
        for x in range(idx, 129):
            if time.time() > deadline:
                break
            res = try_add(x, current_seq, current_prods, current_suffix)
            if res:
                current_seq, current_prods, current_suffix = res
        
        return current_seq

    def search(idx, seq, prods, suffix_prods, skip_budget):
        if time.time() > deadline:
            return greedy_fill(idx, seq, prods, suffix_prods)
        
        if idx > 128:
            return seq
        
        if skip_budget == 0:
            return greedy_fill(idx, seq, prods, suffix_prods)
        
        add_res = try_add(idx, seq, prods, suffix_prods)
        
        if add_res:
            new_seq, new_prods, new_suffix = add_res
            res_inc = search(idx + 1, new_seq, new_prods, new_suffix, skip_budget)
            res_skip = search(idx + 1, seq, prods, suffix_prods, skip_budget - 1)
            
            if res_inc and res_skip:
                return res_inc if len(res_inc) > len(res_skip) else res_skip
            elif res_inc:
                return res_inc
            else:
                return res_skip
        else:
            return search(idx + 1, seq, prods, suffix_prods, skip_budget)

    # Start with a reasonable skip budget
    # 10 allows 1024 paths.
    # 12 allows 4096 paths.
    # Let's try 11.
    best_seq = search(2, [], set(), [], 11)
    
    return best_seq
