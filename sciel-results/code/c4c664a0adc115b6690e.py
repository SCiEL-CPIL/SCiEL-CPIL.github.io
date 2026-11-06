import time

def construct():
    start_time = time.time()
    deadline = start_time + 58.0

    def is_valid_incremental(seq, new_val, existing_products):
        n = len(seq)
        p = new_val
        new_prods = set()
        # Check collision with existing products
        for i in range(n - 1, -1, -1):
            if p in existing_products:
                return False, None
            new_prods.add(p)
            if i > 0:
                p *= seq[i]
            else:
                # Last iteration, p was multiplied by seq[0] in previous step?
                # No, loop structure:
                # i = n-1: p = new_val. Check. Add. p *= seq[n-1].
                # i = n-2: p = new_val * seq[n-1]. Check. Add. p *= seq[n-2].
                # ...
                # i = 0: p = new_val * ... * seq[1]. Check. Add. p *= seq[0].
                # Wait, the product for block (0, n) is seq[0]*...*seq[n-1]*new_val.
                # My loop:
                # Start p = new_val.
                # i = n-1: check new_val. add new_val. p = new_val * seq[n-1].
                # i = n-2: check new_val*seq[n-1]. add. p = new_val*seq[n-1]*seq[n-2].
                # ...
                # i = 0: check new_val*...*seq[1]. add. p = new_val*...*seq[0].
                # The loop ends. The last product added is for block (0, n-1) ending at n?
                # No, block (0, n) is seq[0]...seq[n-1]new_val.
                # In the iteration i=0, we check `p` which is `new_val * ... * seq[1]`.
                # This corresponds to block (1, n).
                # We need to also check block (0, n).
                # So after the loop, or inside, we need to handle the full product.
                
                # Let's rewrite the loop clearly.
                pass
        # The logic above is slightly off. Let's trace.
        # We want products:
        # 1. new_val (block [n])
        # 2. seq[n-1]*new_val (block [n-1, n])
        # ...
        # n. seq[0]*...*seq[n-1]*new_val (block [0, n])
        
        # Correct loop:
        p = new_val
        new_prods = set()
        for i in range(n - 1, -1, -1):
            if p in existing_products:
                return False, None
            new_prods.add(p)
            p *= seq[i]
        # After loop, p is seq[0]*...*seq[n-1]*new_val.
        # Did we check it?
        # i=0: checked p (which was seq[1]*...*seq[n-1]*new_val).
        # Then p *= seq[0].
        # So the last product `p` (full block) was NOT checked in the `if p in existing` inside the loop for i=0?
        # Wait.
        # Iteration i=0:
        # p currently holds product for block [1, n].
        # Check p.
        # Add p.
        # p *= seq[0]. Now p holds product for block [0, n].
        # Loop ends.
        # So we missed checking the full block product!
        
        # Fix: Check after loop? Or restructure.
        
        # Restructure:
        p = new_val
        new_prods = set()
        # Check block [n, n]
        if p in existing_products:
            return False, None
        new_prods.add(p)
        
        for i in range(n - 1, -1, -1):
            p *= seq[i]
            if p in existing_products:
                return False, None
            new_prods.add(p)
            
        return True, new_prods

    def greedy_construct():
        values = []
        prods = set()
        for x in range(2, 129):
            valid, new_prods = is_valid_incremental(values, x, prods)
            if valid:
                values.append(x)
                prods.update(new_prods)
        return values

    best_seq = greedy_construct()
    best_len = len(best_seq)

    def dfs(idx, current_seq, current_prods, current_len):
        nonlocal best_seq, best_len
        
        if time.time() > deadline:
            return

        # Pruning
        if current_len + (128 - idx + 1) <= best_len:
            return

        if idx > 128:
            if current_len > best_len:
                best_len = current_len
                best_seq = current_seq[:]
            return

        # Branch 1: Include idx
        valid, new_prods = is_valid_incremental(current_seq, idx, current_prods)
        if valid:
            new_seq = current_seq + [idx]
            new_prods_set = current_prods | new_prods
            dfs(idx + 1, new_seq, new_prods_set, current_len + 1)
            if time.time() > deadline:
                return

        # Branch 2: Exclude idx
        dfs(idx + 1, current_seq, current_prods, current_len)

    dfs(2, [], set(), 0)

    return best_seq
