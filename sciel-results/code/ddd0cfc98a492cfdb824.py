def construct():
    import time
    from math import gcd

    deadline = time.time() + 85.0  # Safety margin for 90s limit
    
    # 1. Precompute GCD table
    gcd_table = [[0] * 129 for _ in range(129)]
    for a in range(1, 129):
        for b in range(a, 129):
            g = gcd(a, b)
            gcd_table[a][b] = g
            gcd_table[b][a] = g

    # 2. Identify all "bad" triples
    bad_triples = []
    for a in range(1, 129):
        for b in range(a + 1, 129):
            gab = gcd_table[a][b]
            for c in range(b + 1, 129):
                if gcd_table[a][c] == gab and gcd_table[b][c] == gab:
                    bad_triples.append((a - 1, b - 1, c - 1))
    
    # Map vertex to list of triple indices
    v_to_t_idx = [[] for _ in range(128)]
    triple_masks = []
    for i, t in enumerate(bad_triples):
        mask = 0
        for v in t:
            mask |= (1 << v)
            v_to_t_idx[v].append(i)
        triple_masks.append(mask)

    # 3. Greedy Initial Solution
    def greedy_solution():
        selected = set()
        # Sort by degree descending
        order = sorted(range(128), key=lambda x: len(v_to_t_idx[x]), reverse=True)
        for v in order:
            can_add = True
            for t_idx in v_to_t_idx[v]:
                t = bad_triples[t_idx]
                count = 0
                for u in t:
                    if u == v: continue
                    if u in selected:
                        count += 1
                if count == 2:
                    can_add = False
                    break
            if can_add:
                selected.add(v)
        return selected

    best_set = greedy_solution()
    best_size = len(best_set)
    
    # 4. Branch and Bound with Bitmasks
    def propagate_mask(selected, candidates, forbidden):
        changed = True
        while changed:
            changed = False
            for mask in triple_masks:
                sel_count = (mask & selected).bit_count()
                if sel_count == 3:
                    return False
                if sel_count == 2:
                    rem = mask & candidates
                    if rem:
                        candidates &= ~rem
                        forbidden |= rem
                        changed = True
        return True

    def solve_mask(selected, candidates, forbidden):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
        
        if not propagate_mask(selected, candidates, forbidden):
            return
            
        if candidates == 0:
            size = selected.bit_count()
            if size > best_size:
                best_size = size
                s = set()
                for i in range(128):
                    if (selected >> i) & 1:
                        s.add(i + 1)
                best_set = s
            return
            
        if selected.bit_count() + candidates.bit_count() <= best_size:
            return
            
        # Heuristic: Pick vertex with max active degree
        pivot = -1
        max_active_deg = -1
        
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            
            active_deg = 0
            for t_idx in v_to_t_idx[v]:
                mask = triple_masks[t_idx]
                if (mask & selected).bit_count() < 2 and (mask & forbidden) == 0:
                    active_deg += 1
                        
            if active_deg > max_active_deg:
                max_active_deg = active_deg
                pivot = v
                
            c ^= lsb
            
        if pivot == -1:
            return
            
        bit = 1 << pivot
        
        # Branch 1: Select
        solve_mask(selected | bit, candidates & ~bit, forbidden)
        
        # Branch 2: Exclude
        solve_mask(selected, candidates & ~bit, forbidden | bit)

    # Run B&B
    all_candidates = (1 << 128) - 1
    solve_mask(0, all_candidates, 0)
    
    if not best_set:
        return [1]
        
    return list(best_set)
