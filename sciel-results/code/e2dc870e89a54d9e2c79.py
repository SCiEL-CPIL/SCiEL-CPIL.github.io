import time

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s/90s limit

    best_seq = []
    best_len = 0

    # Greedy initialization to get a decent starting point
    def greedy_init():
        vals = []
        sums = set()
        for x in range(1, 257):
            new_sums = [x]
            total = x
            for a in reversed(vals):
                total += a
                new_sums.append(total)
            if not sums.intersection(new_sums):
                vals.append(x)
                sums.update(new_sums)
        return vals

    initial_seq = greedy_init()
    if initial_seq:
        best_seq = initial_seq.copy()
        best_len = len(initial_seq)

    # DFS with maintained suffix sums for O(n) new-sum computation
    # suffix_sums[i] = seq[-1] + seq[-2] + ... + seq[-i] (suffix of length i)
    # When adding x, new sums are: x, x+suffix_sums[1], x+suffix_sums[2], ...
    
    def dfs(seq, sums_mask, last_val, suffix_sums):
        nonlocal best_seq, best_len, start_time, time_limit

        cur_len = len(seq)
        if cur_len > best_len:
            best_seq = seq.copy()
            best_len = cur_len

        # Time check
        if time.time() - start_time > time_limit:
            return

        # Pruning: upper bound on how many more elements we can add
        # With n total elements, we need n(n+1)/2 distinct sums in [1, 256*n]
        # So n(n+1)/2 <= 256*n => n+1 <= 512 => n <= 511
        # But we're constrained by values in [1,256], so max length is 256
        # Better bound: if current length + remaining_candidates < best_len, prune
        remaining = 256 - last_val
        if cur_len + remaining <= best_len:
            return

        # Try candidates from large to small (larger values leave more room)
        # But we also want to keep sums small to avoid collisions
        # Compromise: try a mix, but bias toward larger
        # Actually, let's try from last_val+1 to 256, but we can reorder
        # For now, try ascending (smaller first keeps sums lower)
        # But inspiration says larger first is better. Let's try descending.
        
        for x in range(256, last_val, -1):
            # Compute new sums mask
            new_mask = 1 << x
            total = x
            for ss in suffix_sums:
                total = ss + x
                new_mask |= (1 << total)

            if sums_mask & new_mask:
                continue

            # Build new suffix sums
            # Old suffix_sums = [s1, s2, ..., sk] where si = sum of last i elements
            # New suffix sums: [x, x+s1, x+s2, ..., x+sk]
            new_suffix_sums = [x] + [ss + x for ss in suffix_sums]

            dfs(seq + [x], sums_mask | new_mask, x, new_suffix_sums)

            if time.time() - start_time > time_limit:
                return

    # Start DFS from empty
    dfs([], 0, 0, [])

    return best_seq
