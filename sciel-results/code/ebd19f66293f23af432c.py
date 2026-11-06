import time

def construct():
    start_time = time.time()
    timeout = 82.0  # safety margin before 90s kill

    n = 256

    # Build conflict masks: conflict[i] has bit j set if (i+1) and (j+1) conflict
    conflict = [0] * n
    for i in range(n):
        a = i + 1
        mask_i = 0
        for j in range(n):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                mask_i |= (1 << j)
        conflict[i] = mask_i

    # Precompute degrees and reorder vertices by degree descending
    degrees = [conflict[i].bit_count() for i in range(n)]
    order = sorted(range(n), key=lambda i: degrees[i], reverse=True)
    pos_in_order = [0] * n
    for idx, orig in enumerate(order):
        pos_in_order[orig] = idx

    # Reorder conflict masks to match the new ordering
    # new_conflict[i] corresponds to original vertex order[i]
    new_conflict = [0] * n
    for i in range(n):
        orig_i = order[i]
        mask = 0
        for j in range(n):
            if conflict[orig_i] & (1 << j):
                mask |= (1 << pos_in_order[j])
        new_conflict[i] = mask

    best_size = [0]
    best_mask = [0]

    def color_bound(candidates):
        """Greedy coloring of complement of conflict graph on candidates.
        Returns number of colors (upper bound on independent set size in candidates)."""
        uncolored = candidates
        colors = 0
        while uncolored:
            colors += 1
            available = uncolored
            chosen = 0
            while available:
                lsb = available & -available
                i = lsb.bit_length() - 1
                chosen |= lsb
                available &= ~(new_conflict[i] | lsb)
            uncolored &= ~chosen
        return colors

    def solve(candidates, current_size, current_mask):
        if time.time() - start_time > timeout:
            return

        if candidates == 0:
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_mask[0] = current_mask
            return

        # Upper bound via coloring
        ub = color_bound(candidates)
        if current_size + ub <= best_size[0]:
            return

        # Select vertex: pick from the last color class (highest color = most constrained)
        # Re-run coloring to get the last class
        uncolored = candidates
        last_class = 0
        while uncolored:
            available = uncolored
            chosen = 0
            while available:
                lsb = available & -available
                i = lsb.bit_length() - 1
                chosen |= lsb
                available &= ~(new_conflict[i] | lsb)
            last_class = chosen
            uncolored &= ~chosen

        # Pick a vertex from last_class
        # Choose the one with max degree in candidates for better pruning
        v_idx = -1
        max_deg = -1
        c = last_class
        while c:
            lsb = c & -c
            i = lsb.bit_length() - 1
            deg = (new_conflict[i] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                v_idx = i
            c ^= lsb

        if v_idx == -1:
            return

        v_bit = 1 << v_idx

        # Branch 1: Include v_idx
        new_cand_inc = (candidates & ~v_bit) & ~new_conflict[v_idx]
        solve(new_cand_inc, current_size + 1, current_mask | v_bit)

        # Branch 2: Exclude v_idx
        solve(candidates & ~v_bit, current_size, current_mask)

    initial_cand = (1 << n) - 1
    solve(initial_cand, 0, 0)

    if best_size[0] == 0:
        return [1]

    result = []
    for i in range(n):
        if (best_mask[0] >> i) & 1:
            result.append(order[i] + 1)

    return result
