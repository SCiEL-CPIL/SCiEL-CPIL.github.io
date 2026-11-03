import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0  # Safety margin

    def elapsed():
        return time.time() - start_time

    # 1. Precomputation
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}

    all_blocks = []
    block_masks = []
    for block in itertools.combinations(range(v), k):
        mask = 0
        for ts in itertools.combinations(block, t):
            mask |= (1 << tset_index[ts])
        all_blocks.append(block)
        block_masks.append(mask)

    num_blocks = len(all_blocks)
    full_mask = (1 << num_tsets) - 1

    # Build tset_to_blocks for fast candidate lookup
    tset_to_blocks = [[] for _ in range(num_tsets)]
    for b_idx, mask in enumerate(block_masks):
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            tset_to_blocks[idx].append(b_idx)
            m ^= lsb

    # 2. Proper Greedy Set Cover
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        used = [False] * num_blocks

        while uncovered:
            if elapsed() > deadline - 15:
                break
            best_idx = -1
            best_count = -1

            for i in range(num_blocks):
                if used[i]:
                    continue
                count = (block_masks[i] & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = i

            if best_idx == -1 or best_count == 0:
                break

            used[best_idx] = True
            selected.append(best_idx)
            uncovered &= ~block_masks[best_idx]

        return selected

    # 3. Local Search with 1-for-1 and 2-for-1 swaps
    def local_search(initial_sol, seed):
        rng = random.Random(seed)
        sol = list(initial_sol)
        in_sol = [False] * num_blocks
        for b in sol:
            in_sol[b] = True

        cover_counts = [0] * num_tsets

        def update_counts(block_idx, delta):
            m = block_masks[block_idx]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_counts[idx] += delta
                m ^= lsb

        for b in sol:
            update_counts(b, 1)

        best_sol = list(sol)
        best_size = len(sol)
        no_improve = 0
        max_no_improve = 120

        while elapsed() < deadline - 5 and no_improve < max_no_improve:
            improved = False

            # Phase 1: Remove redundant blocks (repeat until none found)
            removed_any = True
            while removed_any and elapsed() < deadline - 5:
                removed_any = False
                order = list(range(len(sol)))
                rng.shuffle(order)
                for pos in order:
                    if pos >= len(sol):
                        continue
                    b = sol[pos]
                    m = block_masks[b]
                    redundant = True
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cover_counts[idx] == 1:
                            redundant = False
                            break
                        m ^= lsb

                    if redundant:
                        update_counts(b, -1)
                        in_sol[b] = False
                        sol.pop(pos)
                        removed_any = True
                        if len(sol) < best_size:
                            best_size = len(sol)
                            best_sol = list(sol)
                        break

            if removed_any:
                improved = True
                no_improve = 0
                continue

            # Phase 2: 1-for-1 Swap
            if sol:
                found_swap = False
                for _ in range(20):
                    if not sol or elapsed() > deadline - 5:
                        break
                    pos = rng.randint(0, len(sol) - 1)
                    b_out = sol[pos]

                    critical_mask = 0
                    m = block_masks[b_out]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cover_counts[idx] == 1:
                            critical_mask |= lsb
                        m ^= lsb

                    if critical_mask == 0:
                        update_counts(b_out, -1)
                        in_sol[b_out] = False
                        sol.pop(pos)
                        improved = True
                        found_swap = True
                        break

                    # Find candidate blocks covering all critical t-sets
                    crit_ts_list = []
                    m = critical_mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        crit_ts_list.append(idx)
                        m ^= lsb

                    if not crit_ts_list:
                        continue

                    # Intersect tset_to_blocks for each critical t-set
                    candidates = set(tset_to_blocks[crit_ts_list[0]])
                    for ts_idx in crit_ts_list[1:]:
                        if not candidates:
                            break
                        candidates &= set(tset_to_blocks[ts_idx])

                    # Filter out blocks already in solution
                    candidates = {c for c in candidates if not in_sol[c]}

                    if candidates:
                        b_in = rng.choice(list(candidates))
                        update_counts(b_out, -1)
                        in_sol[b_out] = False
                        update_counts(b_in, 1)
                        in_sol[b_in] = True
                        sol[pos] = b_in
                        found_swap = True
                        improved = True
                        break

                if found_swap:
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    no_improve = 0
                    continue
            else:
                break

            # Phase 3: 2-for-1 Swap (remove 2 blocks, add 1 block covering their union of critical t-sets)
            if len(sol) >= 2:
                found_21 = False
                for _ in range(15):
                    if elapsed() > deadline - 5:
                        break
                    pos1 = rng.randint(0, len(sol) - 1)
                    pos2 = rng.randint(0, len(sol) - 1)
                    if pos1 == pos2:
                        continue

                    b_out1 = sol[pos1]
                    b_out2 = sol[pos2]

                    # Compute union of critical t-sets from both blocks
                    critical_mask = 0
                    m = block_masks[b_out1]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cover_counts[idx] == 1:
                            critical_mask |= lsb
                        m ^= lsb

                    m = block_masks[b_out2]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cover_counts[idx] == 1:
                            critical_mask |= lsb
                        m ^= lsb

                    if critical_mask == 0:
                        # Both blocks are redundant
                        update_counts(b_out1, -1)
                        in_sol[b_out1] = False
                        update_counts(b_out2, -1)
                        in_sol[b_out2] = False
                        # Remove in reverse order to preserve indices
                        if pos1 > pos2:
                            pos1, pos2 = pos2, pos1
                        sol.pop(pos2)
                        sol.pop(pos1)
                        improved = True
                        found_21 = True
                        break

                    # Find candidate blocks covering all critical t-sets
                    crit_ts_list = []
                    m = critical_mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        crit_ts_list.append(idx)
                        m ^= lsb

                    if not crit_ts_list:
                        continue

                    # Intersect tset_to_blocks for each critical t-set
                    candidates = set(tset_to_blocks[crit_ts_list[0]])
                    for ts_idx in crit_ts_list[1:]:
                        if not candidates:
                            break
                        candidates &= set(tset_to_blocks[ts_idx])

                    # Filter out blocks already in solution
                    candidates = {c for c in candidates if not in_sol[c]}

                    if candidates:
                        b_in = rng.choice(list(candidates))
                        update_counts(b_out1, -1)
                        in_sol[b_out1] = False
                        update_counts(b_out2, -1)
                        in_sol[b_out2] = False
                        # Remove in reverse order to preserve indices
                        if pos1 > pos2:
                            pos1, pos2 = pos2, pos1
                        sol.pop(pos2)
                        sol.pop(pos1)
                        update_counts(b_in, 1)
                        in_sol[b_in] = True
                        sol.append(b_in)
                        found_21 = True
                        improved = True
                        break

                if found_21:
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    no_improve = 0
                    continue

            no_improve += 1

        return best_sol

    # Main Loop
    best_solution = None
    best_size = float('inf')

    # Run greedy restarts with local search
    num_restarts = 8
    for seed in range(num_restarts):
        if elapsed() > deadline - 20:
            break
        sol = greedy_cover(seed)
        if sol:
            ls_sol = local_search(sol, seed)
            if len(ls_sol) < best_size:
                best_size = len(ls_sol)
                best_solution = ls_sol

    if best_solution is None:
        return []

    return [list(all_blocks[i]) for i in best_solution]
