<<<<<<< SEARCH
"""Improved covering design C(24,7,3): fast vectorized greedy + conflict-graph redundancy removal + local search."""
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    # Precompute all t-subsets
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_to_id = {ts: i for i, ts in enumerate(all_tsets)}

    # Precompute for each t-set, the list of (t-1)-subsets it contains
    tset_to_subs = {}
    for ts in all_tsets:
        tset_to_subs[ts] = list(itertools.combinations(ts, t - 1))

    # Precompute: for each (t-1)-subset, which t-sets contain it
    sub_to_tsets = {}
    for ts in all_tsets:
        for sub in tset_to_subs[ts]:
            if sub not in sub_to_tsets:
                sub_to_tsets[sub] = []
            sub_to_tsets[sub].append(ts)

    # For each (t-1)-subset, precompute the t-set IDs
    sub_to_tset_ids = {}
    for sub, tsets in sub_to_tsets.items():
        sub_to_tset_ids[sub] = np.array([tset_to_id[ts] for ts in tsets], dtype=np.int32)

    # Precompute all 7-subsets of 24 and their t-set ID arrays for fast lookup
    # There are C(24,7) = 346104 possible blocks; precompute all of them
    # Actually that's too many. Instead, cache on demand.
    _block_tset_cache = {}

    def block_tset_ids(block):
        key = tuple(block)
        if key in _block_tset_cache:
            return _block_tset_cache[key]
        ids = [tset_to_id[ts] for ts in itertools.combinations(block, t)]
        _block_tset_cache[key] = ids
        return ids

    def block_tset_ids_array(block):
        return np.array(block_tset_ids(block), dtype=np.int32)

    def block_tset_set(block):
        """Return a frozenset of t-set IDs for this block (cached)."""
        key = tuple(block)
        if key not in _block_tset_cache:
            ids = [tset_to_id[ts] for ts in itertools.combinations(block, t)]
            _block_tset_cache[key] = ids
        return frozenset(_block_tset_cache[key])

    # Fast greedy: vectorized gain computation
    def greedy_construct(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsets, dtype=bool)
        blocks = []

        while uncovered.any():
            if time.time() > deadline - 60:
                break
            unc_idx = np.where(uncovered)[0]
            start_id = unc_idx[rng.randint(0, len(unc_idx) - 1)]
            start_ts = all_tsets[start_id]

            block = list(start_ts)
            block_set = set(block)

            while len(block) < k:
                candidates = [p for p in range(v) if p not in block_set]
                subs_in_block = list(itertools.combinations(block, t - 1))

                # Vectorized gain: for each candidate, sum uncovered counts over (t-1)-subs
                best_point = -1
                max_gain = -1
                for p in candidates:
                    gain = 0
                    for sub in subs_in_block:
                        ts = tuple(sorted(sub + (p,)))
                        tid = tset_to_id[ts]
                        if uncovered[tid]:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_point = p

                if max_gain == 0:
                    best_point = rng.choice(candidates)

                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            blocks.append(block)
            for tid in block_tset_ids(block):
                uncovered[tid] = False

        return blocks

    # Phase 1: Multiple greedy restarts
    best_blocks = None
    best_count = float('inf')

    for seed in range(40):
        if time.time() > deadline - 50:
            break
        blocks = greedy_construct(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_construct(0)

    # Phase 2: Local search with redundancy detection
    count = np.zeros(n_tsets, dtype=np.int32)

    def update_counts(blocks_list, delta=1):
        for block in blocks_list:
            for tid in block_tset_ids(block):
                count[tid] += delta

    update_counts(best_blocks, 1)
    current_blocks = list(best_blocks)
    current_count = len(current_blocks)

    def is_redundant(block):
        for tid in block_tset_ids(block):
            if count[tid] == 1:
                return False
        return True

    def try_remove_redundant(blocks):
        """Remove all blocks whose every t-set is covered by at least one other block.
        Uses a batch approach: mark redundant blocks, then remove all at once."""
        n = len(blocks)
        # Precompute t-set sets for all blocks
        block_sets = [block_tset_set(b) for b in blocks]
        
        # A block is redundant if for every t-set in it, count[tid] >= 2
        # i.e., no t-set is uniquely covered by this block
        redundant = []
        for i, b in enumerate(blocks):
            ts_set = block_sets[i]
            all_covered = True
            for tid in ts_set:
                if count[tid] == 1:
                    all_covered = False
                    break
            if all_covered:
                redundant.append(i)
        
        if not redundant:
            return blocks, False
        
        # Remove all redundant blocks at once
        redundant_set = set(redundant)
        new_blocks = [b for i, b in enumerate(blocks) if i not in redundant_set]
        for i in redundant:
            for tid in block_sets[i]:
                count[tid] -= 1
        
        return new_blocks, True

    def try_swap_2_for_1(blocks):
        n = len(blocks)
        if n < 2:
            return blocks, False

        # Precompute t-set sets for all blocks
        block_sets = [block_tset_set(b) for b in blocks]

        for i in range(n):
            if time.time() > deadline - 20:
                return blocks, False
            for j in range(i + 1, n):
                ids_b1 = block_sets[i]
                ids_b2 = block_sets[j]
                check_ids = ids_b1 | ids_b2

                u_need = []
                for tid in check_ids:
                    c = count[tid]
                    cov_b1 = 1 if tid in ids_b1 else 0
                    cov_b2 = 1 if tid in ids_b2 else 0
                    if c == cov_b1 + cov_b2:
                        u_need.append(all_tsets[tid])

                if not u_need:
                    # Both blocks are fully redundant together
                    for tid in ids_b1:
                        count[tid] -= 1
                    for tid in ids_b2:
                        count[tid] -= 1
                    return [b for idx, b in enumerate(blocks) if idx != i and idx != j], True

                p_need = set()
                for ts in u_need:
                    p_need.update(ts)

                if len(p_need) <= k:
                    new_block = list(p_need)
                    rng = random.Random(i * 1000 + j)
                    while len(new_block) < k:
                        p = rng.randint(0, v - 1)
                        if p not in new_block:
                            new_block.append(p)
                    new_block.sort()

                    # Update counts
                    for tid in ids_b1:
                        count[tid] -= 1
                    for tid in ids_b2:
                        count[tid] -= 1
                    for tid in block_tset_ids(new_block):
                        count[tid] += 1
                    new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                    new_blocks.append(new_block)
                    return new_blocks, True

        return blocks, False

    def try_swap_1_for_1(blocks):
        n = len(blocks)
        if n < 1:
            return blocks, False

        for i in range(n):
            if time.time() > deadline - 20:
                return blocks, False

            b = blocks[i]
            ids_b = block_tset_set(b)

            u_need = []
            for tid in ids_b:
                if count[tid] == 1:
                    u_need.append(all_tsets[tid])

            if not u_need:
                # Block is fully redundant
                for tid in ids_b:
                    count[tid] -= 1
                return [x for idx, x in enumerate(blocks) if idx != i], True

            p_need = set()
            for ts in u_need:
                p_need.update(ts)

            if len(p_need) <= k:
                new_block = list(p_need)
                rng = random.Random(i * 7919)
                while len(new_block) < k:
                    p = rng.randint(0, v - 1)
                    if p not in new_block:
                        new_block.append(p)
                new_block.sort()

                if new_block != b:
                    for tid in ids_b:
                        count[tid] -= 1
                    for tid in block_tset_ids(new_block):
                        count[tid] += 1
                    new_blocks = [x for idx, x in enumerate(blocks) if idx != i]
                    new_blocks.append(new_block)
                    return new_blocks, True

        return blocks, False

    # Conflict-graph based batch redundancy removal
    def conflict_graph_removal(blocks):
        """Build a conflict graph where edge (i,j) means block i's unique triples
        are all covered by block j. Find maximal sets of mutually-redundant blocks
        and remove them."""
        n = len(blocks)
        if n < 2:
            return blocks, False

        block_sets = [block_tset_set(b) for b in blocks]
        
        # For each block, find which other blocks cover all of its unique triples
        # A block is "covered by" block j if every t-set in block i that has count==1
        # is also in block j
        can_remove = [True] * n  # Initially all can be removed if redundant
        
        # First pass: identify blocks that are individually redundant
        individually_redundant = []
        for i in range(n):
            ts_set = block_sets[i]
            is_red = True
            for tid in ts_set:
                if count[tid] == 1:
                    is_red = False
                    break
            if is_red:
                individually_redundant.append(i)
            else:
                can_remove[i] = False
        
        if not individually_redundant:
            return blocks, False
        
        # Remove all individually redundant blocks
        red_set = set(individually_redundant)
        new_blocks = [b for i, b in enumerate(blocks) if i not in red_set]
        for i in individually_redundant:
            for tid in block_sets[i]:
                count[tid] -= 1
        
        return new_blocks, True

    # Global greedy refinement: try to replace a block with a better one
    def global_greedy_refine(blocks):
        """Try to improve by replacing individual blocks with better alternatives."""
        n = len(blocks)
        improved = False
        for i in range(n):
            if time.time() > deadline - 25:
                break
            b = blocks[i]
            ids_b = block_tset_set(b)
            # Find triples only covered by this block
            unique_triples = []
            for tid in ids_b:
                if count[tid] == 1:
                    unique_triples.append(all_tsets[tid])

            if not unique_triples:
                # Block is fully redundant
                for tid in ids_b:
                    count[tid] -= 1
                new_blocks = [x for idx, x in enumerate(blocks) if idx != i]
                return new_blocks, True

            # Try to find a better block that covers unique_triples and maximizes other coverage
            p_need = set()
            for ts in unique_triples:
                p_need.update(ts)

            if len(p_need) > k:
                continue

            # Try several candidate blocks containing p_need
            best_new_block = None
            best_gain = -1
            rng = random.Random(i * 131 + 17)
            for attempt in range(20):
                cand = list(p_need)
                while len(cand) < k:
                    p = rng.randint(0, v - 1)
                    if p not in cand:
                        cand.append(p)
                cand.sort()
                # Compute gain: how many additional triples does this cover?
                cand_ids = block_tset_set(cand)
                # Gain = triples in cand not in b (that are uncovered or have low count)
                gain = 0
                for tid in cand_ids:
                    if tid not in ids_b:
                        if count[tid] == 0:
                            gain += 1
                        elif count[tid] <= 2:
                            gain += 0.5
                if gain > best_gain:
                    best_gain = gain
                    best_new_block = cand

            if best_new_block is not None and best_gain > 0 and best_new_block != b:
                for tid in ids_b:
                    count[tid] -= 1
                for tid in block_tset_ids(best_new_block):
                    count[tid] += 1
                new_blocks = [x for idx, x in enumerate(blocks) if idx != i]
                new_blocks.append(best_new_block)
                return new_blocks, True

        return blocks, False

    # Iterative local search
    improved = True
    max_rounds = 300
    for _ in range(max_rounds):
        if time.time() > deadline - 40:
            break
        if not improved:
            break
        improved = False

        new_blocks, rem = try_remove_redundant(current_blocks)
        if rem:
            current_blocks = new_blocks
            improved = True

        new_blocks, swp = try_swap_2_for_1(current_blocks)
        if swp:
            current_blocks = new_blocks
            improved = True

        new_blocks, swp = try_swap_1_for_1(current_blocks)
        if swp:
            current_blocks = new_blocks
            improved = True

        new_blocks, swp = global_greedy_refine(current_blocks)
        if swp:
            current_blocks = new_blocks
            improved = True

        new_blocks, rem = conflict_graph_removal(current_blocks)
        if rem:
            current_blocks = new_blocks
            improved = True

    # Additional local search rounds after conflict graph removal
    if time.time() < deadline - 30:
        improved = True
        for _ in range(100):
            if time.time() > deadline - 35:
                break
            if not improved:
                break
            improved = False
            new_blocks, rem = try_remove_redundant(current_blocks)
            if rem:
                current_blocks = new_blocks
                improved = True
            new_blocks, swp = try_swap_2_for_1(current_blocks)
            if swp:
                current_blocks = new_blocks
                improved = True
            new_blocks, swp = try_swap_1_for_1(current_blocks)
            if swp:
                current_blocks = new_blocks
                improved = True
            new_blocks, rem = conflict_graph_removal(current_blocks)
            if rem:
                current_blocks = new_blocks
                improved = True

    # Final: try more aggressive 2-for-1 and 1-for-0 passes
    for _ in range(30):
        if time.time() > deadline - 15:
            break
        improved = False
        new_blocks, swp = try_swap_2_for_1(current_blocks)
        if swp:
            current_blocks = new_blocks
            improved = True
        new_blocks, rem = try_remove_redundant(current_blocks)
        if rem:
            current_blocks = new_blocks
            improved = True
        new_blocks, rem = conflict_graph_removal(current_blocks)
        if rem:
            current_blocks = new_blocks
            improved = True
        if not improved:
            break

    # Final verification
    covered = set()
    for block in current_blocks:
        for tid in block_tset_ids(block):
            covered.add(tid)

    if len(covered) < n_tsets:
        return best_blocks

    return current_blocks
=======
"""Improved covering design C(24,7,3): bitset-accelerated greedy + VNS perturbation + smooth local search."""
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s limit

    # Precompute all t-subsets
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_to_id = {ts: i for i, ts in enumerate(all_tsets)}

    # For each (t-1)-subset, precompute which t-sets contain it (as array of IDs)
    sub_to_tset_ids = {}
    for ts in all_tsets:
        for sub in itertools.combinations(ts, t - 1):
            if sub not in sub_to_tset_ids:
                sub_to_tset_ids[sub] = []
            sub_to_tset_ids[sub].append(tset_to_id[ts])
    for sub in sub_to_tset_ids:
        sub_to_tset_ids[sub] = np.array(sub_to_tset_ids[sub], dtype=np.int32)

    # Cache for block -> frozenset of t-set IDs
    _block_tset_cache = {}

    def block_tset_ids(block):
        key = tuple(block)
        if key in _block_tset_cache:
            return _block_tset_cache[key]
        ids = [tset_to_id[ts] for ts in itertools.combinations(block, t)]
        _block_tset_cache[key] = ids
        return ids

    def block_tset_set(block):
        key = tuple(block)
        if key not in _block_tset_cache:
            ids = [tset_to_id[ts] for ts in itertools.combinations(block, t)]
            _block_tset_cache[key] = ids
        return frozenset(_block_tset_cache[key])

    # Bitset-based greedy: use numpy boolean array for uncovered
    def greedy_construct(seed=0, perturb=False):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsets, dtype=bool)
        blocks = []

        while uncovered.any():
            if time.time() > deadline - 55:
                break

            # Pick a random uncovered t-set as seed
            unc_idx = np.where(uncovered)[0]
            start_id = unc_idx[rng.randint(0, len(unc_idx) - 1)]
            start_ts = all_tsets[start_id]

            block = list(start_ts)
            block_set = set(block)

            # Build (t-1)-subsets of current block incrementally
            # For each candidate point, compute gain via precomputed sub_to_tset_ids
            while len(block) < k:
                candidates = [p for p in range(v) if p not in block_set]
                subs_in_block = list(itertools.combinations(block, t - 1))

                # For each candidate, compute number of newly covered t-sets
                best_point = -1
                max_gain = -1
                for p in candidates:
                    gain = 0
                    for sub in subs_in_block:
                        ts = tuple(sorted(sub + (p,)))
                        tid = tset_to_id[ts]
                        if uncovered[tid]:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_point = p

                if max_gain == 0:
                    # No gain; pick randomly to diversify
                    best_point = rng.choice(candidates)

                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            blocks.append(block)
            for tid in block_tset_ids(block):
                uncovered[tid] = False

        # Optional perturbation for VNS: remove some blocks and rebuild
        if perturb and len(blocks) > 5:
            n_remove = max(1, len(blocks) // 20)
            for _ in range(n_remove):
                idx = rng.randint(0, len(blocks) - 1)
                b = blocks[idx]
                for tid in block_tset_ids(b):
                    uncovered[tid] = True
                blocks.pop(idx)
            # Rebuild
            while uncovered.any():
                if time.time() > deadline - 55:
                    break
                unc_idx = np.where(uncovered)[0]
                start_id = unc_idx[rng.randint(0, len(unc_idx) - 1)]
                start_ts = all_tsets[start_id]
                block = list(start_ts)
                block_set = set(block)
                while len(block) < k:
                    candidates = [p for p in range(v) if p not in block_set]
                    subs_in_block = list(itertools.combinations(block, t - 1))
                    best_point = -1
                    max_gain = -1
                    for p in candidates:
                        gain = 0
                        for sub in subs_in_block:
                            ts = tuple(sorted(sub + (p,)))
                            tid = tset_to_id[ts]
                            if uncovered[tid]:
                                gain += 1
                        if gain > max_gain:
                            max_gain = gain
                            best_point = p
                    if max_gain == 0:
                        best_point = rng.choice(candidates)
                    block.append(best_point)
                    block_set.add(best_point)
                block.sort()
                blocks.append(block)
                for tid in block_tset_ids(block):
                    uncovered[tid] = False

        return blocks

    # Phase 1: Multiple greedy restarts with some perturbed
    best_blocks = None
    best_count = float('inf')

    n_restarts = 50
    for seed in range(n_restarts):
        if time.time() > deadline - 50:
            break
        perturb = (seed % 5 == 4)  # every 5th is perturbed
        blocks = greedy_construct(seed, perturb=perturb)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_construct(0)

    # Phase 2: Local search with count-based redundancy
    count = np.zeros(n_tsets, dtype=np.int32)

    def update_counts(blocks_list, delta=1):
        for block in blocks_list:
            for tid in block_tset_ids(block):
                count[tid] += delta

    update_counts(best_blocks, 1)
    current_blocks = list(best_blocks)

    def try_remove_redundant(blocks):
        """Remove all blocks whose every t-set is covered by at least one other block."""
        n = len(blocks)
        block_sets = [block_tset_set(b) for b in blocks]

        redundant = []
        for i, b in enumerate(blocks):
            ts_set = block_sets[i]
            all_covered = True
            for tid in ts_set:
                if count[tid] == 1:
                    all_covered = False
                    break
            if all_covered:
                redundant.append(i)

        if not redundant:
            return blocks, False

        redundant_set = set(redundant)
        new_blocks = [b for i, b in enumerate(blocks) if i not in redundant_set]
        for i in redundant:
            for tid in block_sets[i]:
                count[tid] -= 1

        return new_blocks, True

    def try_swap_2_for_1(blocks):
        """Try to replace two blocks with one that covers all their unique t-sets."""
        n = len(blocks)
        if n < 2:
            return blocks, False

        block_sets = [block_tset_set(b) for b in blocks]

        for i in range(n):
            if time.time() > deadline - 20:
                return blocks, False
            for j in range(i + 1, n):
                ids_b1 = block_sets[i]
                ids_b2 = block_sets[j]
                check_ids = ids_b1 | ids_b2

                u_need = []
                for tid in check_ids:
                    c = count[tid]
                    cov_b1 = 1 if tid in ids_b1 else 0
                    cov_b2 = 1 if tid in ids_b2 else 0
                    if c == cov_b1 + cov_b2:
                        u_need.append(all_tsets[tid])

                if not u_need:
                    # Both blocks are fully redundant together
                    for tid in ids_b1:
                        count[tid] -= 1
                    for tid in ids_b2:
                        count[tid] -= 1
                    return [b for idx, b in enumerate(blocks) if idx != i and idx != j], True

                p_need = set()
                for ts in u_need:
                    p_need.update(ts)

                if len(p_need) <= k:
                    # Try to build a good replacement block
                    new_block = list(p_need)
                    rng = random.Random(i * 1000 + j)
                    # Fill remaining slots with points that maximize coverage
                    remaining = [p for p in range(v) if p not in new_block]
                    while len(new_block) < k and remaining:
                        best_p = -1
                        best_gain = -1
                        for p in remaining:
                            cand = new_block + [p]
                            cand_set = set(cand)
                            # Count how many new t-sets this would cover
                            gain = 0
                            for ts in itertools.combinations(cand, t):
                                tid = tset_to_id[ts]
                                if count[tid] == 0:
                                    gain += 2
                                elif count[tid] == 1:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_p = p
                        if best_p == -1 or best_gain <= 0:
                            best_p = rng.choice(remaining)
                        new_block.append(best_p)
                        remaining.remove(best_p)
                    new_block.sort()

                    # Update counts
                    for tid in ids_b1:
                        count[tid] -= 1
                    for tid in ids_b2:
                        count[tid] -= 1
                    for tid in block_tset_ids(new_block):
                        count[tid] += 1
                    new_blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                    new_blocks.append(new_block)
                    return new_blocks, True

        return blocks, False

    def try_swap_1_for_1(blocks):
        """Try to replace a block with a better one that covers its unique t-sets."""
        n = len(blocks)
        if n < 1:
            return blocks, False

        for i in range(n):
            if time.time() > deadline - 20:
                return blocks, False

            b = blocks[i]
            ids_b = block_tset_set(b)

            u_need = []
            for tid in ids_b:
                if count[tid] == 1:
                    u_need.append(all_tsets[tid])

            if not u_need:
                # Block is fully redundant
                for tid in ids_b:
                    count[tid] -= 1
                return [x for idx, x in enumerate(blocks) if idx != i], True

            p_need = set()
            for ts in u_need:
                p_need.update(ts)

            if len(p_need) <= k:
                # Try multiple candidate replacements
                best_new_block = None
                best_gain = -1
                rng = random.Random(i * 7919)
                for attempt in range(15):
                    cand = list(p_need)
                    remaining = [p for p in range(v) if p not in cand]
                    while len(cand) < k and remaining:
                        best_p = -1
                        best_p_gain = -1
                        for p in remaining:
                            temp_cand = cand + [p]
                            gain = 0
                            for ts in itertools.combinations(temp_cand, t):
                                tid = tset_to_id[ts]
                                if tid not in ids_b:
                                    if count[tid] == 0:
                                        gain += 2
                                    elif count[tid] == 1:
                                        gain += 1
                            if gain > best_p_gain:
                                best_p_gain = gain
                                best_p = p
                        if best_p == -1 or best_p_gain <= 0:
                            best_p = rng.choice(remaining)
                        cand.append(best_p)
                        remaining.remove(best_p)
                    cand.sort()
                    if cand != b:
                        cand_ids = block_tset_set(cand)
                        gain = 0
                        for tid in cand_ids:
                            if tid not in ids_b:
                                if count[tid] == 0:
                                    gain += 2
                                elif count[tid] == 1:
                                    gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_new_block = cand

                if best_new_block is not None and best_gain > 0:
                    for tid in ids_b:
                        count[tid] -= 1
                    for tid in block_tset_ids(best_new_block):
                        count[tid] += 1
                    new_blocks = [x for idx, x in enumerate(blocks) if idx != i]
                    new_blocks.append(best_new_block)
                    return new_blocks, True

        return blocks, False

    # VNS-style: multiple neighborhood structures
    def vns_local_search(blocks, max_iter=200):
        """Variable neighborhood search with multiple neighborhood structures."""
        n_blocks = len(blocks)
        best = list(blocks)
        best_len = len(blocks)
        rng = random.Random(42)

        for it in range(max_iter):
            if time.time() > deadline - 35:
                break

            # Neighborhood 1: single block replacement
            new_b, changed = try_swap_1_for_1(best)
            if changed and len(new_b) < best_len:
                best = new_b
                best_len = len(best)
                continue

            # Neighborhood 2: pair-to-one
            new_b, changed = try_swap_2_for_1(best)
            if changed and len(new_b) < best_len:
                best = new_b
                best_len = len(best)
                continue

            # Neighborhood 3: redundancy removal
            new_b, changed = try_remove_redundant(best)
            if changed:
                best = new_b
                best_len = len(best)
                continue

            # Perturbation: remove a random block and re-add greedily
            if len(best) > 3:
                idx = rng.randint(0, len(best) - 1)
                removed_block = best[idx]
                for tid in block_tset_ids(removed_block):
                    count[tid] -= 1
                best.pop(idx)

                # Re-add greedily
                uncovered = np.zeros(n_tsets, dtype=bool)
                for tid in range(n_tsets):
                    if count[tid] == 0:
                        uncovered[tid] = True

                if uncovered.any():
                    unc_idx = np.where(uncovered)[0]
                    start_id = unc_idx[rng.randint(0, len(unc_idx) - 1)]
                    start_ts = all_tsets[start_id]
                    new_block = list(start_ts)
                    block_set = set(new_block)
                    while len(new_block) < k:
                        candidates = [p for p in range(v) if p not in block_set]
                        subs_in_block = list(itertools.combinations(new_block, t - 1))
                        best_p = -1
                        max_gain = -1
                        for p in candidates:
                            gain = 0
                            for sub in subs_in_block:
                                ts = tuple(sorted(sub + (p,)))
                                tid = tset_to_id[ts]
                                if uncovered[tid]:
                                    gain += 1
                            if gain > max_gain:
                                max_gain = gain
                                best_p = p
                        if max_gain == 0:
                            best_p = rng.choice(candidates)
                        new_block.append(best_p)
                        block_set.add(best_p)
                    new_block.sort()
                    best.append(new_block)
                    for tid in block_tset_ids(new_block):
                        count[tid] += 1
                        if count[tid] > 0:
                            uncovered[tid] = False

                # Local polish
                for _ in range(10):
                    if time.time() > deadline - 35:
                        break
                    new_b, c1 = try_remove_redundant(best)
                    if c1:
                        best = new_b
                    new_b, c2 = try_swap_1_for_1(best)
                    if c2:
                        best = new_b
                    if not (c1 or c2):
                        break

                if len(best) < best_len:
                    best_len = len(best)
                else:
                    # Revert perturbation if it didn't help
                    pass  # Already modified in place; accept as-is

        return best

    # Run local search phases
    improved = True
    for _ in range(100):
        if time.time() > deadline - 45:
            break
        if not improved:
            break
        improved = False

        new_blocks, rem = try_remove_redundant(current_blocks)
        if rem:
            current_blocks = new_blocks
            improved = True

        new_blocks, swp = try_swap_1_for_1(current_blocks)
        if swp:
            current_blocks = new_blocks
            improved = True

        new_blocks, swp = try_swap_2_for_1(current_blocks)
        if swp:
            current_blocks = new_blocks
            improved = True

    # VNS phase
    if time.time() < deadline - 40:
        current_blocks = vns_local_search(current_blocks, max_iter=150)

    # Final polish
    for _ in range(50):
        if time.time() > deadline - 25:
            break
        improved = False
        new_blocks, rem = try_remove_redundant(current_blocks)
        if rem:
            current_blocks = new_blocks
            improved = True
        new_blocks, swp = try_swap_1_for_1(current_blocks)
        if swp:
            current_blocks = new_blocks
            improved = True
        new_blocks, swp = try_swap_2_for_1(current_blocks)
        if swp:
            current_blocks = new_blocks
            improved = True
        if not improved:
            break

    # Final verification
    covered = set()
    for block in current_blocks:
        for tid in block_tset_ids(block):
            covered.add(tid)

    if len(covered) < n_tsets:
        # Fallback to best found
        return best_blocks

    return current_blocks
>>>>>>> REPLACE
