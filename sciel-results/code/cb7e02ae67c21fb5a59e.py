import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit

    # Precompute
    t_subsets = list(itertools.combinations(range(v), t))
    t_subsets_set = {tuple(s): i for i, s in enumerate(t_subsets)}
    num_tsubs = len(t_subsets)
    k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(k_subsets)

    # For each block, list of t-subset indices it covers
    block_to_tsubs = []
    for block in k_subsets:
        covered = []
        for tsub in itertools.combinations(block, t):
            covered.append(t_subsets_set[tsub])
        block_to_tsubs.append(covered)

    # For each t-subset, list of blocks covering it
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for i, tsubs in enumerate(block_to_tsubs):
        for s in tsubs:
            tsub_to_blocks[s].append(i)
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    tsub_to_blocks_set = [set(lst) for lst in tsub_to_blocks]

    k_choose_t = len(block_to_tsubs[0])

    # ---------- CORE-PERIPHERY PHASE 1: Low-collision greedy core ----------
    # Goal: pick a "core" of blocks that collectively cover most t-subsets
    # while minimizing pairwise 4-overlap (i.e., each new block should
    # overlap little with the union of t-subsets already covered by core).
    # This yields a sparse, low-redundancy core that is easier to polish.

    covered_mask = np.zeros(num_tsubs, dtype=bool)  # t-subsets covered by core
    core_blocks = []

    # We'll pick blocks greedily: at each step, choose the block that
    # covers the most *new* t-subsets (standard greedy), but with a
    # secondary criterion: prefer blocks with fewer shared t-subsets
    # with already-selected core blocks (lower collision).
    #
    # To make this efficient, we maintain a gain array and a collision
    # penalty. The effective score = gain - lambda * collision_count.
    # We use a small lambda so the primary objective is still coverage.

    # Precompute block-block overlap counts? Too expensive (C(19,7)^2 ~ 50M).
    # Instead, we track for each t-subset how many core blocks cover it,
    # and use that to compute collision on the fly.

    core_cover_count = np.zeros(num_tsubs, dtype=np.int32)
    gains = np.full(num_blocks, k_choose_t, dtype=np.int32)

    # Phase 1: Build core until coverage reaches a threshold or time runs out
    core_time_budget = 40  # seconds
    core_deadline = start_time + core_time_budget

    while time.time() < core_deadline:
        if covered_mask.all():
            break
        b = int(np.argmax(gains))
        if gains[b] == 0:
            break
        core_blocks.append(b)
        for s in block_to_tsubs[b]:
            if not covered_mask[s]:
                covered_mask[s] = True
                core_cover_count[s] += 1
                # Update gains for all blocks covering this t-subset
                gains[tsub_to_blocks_np[s]] -= 1

    # ---------- PHASE 2: Repair residual gaps ----------
    # Now cover remaining uncovered t-subsets. Use a more aggressive
    # greedy that doesn't care about collision (just max coverage).
    uncovered = ~covered_mask
    repair_blocks = []

    # Reset gains to count only uncovered t-subsets
    gains2 = np.zeros(num_blocks, dtype=np.int32)
    for b in range(num_blocks):
        cnt = 0
        for s in block_to_tsubs[b]:
            if uncovered[s]:
                cnt += 1
        gains2[b] = cnt

    repair_deadline = deadline - 50  # Leave 50s for local search
    while time.time() < repair_deadline:
        if not uncovered.any():
            break
        b = int(np.argmax(gains2))
        if gains2[b] == 0:
            break
        repair_blocks.append(b)
        for s in block_to_tsubs[b]:
            if uncovered[s]:
                uncovered[s] = False
                gains2[tsub_to_blocks_np[s]] -= 1

    # If still uncovered (shouldn't happen), force-complete
    while uncovered.any():
        best_b = -1
        best_gain = 0
        for b in range(num_blocks):
            g = 0
            for s in block_to_tsubs[b]:
                if uncovered[s]:
                    g += 1
            if g > best_gain:
                best_gain = g
                best_b = b
        if best_b < 0 or best_gain == 0:
            break
        repair_blocks.append(best_b)
        for s in block_to_tsubs[best_b]:
            if uncovered[s]:
                uncovered[s] = False

    selected_block_indices = core_blocks + repair_blocks

    # ---------- PHASE 3: Local Search Polish ----------
    # Build cover_count for the full selected set
    cover_count = np.zeros(num_tsubs, dtype=np.int32)
    for b in selected_block_indices:
        for s in block_to_tsubs[b]:
            cover_count[s] += 1

    def is_redundant(b_idx):
        for s in block_to_tsubs[b_idx]:
            if cover_count[s] == 1:
                return False
        return True

    def find_swap_in(tsubs_to_cover, exclude_set):
        if not tsubs_to_cover:
            return None
        first_s = tsubs_to_cover[0]
        candidates = tsub_to_blocks_set[first_s]
        for c in candidates:
            if c in exclude_set:
                continue
            covers_all = True
            for s in tsubs_to_cover[1:]:
                if c not in tsub_to_blocks_set[s]:
                    covers_all = False
                    break
            if covers_all:
                return c
        return None

    local_deadline = deadline - 5
    improved = True
    while improved and time.time() < local_deadline:
        improved = False

        # Step A: Redundancy removal
        redundant_blocks = []
        for b in selected_block_indices:
            if is_redundant(b):
                redundant_blocks.append(b)

        if redundant_blocks:
            for b in redundant_blocks:
                if b in selected_block_indices:
                    selected_block_indices.remove(b)
                    for s in block_to_tsubs[b]:
                        cover_count[s] -= 1
            improved = True
            continue

        # Step B: 1-for-1 swaps
        unique_counts = {}
        unique_tsubs = {}
        for b in selected_block_indices:
            u_list = []
            for s in block_to_tsubs[b]:
                if cover_count[s] == 1:
                    u_list.append(s)
            unique_counts[b] = len(u_list)
            unique_tsubs[b] = u_list

        sorted_blocks = sorted(selected_block_indices, key=lambda b: unique_counts[b])

        swap_made = False
        for b_out in sorted_blocks:
            if time.time() > local_deadline:
                break
            s_uniq = unique_tsubs[b_out]
            if not s_uniq:
                continue
            exclude_set = set(selected_block_indices)
            b_in = find_swap_in(s_uniq, exclude_set)
            if b_in is not None:
                selected_block_indices.remove(b_out)
                selected_block_indices.append(b_in)
                for s in block_to_tsubs[b_out]:
                    cover_count[s] -= 1
                for s in block_to_tsubs[b_in]:
                    cover_count[s] += 1
                swap_made = True
                improved = True
                break

        if swap_made:
            continue

        # Step C: 2-for-1 swaps
        if time.time() > local_deadline - 10:
            break

        n_sel = len(selected_block_indices)
        found_2for1 = False
        for i in range(n_sel):
            if time.time() > local_deadline:
                break
            b1 = selected_block_indices[i]
            u1 = unique_tsubs[b1]
            if len(u1) > 5:
                continue
            for j in range(i + 1, n_sel):
                if time.time() > local_deadline:
                    break
                b2 = selected_block_indices[j]
                u2 = unique_tsubs[b2]
                if len(u2) > 5:
                    continue

                u_union = set(u1) | set(u2)
                if len(u_union) > 10:
                    continue

                exclude_set = set(selected_block_indices)
                b3 = find_swap_in(list(u_union), exclude_set)

                if b3 is not None:
                    selected_block_indices.remove(b1)
                    selected_block_indices.remove(b2)
                    selected_block_indices.append(b3)

                    for s in block_to_tsubs[b1]:
                        cover_count[s] -= 1
                    for s in block_to_tsubs[b2]:
                        cover_count[s] -= 1
                    for s in block_to_tsubs[b3]:
                        cover_count[s] += 1

                    found_2for1 = True
                    improved = True
                    break
            if found_2for1:
                break

        if not found_2for1:
            break

    # Convert to result
    result = [list(k_subsets[i]) for i in selected_block_indices]
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(sorted(blk))
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))
    return dedup
