import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 142  # 8s safety margin

    # Precompute all t-subsets and map to IDs
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}

    # Precompute all k-subsets (blocks) and their coverage bitmasks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Build bitmask for each block: bit i set if block covers triplet i
    # Use Python ints as arbitrary-precision bitsets
    block_masks = [0] * n_blocks
    for idx, block in enumerate(all_blocks):
        mask = 0
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
        block_masks[idx] = mask

    # ---------- Fast Greedy Construction ----------
    def greedy_fast(seed=0):
        rng = random.Random(seed)
        is_uncovered = np.ones(n_triplets, dtype=bool)
        uncovered_list = list(range(n_triplets))
        pos = np.arange(n_triplets, dtype=np.int32)
        blocks = []

        while uncovered_list:
            if time.time() > deadline - 30:
                break

            # Pick a random uncovered triplet
            idx = rng.randrange(len(uncovered_list))
            start_tid = uncovered_list[idx]
            start_triplet = all_triplets[start_tid]

            block = list(start_triplet)
            block_set = set(block)

            # Greedily add points to maximize coverage of uncovered triplets
            while len(block) < k:
                best_point = -1
                max_gain = -1
                candidates = [p for p in range(v) if p not in block_set]

                # Precompute (t-1)-subsets of current block
                pairs_in_block = list(itertools.combinations(block, t - 1))

                for p in candidates:
                    gain = 0
                    for pair in pairs_in_block:
                        triplet = tuple(sorted(pair + (p,)))
                        tid = triplet_to_id[triplet]
                        if is_uncovered[tid]:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_point = p
                        if max_gain == len(pairs_in_block):
                            break

                if max_gain <= 0:
                    best_point = rng.choice(candidates)

                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            blocks.append(block)

            # Update uncovered
            for triplet in itertools.combinations(block, t):
                tid = triplet_to_id[triplet]
                if is_uncovered[tid]:
                    is_uncovered[tid] = False
                    cur_idx = pos[tid]
                    last_idx = len(uncovered_list) - 1
                    if cur_idx != last_idx:
                        last_tid = uncovered_list[last_idx]
                        uncovered_list[cur_idx] = last_tid
                        pos[last_tid] = cur_idx
                    uncovered_list.pop()

        return blocks

    # ---------- Local Search ----------
    def local_search(blocks, time_limit=10):
        """Perform redundancy removal, 2-for-1 merges, and 1-for-1 swaps."""
        ls_start = time.time()
        ls_deadline = ls_start + time_limit

        # Build count array
        count = np.zeros(n_triplets, dtype=np.int32)

        def update_counts(block_list, delta):
            for b in block_list:
                for triplet in itertools.combinations(b, t):
                    tid = triplet_to_id[triplet]
                    count[tid] += delta

        def get_unique_triplets(block):
            unique = []
            for triplet in itertools.combinations(block, t):
                tid = triplet_to_id[triplet]
                if count[tid] == 1:
                    unique.append(tid)
            return unique

        # Initialize counts
        update_counts(blocks, 1)

        progress = True
        while progress and time.time() < ls_deadline:
            progress = False

            # Pass 1: Remove redundant blocks
            new_blocks = []
            for b in blocks:
                redundant = True
                for triplet in itertools.combinations(b, t):
                    tid = triplet_to_id[triplet]
                    if count[tid] == 1:
                        redundant = False
                        break
                if redundant:
                    update_counts([b], -1)
                    progress = True
                else:
                    new_blocks.append(b)
            blocks = new_blocks

            # Pass 2: 2-for-1 merge
            n = len(blocks)
            if n >= 2:
                block_unique = [get_unique_triplets(b) for b in blocks]
                merged = False
                for i in range(n):
                    if time.time() > ls_deadline:
                        break
                    if not block_unique[i]:
                        continue
                    for j in range(i + 1, n):
                        if time.time() > ls_deadline:
                            break
                        if not block_unique[j]:
                            continue
                        all_unique = set(block_unique[i]) | set(block_unique[j])
                        points_needed = set()
                        for tid in all_unique:
                            points_needed.update(all_triplets[tid])
                        if len(points_needed) <= k:
                            new_block = list(points_needed)
                            remaining = [p for p in range(v) if p not in points_needed]
                            while len(new_block) < k and remaining:
                                p = remaining.pop(0)
                                new_block.append(p)
                            new_block.sort()
                            update_counts([blocks[i], blocks[j]], -1)
                            update_counts([new_block], 1)
                            blocks = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                            blocks.append(new_block)
                            progress = True
                            merged = True
                            break
                    if merged:
                        break
                if merged:
                    continue

            # Pass 3: 1-for-1 swap (replace one block with another that covers its unique triplets)
            n = len(blocks)
            if n >= 1:
                swapped = False
                for i in range(n):
                    if time.time() > ls_deadline:
                        break
                    unique_i = get_unique_triplets(blocks[i])
                    if not unique_i:
                        continue
                    # Try to find a block (not in current set) that covers all unique_i
                    points_needed = set()
                    for tid in unique_i:
                        points_needed.update(all_triplets[tid])
                    if len(points_needed) > k:
                        continue
                    # Search for a block containing points_needed
                    # Build required set
                    req_set = frozenset(points_needed)
                    found = -1
                    for bi, b in enumerate(all_blocks):
                        if frozenset(b) >= req_set:
                            # Check if this block is already in our list
                            if bi not in [all_blocks.index(blocks[j]) for j in range(n)]:
                                found = bi
                                break
                    if found == -1:
                        continue
                    new_block = list(all_blocks[found])
                    # Verify it covers all unique_i
                    covers_all = True
                    for tid in unique_i:
                        triplet = all_triplets[tid]
                        if not all(p in new_block for p in triplet):
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    # Perform swap
                    update_counts([blocks[i]], -1)
                    update_counts([new_block], 1)
                    blocks[i] = new_block
                    progress = True
                    swapped = True
                    break
                if swapped:
                    continue

        return blocks

    # ---------- Main ----------
    # Run greedy with a few seeds, keep best
    best_blocks = None
    best_count = float('inf')

    for seed in range(5):
        if time.time() > deadline - 40:
            break
        blocks = greedy_fast(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    # Local search on best
    if best_blocks is not None and time.time() < deadline - 20:
        best_blocks = local_search(best_blocks, time_limit=min(20, deadline - time.time() - 10))

    # Final greedy polish: try to replace blocks with better ones
    if best_blocks is not None and time.time() < deadline - 10:
        # Convert to set of frozensets for quick lookup
        current_set = set(frozenset(b) for b in best_blocks)
        # Build coverage count
        count = np.zeros(n_triplets, dtype=np.int32)
        for b in best_blocks:
            for triplet in itertools.combinations(b, t):
                tid = triplet_to_id[triplet]
                count[tid] += 1

        # Try 1-for-1 swaps with precomputed block masks
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            n = len(best_blocks)
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                # Get unique triplets of block i
                unique = []
                for triplet in itertools.combinations(best_blocks[i], t):
                    tid = triplet_to_id[triplet]
                    if count[tid] == 1:
                        unique.append(tid)
                if not unique:
                    continue
                # Find a block mask that covers all unique triplets
                req_mask = 0
                for tid in unique:
                    req_mask |= (1 << tid)
                # Search for a block that covers req_mask and is not already in our set
                for bi in range(n_blocks):
                    if (block_masks[bi] & req_mask) == req_mask:
                        fs = frozenset(all_blocks[bi])
                        if fs not in current_set:
                            # Perform swap
                            old_block = best_blocks[i]
                            new_block = list(all_blocks[bi])
                            # Update counts
                            for triplet in itertools.combinations(old_block, t):
                                tid = triplet_to_id[triplet]
                                count[tid] -= 1
                            for triplet in itertools.combinations(new_block, t):
                                tid = triplet_to_id[triplet]
                                count[tid] += 1
                            current_set.discard(frozenset(old_block))
                            current_set.add(fs)
                            best_blocks[i] = new_block
                            improved = True
                            break
                if improved:
                    break

    return best_blocks if best_blocks is not None else []

if __name__ == "__main__":
    pass
