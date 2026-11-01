import itertools
import random
import time

def construct(v, k, t):
    start_total = time.time()
    deadline = start_total + 140.0  # safety margin before 150s kill

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)

    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)

    # Map each block to the indices of the t-subsets it covers
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_indices.append(indices)

    # Map each t-subset to the blocks that cover it
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)

    # Precompute degrees of t-subsets
    t_subset_degrees = [len(lst) for lst in t_subset_blocks]

    # 2. Phase 1: Greedy Set Cover (direct, not packing)
    # Pick blocks that cover the most uncovered t-subsets
    cover_count = [0] * num_t_subsets
    active_blocks = set()

    while True:
        if time.time() > deadline - 60:
            break
        best_block = -1
        max_gain = 0
        # Sample-based greedy to save time
        sample_size = min(num_k_subsets, 5000)
        if num_k_subsets > sample_size:
            candidates = random.sample(range(num_k_subsets), sample_size)
        else:
            candidates = range(num_k_subsets)

        for b_idx in candidates:
            if b_idx in active_blocks:
                continue
            gain = 0
            for t_idx in block_t_indices[b_idx]:
                if cover_count[t_idx] == 0:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_block = b_idx

        if best_block == -1 or max_gain == 0:
            break

        active_blocks.add(best_block)
        for t_idx in block_t_indices[best_block]:
            cover_count[t_idx] += 1

    # Verify coverage, add remaining if needed
    for t_idx in range(num_t_subsets):
        if cover_count[t_idx] == 0:
            # Find any block covering this
            for b_idx in t_subset_blocks[t_idx]:
                if b_idx not in active_blocks:
                    active_blocks.add(b_idx)
                    for t2 in block_t_indices[b_idx]:
                        cover_count[t2] += 1
                    break

    # 3. Minimize: remove redundant blocks
    def minimize(active_set, counts):
        changed = True
        while changed:
            changed = False
            items = list(active_set)
            random.shuffle(items)
            for b_idx in items:
                if b_idx not in active_set:
                    continue
                can_remove = True
                for t_idx in block_t_indices[b_idx]:
                    if counts[t_idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    active_set.discard(b_idx)
                    for t_idx in block_t_indices[b_idx]:
                        counts[t_idx] -= 1
                    changed = True
        return active_set, counts

    active_blocks, cover_count = minimize(active_blocks, cover_count)

    # 4. Local Search
    def try_remove(active_set, counts, b_idx):
        if b_idx not in active_set:
            return False
        for t_idx in block_t_indices[b_idx]:
            if counts[t_idx] <= 1:
                return False
        active_set.discard(b_idx)
        for t_idx in block_t_indices[b_idx]:
            counts[t_idx] -= 1
        return True

    def try_swap(active_set, counts, old_idx, new_idx):
        if old_idx not in active_set or new_idx in active_set:
            return False
        unique = [t_idx for t_idx in block_t_indices[old_idx] if counts[t_idx] == 1]
        for t_idx in unique:
            if t_idx not in block_t_indices[new_idx]:
                return False
        active_set.discard(old_idx)
        active_set.add(new_idx)
        for t_idx in block_t_indices[old_idx]:
            counts[t_idx] -= 1
        for t_idx in block_t_indices[new_idx]:
            counts[t_idx] += 1
        return True

    # Precompute for each block the set of t-subsets it covers (as a set for fast lookup)
    block_t_set = [set(indices) for indices in block_t_indices]

    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_blocks = list(active_blocks)
        random.shuffle(current_blocks)

        for B in current_blocks:
            if B not in active_blocks:
                continue

            # Try remove
            if try_remove(active_blocks, cover_count, B):
                active_blocks, cover_count = minimize(active_blocks, cover_count)
                improved = True
                continue

            # Try swap
            unique_subsets = [t_idx for t_idx in block_t_indices[B] if cover_count[t_idx] == 1]

            if unique_subsets:
                # Find blocks that cover all unique subsets
                candidates = None
                for t_idx in unique_subsets:
                    blocks_list = t_subset_blocks[t_idx]
                    if candidates is None:
                        candidates = set(blocks_list)
                    else:
                        candidates.intersection_update(blocks_list)
                    if not candidates:
                        break

                if candidates:
                    valid = [c for c in candidates if c != B and c not in active_blocks]
                    if valid:
                        B_new = random.choice(valid)
                        if try_swap(active_blocks, cover_count, B, B_new):
                            active_blocks, cover_count = minimize(active_blocks, cover_count)
                            improved = True
            else:
                # No unique subsets: try swapping with a random inactive block
                # that might enable further removals
                if len(active_blocks) > 100:
                    inactive = random.sample(
                        [i for i in range(num_k_subsets) if i not in active_blocks],
                        min(10, num_k_subsets - len(active_blocks))
                    )
                    for B_new in inactive:
                        if try_swap(active_blocks, cover_count, B, B_new):
                            active_blocks, cover_count = minimize(active_blocks, cover_count)
                            if len(active_blocks) < len(current_blocks):
                                improved = True
                            break

    # 5. Final polish: multiple minimization passes
    for _ in range(3):
        active_blocks, cover_count = minimize(active_blocks, cover_count)

    result = [list(all_k_subsets[b_idx]) for b_idx in active_blocks]
    return result
