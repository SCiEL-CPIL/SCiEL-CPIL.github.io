"""Combined program: greedy + local search (deletion + swap) for covering design C(19,8,5)."""
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Enumerate all t-subsets and assign indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Enumerate all k-subsets (candidate blocks)
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # Build incidence: block_to_tsubs[i] = list of t-subset indices covered by block i
    # t_sub_to_blocks[j] = list of block indices covering t-subset j
    block_to_tsubs = [[] for _ in range(n_k)]
    t_sub_to_blocks = [[] for _ in range(n_t)]

    for i, ksub in enumerate(all_k_subs):
        for tsub in itertools.combinations(ksub, t):
            j = t_sub_to_idx[tsub]
            block_to_tsubs[i].append(j)
            t_sub_to_blocks[j].append(i)

    # Convert to numpy arrays for fast access
    block_to_tsubs_np = [np.array(lst, dtype=np.int32) for lst in block_to_tsubs]
    t_sub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in t_sub_to_blocks]

    # Precompute block sizes (all same: C(k,t))
    block_sizes = np.array([len(lst) for lst in block_to_tsubs], dtype=np.int32)

    def is_complete(blocks):
        """Check if blocks cover all t-subsets."""
        cov = np.zeros(n_t, dtype=np.uint8)
        for block in blocks:
            bid = all_k_subs.index(tuple(block))
            for j in block_to_tsubs[bid]:
                cov[j] = 1
        return np.all(cov == 1)

    def greedy_cover(seed=None):
        """Randomized greedy set cover."""
        if seed is not None:
            random.seed(seed)

        covered = np.zeros(n_t, dtype=np.uint8)
        cov_count = np.zeros(n_t, dtype=np.int32)
        gains = block_sizes.copy()
        selected = []
        used_blocks_set = set()

        while True:
            if time.time() > deadline - 5:
                break
            uncovered_idx = np.where(covered == 0)[0]
            if len(uncovered_idx) == 0:
                break

            # Find best block maximizing gain among unused blocks
            # Zero out gains for used blocks
            active_gains = gains.copy()
            for bi in used_blocks_set:
                active_gains[bi] = 0

            best_block = int(np.argmax(active_gains))
            best_gain = int(active_gains[best_block])

            if best_gain <= 0:
                break

            selected.append(best_block)
            used_blocks_set.add(best_block)

            # Update coverage
            for j in block_to_tsubs[best_block]:
                if cov_count[j] == 0:
                    covered[j] = 1
                cov_count[j] += 1
                # Update gains for all blocks covering this t-subset
                for b in t_sub_to_blocks[j]:
                    if b not in used_blocks_set:
                        gains[b] -= 1

        return selected

    def try_reduce(block_ids):
        """Remove redundant blocks via deletion pass."""
        if not block_ids:
            return block_ids

        cov_count = np.zeros(n_t, dtype=np.int32)
        for bid in block_ids:
            for j in block_to_tsubs[bid]:
                cov_count[j] += 1

        changed = True
        while changed:
            changed = False
            if time.time() > deadline - 10:
                break
            # Try each block in random order
            indices = list(range(len(block_ids)))
            random.shuffle(indices)
            for i in indices:
                bid = block_ids[i]
                # Check if any t-subset is only covered by this block
                redundant = True
                for j in block_to_tsubs[bid]:
                    if cov_count[j] == 1:
                        redundant = False
                        break
                if redundant:
                    for j in block_to_tsubs[bid]:
                        cov_count[j] -= 1
                    block_ids.pop(i)
                    changed = True
                    break  # Restart

        return block_ids

    def try_swap(block_ids):
        """Try to swap blocks to enable further reductions."""
        if not block_ids:
            return block_ids

        cov_count = np.zeros(n_t, dtype=np.int32)
        for bid in block_ids:
            for j in block_to_tsubs[bid]:
                cov_count[j] += 1

        current_set = set(block_ids)
        swaps_done = 0
        max_swaps = 30

        indices = list(range(len(block_ids)))
        random.shuffle(indices)

        for i in indices:
            if time.time() > deadline - 15 or swaps_done >= max_swaps:
                break

            old_block = block_ids[i]

            # Find unique subsets covered only by this block
            unique_subs = []
            for s in block_to_tsubs[old_block]:
                if cov_count[s] == 1:
                    unique_subs.append(s)

            if not unique_subs:
                # Block is not critical; try swapping with a random block
                new_block = random.randint(0, n_k - 1)
                if new_block == old_block or new_block in current_set:
                    continue

                # Perform swap
                for s in block_to_tsubs[old_block]:
                    cov_count[s] -= 1
                for s in block_to_tsubs[new_block]:
                    cov_count[s] += 1
                current_set.discard(old_block)
                current_set.add(new_block)
                block_ids[i] = new_block
                swaps_done += 1
            else:
                # Find candidates that cover all unique_subs
                candidates = set(t_sub_to_blocks[unique_subs[0]])
                for s in unique_subs[1:]:
                    candidates.intersection_update(set(t_sub_to_blocks[s]))
                    if not candidates:
                        break

                if candidates:
                    candidates.discard(old_block)
                    candidates -= current_set
                    if candidates:
                        new_block = random.choice(list(candidates))
                        # Perform swap
                        for s in block_to_tsubs[old_block]:
                            cov_count[s] -= 1
                        for s in block_to_tsubs[new_block]:
                            cov_count[s] += 1
                        current_set.discard(old_block)
                        current_set.add(new_block)
                        block_ids[i] = new_block
                        swaps_done += 1

        return block_ids

    def local_search(initial_blocks, time_limit=30):
        """Local search: deletion + swap passes."""
        block_ids = list(initial_blocks)
        best_len = len(block_ids)
        best_blocks = list(block_ids)
        search_start = time.time()

        while time.time() - search_start < time_limit and time.time() < deadline - 10:
            # Deletion pass
            block_ids = try_reduce(block_ids)

            if len(block_ids) < best_len:
                best_len = len(block_ids)
                best_blocks = list(block_ids)

            # Swap pass
            block_ids = try_swap(block_ids)

            # Another deletion pass after swaps
            block_ids = try_reduce(block_ids)

            if len(block_ids) < best_len:
                best_len = len(block_ids)
                best_blocks = list(block_ids)

        return best_blocks

    # Run multiple greedy attempts with different seeds
    best_blocks = None
    best_len = float('inf')

    for seed in range(20):
        if time.time() > deadline - 20:
            break

        blocks = greedy_cover(seed=seed)
        if not blocks:
            continue

        # Convert to actual block lists
        block_lists = [list(all_k_subs[bid]) for bid in blocks]

        if not is_complete(block_lists):
            continue

        if len(block_lists) < best_len:
            best_len = len(block_lists)
            best_blocks = block_lists

    if best_blocks is None:
        # Fallback: simple greedy
        covered = np.zeros(n_t, dtype=np.uint8)
        blocks = []
        while True:
            uncovered_idx = np.where(covered == 0)[0]
            if len(uncovered_idx) == 0:
                break
            first_uncovered = uncovered_idx[0]
            found = False
            for bid in t_sub_to_blocks[first_uncovered]:
                block = list(all_k_subs[bid])
                if block not in blocks:
                    blocks.append(block)
                    for j in block_to_tsubs[bid]:
                        covered[j] = 1
                    found = True
                    break
            if not found:
                break
        best_blocks = blocks
        best_len = len(blocks)

    # Local search on the best solution found
    if best_blocks is not None and time.time() < deadline - 15:
        best_bid_list = [all_k_subs.index(tuple(b)) for b in best_blocks]
        improved = local_search(best_bid_list, time_limit=35)
        if len(improved) < best_len:
            best_blocks = [list(all_k_subs[bid]) for bid in improved]
            best_len = len(improved)

    # Final verification
    if best_blocks is None or not is_complete(best_blocks):
        # Emergency fallback
        covered = np.zeros(n_t, dtype=np.uint8)
        blocks = []
        while True:
            uncovered_idx = np.where(covered == 0)[0]
            if len(uncovered_idx) == 0:
                break
            first_uncovered = uncovered_idx[0]
            found = False
            for bid in t_sub_to_blocks[first_uncovered]:
                block = list(all_k_subs[bid])
                if block not in blocks:
                    blocks.append(block)
                    for j in block_to_tsubs[bid]:
                        covered[j] = 1
                    found = True
                    break
            if not found:
                break
        return blocks

    return best_blocks
