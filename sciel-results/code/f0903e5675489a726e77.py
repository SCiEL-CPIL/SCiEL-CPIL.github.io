import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Precomputation
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {tset: i for i, tset in enumerate(all_tsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    t_subsets_per_block = len(list(itertools.combinations(range(k), t)))

    # Build BlockTsets: for each block, the indices of t-subsets it covers
    BlockTsets = np.zeros((num_blocks, t_subsets_per_block), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        count = 0
        for sub in itertools.combinations(block, t):
            BlockTsets[i, count] = tset_to_idx[sub]
            count += 1

    # Build inverted index: for each t-set, list of blocks that cover it
    # This is expensive but enables efficient local search
    tset_to_blocks = [[] for _ in range(num_tsets)]
    for i in range(num_blocks):
        for ts in BlockTsets[i]:
            tset_to_blocks[ts].append(i)

    def polish_blocks(blocks):
        """Remove redundant blocks (those whose t-sets are all covered by other blocks)."""
        if not blocks:
            return blocks

        cover_count = np.zeros(num_tsets, dtype=np.int32)
        for b in blocks:
            cover_count[BlockTsets[b]] += 1

        i = len(blocks) - 1
        while i >= 0:
            b = blocks[i]
            can_remove = True
            for ts in BlockTsets[b]:
                if cover_count[ts] == 1:
                    can_remove = False
                    break
            if can_remove:
                for ts in BlockTsets[b]:
                    cover_count[ts] -= 1
                blocks.pop(i)
            else:
                i -= 1
        return blocks

    def greedy_construct():
        """Standard greedy set cover."""
        uncovered = np.ones(num_tsets, dtype=bool)
        blocks = []
        while uncovered.any():
            gains = uncovered[BlockTsets].sum(axis=1)
            max_gain = gains.max()
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best_block_idx = candidates[np.random.randint(len(candidates))]
            blocks.append(best_block_idx)
            uncovered[BlockTsets[best_block_idx]] = False
        return blocks

    def random_greedy_with_noise():
        """Greedy with random tie-breaking and occasional suboptimal choices."""
        uncovered = np.ones(num_tsets, dtype=bool)
        blocks = []
        while uncovered.any():
            gains = uncovered[BlockTsets].sum(axis=1)
            max_gain = gains.max()
            if max_gain == 0:
                break
            # With some probability, pick from top few rather than just max
            if random.random() < 0.1:
                # Pick from top 10% of candidates
                threshold = max_gain * 0.8
                candidates = np.where(gains >= threshold)[0]
            else:
                candidates = np.where(gains == max_gain)[0]
            best_block_idx = candidates[np.random.randint(len(candidates))]
            blocks.append(best_block_idx)
            uncovered[BlockTsets[best_block_idx]] = False
        return blocks

    def local_search_swap(blocks):
        """Try to replace one block with another to improve coverage efficiency."""
        n = len(blocks)
        if n < 2:
            return blocks, False

        cover_count = np.zeros(num_tsets, dtype=np.int32)
        for b in blocks:
            cover_count[BlockTsets[b]] += 1

        improved = False
        # Try a limited number of swap attempts
        num_attempts = min(20, n)
        for _ in range(num_attempts):
            # Pick a random block to potentially replace
            i = np.random.randint(n)
            b_old = blocks[i]

            # Find t-sets uniquely covered by b_old
            unique_ts = []
            for ts in BlockTsets[b_old]:
                if cover_count[ts] == 1:
                    unique_ts.append(ts)

            if not unique_ts:
                # This block is redundant, remove it
                for ts in BlockTsets[b_old]:
                    cover_count[ts] -= 1
                blocks.pop(i)
                improved = True
                break

            # Try to find a block that covers all unique_ts and doesn't add too much
            # Use inverted index to find candidate blocks
            candidate_set = set()
            for ts in unique_ts:
                candidate_set.update(tset_to_blocks[ts])
            candidate_set.discard(b_old)

            if not candidate_set:
                continue

            # Find the best replacement
            best_block = -1
            best_score = float('inf')
            for b_new in candidate_set:
                # Check if b_new covers all unique_ts
                new_ts = set(BlockTsets[b_new].tolist())
                if not set(unique_ts).issubset(new_ts):
                    continue
                # Score: number of new t-sets added (should be minimized)
                new_unique = 0
                for ts in BlockTsets[b_new]:
                    if cover_count[ts] == 0:
                        new_unique += 1
                if new_unique < best_score:
                    best_score = new_unique
                    best_block = b_new

            if best_block != -1:
                # Perform the swap
                for ts in BlockTsets[b_old]:
                    cover_count[ts] -= 1
                for ts in BlockTsets[best_block]:
                    cover_count[ts] += 1
                blocks[i] = best_block
                improved = True
                break

        return blocks, improved

    def hill_climbing(blocks, max_iterations=50):
        """Iteratively apply local search swaps until no improvement."""
        for _ in range(max_iterations):
            if time.time() > deadline - 5.0:
                break
            blocks, improved = local_search_swap(blocks)
            if not improved:
                break
            blocks = polish_blocks(blocks)
        return blocks

    best_blocks = None
    best_len = float('inf')

    # Multiple restarts with different strategies
    num_restarts = 0
    while time.time() < deadline - 10.0:
        num_restarts += 1
        # Alternate between different initialization strategies
        if num_restarts % 3 == 1:
            blocks = greedy_construct()
        elif num_restarts % 3 == 2:
            blocks = random_greedy_with_noise()
        else:
            # Weighted greedy: prefer blocks that cover more uncovered t-sets
            # but with some randomness
            uncovered = np.ones(num_tsets, dtype=bool)
            blocks = []
            while uncovered.any():
                gains = uncovered[BlockTsets].sum(axis=1)
                # Add noise to gains
                noisy_gains = gains + np.random.normal(0, 0.1, size=num_blocks)
                noisy_gains[~uncovered.any()] = 0  # mask
                max_gain = noisy_gains.max()
                if max_gain <= 0:
                    break
                candidates = np.where(noisy_gains == max_gain)[0]
                best_block_idx = candidates[np.random.randint(len(candidates))]
                blocks.append(best_block_idx)
                uncovered[BlockTsets[best_block_idx]] = False

        blocks = polish_blocks(blocks)
        blocks = hill_climbing(blocks)

        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]

    if best_blocks is None:
        return []

    result = [list(all_blocks[b]) for b in best_blocks]
    return result
