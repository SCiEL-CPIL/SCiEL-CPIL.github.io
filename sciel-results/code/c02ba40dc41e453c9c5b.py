import itertools
import time
import random
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    # 1. Precompute all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_index_map = {s: i for i, s in enumerate(t_subsets)}
    num_t = len(t_subsets)

    # 2. Precompute all k-blocks and their coverage masks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)

    # Precompute t-subsets within each block for fast redundancy check
    block_tsubs = []
    block_masks = [0] * num_k
    for idx, block in enumerate(k_blocks):
        mask = 0
        tsubs_in_block = []
        for combo in itertools.combinations(block, t):
            i = t_index_map[combo]
            mask |= (1 << i)
            tsubs_in_block.append(i)
        block_masks[idx] = mask
        block_tsubs.append(tsubs_in_block)

    full_mask = (1 << num_t) - 1

    def _greedy_pass(seed, strategy='max_gain'):
        """One greedy set cover pass with different strategies."""
        if time.time() > deadline:
            return None

        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        selected_set = set()

        while uncovered:
            if time.time() > deadline:
                return None

            best_idx = -1
            best_count = -1

            if strategy == 'max_gain':
                # Standard max gain with random tie-breaking
                best_indices = []
                for i in range(num_k):
                    if i in selected_set:
                        continue
                    gain = (uncovered & block_masks[i]).bit_count()
                    if gain > best_count:
                        best_count = gain
                        best_indices = [i]
                    elif gain == best_count and gain > 0:
                        best_indices.append(i)
                if best_indices:
                    if len(best_indices) > 1:
                        best_idx = rng.choice(best_indices)
                    else:
                        best_idx = best_indices[0]
            elif strategy == 'randomized':
                # Randomly weight gains to escape local optima
                best_indices = []
                weighted_best = -1
                for i in range(num_k):
                    if i in selected_set:
                        continue
                    gain = (uncovered & block_masks[i]).bit_count()
                    if gain == 0:
                        continue
                    # Add random perturbation
                    w = gain + rng.random() * 0.5
                    if w > weighted_best:
                        weighted_best = w
                        best_idx = i
            else:  # 'first_random'
                # Pick randomly from top-10 gain blocks
                gains = []
                for i in range(num_k):
                    if i in selected_set:
                        continue
                    gain = (uncovered & block_masks[i]).bit_count()
                    if gain > 0:
                        gains.append((gain, i))
                if not gains:
                    break
                gains.sort(key=lambda x: -x[0])
                top_n = min(10, len(gains))
                best_idx = rng.choice(gains[:top_n])[1]
                best_count = gains[0][0]

            if best_idx == -1 or best_count <= 0:
                break

            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~block_masks[best_idx]

        return selected

    def _remove_redundant_fast(selected_indices):
        """Remove redundant blocks using t-subset counts (faster than mask OR)."""
        if not selected_indices:
            return selected_indices

        # Count how many selected blocks cover each t-subset
        tsub_count = Counter()
        for idx in selected_indices:
            for ts in block_tsubs[idx]:
                tsub_count[ts] += 1

        # Keep only blocks that cover at least one t-subset uniquely
        kept = []
        for idx in selected_indices:
            if time.time() > deadline:
                kept.append(idx)
                continue
            has_unique = False
            for ts in block_tsubs[idx]:
                if tsub_count[ts] == 1:
                    has_unique = True
                    break
            if has_unique:
                kept.append(idx)

        return kept

    def _local_search_polish(selected_indices, max_swaps=50):
        """Try to improve by swapping blocks in/out."""
        if time.time() > deadline:
            return selected_indices

        current = list(selected_indices)
        current_mask = 0
        for idx in current:
            current_mask |= block_masks[idx]

        # Verify validity
        if current_mask != full_mask:
            return current

        improved = True
        iterations = 0
        while improved and iterations < max_swaps and time.time() < deadline:
            improved = False
            iterations += 1

            # Try removing each block and see if we can add a better one
            for i in range(len(current)):
                if time.time() > deadline - 2.0:
                    return current

                # Remove block i
                test_mask = current_mask & ~block_masks[current[i]]
                # Check if still valid
                if test_mask != full_mask:
                    continue

                # Try to find a block that covers the missing parts with fewer blocks
                missing = full_mask & ~test_mask
                # Find block with max coverage of missing
                best_add = -1
                best_gain = 0
                for j in range(num_k):
                    if j in current and j != current[i]:
                        continue
                    gain = (missing & block_masks[j]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_add = j

                if best_add != -1 and best_gain > 0:
                    # Check if adding best_add covers all missing
                    new_mask = test_mask | block_masks[best_add]
                    if new_mask == full_mask:
                        # We can swap: remove i, add best_add
                        # But we need same count or less; here it's same count
                        # Only do if it helps future swaps
                        # For now, skip same-count swaps unless forced
                        pass

            # Try adding a block and removing two others (net -1)
            for i in range(len(current)):
                for j in range(i + 1, len(current)):
                    if time.time() > deadline - 2.0:
                        return current

                    test_mask = current_mask & ~block_masks[current[i]] & ~block_masks[current[j]]
                    if test_mask == full_mask:
                        # Both i and j are redundant!
                        new_current = []
                        for idx in current:
                            if idx != current[i] and idx != current[j]:
                                new_current.append(idx)
                        current = new_current
                        current_mask = 0
                        for idx in current:
                            current_mask |= block_masks[idx]
                        improved = True
                        break
                if improved:
                    break

        return current

    best_blocks = None
    best_len = float('inf')

    # Run multiple diverse greedy passes
    strategies = ['max_gain', 'randomized', 'first_random']
    attempt = 0
    while time.time() < deadline - 15.0:  # Leave 15s for polish
        strategy = strategies[attempt % len(strategies)]
        seed = attempt * 7 + 42
        selected = _greedy_pass(seed, strategy)
        if selected is None:
            break

        # Remove redundant blocks (fast method)
        selected = _remove_redundant_fast(selected)

        # Local search polish
        selected = _local_search_polish(selected)

        if len(selected) < best_len:
            best_len = len(selected)
            best_blocks = [list(k_blocks[i]) for i in selected]

        # Early stop if we've found a very good solution
        if best_len <= 335:
            break

        attempt += 1

    # Fallback if no solution found
    if best_blocks is None:
        uncovered = full_mask
        selected = []
        selected_set = set()
        while uncovered and time.time() < deadline:
            best_idx = -1
            best_count = -1
            for i in range(num_k):
                if i in selected_set:
                    continue
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_count:
                    best_count = gain
                    best_idx = i
            if best_idx == -1 or best_count == 0:
                break
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~block_masks[best_idx]

        if selected:
            selected = _remove_redundant_fast(selected)
            best_blocks = [list(k_blocks[i]) for i in selected]
        else:
            best_blocks = []

    return best_blocks
