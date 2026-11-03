import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    # 1. Precompute all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_index_map = {s: i for i, s in enumerate(t_subsets)}
    num_t = len(t_subsets)

    # 2. Precompute all k-blocks and their coverage as integer lists
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)

    # For each k-block, store the list of t-subset indices it covers
    block_t_lists = []
    for block in k_blocks:
        t_list = [t_index_map[combo] for combo in itertools.combinations(block, t)]
        block_t_lists.append(t_list)

    # Build inverted index: for each t-subset, which k-blocks cover it
    t_to_blocks = [[] for _ in range(num_t)]
    for i, t_list in enumerate(block_t_lists):
        for ti in t_list:
            t_to_blocks[ti].append(i)

    full_mask = (1 << num_t) - 1

    def _greedy_pass(seed):
        """One greedy set cover pass with optional randomization."""
        if time.time() > deadline:
            return None

        rng = random.Random(seed)
        uncovered = set(range(num_t))
        selected = []
        selected_set = set()

        while uncovered:
            if time.time() > deadline:
                return None

            # Find block with max coverage of uncovered subsets
            best_idx = -1
            best_count = -1
            best_t_list = None

            # Use randomized order for diversity
            all_indices = list(range(num_k))
            if seed % 2 == 0:
                rng.shuffle(all_indices)

            for i in all_indices:
                if i in selected_set:
                    continue
                # Count how many uncovered t-subsets this block covers
                count = 0
                for ti in block_t_lists[i]:
                    if ti in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_idx = i
                    best_t_list = block_t_lists[i]

            if best_idx == -1 or best_count == 0:
                break

            selected.append(best_idx)
            selected_set.add(best_idx)
            # Remove covered t-subsets from uncovered
            for ti in best_t_list:
                uncovered.discard(ti)

        return selected

    def _greedy_pass_fast(seed):
        """Faster greedy using inverted index to only consider relevant blocks."""
        if time.time() > deadline:
            return None

        rng = random.Random(seed)
        uncovered = set(range(num_t))
        selected = []
        selected_set = set()

        while uncovered:
            if time.time() > deadline:
                return None

            # Find the uncovered t-subset with fewest covering blocks (most constrained)
            # This is a heuristic: pick a random uncovered t-subset, then find best block covering it
            # Actually, let's use: pick the uncovered t-subset with minimum number of available covering blocks
            
            # For efficiency, randomly sample from uncovered to find a good pivot
            uncovered_list = list(uncovered)
            
            # Find the most constrained uncovered t-subset (fewest blocks can cover it)
            # Sample up to 50 to keep it fast
            sample_size = min(50, len(uncovered_list))
            if seed % 2 == 0:
                rng.shuffle(uncovered_list)
            pivot_candidates = uncovered_list[:sample_size]
            
            best_pivot = None
            best_pivot_count = float('inf')
            for ti in pivot_candidates:
                # Count how many blocks not yet selected cover this t-subset
                count = 0
                for bi in t_to_blocks[ti]:
                    if bi not in selected_set:
                        count += 1
                if count < best_pivot_count:
                    best_pivot_count = count
                    best_pivot = ti
                if count == 1:  # Can't get better
                    break

            if best_pivot is None or best_pivot_count == 0:
                break

            # Now find the best block covering best_pivot that maximizes total uncovered coverage
            best_idx = -1
            best_count = -1
            for i in t_to_blocks[best_pivot]:
                if i in selected_set:
                    continue
                count = 0
                for ti in block_t_lists[i]:
                    if ti in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_idx = i

            if best_idx == -1 or best_count == 0:
                break

            selected.append(best_idx)
            selected_set.add(best_idx)
            for ti in block_t_lists[best_idx]:
                uncovered.discard(ti)

        return selected

    def _remove_redundant(selected_indices):
        """Remove redundant blocks from a valid covering."""
        if not selected_indices:
            return selected_indices

        kept = list(selected_indices)
        changed = True

        while changed and time.time() < deadline:
            changed = False
            new_kept = []
            for i in range(len(kept)):
                idx = kept[i]
                # Check if this block's coverage is fully covered by others
                # Build union of all other blocks' t-subsets
                others_set = set()
                for j in range(len(kept)):
                    if j != i:
                        for ti in block_t_lists[kept[j]]:
                            others_set.add(ti)

                # If all t-subsets covered by this block are also covered by others
                redundant = True
                for ti in block_t_lists[idx]:
                    if ti not in others_set:
                        redundant = False
                        break

                if redundant:
                    changed = True
                else:
                    new_kept.append(idx)

            if changed:
                kept = new_kept

        return kept

    best_blocks = None
    best_len = float('inf')

    # Run multiple greedy passes with different strategies
    attempt = 0
    while time.time() < deadline - 5.0:
        if attempt % 2 == 0:
            selected = _greedy_pass_fast(attempt)
        else:
            selected = _greedy_pass(attempt)
        
        if selected is None:
            break

        # Remove redundant blocks
        selected = _remove_redundant(selected)

        if len(selected) < best_len:
            best_len = len(selected)
            best_blocks = [list(k_blocks[i]) for i in selected]

        # Early stop if we've found a very good solution
        if best_len <= 340:
            break

        attempt += 1

    # Fallback if no solution found
    if best_blocks is None:
        selected = _greedy_pass_fast(0)
        if selected is not None:
            selected = _remove_redundant(selected)
            best_blocks = [list(k_blocks[i]) for i in selected]
        else:
            best_blocks = []

    return best_blocks
