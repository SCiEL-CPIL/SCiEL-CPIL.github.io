"""Covering design C(24,7,3) - Greedy independent-set on conflict graph + iterative pruning."""
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 145.0
    start_time = time.time()

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}

    # All possible blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, precompute the bitmask of t-subsets it covers
    # n_t = C(24,3) = 2024, fits in a Python int (arbitrary precision)
    block_cover_mask = []
    for block in all_blocks:
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
        block_cover_mask.append(mask)

    full_mask = (1 << n_t) - 1

    # Frequency of each t-subset: how many blocks cover it
    t_freq = np.zeros(n_t, dtype=np.int32)
    for mask in block_cover_mask:
        # Count bits per position is expensive; instead build reverse index
        pass

    # Build reverse index: for each t-subset index, list of block indices covering it
    t_to_blocks = [[] for _ in range(n_t)]
    for b_idx, mask in enumerate(block_cover_mask):
        # Iterate set bits
        m = mask
        while m:
            # Get lowest set bit
            lb = m & (-m)
            idx = lb.bit_length() - 1
            t_to_blocks[idx].append(b_idx)
            m ^= lb

    # Weight for each t-subset: 1/frequency (rarer triples are harder to cover)
    t_weights = np.zeros(n_t, dtype=np.float64)
    for i in range(n_t):
        freq = len(t_to_blocks[i])
        t_weights[i] = 1.0 / max(freq, 1)

    def greedy_weighted_cover(seed=0, time_budget=30.0):
        """
        Weighted greedy: repeatedly pick the block that maximizes weighted coverage
        of uncovered triples. Weight = sum of t_weights for newly covered triples.
        """
        rng = random.Random(seed)
        uncovered_mask = full_mask
        selected = []
        end_time = time.time() + time_budget

        while uncovered_mask and time.time() < end_time:
            # Pick a random uncovered triple to guide selection
            # Find an uncovered triple index
            m = uncovered_mask
            # Randomly pick a set bit
            bits = []
            tmp = m
            while tmp and len(bits) < 100:  # sample up to 100 uncovered triples
                lb = tmp & (-tmp)
                bits.append(lb.bit_length() - 1)
                tmp ^= lb
            if not bits:
                break

            # For each sampled uncovered triple, consider blocks covering it
            # Pick the block with highest weighted marginal gain
            best_block = -1
            best_gain = -1.0

            # Sample from the union of blocks covering sampled triples
            candidate_set = set()
            for idx in bits:
                for b in t_to_blocks[idx]:
                    candidate_set.add(b)
                if len(candidate_set) > 5000:
                    break

            if not candidate_set:
                # Fallback: try all blocks (slow)
                candidate_set = set(range(n_blocks))

            for b in candidate_set:
                new_cover = block_cover_mask[b] & uncovered_mask
                if new_cover == 0:
                    continue
                # Compute weighted gain
                gain = 0.0
                nm = new_cover
                while nm:
                    lb = nm & (-nm)
                    idx = lb.bit_length() - 1
                    gain += t_weights[idx]
                    nm ^= lb
                if gain > best_gain:
                    best_gain = gain
                    best_block = b

            if best_block < 0:
                break

            selected.append(all_blocks[best_block])
            uncovered_mask &= ~block_cover_mask[best_block]

        return selected

    def prune_redundant(blocks, time_budget=20.0):
        """
        Iteratively remove any block whose removal still leaves full coverage.
        This is a local-search step that can significantly reduce block count.
        """
        if not blocks:
            return blocks

        # Build coverage map: for each triple, which selected blocks cover it
        # Use a dict: triple_idx -> set of block indices in selected
        block_indices = list(range(len(blocks)))
        triple_to_sel_blocks = [[] for _ in range(n_t)]

        for bi, block in enumerate(blocks):
            for ts in itertools.combinations(block, t):
                idx = t_subset_to_idx[ts]
                triple_to_sel_blocks[idx].append(bi)

        selected_set = set(block_indices)
        end_time = time.time() + time_budget
        changed = True

        while changed and time.time() < end_time:
            changed = False
            # Try removing each block
            for bi in list(selected_set):
                if time.time() > end_time:
                    break
                # Check if removing block bi still covers all triples
                # A triple is still covered if some other selected block covers it
                # For each triple covered by block bi, check if another block covers it
                block = blocks[bi]
                removable = True
                for ts in itertools.combinations(block, t):
                    idx = t_subset_to_idx[ts]
                    # Count how many selected blocks (other than bi) cover idx
                    count = 0
                    for other_bi in triple_to_sel_blocks[idx]:
                        if other_bi != bi and other_bi in selected_set:
                            count += 1
                            if count >= 1:
                                break
                    if count == 0:
                        removable = False
                        break

                if removable:
                    selected_set.discard(bi)
                    changed = True

        # Reconstruct block list
        result = [blocks[bi] for bi in sorted(selected_set)]
        return result

    def local_search_improve(blocks, time_budget=40.0):
        """
        Try to replace pairs of blocks with a single block that covers their union
        of triples, or swap a block for a better one.
        """
        if not blocks:
            return blocks

        end_time = time.time() + time_budget
        current = list(blocks)
        current_len = len(current)

        # Build coverage masks for current blocks
        current_masks = [block_cover_mask[all_blocks.index(b)] if b in all_blocks else 0 for b in current]
        # Actually, let's map blocks to their indices properly
        block_to_idx = {b: i for i, b in enumerate(all_blocks)}
        current_masks = []
        for b in current:
            idx = block_to_idx.get(b, -1)
            if idx >= 0:
                current_masks.append(block_cover_mask[idx])
            else:
                current_masks.append(0)

        full_cov = 0
        for m in current_masks:
            full_cov |= m

        # Try removing blocks one by one (already done in prune, but with different order)
        # Try swapping: for each block, try replacing with a different block
        # that covers at least as many uncovered triples
        for iter_round in range(3):
            if time.time() > end_time:
                break

            # Try single-block replacement
            for i in range(len(current)):
                if time.time() > end_time:
                    break
                # What does block i cover that others don't?
                others_cov = 0
                for j in range(len(current)):
                    if j != i:
                        others_cov |= current_masks[j]
                unique_to_i = current_masks[i] & ~others_cov

                if unique_to_i == 0:
                    # Block i is redundant, remove it
                    current.pop(i)
                    current_masks.pop(i)
                    i -= 1
                    continue

                # Try to find a block that covers unique_to_i and more
                # Sample candidate blocks
                # Pick a triple from unique_to_i
                m = unique_to_i
                if m:
                    lb = m & (-m)
                    tri_idx = lb.bit_length() - 1
                    # Blocks covering this triple
                    candidates = t_to_blocks[tri_idx]
                    rng = random.Random(i + iter_round)
                    rng.shuffle(candidates)
                    for cand_idx in candidates[:50]:
                        cand_mask = block_cover_mask[cand_idx]
                        # Check if replacing block i with cand_idx maintains coverage
                        new_others = 0
                        for j in range(len(current)):
                            if j != i:
                                new_others |= current_masks[j]
                        # New total coverage
                        new_total = new_others | cand_mask
                        if (new_total & full_cov) == full_cov:
                            # Replacement is valid
                            current[i] = all_blocks[cand_idx]
                            current_masks[i] = cand_mask
                            break

            # Prune again after swaps
            current = prune_redundant(current, time_budget=5.0)
            if len(current) < current_len:
                current_len = len(current)

        return current

    # Main strategy: multiple greedy starts with different seeds, then prune and local search
    best_blocks = None
    best_len = float('inf')

    # Phase 1: Weighted greedy with multiple seeds
    for seed in range(20):
        if time.time() > deadline - 60.0:
            break
        blocks = greedy_weighted_cover(seed=seed, time_budget=15.0)
        if not blocks:
            continue
        # Prune
        blocks = prune_redundant(blocks, time_budget=10.0)
        # Validate
        cov = 0
        for b in blocks:
            idx = all_blocks.index(b) if b in all_blocks else -1
            if idx >= 0:
                cov |= block_cover_mask[idx]
        if cov == full_mask and len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks

    # Phase 2: Local search on the best solution
    if best_blocks is not None and time.time() < deadline - 30.0:
        improved = local_search_improve(best_blocks, time_budget=40.0)
        # Validate
        cov = 0
        for b in improved:
            idx = all_blocks.index(b) if b in all_blocks else -1
            if idx >= 0:
                cov |= block_cover_mask[idx]
        if cov == full_mask and len(improved) < best_len:
            best_len = len(improved)
            best_blocks = improved

    # Final fallback: simple greedy if nothing worked
    if best_blocks is None or len(best_blocks) == 0:
        best_blocks = greedy_weighted_cover(seed=0, time_budget=30.0)
        best_blocks = prune_redundant(best_blocks, time_budget=10.0)

    # Final validation
    if best_blocks is None:
        return []

    cov = 0
    for b in best_blocks:
        try:
            idx = all_blocks.index(b)
        except ValueError:
            idx = -1
        if idx >= 0:
            cov |= block_cover_mask[idx]

    if cov != full_mask:
        # Emergency fallback: brute force greedy
        uncovered = full_mask
        final_blocks = []
        for b_idx in range(n_blocks):
            if uncovered == 0:
                break
            new_cov = block_cover_mask[b_idx] & uncovered
            if new_cov:
                final_blocks.append(all_blocks[b_idx])
                uncovered &= ~block_cover_mask[b_idx]
        return final_blocks

    return best_blocks
