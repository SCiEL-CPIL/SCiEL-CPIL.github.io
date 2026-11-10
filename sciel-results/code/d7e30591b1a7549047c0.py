# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Uses bitmask-based greedy set cover with local search (redundant block removal).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # Precompute all t-subsets and index them
    all_t = list(itertools.combinations(range(v), t))
    t_index = {ts: i for i, ts in enumerate(all_t)}
    n_t = len(all_t)

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Build a mapping from block tuple to its index for O(1) lookup
    block_to_idx = {block: i for i, block in enumerate(all_blocks)}

    # Precompute coverage masks as Python integers (bitmasks)
    # masks[i] is a bitmask where bit j is 1 if t-subset j is covered by k-subset i
    masks = [0] * n_blocks
    for i, block in enumerate(all_blocks):
        m = 0
        for ts in itertools.combinations(block, t):
            idx = t_index[ts]
            m |= (1 << idx)
        masks[i] = m

    # For each t-subset, list of block indices that contain it
    blocks_containing = [[] for _ in range(n_t)]
    for b_idx in range(n_blocks):
        m = masks[b_idx]
        # Extract set bits
        while m:
            lsb = m & (-m)
            t_idx = lsb.bit_length() - 1
            blocks_containing[t_idx].append(b_idx)
            m ^= lsb

    full_mask = (1 << n_t) - 1

    def greedy_construct(rng, deadline):
        """
        Randomized greedy: repeatedly pick the block covering the most uncovered t-subsets.
        Uses bitmask operations for fast gain computation.
        """
        uncovered = full_mask
        selected = []

        while uncovered:
            if time.time() > deadline:
                return None

            best_block = -1
            best_gain = 0
            best_gain_mask = 0

            # Scan all blocks to find max gain
            for b_idx in range(n_blocks):
                gain_mask = uncovered & masks[b_idx]
                gain = gain_mask.bit_count()

                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    best_gain_mask = gain_mask
                elif gain == best_gain and gain > 0:
                    # Random tie-breaking
                    if rng.random() < 0.1:
                        best_block = b_idx
                        best_gain_mask = gain_mask

            if best_block == -1 or best_gain <= 0:
                return None  # No improvement possible

            selected.append(best_block)
            uncovered &= ~masks[best_block]

        return selected

    def local_search_remove(selected, deadline):
        """
        Remove redundant blocks: a block is redundant if all its t-subsets are covered by other blocks.
        """
        if not selected:
            return selected

        # Compute coverage count: how many times each t-subset is covered
        coverage_count = [0] * n_t
        for b_idx in selected:
            m = masks[b_idx]
            while m:
                lsb = m & (-m)
                t_idx = lsb.bit_length() - 1
                coverage_count[t_idx] += 1
                m ^= lsb

        new_selected = list(selected)
        changed = True

        while changed:
            changed = False
            # Try removing each block (iterate in reverse for safe removal)
            for i in range(len(new_selected) - 1, -1, -1):
                if time.time() > deadline:
                    break

                b_idx = new_selected[i]
                m = masks[b_idx]
                is_redundant = True
                while m:
                    lsb = m & (-m)
                    t_idx = lsb.bit_length() - 1
                    if coverage_count[t_idx] == 1:
                        is_redundant = False
                        break
                    m ^= lsb

                if is_redundant:
                    # Remove this block
                    new_selected.pop(i)
                    m = masks[b_idx]
                    while m:
                        lsb = m & (-m)
                        t_idx = lsb.bit_length() - 1
                        coverage_count[t_idx] -= 1
                        m ^= lsb
                    changed = True

        return new_selected

    # Main loop: try multiple restarts
    best_selected = None
    best_len = float('inf')

    restart_count = 0
    while time.time() < deadline - 10.0:
        restart_count += 1
        rng_restart = random.Random(42 + restart_count)

        selected = greedy_construct(rng_restart, deadline)

        if selected is not None:
            # Local search
            selected = local_search_remove(selected, deadline)

            if len(selected) < best_len:
                best_len = len(selected)
                best_selected = selected

                # If we've used a lot of time, stop
                if time.time() > deadline - 5.0:
                    break

        # If we've used a lot of time, stop
        if time.time() > deadline - 15.0:
            break

    # Fallback: deterministic greedy if nothing found
    if best_selected is None:
        uncovered = full_mask
        selected = []
        while uncovered:
            if time.time() > deadline:
                break
            # Find any uncovered t-subset
            lsb = uncovered & (-uncovered)
            t_idx = lsb.bit_length() - 1

            # Find the block covering the most uncovered t-subsets that contains t_sub
            best_block = -1
            best_gain = -1
            for b_idx in blocks_containing[t_idx]:
                gain_mask = uncovered & masks[b_idx]
                gain = gain_mask.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx

            if best_block != -1:
                selected.append(best_block)
                uncovered &= ~masks[best_block]

        if uncovered != 0:
            return [list(all_blocks[i]) for i in selected]

        best_selected = selected

    return [list(all_blocks[i]) for i in best_selected]
# EVOLVE-BLOCK-END
