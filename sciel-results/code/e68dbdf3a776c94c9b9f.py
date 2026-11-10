import time
import itertools
import random

def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0  # Safety margin before 150s kill

    # Precompute all t-subsets and their indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {s: i for i, s in enumerate(all_tsubs)}

    # Precompute bitmasks for all possible k-blocks
    # For v=24, k=7: C(24,7) = 346104 blocks, each covering C(7,3)=35 t-subsets
    block_bitmasks = {}
    all_blocks = list(itertools.combinations(range(v), k))
    for block in all_blocks:
        mask = 0
        for tsub in itertools.combinations(block, t):
            idx = tsub_index[tsub]
            mask |= (1 << idx)
        block_bitmasks[block] = mask

    # For fast lookup: map each t-subset to list of block indices containing it
    # We'll use block indices into all_blocks for efficiency
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for bi, block in enumerate(all_blocks):
        for tsub in itertools.combinations(block, t):
            idx = tsub_index[tsub]
            tsub_to_blocks[idx].append(bi)

    # Convert block_bitmasks to a list indexed by block index
    block_masks_list = [block_bitmasks[b] for b in all_blocks]
    full_mask = (1 << n_tsubs) - 1

    best_blocks = None
    best_count = float('inf')

    max_restarts = 500  # Will be limited by time

    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break

        rng = random.Random(restart * 7919 + 13)

        covered = 0
        blocks = []  # Store block indices

        # Greedy phase
        while covered != full_mask:
            if time.time() > deadline - 3.0:
                break

            uncovered = full_mask ^ covered

            # Find uncovered t-subsets
            uncovered_list = []
            for i in range(n_tsubs):
                if not (covered >> i) & 1:
                    uncovered_list.append(i)

            if not uncovered_list:
                break

            # Pick a random uncovered t-subset
            focus_idx = uncovered_list[rng.randint(0, len(uncovered_list) - 1)]

            # Consider all blocks containing this t-subset
            candidates = tsub_to_blocks[focus_idx]

            if not candidates:
                break

            # Find the block with maximum new coverage
            best_block_idx = -1
            best_gain = -1
            best_gain_ties = []

            for bi in candidates:
                gain = bin(block_masks_list[bi] & uncovered).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = bi
                    best_gain_ties = [bi]
                elif gain == best_gain:
                    best_gain_ties.append(bi)

            if best_block_idx == -1:
                break

            # Random tie-breaking
            chosen = best_gain_ties[rng.randint(0, len(best_gain_ties) - 1)]
            blocks.append(chosen)
            covered |= block_masks_list[chosen]

        # Check if we have a complete covering
        if covered == full_mask:
            # Local search: remove redundant blocks
            final_blocks = list(blocks)
            improved = True
            while improved and time.time() < deadline - 1.0:
                improved = False

                # Compute coverage counts for each t-subset
                cov_count = [0] * n_tsubs
                for bi in final_blocks:
                    mask = block_masks_list[bi]
                    # Count set bits in mask
                    m = mask
                    while m:
                        # Find lowest set bit
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        cov_count[idx] += 1
                        m ^= lsb

                # Try to remove blocks that are redundant
                for i in range(len(final_blocks) - 1, -1, -1):
                    if time.time() > deadline - 1.0:
                        break

                    bi = final_blocks[i]
                    mask = block_masks_list[bi]

                    # Check if all t-subsets in this block are covered by other blocks
                    # i.e., cov_count[s] > 1 for all s in this block
                    m = mask
                    redundant = True
                    while m:
                        lsb = m & (-m)
                        idx = lsb.bit_length() - 1
                        if cov_count[idx] <= 1:
                            redundant = False
                            break
                        m ^= lsb

                    if redundant:
                        # Remove this block
                        final_blocks.pop(i)
                        # Update coverage counts
                        mask = block_masks_list[bi]
                        m = mask
                        while m:
                            lsb = m & (-m)
                            idx = lsb.bit_length() - 1
                            cov_count[idx] -= 1
                            m ^= lsb
                        improved = True
                        break  # Restart the loop as coverage changed

            if len(final_blocks) < best_count:
                best_count = len(final_blocks)
                best_blocks = [list(all_blocks[bi]) for bi in final_blocks]

    # If no complete covering found, use a simpler fallback
    if best_blocks is None:
        covered = 0
        blocks = []
        rng = random.Random(42)
        while covered != full_mask:
            uncovered_list = []
            for i in range(n_tsubs):
                if not (covered >> i) & 1:
                    uncovered_list.append(i)
            if not uncovered_list:
                break
            focus_idx = uncovered_list[rng.randint(0, len(uncovered_list) - 1)]
            candidates = tsub_to_blocks[focus_idx]
            best_block_idx = -1
            best_gain = -1
            for bi in candidates:
                gain = bin(block_masks_list[bi] & (full_mask ^ covered)).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = bi
            if best_block_idx == -1:
                break
            blocks.append(best_block_idx)
            covered |= block_masks_list[best_block_idx]
        best_blocks = [list(all_blocks[bi]) for bi in blocks]
        best_count = len(blocks)

    return best_blocks
