import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}

    # Precompute all k-blocks and their t-subset coverage
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, store the list of subset indices it covers
    block_to_subsets = [[] for _ in range(n_blocks)]
    # For each subset, store which blocks cover it
    subset_to_blocks = [[] for _ in range(n_subsets)]

    for b_idx, block in enumerate(all_blocks):
        for subset in itertools.combinations(block, t):
            u_idx = subset_to_idx[subset]
            block_to_subsets[b_idx].append(u_idx)
            subset_to_blocks[u_idx].append(b_idx)

    # Convert to numpy arrays for fast operations
    # block_subsets as a list of arrays
    block_subsets_arr = [np.array(bs, dtype=np.int32) for bs in block_to_subsets]
    subset_blocks_arr = [np.array(sb, dtype=np.int32) for sb in subset_to_blocks]

    # Initialize gain vector: gain[b] = number of currently uncovered subsets in block b
    uncovered = np.ones(n_subsets, dtype=bool)
    gains = np.zeros(n_blocks, dtype=np.int32)

    # Initial gains: for each subset, add 1 to all blocks containing it
    for u in range(n_subsets):
        for b in subset_to_blocks[u]:
            gains[b] += 1

    blocks_found = []
    used_blocks = set()

    while True:
        if time.time() > deadline:
            break

        if not uncovered.any():
            break

        # Find block with max gain
        b_star = int(np.argmax(gains))

        if gains[b_star] <= 0:
            break

        # Add this block
        blocks_found.append(list(all_blocks[b_star]))
        used_blocks.add(b_star)

        # Update uncovered and gains
        for u in block_to_subsets[b_star]:
            if uncovered[u]:
                uncovered[u] = False
                # Decrement gain for all blocks containing u
                for b in subset_to_blocks[u]:
                    gains[b] -= 1

        # Early exit if all covered
        if not uncovered.any():
            break

    # Polish pass: try to improve by checking if we can replace some blocks
    # with better ones, or add targeted blocks for remaining uncovered
    if time.time() < deadline - 5.0 and uncovered.any():
        # For each uncovered subset, find the block with max gain among unused blocks
        # that covers it, and add it
        uncov_indices = np.where(uncovered)[0]
        for u_idx in uncov_indices:
            if time.time() > deadline - 2.0:
                break
            best_b = -1
            best_g = 0
            for b in subset_to_blocks[u_idx]:
                if b not in used_blocks:
                    g = gains[b]
                    if g > best_g:
                        best_g = g
                        best_b = b
            if best_b >= 0 and best_g > 0:
                blocks_found.append(list(all_blocks[best_b]))
                used_blocks.add(best_b)
                for u in block_to_subsets[best_b]:
                    if uncovered[u]:
                        uncovered[u] = False
                        for b in subset_to_blocks[u]:
                            gains[b] -= 1
                if not uncovered.any():
                    break

    return blocks_found
