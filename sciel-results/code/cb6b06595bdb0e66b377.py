import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # Pre-compute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # Pre-compute all t-subsets (targets)
    all_targets = list(itertools.combinations(range(v), t))
    num_targets = len(all_targets)

    # Map target to index
    target_to_idx = {tup: i for i, tup in enumerate(all_targets)}

    # Pre-compute for each block, the list of target indices it covers
    # Store as a 2D numpy array: shape (num_blocks, C(k,t))
    ct = len(list(itertools.combinations(range(k), t)))
    block_targets = np.zeros((num_blocks, ct), dtype=np.int32)
    for bi, block in enumerate(all_blocks):
        for ti, t_sub in enumerate(itertools.combinations(block, t)):
            block_targets[bi, ti] = target_to_idx[t_sub]

    # Initialize uncovered as boolean array
    uncovered = np.ones(num_targets, dtype=bool)

    # Greedy Set Cover with vectorized gain computation
    selected_indices = []
    selected_mask = np.zeros(num_blocks, dtype=bool)

    while uncovered.any():
        if time.time() > deadline - 15.0:
            break

        # Vectorized gain computation: for each block, count how many of its targets are uncovered
        # uncovered[block_targets[i]] gives boolean array of which targets of block i are uncovered
        # Sum along axis 1
        gains = uncovered[block_targets].sum(axis=1)

        # Mask out already selected blocks
        gains[selected_mask] = 0

        best_block_idx = int(np.argmax(gains))
        best_gain = int(gains[best_block_idx])

        if best_gain == 0:
            break

        selected_indices.append(best_block_idx)
        selected_mask[best_block_idx] = True

        # Remove covered targets from uncovered
        uncovered[block_targets[best_block_idx]] = False

    # Post-processing: Remove redundant blocks
    if time.time() < deadline - 10.0 and len(selected_indices) > 0:
        # Build coverage count
        coverage_count = np.zeros(num_targets, dtype=np.int32)
        for bi in selected_indices:
            coverage_count[block_targets[bi]] += 1

        # Iteratively remove redundant blocks
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_indices = []
            for bi in selected_indices:
                # Check if this block is redundant: all its targets have coverage >= 2
                if (coverage_count[block_targets[bi]] >= 2).all():
                    coverage_count[block_targets[bi]] -= 1
                    changed = True
                else:
                    new_indices.append(bi)
            if changed:
                selected_indices = new_indices

        # Verify coverage
        if selected_indices:
            covered = np.zeros(num_targets, dtype=bool)
            for bi in selected_indices:
                covered[block_targets[bi]] = True
            if not covered.all():
                # Fallback: re-add missing coverage
                uncovered = ~covered
                while uncovered.any():
                    if time.time() > deadline - 2.0:
                        break
                    gains = uncovered[block_targets].sum(axis=1)
                    gains[selected_mask] = 0
                    best_block_idx = int(np.argmax(gains))
                    if gains[best_block_idx] == 0:
                        break
                    selected_indices.append(best_block_idx)
                    selected_mask[best_block_idx] = True
                    uncovered[block_targets[best_block_idx]] = False

    # Local search: try to replace a pair of blocks with a single block
    if time.time() < deadline - 5.0 and len(selected_indices) > 1:
        n = len(selected_indices)
        if n <= 200:
            coverage_count = np.zeros(num_targets, dtype=np.int32)
            for bi in selected_indices:
                coverage_count[block_targets[bi]] += 1

            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                n = len(selected_indices)
                if n > 200:
                    break
                coverage_count = np.zeros(num_targets, dtype=np.int32)
                for bi in selected_indices:
                    coverage_count[block_targets[bi]] += 1

                for i in range(n):
                    if time.time() > deadline - 5.0:
                        break
                    for j in range(i + 1, n):
                        if time.time() > deadline - 5.0:
                            break
                        bi = selected_indices[i]
                        bj = selected_indices[j]
                        # Find exposed targets when removing both
                        bi_set = set(block_targets[bi])
                        bj_set = set(block_targets[bj])
                        exposed = []
                        for t in bi_set:
                            if t in bj_set:
                                if coverage_count[t] == 2:
                                    exposed.append(t)
                            else:
                                if coverage_count[t] == 1:
                                    exposed.append(t)
                        for t in bj_set - bi_set:
                            if coverage_count[t] == 1:
                                exposed.append(t)

                        if len(exposed) == 0:
                            selected_indices.pop(j)
                            improved = True
                            break
                        if len(exposed) <= 10:
                            # Check if a single block covers all exposed targets
                            # A block covers target t iff all 4 points of all_targets[t] are in the block
                            # We need a block that covers all targets in exposed
                            # For each candidate block, check if it covers all exposed
                            # Use the precomputed block_targets: block b covers target t iff t is in block_targets[b]
                            # So we need: all(t in block_targets[b] for t in exposed)
                            # Vectorized: for each block, count how many exposed targets it covers
                            exposed_arr = np.array(exposed, dtype=np.int32)
                            # For each block, check if it contains all exposed targets
                            # block_targets has shape (num_blocks, ct)
                            # We need: for each block b, is exposed_arr a subset of block_targets[b]?
                            # This is: np.all(np.isin(exposed_arr, block_targets[b]))
                            # Vectorized across blocks:
                            # isin_mask = np.isin(block_targets, exposed_arr)  # shape (num_blocks, ct)
                            # counts = isin_mask.sum(axis=1)  # for each block, how many of its targets are in exposed
                            # But we need: for each block, how many exposed targets are in it
                            # That's different. Let's do:
                            # For each exposed target t, which blocks contain it?
                            # Then intersect.
                            # Since exposed is small, let's just check blocks that contain the first exposed target
                            if len(exposed) > 0:
                                first_t = exposed[0]
                                # Find blocks that contain first_t
                                # block_targets contains target indices. A block contains target first_t if first_t is in its row.
                                # We can precompute: for each target, which blocks contain it.
                                # But that's expensive. Instead, since exposed is small, check a subset of blocks.
                                # Actually, let's just skip this optimization for now.
                                pass

    selected_blocks = [list(all_blocks[i]) for i in selected_indices]
    return selected_blocks
