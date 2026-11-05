import itertools
import random
import time
import math
import numpy as np

# Global Precomputation
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets and their masks
all_blocks = []
all_masks = []
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]
subset_frequencies = [0] * N_SUBSETS

for block in itertools.combinations(range(V), K):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(len(all_blocks))
        subset_frequencies[idx] += 1
    all_blocks.append(block)
    all_masks.append(mask)

N_BLOCKS = len(all_blocks)

# Precompute per-block list of covered subset indices (35 each) for fast iteration
block_bit_indices = []
for mask in all_masks:
    bits = []
    mm = mask
    while mm:
        lsb = mm & -mm
        bits.append(lsb.bit_length() - 1)
        mm ^= lsb
    block_bit_indices.append(bits)

# Weighted greedy weights: rare subsets get higher priority
weights = [1.0 / f for f in subset_frequencies]

# Precompute coverage matrix as numpy array for fast vectorized greedy
# coverage_matrix[i, j] = 1 if block i covers subset j
coverage_matrix = np.zeros((N_BLOCKS, N_SUBSETS), dtype=np.int8)
for i, bits in enumerate(block_bit_indices):
    for b in bits:
        coverage_matrix[i, b] = 1

# weights as numpy array
weights_np = np.array(weights, dtype=np.float64)
weights_uniform = np.ones(N_SUBSETS, dtype=np.float64)
weights_sqrt_inv = np.array([1.0 / math.sqrt(f) for f in subset_frequencies], dtype=np.float64)


def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 112  # Safety margin below 120s kill

    def fast_greedy(weight_vector):
        """Vectorized weighted greedy using numpy for speed."""
        uncovered = np.ones(N_SUBSETS, dtype=np.int8)
        chosen = []
        remaining = N_SUBSETS

        while remaining > 0:
            gains = coverage_matrix @ (weight_vector * uncovered)
            best_idx = int(np.argmax(gains))
            best_gain = gains[best_idx]

            if best_gain <= 0:
                break

            chosen.append(best_idx)

            # Update uncovered
            newly_covered = coverage_matrix[best_idx]
            still_uncovered = uncovered & (1 - newly_covered)
            remaining = int(np.sum(still_uncovered))
            uncovered = still_uncovered

        return chosen

    # 1. Try multiple greedy initializations with different weightings
    greedy_results = []
    for w_vec in [weights_np, weights_uniform, weights_sqrt_inv]:
        if time.time() > deadline - 5:
            break
        result = fast_greedy(w_vec)
        if result:
            greedy_results.append(result)

    # Pick the best greedy result
    if greedy_results:
        best_greedy = min(greedy_results, key=len)
    else:
        best_greedy = fast_greedy(weights_np)

    current_blocks = best_greedy.copy()
    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()

    # Build counts
    counts = [0] * N_SUBSETS
    for idx in current_blocks:
        for bit in block_bit_indices[idx]:
            counts[bit] += 1

    current_set = set(current_blocks)

    def add_block(idx):
        current_blocks.append(idx)
        current_set.add(idx)
        for bit in block_bit_indices[idx]:
            counts[bit] += 1

    def remove_block(pos):
        idx = current_blocks[pos]
        for bit in block_bit_indices[idx]:
            counts[bit] -= 1
        current_blocks.pop(pos)
        current_set.discard(idx)

    def is_redundant(pos):
        idx = current_blocks[pos]
        for bit in block_bit_indices[idx]:
            if counts[bit] == 1:
                return False
        return True

    # 2. Local search
    no_improve = 0
    max_no_improve = 40

    while time.time() < deadline:
        # Phase A: prune all redundant blocks
        progress = True
        while progress:
            progress = False
            for pos in range(len(current_blocks)):
                idx = current_blocks[pos]
                redundant = True
                for bit in block_bit_indices[idx]:
                    if counts[bit] == 1:
                        redundant = False
                        break
                if redundant:
                    for bit in block_bit_indices[idx]:
                        counts[bit] -= 1
                    current_blocks.pop(pos)
                    current_set.discard(idx)
                    progress = True
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = current_blocks.copy()
                    no_improve = 0
                    break

        if len(current_blocks) == 0:
            break

        # Phase B: try a 1-for-1 swap
        best_swap_candidate = -1
        min_unique = float('inf')

        n_sample = min(len(current_blocks), 30)
        sampled_indices = random.sample(range(len(current_blocks)), n_sample)

        for i in sampled_indices:
            idx_i = current_blocks[i]
            n_unique = 0
            for bit in block_bit_indices[idx_i]:
                if counts[bit] == 1:
                    n_unique += 1
            if 0 < n_unique < min_unique:
                min_unique = n_unique
                best_swap_candidate = i

        if best_swap_candidate >= 0:
            i = best_swap_candidate
            idx_i = current_blocks[i]
            uncovered_bits = [bit for bit in block_bit_indices[idx_i] if counts[bit] == 1]
            success = False
            if len(uncovered_bits) <= 6:
                uncovered_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
                candidates = blocks_containing_subset[uncovered_bits[0]]
                for b in uncovered_bits[1:]:
                    if not candidates:
                        break
                    set_cands = set(candidates)
                    set_b = set(blocks_containing_subset[b])
                    candidates = list(set_cands & set_b)
                if candidates:
                    valid_candidates = [c for c in candidates if c != idx_i and c not in current_set]
                    if valid_candidates:
                        j = random.choice(valid_candidates)
                        remove_block(i)
                        add_block(j)
                        no_improve = 0
                        success = True
            if success:
                continue

        # Phase C: try removing a redundant block (shouldn't happen after Phase A, but check)
        i = random.randint(0, len(current_blocks) - 1)
        idx_i = current_blocks[i]
        uncovered_bits = [bit for bit in block_bit_indices[idx_i] if counts[bit] == 1]
        if not uncovered_bits:
            remove_block(i)
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
            no_improve = 0
            continue

        # Phase D: try 1-exchange with the randomly chosen block
        success = False
        if len(uncovered_bits) <= 5:
            uncovered_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
            candidates = blocks_containing_subset[uncovered_bits[0]]
            for b in uncovered_bits[1:]:
                if not candidates:
                    break
                set_cands = set(candidates)
                set_b = set(blocks_containing_subset[b])
                candidates = list(set_cands & set_b)
            if candidates:
                valid_candidates = [c for c in candidates if c != idx_i and c not in current_set]
                if valid_candidates:
                    j = random.choice(valid_candidates)
                    remove_block(i)
                    add_block(j)
                    no_improve = 0
                    success = True

        if success:
            continue

        # Phase E: try 2-exchange perturbation when stuck
        if not success and no_improve > 100:
            for _ in range(15):
                if time.time() > deadline:
                    break
                j1 = random.randint(0, N_BLOCKS - 1)
                j2 = random.randint(0, N_BLOCKS - 1)
                if j1 == j2 or j1 == idx_i or j2 == idx_i:
                    continue
                if j1 in current_set or j2 in current_set:
                    continue
                combined_mask = all_masks[j1] | all_masks[j2]
                covers_all = True
                for bit_idx in uncovered_bits:
                    if not (combined_mask & (1 << bit_idx)):
                        covers_all = False
                        break
                if covers_all:
                    remove_block(i)
                    add_block(j1)
                    add_block(j2)
                    no_improve = 0
                    success = True
                    break

        if not success:
            no_improve += 1
            if no_improve > max_no_improve:
                # Perturbation: Add a random block, then try to prune
                c = random.randint(0, N_BLOCKS - 1)
                if c not in current_set:
                    add_block(c)
                no_improve = 0

    # Final cleanup: remove any remaining redundant blocks from best state
    counts_final = [0] * N_SUBSETS
    for idx in best_blocks:
        for bit in block_bit_indices[idx]:
            counts_final[bit] += 1

    i = len(best_blocks) - 1
    while i >= 0:
        idx = best_blocks[i]
        redundant = True
        for bit in block_bit_indices[idx]:
            if counts_final[bit] == 1:
                redundant = False
                break
        if redundant:
            for bit in block_bit_indices[idx]:
                counts_final[bit] -= 1
            best_blocks.pop(i)
        i -= 1

    return [list(all_blocks[i]) for i in best_blocks]
