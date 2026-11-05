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

# Precompute alternative weightings for diverse greedy runs
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
    deadline = start_time + 115  # Safety margin

    def fast_greedy(weight_vector=None):
        """Vectorized weighted greedy using numpy for speed."""
        if weight_vector is None:
            weight_vector = weights_np
        uncovered = np.ones(N_SUBSETS, dtype=np.int8)
        chosen = []
        remaining = N_SUBSETS

        while remaining > 0:
            # Compute gain for each block: sum of weights of uncovered subsets it covers
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

    current_blocks = best_greedy
    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()

    # Build counts (list of ints, fast)
    counts = [0] * N_SUBSETS
    for idx in current_blocks:
        for bit in block_bit_indices[idx]:
            counts[bit] += 1

    def add_block(idx):
        current_blocks.append(idx)
        for bit in block_bit_indices[idx]:
            counts[bit] += 1

    def remove_block(pos):
        idx = current_blocks[pos]
        for bit in block_bit_indices[idx]:
            counts[bit] -= 1
        current_blocks.pop(pos)

    def is_redundant(pos):
        idx = current_blocks[pos]
        for bit in block_bit_indices[idx]:
            if counts[bit] == 1:
                return False
        return True

    # 2. Local search: repeatedly remove redundant blocks, then do swaps
    no_improve = 0
    max_no_improve = 40

    while time.time() < deadline:
        # Phase A: prune all redundant blocks
        progress = True
        while progress:
            progress = False
            # Scan in random order to break symmetry
            indices = list(range(len(current_blocks)))
            random.shuffle(indices)
            for pos in indices:
                if time.time() > deadline:
                    break
                if is_redundant(pos):
                    remove_block(pos)
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = current_blocks.copy()
                    progress = True
                    no_improve = 0
                    break

        if len(current_blocks) == 0:
            break

        # Phase B: try a 1-for-1 swap
        # Sample a few candidates to find one with few unique subsets
        best_swap_candidate = -1
        min_unique = float('inf')
        n_sample = min(len(current_blocks), 20)
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
        else:
            i = random.randint(0, len(current_blocks) - 1)

        idx_i = current_blocks[i]
        uncovered_bits = [bit for bit in block_bit_indices[idx_i] if counts[bit] == 1]

        if not uncovered_bits:
            # Redundant, should have been caught in Phase A, but handle just in case
            remove_block(i)
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
            no_improve = 0
            continue

        # We need to cover uncovered_bits
        if len(uncovered_bits) > 4:
            # Too many unique bits, hard to replace with one block
            no_improve += 1
            if no_improve > max_no_improve:
                # Perturbation: Add a random block, then try to prune
                c = random.randint(0, N_BLOCKS - 1)
                if c not in current_blocks:
                    add_block(c)
                    no_improve = 0
                continue
        else:
            # Find a block that covers all uncovered_bits
            # Sort by rarity to speed up intersection
            uncovered_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
            candidates = blocks_containing_subset[uncovered_bits[0]]

            for b in uncovered_bits[1:]:
                if not candidates:
                    break
                # Intersection of lists
                set_cands = set(candidates)
                set_b = set(blocks_containing_subset[b])
                candidates = list(set_cands.intersection(set_b))

            if candidates:
                valid_candidates = [c for c in candidates if c != i]
                if valid_candidates:
                    # Choose candidate that maximizes "new" coverage relative to current set (excluding i)
                    # Or just pick one that covers the most total subsets?
                    # For now, random choice is fast.
                    # Let's pick the one with the highest total coverage (popcount of mask) among candidates?
                    # Or the one that overlaps least with current set (excluding i)?
                    # To improve, we can score candidates by how many NEW subsets they cover (excluding those covered by i and other blocks)
                    # But that's expensive. Let's stick to random or simple heuristic.
                    # Heuristic: pick the one with the most subsets covered in total (mask bit count is constant 35, so this doesn't help)
                    # How about: pick the one that covers the most subsets that are currently covered by exactly 1 block (other than i)?
                    # That would strengthen the coverage.
                    best_cand = -1
                    best_score = -1
                    for c in valid_candidates:
                        score = 0
                        for bit in block_bit_indices[c]:
                            if counts[bit] == 1:
                                score += 1
                        if score > best_score:
                            best_score = score
                            best_cand = c
                    j = best_cand
                    remove_block(i)
                    add_block(j)
                    no_improve = 0
                    continue
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    c = random.randint(0, N_BLOCKS - 1)
                    if c not in current_blocks:
                        add_block(c)
                    no_improve = 0
                    continue

    # Final cleanup: remove any remaining redundant blocks from best solution
    final_counts = [0] * N_SUBSETS
    for idx in best_blocks:
        for bit in block_bit_indices[idx]:
            final_counts[bit] += 1

    final_blocks = best_blocks.copy()
    improved = True
    while improved:
        improved = False
        for i in range(len(final_blocks) - 1, -1, -1):
            idx = final_blocks[i]
            redundant = True
            for bit in block_bit_indices[idx]:
                if final_counts[bit] == 1:
                    redundant = False
                    break
            if redundant:
                for bit in block_bit_indices[idx]:
                    final_counts[bit] -= 1
                final_blocks.pop(i)
                improved = True
                if time.time() > deadline:
                    break

    return [list(all_blocks[i]) for i in final_blocks]
