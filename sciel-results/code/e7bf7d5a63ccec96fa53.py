import itertools
import random
import time
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

for block in itertools.combinations(range(V), K):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(len(all_blocks))
    all_blocks.append(block)
    all_masks.append(mask)

N_BLOCKS = len(all_blocks)

# Precompute mask bit positions for fast iteration
mask_bit_positions = []
for m in all_masks:
    positions = []
    mm = m
    while mm:
        lsb = mm & -mm
        positions.append(lsb.bit_length() - 1)
        mm ^= lsb
    mask_bit_positions.append(positions)

# Numpy version of masks for fast greedy
masks_np = np.zeros((N_BLOCKS, N_SUBSETS), dtype=np.uint8)
for i, m in enumerate(all_masks):
    for pos in mask_bit_positions[i]:
        masks_np[i, pos] = 1

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters (not expected in this task)
        # Simple greedy
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
    deadline = start_time + 118  # Safety margin

    # 1. Greedy Initialization (numpy-accelerated)
    counts = [0] * N_SUBSETS
    current_blocks = []
    current_masks = []

    def add_block(idx):
        current_blocks.append(idx)
        current_masks.append(all_masks[idx])
        for pos in mask_bit_positions[idx]:
            counts[pos] += 1

    def remove_block(i):
        for pos in mask_bit_positions[current_blocks[i]]:
            counts[pos] -= 1
        current_blocks.pop(i)
        current_masks.pop(i)

    def is_redundant(i):
        for pos in mask_bit_positions[current_blocks[i]]:
            if counts[pos] == 1:
                return False
        return True

    # Fast greedy using numpy
    uncovered = np.ones(N_SUBSETS, dtype=np.uint8)
    while uncovered.any():
        gains = (masks_np * uncovered).sum(axis=1)
        best_idx = int(gains.argmax())
        if gains[best_idx] == 0:
            break
        add_block(best_idx)
        uncovered &= ~masks_np[best_idx].astype(bool)

    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()

    # 2. Local Search with Perturbation (ILS style)
    no_improve_count = 0
    max_no_improve = 30

    while time.time() < deadline:
        # Phase 1: Prune
        removed_any = False
        indices = list(range(len(current_blocks)))
        random.shuffle(indices)

        for i in indices:
            if time.time() > deadline:
                break
            if is_redundant(i):
                remove_block(i)
                removed_any = True
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                break

        if removed_any:
            no_improve_count = 0
            continue

        if len(current_blocks) == 0:
            break

        # Phase 2: Try 2-exchange (remove 2, add 1)
        n = len(current_blocks)
        if n >= 2 and random.random() < 0.3:
            i, j = random.sample(range(n), 2)
            # Check if removing both and adding one block works
            m_i = current_masks[i]
            m_j = current_masks[j]
            combined = m_i | m_j

            # Find subsets that would become uncovered
            uncovered_bits = []
            mm = combined
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] <= 2:
                    uncovered_bits.append(bit_idx)
                mm ^= lsb

            if uncovered_bits:
                # Find a block that covers all uncovered_bits
                uncovered_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
                candidates = blocks_containing_subset[uncovered_bits[0]]
                for b in uncovered_bits[1:]:
                    if not candidates:
                        break
                    set_cands = set(candidates)
                    set_b = set(blocks_containing_subset[b])
                    candidates = list(set_cands.intersection(set_b))

                valid_candidates = [c for c in candidates if c not in (i, j)]
                if valid_candidates:
                    # Use smoothed potential: prefer blocks that cover many low-count subsets
                    def score_block(c):
                        s = 0.0
                        for pos in mask_bit_positions[c]:
                            cnt = counts[pos]
                            # Smoothed gain: log(cnt + 1) as penalty for over-coverage
                            s += 1.0 / (1.0 + cnt)
                        return s

                    j_new = max(valid_candidates, key=score_block)
                    remove_block(j)  # Remove later index first
                    remove_block(i)
                    add_block(j_new)
                    no_improve_count = 0
                    continue
            else:
                # Both are redundant together
                remove_block(j)
                remove_block(i)
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                no_improve_count = 0
                continue

        # Phase 2b: Single swap with smoothed potential
        i = random.randint(0, len(current_blocks) - 1)

        m_i = current_masks[i]
        uncovered_bits = []
        mm = m_i
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            if counts[bit_idx] == 1:
                uncovered_bits.append(bit_idx)
            mm ^= lsb

        if not uncovered_bits:
            remove_block(i)
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
            no_improve_count = 0
            continue

        if len(uncovered_bits) > 5:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturbation: Add a block with high smoothed potential
                def perturbation_score(c):
                    s = 0.0
                    for pos in mask_bit_positions[c]:
                        cnt = counts[pos]
                        s += 1.0 / (1.0 + cnt)
                    return s

                c = random.randint(0, N_BLOCKS - 1)
                if c not in current_blocks:
                    add_block(c)
                    no_improve_count = 0
                    max_no_improve = min(150, max_no_improve * 2)
                continue
        else:
            uncovered_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
            candidates = blocks_containing_subset[uncovered_bits[0]]
            for b in uncovered_bits[1:]:
                if not candidates:
                    break
                set_cands = set(candidates)
                set_b = set(blocks_containing_subset[b])
                candidates = list(set_cands.intersection(set_b))

            valid_candidates = [c for c in candidates if c != i]
            if valid_candidates:
                # Smoothed potential: prefer blocks covering many low-count subsets
                def swap_score(c):
                    s = 0.0
                    for pos in mask_bit_positions[c]:
                        cnt = counts[pos]
                        # Reward covering under-covered subsets, penalize over-covered
                        s += 1.0 / (1.0 + cnt)
                    return s

                j = max(valid_candidates, key=swap_score)
                remove_block(i)
                add_block(j)
                no_improve_count = 0
                continue
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    def perturbation_score(c):
                        s = 0.0
                        for pos in mask_bit_positions[c]:
                            cnt = counts[pos]
                            s += 1.0 / (1.0 + cnt)
                        return s

                    c = random.randint(0, N_BLOCKS - 1)
                    if c not in current_blocks:
                        add_block(c)
                        no_improve_count = 0
                        max_no_improve = min(150, max_no_improve * 2)
                    continue

    return [list(all_blocks[i]) for i in best_blocks]
