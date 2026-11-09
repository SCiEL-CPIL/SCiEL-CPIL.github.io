import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using randomized greedy set cover
    with multiple restarts and redundancy removal.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # 1. Precompute all t-subsets and assign them indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}

    # 2. Precompute coverage bitmask for every possible k-block
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    block_masks = []
    for block in all_blocks:
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks.append(mask)

    full_mask = (1 << num_t_subsets) - 1

    def count_bits(x):
        if hasattr(x, 'bit_count'):
            return x.bit_count()
        return bin(x).count('1')

    def greedy_cover(seed, temperature=1.0, top_k=5):
        """
        Randomized greedy: at each step, pick among top-k gain blocks
        with probability proportional to gain^temperature.
        Returns list of chosen block indices.
        """
        rng = random.Random(seed)
        current_uncovered = full_mask
        chosen = []
        current_union = 0

        while current_uncovered != 0:
            if time.time() > deadline - 2.0:
                break

            # Compute gains for all blocks
            gains = []
            for i in range(num_blocks):
                intersection = block_masks[i] & current_uncovered
                gains.append(count_bits(intersection))

            max_gain = max(gains)
            if max_gain == 0:
                break

            # Find top-k blocks by gain
            # For efficiency, find all blocks with gain close to max
            threshold = max_gain - 1  # consider blocks within 1 of max
            candidates = [i for i, g in enumerate(gains) if g >= threshold]

            if not candidates:
                candidates = [i for i, g in enumerate(gains) if g == max_gain]

            if len(candidates) <= 1:
                best_idx = candidates[0] if candidates else 0
            else:
                # Weighted random selection from candidates
                # Weight = gain^temperature (higher gain more likely)
                # Add small weight for diversity: prefer less overlap with current_union
                weights = []
                for i in candidates:
                    g = gains[i]
                    overlap = count_bits(block_masks[i] & current_union)
                    # Higher gain and lower overlap -> higher weight
                    w = (g ** temperature) * (1.0 / (1.0 + 0.1 * overlap))
                    weights.append(w)

                # Normalize and sample
                total_w = sum(weights)
                if total_w == 0:
                    best_idx = candidates[0]
                else:
                    r = rng.random() * total_w
                    cum = 0.0
                    best_idx = candidates[-1]
                    for idx, w in zip(candidates, weights):
                        cum += w
                        if r <= cum:
                            best_idx = idx
                            break

            chosen.append(best_idx)
            current_uncovered &= ~block_masks[best_idx]
            current_union |= block_masks[best_idx]

        return chosen

    def remove_redundant(chosen_indices):
        """Remove blocks that are redundant (covered by others)."""
        if not chosen_indices:
            return []

        masks = [block_masks[i] for i in chosen_indices]
        n = len(masks)
        result = list(range(n))

        changed = True
        while changed:
            changed = False
            for i in range(len(result)):
                if time.time() > deadline - 2.0:
                    break
                mask_i = masks[result[i]]
                union_others = 0
                for j in range(len(result)):
                    if i != j:
                        union_others |= masks[result[j]]
                if (mask_i & ~union_others) == 0:
                    result.pop(i)
                    changed = True
                    break

        return [chosen_indices[i] for i in result]

    # Run multiple randomized greedy trials
    best_solution = None
    best_count = float('inf')

    # First, run deterministic greedy (temperature=0 equivalent: always pick max)
    # Actually, let's just run several randomized trials with different seeds
    trial_count = 0
    max_trials = 50  # Will be limited by time

    while time.time() < deadline - 5.0:
        seed = trial_count * 12345 + 67890
        # Vary temperature and top_k across trials
        temp = 0.5 + (trial_count % 5) * 0.5  # 0.5, 1.0, 1.5, 2.0, 2.5
        chosen = greedy_cover(seed, temperature=temp)

        if not chosen:
            trial_count += 1
            continue

        # Remove redundancies
        pruned = remove_redundant(chosen)

        if len(pruned) < best_count:
            best_count = len(pruned)
            best_solution = pruned

        trial_count += 1

        # If we've done several trials, check if we can afford more
        if trial_count >= max_trials:
            break

    if best_solution is None:
        # Fallback: deterministic greedy
        current_uncovered = full_mask
        chosen = []
        while current_uncovered != 0:
            if time.time() > deadline - 2.0:
                break
            best_gain = -1
            best_idx = -1
            for i in range(num_blocks):
                intersection = block_masks[i] & current_uncovered
                gain = count_bits(intersection)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            chosen.append(best_idx)
            current_uncovered &= ~block_masks[best_idx]
        best_solution = remove_redundant(chosen)

    return [list(all_blocks[idx]) for idx in best_solution]
