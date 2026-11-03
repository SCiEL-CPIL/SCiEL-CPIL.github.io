import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # 8s safety margin

    points = list(range(v))
    all_t_subs = list(itertools.combinations(points, t))
    num_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    all_k_blocks = list(itertools.combinations(points, k))
    num_k = len(all_k_blocks)

    # Precompute: for each block, which t-subset indices it covers
    # Use numpy arrays for speed
    block_t_masks = np.zeros((num_k, num_t), dtype=np.uint8)
    t_to_blocks = [[] for _ in range(num_t)]
    for b_idx, block in enumerate(all_k_blocks):
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            block_t_masks[b_idx, idx] = 1
            t_to_blocks[idx].append(b_idx)

    # Convert to numpy for faster ops
    block_t_masks = np.ascontiguousarray(block_t_masks, dtype=np.uint8)

    # Greedy initialization
    def greedy_init(seed):
        random.seed(seed)
        uncovered = np.ones(num_t, dtype=bool)
        gains = np.zeros(num_k, dtype=np.int32)
        for s_idx in range(num_t):
            if uncovered[s_idx]:
                for b_idx in t_to_blocks[s_idx]:
                    gains[b_idx] += 1

        blocks = []
        while True:
            if time.time() > deadline:
                break
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            chosen = random.choice(candidates)
            blocks.append(chosen)
            for s_idx in np.where(block_t_masks[chosen])[0]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    for b_idx in t_to_blocks[s_idx]:
                        gains[b_idx] -= 1
        return blocks

    # Simulated annealing
    def simulated_annealing(initial_blocks, seed):
        random.seed(seed)
        blocks = list(initial_blocks)
        block_set = set(blocks)

        # Compute coverage counts
        cover_count = np.zeros(num_t, dtype=np.int32)
        for b_idx in blocks:
            cover_count += block_t_masks[b_idx]

        def is_valid():
            return np.all(cover_count >= 1)

        if not is_valid():
            return blocks, len(blocks)

        current_energy = len(blocks)
        best_blocks = blocks.copy()
        best_energy = current_energy

        # Temperature schedule
        t0 = 1.0
        t_end = 0.01
        cooling = 0.998

        temp = t0
        while time.time() < deadline and temp > t_end:
            # Random swap: pick a block in, pick a block out
            i = random.randint(0, len(blocks) - 1)
            old_b = blocks[i]

            # Try to find a new block that covers all unique t-subsets of old_b
            unique_t = np.where((cover_count == 1) & (block_t_masks[old_b] == 1))[0]

            if len(unique_t) == 0:
                # Can just remove
                new_blocks = blocks[:i] + blocks[i+1:]
                new_set = block_set - {old_b}
                new_cover = cover_count - block_t_masks[old_b]
                new_energy = len(new_blocks)
                if np.all(new_cover >= 1):
                    # Accept
                    blocks = new_blocks
                    block_set = new_set
                    cover_count = new_cover
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_blocks = blocks.copy()
                temp *= cooling
                continue

            # Find candidate blocks that cover all unique_t
            candidates = None
            for ut in unique_t:
                cand_set = set(t_to_blocks[ut])
                if candidates is None:
                    candidates = cand_set
                else:
                    candidates &= cand_set

            if not candidates:
                temp *= cooling
                continue

            candidates = [c for c in candidates if c not in block_set]
            if not candidates:
                temp *= cooling
                continue

            new_b = random.choice(candidates)

            # Compute new coverage
            new_cover = cover_count - block_t_masks[old_b] + block_t_masks[new_b]
            new_energy = current_energy  # same size, swap

            if np.all(new_cover >= 1):
                # Valid swap, always accept (or accept with probability)
                delta_e = new_energy - current_energy  # 0
                if random.random() < 1.0:  # always accept valid swaps
                    blocks[i] = new_b
                    block_set.discard(old_b)
                    block_set.add(new_b)
                    cover_count = new_cover
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_blocks = blocks.copy()
            else:
                # Invalid swap, reject
                pass

            temp *= cooling

            # Occasionally try to add a block if we're stuck (shouldn't happen if valid)
            # Or try to remove redundant blocks more aggressively

        return best_blocks, best_energy

    # Run multiple iterations with different seeds
    best_blocks = None
    best_count = float('inf')

    iter_count = 0
    while time.time() < deadline:
        seed = iter_count
        init_blocks = greedy_init(seed)
        if not init_blocks:
            iter_count += 1
            continue

        sa_blocks, sa_count = simulated_annealing(init_blocks, seed + 1000)

        if sa_count < best_count:
            best_count = sa_count
            best_blocks = sa_blocks.copy()

        iter_count += 1
        # Allow more iterations if we're going fast
        if iter_count % 5 == 0 and time.time() > deadline - 5:
            break

    if best_blocks is None:
        return []

    result = [list(all_k_blocks[b_idx]) for b_idx in best_blocks]
    return result
