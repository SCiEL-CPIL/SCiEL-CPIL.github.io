import itertools
import time
import random
import sys
import math
import numpy as np

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    # Parameters specific to this task: v=19, k=8, t=5
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def get_elapsed():
        return time.time() - start_time

    def check_time(budget=10.0):
        return get_elapsed() > deadline - budget

    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)  # 11628

    # Map each t-subset to an index
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)  # 75582

    # Precompute t-subsets for each block
    block_tsubs = []
    for b in all_blocks:
        tsubs_in_b = [tsub_index[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(tsubs_in_b)

    # Inverse: for each t-subset, which blocks cover it?
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for i, b_tsubs in enumerate(block_tsubs):
        for j in b_tsubs:
            tsub_blocks[j].append(i)

    # NumPy arrays for fast operations
    block_tsubs_np = np.array(block_tsubs, dtype=np.int32)
    tsub_blocks_np = [np.array(tb, dtype=np.int32) for tb in tsub_blocks]

    # Weighted greedy: prioritize blocks that cover rare uncovered t-subsets
    def weighted_greedy(random_seed=None):
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random

        uncovered = np.ones(num_tsubs, dtype=np.uint8)
        uncovered_count = num_tsubs
        selected_blocks = []

        # Frequency of each t-subset among all blocks
        freq = np.zeros(num_tsubs, dtype=np.int32)
        for i in range(num_blocks):
            freq[block_tsubs_np[i]] += 1

        while uncovered_count > 0:
            if check_time(15.0):
                return None

            # Find the rarest uncovered t-subset
            # For efficiency, sample a few uncovered t-subsets and pick the rarest
            if uncovered_count < 200:
                uncov_indices = np.where(uncovered == 1)[0]
                if len(uncov_indices) == 0:
                    break
                # Pick the one with minimum frequency
                sample = uncov_indices[:min(50, len(uncov_indices))]
                s_idx = int(sample[np.argmin(freq[sample])])
            else:
                # Sample a few random uncovered t-subsets
                best_s = -1
                best_freq = float('inf')
                for _ in range(20):
                    while True:
                        s_idx = rng.randint(0, num_tsubs - 1)
                        if uncovered[s_idx]:
                            break
                    if freq[s_idx] < best_freq:
                        best_freq = freq[s_idx]
                        best_s = s_idx
                s_idx = best_s

            # Get candidate blocks
            candidates = tsub_blocks_np[s_idx]

            if len(candidates) == 0:
                return None

            # Compute gain for each candidate block
            cand_tsubs = block_tsubs_np[candidates]
            gains = np.sum(uncovered[cand_tsubs], axis=1)

            # Add a small random perturbation to break ties and explore
            if random_seed is not None:
                perturbation = rng.random(len(gains)) * 0.1
                gains = gains + perturbation

            best_local = int(np.argmax(gains))
            best_block_idx = int(candidates[best_local])

            selected_blocks.append(best_block_idx)

            # Update uncovered
            for j in block_tsubs[best_block_idx]:
                if uncovered[j]:
                    uncovered[j] = 0
                    uncovered_count -= 1

        if uncovered_count > 0:
            return None

        return selected_blocks

    def reduce_blocks(block_indices):
        """
        Reduce the number of blocks while maintaining coverage.
        """
        block_indices = list(block_indices)

        # Build cover count
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in block_indices:
            cover_count[block_tsubs_np[b_idx]] += 1

        changed = True
        max_passes = 30
        for _ in range(max_passes):
            if check_time(10.0):
                break

            changed = False
            new_block_indices = []
            for b_idx in block_indices:
                b_tsubs = block_tsubs_np[b_idx]
                if np.any(cover_count[b_tsubs] == 1):
                    new_block_indices.append(b_idx)
                else:
                    changed = True
                    cover_count[b_tsubs] -= 1

            block_indices = new_block_indices

            if not changed:
                break

        # Try local search: replace a block with another that enables removals
        for _ in range(10):
            if check_time(10.0):
                break

            improved = False
            for i, b_idx in enumerate(block_indices):
                if check_time(10.0):
                    break

                # Try replacing block i with another block
                b_tsubs = block_tsubs_np[b_idx]

                # Find blocks that share many t-subsets with b_idx
                # and could potentially allow removing another block
                # Simple heuristic: try a few random blocks
                for _ in range(5):
                    new_b_idx = random.randint(0, num_blocks - 1)
                    if new_b_idx == b_idx:
                        continue

                    # Check if replacing b_idx with new_b_idx maintains coverage
                    # and potentially allows removing another block
                    # This is expensive, so do a quick check
                    new_cover_count = cover_count.copy()
                    new_cover_count[block_tsubs_np[b_idx]] -= 1
                    new_cover_count[block_tsubs_np[new_b_idx]] += 1

                    if np.any(new_cover_count < 0):
                        continue

                    # Check if this replacement allows removing another block
                    # Find blocks that are now redundant
                    removable = -1
                    for j, other_b_idx in enumerate(block_indices):
                        if j == i:
                            continue
                        other_tsubs = block_tsubs_np[other_b_idx]
                        if np.all(new_cover_count[other_tsubs] >= 2):
                            removable = j
                            break

                    if removable != -1:
                        # Perform the replacement and removal
                        cover_count = new_cover_count
                        block_indices[i] = new_b_idx
                        block_indices.pop(removable)
                        improved = True
                        break

                if improved:
                    break

            if not improved:
                break

        return block_indices

    # Main loop: multiple restarts with different strategies
    best_len = float('inf')
    best_block_list = []

    # First, run a deterministic weighted greedy
    initial = weighted_greedy(random_seed=42)
    if initial is not None:
        reduced = reduce_blocks(initial)
        if len(reduced) < best_len:
            best_len = len(reduced)
            best_block_list = [all_blocks[i] for i in reduced]

    # Now, run randomized restarts
    num_restarts = 150
    for restart in range(num_restarts):
        if check_time(15.0):
            break

        seed = restart * 1000 + 7
        selected = weighted_greedy(random_seed=seed)
        if selected is None:
            continue

        reduced = reduce_blocks(selected)
        if len(reduced) < best_len:
            best_len = len(reduced)
            best_block_list = [all_blocks[i] for i in reduced]

            # Early exit if we hit a very good number
            if best_len <= 320:
                break

    # If we still have time, do more aggressive reduction on the best solution
    if best_block_list and not check_time(30.0):
        best_indices = [all_blocks.index(b) for b in best_block_list]

        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in best_indices:
            cover_count[block_tsubs_np[b_idx]] += 1

        for _ in range(50):
            if check_time(10.0):
                break
            changed = False
            new_indices = []
            for b_idx in best_indices:
                b_tsubs = block_tsubs_np[b_idx]
                if np.any(cover_count[b_tsubs] == 1):
                    new_indices.append(b_idx)
                else:
                    changed = True
                    cover_count[b_tsubs] -= 1
            best_indices = new_indices
            if not changed:
                break

        if len(best_indices) < best_len:
            best_len = len(best_indices)
            best_block_list = [all_blocks[i] for i in best_indices]

    # Return the best solution
    return [list(b) for b in best_block_list]
