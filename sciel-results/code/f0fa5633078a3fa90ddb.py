import itertools
import time
import random
import sys
import math

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    # Parameters specific to this task: v=19, k=8, t=5
    start_time = time.time()
    deadline = start_time + 145.0

    def get_elapsed():
        return time.time() - start_time

    # Precompute all t-subsets and map them to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # Precompute block_tsubs: for each block, list of t-subset indices it contains
    block_tsubs = []
    for b in all_blocks:
        tsubs_in_b = [tsub_index[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(tsubs_in_b)

    # Inverse mapping: for each t-subset, which blocks cover it?
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for i, b_tsubs in enumerate(block_tsubs):
        for j in b_tsubs:
            tsub_blocks[j].append(i)

    # Greedy construction
    def greedy_construct(randomize=False, seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()

        uncovered = bytearray([1] * num_tsubs)
        uncovered_count = num_tsubs
        selected_blocks = []

        while uncovered_count > 0:
            if get_elapsed() > deadline - 10.0:
                break

            # Find an uncovered t-subset
            if randomize:
                # Randomly pick an uncovered t-subset
                if uncovered_count < 1000:
                    uncov_list = [i for i in range(num_tsubs) if uncovered[i]]
                    if not uncov_list:
                        break
                    s_idx = rng.choice(uncov_list)
                else:
                    while True:
                        s_idx = rng.randint(0, num_tsubs - 1)
                        if uncovered[s_idx]:
                            break
            else:
                s_idx = -1
                for i in range(num_tsubs):
                    if uncovered[i]:
                        s_idx = i
                        break
                if s_idx == -1:
                    break

            # Find best block covering s_idx
            candidates = tsub_blocks[s_idx]
            if randomize:
                candidates = list(candidates)
                rng.shuffle(candidates)

            best_block_idx = -1
            best_gain = -1

            for b_idx in candidates:
                gain = 0
                for j in block_tsubs[b_idx]:
                    if uncovered[j]:
                        gain += 1
                        if gain == 56:  # C(8,5)
                            break

                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx

            selected_blocks.append(best_block_idx)

            # Update uncovered
            for j in block_tsubs[best_block_idx]:
                if uncovered[j]:
                    uncovered[j] = 0
                    uncovered_count -= 1

        return selected_blocks

    # Local search: iterative removal of redundant blocks
    def local_search(block_indices):
        cover_count = [0] * num_tsubs
        for b_idx in block_indices:
            for j in block_tsubs[b_idx]:
                cover_count[j] += 1

        changed = True
        while changed:
            changed = False
            new_block_indices = []
            for b_idx in block_indices:
                is_essential = False
                for j in block_tsubs[b_idx]:
                    if cover_count[j] == 1:
                        is_essential = True
                        break

                if is_essential:
                    new_block_indices.append(b_idx)
                else:
                    changed = True
                    for j in block_tsubs[b_idx]:
                        cover_count[j] -= 1

            block_indices = new_block_indices

            if get_elapsed() > deadline - 5.0:
                break

        return block_indices

    # Swap-based local search: try replacing blocks to enable further removals
    def swap_local_search(block_indices):
        block_indices = list(block_indices)
        n = len(block_indices)

        # Compute coverage counts
        cover_count = [0] * num_tsubs
        for b_idx in block_indices:
            for j in block_tsubs[b_idx]:
                cover_count[j] += 1

        improved = True
        while improved:
            improved = False

            # Try to find a pair of blocks where swapping one for another
            # allows removal of the second
            for i in range(n):
                if get_elapsed() > deadline - 5.0:
                    return block_indices
                if i >= len(block_indices):
                    break

                # Try removing block i and see if we can add a replacement
                # that covers all t-subsets that would become uncovered
                current_b = block_indices[i]

                # Find t-subsets that would be uncovered if we remove current_b
                would_be_uncovered = []
                for j in block_tsubs[current_b]:
                    if cover_count[j] == 1:
                        would_be_uncovered.append(j)

                if not would_be_uncovered:
                    # This block is redundant, remove it
                    for j in block_tsubs[current_b]:
                        cover_count[j] -= 1
                    block_indices.pop(i)
                    n = len(block_indices)
                    improved = True
                    break

                # Try to find a replacement block that covers all would_be_uncovered
                # and is not already in block_indices
                current_set = set(block_indices)

                # Candidate blocks: those that cover at least one of would_be_uncovered
                candidate_blocks = set()
                for j in would_be_uncovered:
                    for b in tsub_blocks[j]:
                        if b not in current_set:
                            candidate_blocks.add(b)

                best_replacement = -1
                best_score = -1

                for b_idx in candidate_blocks:
                    # Check if this block covers all would_be_uncovered
                    covers_all = True
                    for j in would_be_uncovered:
                        if j not in block_tsubs[b_idx]:
                            covers_all = False
                            break

                    if not covers_all:
                        continue

                    # Score: how many additional t-subsets it covers (that are currently covered once)
                    extra_gain = 0
                    for j in block_tsubs[b_idx]:
                        if j in cover_count and cover_count[j] == 1:
                            extra_gain += 1

                    if extra_gain > best_score:
                        best_score = extra_gain
                        best_replacement = b_idx

                if best_replacement != -1:
                    # Perform the swap
                    old_b = block_indices[i]
                    block_indices[i] = best_replacement

                    # Update cover counts
                    for j in block_tsubs[old_b]:
                        cover_count[j] -= 1
                    for j in block_tsubs[best_replacement]:
                        cover_count[j] += 1

                    improved = True
                    break

        return block_indices

    # Main: try multiple strategies
    best_blocks = None
    best_len = float('inf')

    # Strategy 1: Deterministic greedy + local search
    initial = greedy_construct(randomize=False)
    optimized = local_search(initial)
    optimized = swap_local_search(optimized)
    if len(optimized) < best_len:
        best_len = len(optimized)
        best_blocks = optimized

    # Strategy 2: Multiple randomized greedy restarts
    num_restarts = 30
    for restart in range(num_restarts):
        if get_elapsed() > deadline - 10.0:
            break

        seed = restart * 7919 + 13
        initial = greedy_construct(randomize=True, seed=seed)
        optimized = local_search(initial)
        optimized = swap_local_search(optimized)

        if len(optimized) < best_len:
            best_len = len(optimized)
            best_blocks = optimized

        if best_len <= 325:
            break

    # Final pass: try to improve best_blocks further
    if best_blocks is not None and get_elapsed() < deadline - 5.0:
        final = local_search(list(best_blocks))
        final = swap_local_search(final)
        if len(final) < best_len:
            best_len = len(final)
            best_blocks = final

    # Convert indices to blocks
    if best_blocks is None:
        # Fallback: simple greedy
        best_blocks = greedy_construct(randomize=False)

    return [list(all_blocks[i]) for i in best_blocks]
