"""Covering design C(v,k,t) via greedy construction with simulated annealing refinement."""
import itertools
import random
import time
import math

def construct(v, k, t):
    deadline = time.time() + 115.0  # safety margin before 120s kill
    rng = random.Random(42)

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)

    # Map each t-subset to an index for fast lookup
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Greedy construction
    uncovered = [True] * num_tsubs
    uncovered_count = num_tsubs
    blocks = []

    # For efficiency, maintain a list of uncovered t-subsets
    uncovered_list = list(all_tsubs)

    while uncovered_list:
        # Pick a random uncovered t-subset (or the first for determinism in first pass)
        # For better initial solution, pick one that has many potential extensions
        # Start with the first uncovered for simplicity
        start_sub = uncovered_list[0]
        block = list(start_sub)

        # Greedily add points to maximize coverage
        while len(block) < k:
            block_set = set(block)
            best_point = None
            best_gain = -1

            # Try all points not in block
            for x in range(v):
                if x in block_set:
                    continue
                # Count how many uncovered t-subsets this point would complete
                gain = 0
                # A t-subset containing x and (t-1) elements from block is covered
                for rest in itertools.combinations(block, t - 1):
                    candidate = tuple(sorted(rest + (x,)))
                    idx = tsub_to_idx.get(candidate)
                    if idx is not None and uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x

            if best_point is None:
                break
            block.append(best_point)

        block.sort()
        blocks.append(tuple(block))

        # Mark all t-subsets in this block as covered
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            if uncovered[idx]:
                uncovered[idx] = False
                uncovered_count -= 1

        # Rebuild uncovered_list
        uncovered_list = [ts for i, ts in enumerate(all_tsubs) if uncovered[i]]

        # Check deadline
        if time.time() > deadline - 60:
            break

    # Convert to list of lists
    blocks = [list(b) for b in blocks]

    # Now try to improve with simulated annealing / local search
    # State: list of blocks
    # Objective: minimize number of blocks (but must remain a valid covering)
    # Move: try to remove a block if the covering remains valid
    # Move: swap a block with a new one that covers more

    # Build coverage count array
    def build_coverage(blocks):
        cov_count = [0] * num_tsubs
        for block in blocks:
            for ts in itertools.combinations(block, t):
                cov_count[tsub_to_idx[ts]] += 1
        return cov_count

    # Check if valid
    def is_valid(cov_count):
        return all(c > 0 for c in cov_count)

    # Current coverage
    cov_count = build_coverage(blocks)

    # Try to remove blocks that are redundant
    # A block is redundant if removing it still leaves all t-subsets covered
    # i.e., every t-subset in the block is covered by at least one other block

    # For each block, compute how many t-subsets it uniquely covers
    # If zero, we can remove it

    # Iteratively remove redundant blocks
    changed = True
    while changed and time.time() < deadline - 30:
        changed = False
        cov_count = build_coverage(blocks)
        for i in range(len(blocks) - 1, -1, -1):
            block = blocks[i]
            # Check if all t-subsets in this block are covered by other blocks
            redundant = True
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                if cov_count[idx] == 1:
                    redundant = False
                    break
            if redundant:
                blocks.pop(i)
                changed = True
                cov_count = build_coverage(blocks)

    # Now try simulated annealing to replace blocks with better ones
    # Temperature schedule
    T = 1.0
    cooling = 0.999
    min_T = 0.01

    current_blocks = list(blocks)
    current_cov = build_coverage(current_blocks)

    # Precompute for each block, the set of t-subset indices it covers
    def block_tsub_indices(block):
        return [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]

    # For SA, we'll try:
    # 1. Remove a block and replace with a new one that covers more uncovered
    # 2. Swap two blocks
    # 3. Add a new block (only if we can remove one later)

    # Simpler: try to find a block that covers more t-subsets than an existing one
    # and replace it, if the covering remains valid

    best_blocks = list(current_blocks)
    best_len = len(best_blocks)

    sa_steps = 0
    while time.time() < deadline - 10 and T > min_T:
        sa_steps += 1

        # Pick a random block to potentially replace
        if not current_blocks:
            break

        i = rng.randrange(len(current_blocks))
        old_block = current_blocks[i]
        old_tsubs = set(block_tsub_indices(old_block))

        # Generate a new candidate block
        # Start from a random t-subset and greedily extend
        start_idx = rng.randrange(num_tsubs)
        start_sub = all_tsubs[start_idx]
        new_block = list(start_sub)

        while len(new_block) < k:
            new_block_set = set(new_block)
            best_point = None
            best_gain = -1
            for x in range(v):
                if x in new_block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(new_block, t - 1):
                    candidate = tuple(sorted(rest + (x,)))
                    idx = tsub_to_idx.get(candidate)
                    if idx is not None:
                        # Count if this would add new coverage
                        if current_cov[idx] == 0:
                            gain += 10  # Weight uncovered more
                        elif current_cov[idx] == 1:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point is None:
                break
            new_block.append(best_point)

        new_block.sort()
        new_block = tuple(new_block)

        if new_block == old_block:
            T *= cooling
            continue

        # Compute coverage change
        new_tsubs = set(block_tsub_indices(new_block))

        # Simulate replacing old_block with new_block
        # t-subsets only in old_block lose one cover
        # t-subsets only in new_block gain one cover
        # t-subsets in both: no change
        # t-subsets in neither: no change

        lost = old_tsubs - new_tsubs
        gained = new_tsubs - old_tsubs

        # Check validity: no t-subset should drop to 0
        valid = True
        for idx in lost:
            if current_cov[idx] == 1:
                valid = False
                break

        if valid:
            # Accept
            current_blocks[i] = list(new_block)
            # Update coverage
            for idx in lost:
                current_cov[idx] -= 1
            for idx in gained:
                current_cov[idx] += 1
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = [list(b) for b in current_blocks]
        else:
            # Reject (or accept with probability exp(-delta/T) but delta is infinite here)
            pass

        T *= cooling

    # Final: try to remove redundant blocks again
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        cov_count = build_coverage(best_blocks)
        for i in range(len(best_blocks) - 1, -1, -1):
            block = best_blocks[i]
            redundant = True
            for ts in itertools.combinations(block, t):
                idx = tsub_to_idx[ts]
                if cov_count[idx] == 1:
                    redundant = False
                    break
            if redundant:
                best_blocks.pop(i)
                changed = True

    return [list(b) for b in best_blocks]
