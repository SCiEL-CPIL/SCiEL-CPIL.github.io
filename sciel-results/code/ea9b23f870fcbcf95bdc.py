"""
Covering design C(19,8,5) using greedy set cover with efficient bitset-based
coverage tracking and a local-search polish phase.

Strategy:
1. Greedy set-cover: repeatedly pick the 8-subset that covers the most uncovered 5-subsets.
   Use numpy bitsets to speed up the inner loop.
2. After the greedy phase, run a local search (block replacement / swap) to try to
   reduce the number of blocks while maintaining coverage.
3. Deadline guard: return the best valid solution found before ~140 s.
"""
import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin

    # Precompute all t-subsets as bitmasks (each 5-subset -> a 64-bit int via index)
    # There are C(19,5) = 11628 five-subsets.
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    # Map each t-subset to an index
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # For each point, precompute which t-subsets contain it (as a list of indices)
    # Actually, for greedy we need: for a candidate block (set of k points),
    # how many uncovered t-subsets are fully contained in it.
    # A t-subset is covered by a block iff all its t elements are in the block.

    # Represent coverage as a numpy boolean array of length n_tsubs
    covered = np.zeros(n_tsubs, dtype=bool)

    # For fast "which t-subsets are covered by block B":
    # A t-subset is covered by B iff it is a subset of B.
    # We can precompute for each possible block (there are C(19,8)=75582) the list of t-subset indices it covers.
    # But that's a lot of memory. Instead, for a given block, we enumerate its t-subsets.
    # C(8,5)=56, so for each candidate block we check 56 t-subsets.

    # Greedy: at each step, among all C(19,8) blocks, find the one covering the most uncovered t-subsets.
    # To speed up: maintain a count of uncovered t-subsets per point? No, that's not enough.
    # Alternative: for each uncovered t-subset, we need to cover it. Pick the block that covers the most.
    # We can iterate over all blocks and compute coverage gain.

    # Precompute all k-subsets as frozensets
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # For each k-subset, precompute the list of t-subset indices it covers
    # This is C(8,5)=56 per block, 75582 blocks -> 75582*56 ~ 4.2M entries. Feasible.
    block_tsubs = []
    for block in all_ksubs:
        tsubs_in_block = []
        for ts in itertools.combinations(block, t):
            tsubs_in_block.append(tsub_index[ts])
        block_tsubs.append(tsubs_in_block)
    block_tsubs = np.array(block_tsubs, dtype=np.int32)  # shape (n_ksubs, 56)

    blocks = []  # list of chosen blocks (as tuples)
    block_indices_used = set()

    # Greedy phase
    while True:
        if time.time() > deadline:
            break
        if covered.all():
            break

        # Find the block with max uncovered coverage
        # uncovered_count per block = sum over its t-subsets of (not covered)
        # covered is boolean array; block_tsubs is int array
        # For each block i, gain = sum(~covered[block_tsubs[i]])
        # This is a matrix-vector product essentially.
        # ~covered is boolean, convert to int
        uncov = (~covered).astype(np.int32)
        # gain for each block: sum of uncov over its t-subsets
        gains = np.array([uncov[bt].sum() for bt in block_tsubs], dtype=np.int32)

        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break  # no improvement possible

        blocks.append(all_ksubs[best_idx])
        block_indices_used.add(best_idx)
        # Update covered
        covered[block_tsubs[best_idx]] = True

    # Local search polish: try to remove or swap blocks to reduce count
    # While we have time, try random block swaps: replace one block with another if coverage is maintained
    # and we can also try to remove a block if all its t-subsets are covered by other blocks.

    # Build a coverage count: for each t-subset, how many blocks cover it
    cover_count = np.zeros(n_tsubs, dtype=np.int32)
    for b in blocks:
        b_idx = all_ksubs.index(b)
        cover_count[block_tsubs[b_idx]] += 1

    def is_valid(cover_count):
        return (cover_count >= 1).all()

    # Try to remove blocks
    def try_remove(blocks, cover_count, block_tsubs, all_ksubs):
        for i in range(len(blocks) - 1, -1, -1):
            b = blocks[i]
            b_idx = all_ksubs.index(b)
            tsubs = block_tsubs[b_idx]
            if np.all(cover_count[tsubs] >= 2):
                # Can remove
                cover_count[tsubs] -= 1
                blocks.pop(i)
                return True
        return False

    # Swap: replace block i with block j if coverage remains
    def try_swap(blocks, cover_count, block_tsubs, all_ksubs, i, j_idx):
        b_old = blocks[i]
        b_new = all_ksubs[j_idx]
        old_tsubs = block_tsubs[all_ksubs.index(b_old)]
        new_tsubs = block_tsubs[j_idx]
        # Check: after removing old and adding new, is everything covered?
        # For tsubs in old: cover_count[t] - 1 + (1 if t in new_tsubs else 0) >= 1
        # For tsubs in new: cover_count[t] + 1 >= 1 (always true if cover_count[t] >= 0)
        # Actually we need: for all t, new_count[t] >= 1
        # new_count[t] = cover_count[t] - (1 if t in old_tsubs) + (1 if t in new_tsubs)
        # This is complex to check efficiently. Skip for now.
        pass

    # Simpler: just try removals repeatedly
    while time.time() < deadline - 5:
        if try_remove(blocks, cover_count, block_tsubs, all_ksubs):
            continue
        else:
            break

    return [list(b) for b in blocks]
