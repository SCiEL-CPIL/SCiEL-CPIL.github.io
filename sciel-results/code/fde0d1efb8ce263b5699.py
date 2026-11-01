"""Combined approach: greedy init + constraint-based 1-exchange local search."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    all_points = list(range(v))
    all_tsubs = list(itertools.combinations(all_points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)

    # Precompute all k-blocks and their covered t-subset indices
    all_blocks = list(itertools.combinations(all_points, k))
    block_to_tsubs = {}
    tsub_to_blocks = [[] for _ in range(n_tsubs)]

    for b in all_blocks:
        covered = []
        for ts in itertools.combinations(b, t):
            idx = tsub_to_idx[ts]
            covered.append(idx)
            tsub_to_blocks[idx].append(b)
        block_to_tsubs[b] = covered

    def elapsed():
        return time.time() - start_time

    def out_of_time():
        return elapsed() > deadline

    # Phase 1: Greedy Set Cover initialization
    def greedy_init():
        coverage = [0] * n_tsubs
        blocks = []
        block_set = set()
        n_uncovered = n_tsubs

        while n_uncovered > 0 and not out_of_time():
            best_block = None
            best_gain = -1
            for b in all_blocks:
                if b in block_set:
                    continue
                gain = 0
                for idx in block_to_tsubs[b]:
                    if coverage[idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
            if best_block is None or best_gain == 0:
                break
            blocks.append(best_block)
            block_set.add(best_block)
            for idx in block_to_tsubs[best_block]:
                if coverage[idx] == 0:
                    n_uncovered -= 1
                coverage[idx] += 1

        return blocks, coverage

    blocks, coverage = greedy_init()

    if out_of_time():
        return blocks

    # Helper functions for local search
    def is_redundant(b):
        for idx in block_to_tsubs[b]:
            if coverage[idx] == 1:
                return False
        return True

    def remove_block(b):
        blocks.remove(b)
        for idx in block_to_tsubs[b]:
            coverage[idx] -= 1

    def add_block(b):
        blocks.append(b)
        for idx in block_to_tsubs[b]:
            coverage[idx] += 1

    # Phase 2: Remove redundant blocks (cascade)
    changed = True
    while changed and not out_of_time():
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            if out_of_time():
                break
            if is_redundant(blocks[i]):
                remove_block(blocks[i])
                changed = True

    # Phase 3: Constraint-based 1-exchange local search
    # For each block, find its critical t-subsets (covered only by this block).
    # A replacement block must cover ALL critical t-subsets.
    # Use intersection of tsub_to_blocks to find candidates quickly.
    def one_exchange_pass():
        nonlocal blocks, coverage
        improved = False
        shuffled = blocks[:]
        random.shuffle(shuffled)

        for b in shuffled:
            if out_of_time():
                break
            if b not in blocks:
                continue  # was removed by a previous swap

            # Find critical t-subsets for b
            critical = [idx for idx in block_to_tsubs[b] if coverage[idx] == 1]

            if not critical:
                # b is redundant now
                remove_block(b)
                improved = True
                continue

            # Find candidate blocks that cover ALL critical t-subsets
            # Intersection of tsub_to_blocks for each critical idx
            candidates = None
            for idx in critical:
                covering = tsub_to_blocks[idx]
                if candidates is None:
                    candidates = set(covering)
                else:
                    candidates.intersection_update(covering)

            if not candidates:
                continue

            # Exclude blocks already in the set
            current_set = set(blocks)
            candidates -= current_set

            if not candidates:
                continue

            # Pick a candidate (prefer one with fewer "wasted" t-subsets, i.e.,
            # t-subsets already covered by other blocks, to minimize redundancy)
            best_cand = None
            best_waste = float('inf')
            for cand in candidates:
                waste = 0
                for idx in block_to_tsubs[cand]:
                    if coverage[idx] >= 1:
                        waste += 1
                if waste < best_waste:
                    best_waste = waste
                    best_cand = cand

            if best_cand is None:
                continue

            # Perform the swap: remove b, add best_cand
            remove_block(b)
            add_block(best_cand)
            improved = True

            # After swap, try to remove any newly-redundant blocks
            for i in range(len(blocks) - 1, -1, -1):
                if out_of_time():
                    break
                if is_redundant(blocks[i]):
                    remove_block(blocks[i])
                    improved = True

        return improved

    # Run multiple passes of 1-exchange until no improvement or time runs out
    pass_count = 0
    while not out_of_time():
        improved = one_exchange_pass()
        pass_count += 1
        if not improved:
            break
        # Occasionally re-run redundancy removal to catch cascades
        if pass_count % 3 == 0:
            changed = True
            while changed and not out_of_time():
                changed = False
                for i in range(len(blocks) - 1, -1, -1):
                    if out_of_time():
                        break
                    if is_redundant(blocks[i]):
                        remove_block(blocks[i])
                        changed = True

    return blocks
# EVOLVE-BLOCK-END
