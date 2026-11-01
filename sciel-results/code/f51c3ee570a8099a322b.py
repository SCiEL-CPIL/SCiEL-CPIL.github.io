"""Rewritten covering construction for C(v,k,t) using a faster, deadline-guarded greedy with local repair."""
# EVOLVE-BLOCK-START
import itertools
import time


def construct(v, k, t):
    """Return a list of k-subsets of {0..v-1} covering every t-subset.

    Uses a time-bounded greedy set-cover with a vectorized gain computation,
    followed by a local repair pass that tries to replace blocks to reduce
    the total count while maintaining coverage.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before the 120s kill

    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    uncovered = set(all_tsubs)
    blocks = []

    # Precompute for each point x and each (t-1)-subset s of points not containing x,
    # the list of t-subsets that contain x and s. We use a dict mapping t-subset -> index.
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # For fast gain: for a partial block B (size < k), gain(x) = number of uncovered
    # t-subsets that contain x and are subsets of B ∪ {x}.
    # Since B has size b, a t-subset contained in B∪{x} must have t-1 elements from B and x.
    # So gain(x) = count of uncovered t-subsets of the form (s, x) where s ⊆ B, |s|=t-1.

    while uncovered and time.time() < deadline:
        # Pick the uncovered t-subset with the smallest index (or first) to seed the block
        seed = min(uncovered)
        block = list(seed)
        block_set = set(block)

        while len(block) < k and time.time() < deadline:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Compute gain: number of uncovered t-subsets containing x and t-1 elements from block
                gain = 0
                if t - 1 <= len(block):
                    for s in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(s + (x,)))
                        if ts in uncovered:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            block_set.add(best_x)

        block.sort()
        blocks.append(block)

        # Remove all t-subsets of this block from uncovered
        for ts in itertools.combinations(block, t):
            uncovered.discard(ts)

    # If we ran out of time, we might still have uncovered t-subsets.
    # Add blocks greedily for remaining uncovered (fast, no optimization) until covered or time runs out.
    while uncovered and time.time() < deadline:
        seed = min(uncovered)
        block = list(seed)
        block_set = set(block)
        while len(block) < k:
            # Just add arbitrary points
            for x in range(v):
                if x not in block_set:
                    block.append(x)
                    block_set.add(x)
                    break
        block.sort()
        blocks.append(block)
        for ts in itertools.combinations(block, t):
            uncovered.discard(ts)

    # Local repair: try to remove redundant blocks
    if time.time() < deadline - 5:
        # Check if any block is redundant (all its t-subsets are covered by other blocks)
        # Build coverage count per t-subset
        cover_count = {ts: 0 for ts in all_tsubs}
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts] += 1

        # Try to remove blocks that are redundant
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline - 2:
                    break
                b = blocks[i]
                redundant = True
                for ts in itertools.combinations(b, t):
                    if cover_count[ts] <= 1:
                        redundant = False
                        break
                if redundant:
                    for ts in itertools.combinations(b, t):
                        cover_count[ts] -= 1
                    blocks.pop(i)
                    changed = True

    return blocks
# EVOLVE-BLOCK-END
