"""Stronger covering design solver: randomized greedy + local search + deletion."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import Counter

# Deadline guard: kill before 150s wall-clock, with safety margin
_DEADLINE = 145.0  # seconds from start of construct()
_start_time = None


def _elapsed():
    return time.time() - _start_time if _start_time is not None else 0.0


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _all_t_subsets(v, t):
    """Return list of all t-subsets as tuples."""
    return list(itertools.combinations(range(v), t))


def _subsets_in_block(block, t):
    """Return set of t-subsets contained in block."""
    return set(itertools.combinations(block, t))


def _greedy_build(v, k, t, rng, uncovered_set, uncovered_count):
    """
    Greedy: repeatedly pick the k-subset that covers the most uncovered t-subsets.
    Uses a fast heuristic: for each candidate point, estimate marginal gain.
    """
    blocks = []
    # Work with a mutable set of uncovered t-subsets
    unc = set(uncovered_set)
    unc_count = uncovered_count

    while unc and _remaining() > 0.5:
        # Pick a random uncovered t-subset to anchor the block
        anchor = next(iter(rng.sample(list(unc), 1)))
        block = list(anchor)
        rng.shuffle(block)

        # Greedily add points to maximize coverage
        while len(block) < k and _remaining() > 0.3:
            best_x = -1
            best_gain = -1
            # Candidate points not yet in block
            candidates = [x for x in range(v) if x not in block]
            # Shuffle for randomization
            rng.shuffle(candidates)

            # For speed, limit candidates considered per step
            for x in candidates:
                # Compute how many new t-subsets this point would cover
                # A t-subset is covered if it's a subset of block+[x]
                # i.e., it contains x and (t-1) elements from block
                gain = 0
                block_set = set(block)
                for rest in itertools.combinations(block, t - 1):
                    s = tuple(sorted(rest + (x,)))
                    if s in unc:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                # Early break if we found a perfect gain
                if best_gain >= t:  # can't do better than t for a single point
                    break

            if best_x == -1:
                break
            block.append(best_x)

        block.sort()
        # Remove covered subsets
        for s in itertools.combinations(block, t):
            unc.discard(s)
        blocks.append(block)

    return blocks, unc


def _local_search_improve(v, k, t, blocks, rng, max_iters=50):
    """
    Local search: try swapping one point in a block with another to improve coverage,
    or remove redundant blocks.
    """
    if not blocks:
        return blocks

    # Build coverage count for each t-subset
    all_ts = _all_t_subsets(v, t)
    ts_index = {ts: i for i, ts in enumerate(all_ts)}
    n_ts = len(all_ts)

    # coverage[i] = number of blocks covering t-subset i
    coverage = [0] * n_ts
    for blk in blocks:
        for s in itertools.combinations(blk, t):
            coverage[ts_index[s]] += 1

    improved = True
    iters = 0
    while improved and iters < max_iters and _remaining() > 1.0:
        improved = False
        iters += 1

        # Try to remove blocks that are redundant (every t-subset they cover is covered by others)
        for i in range(len(blocks) - 1, -1, -1):
            if _remaining() < 0.5:
                break
            blk = blocks[i]
            redundant = True
            for s in itertools.combinations(blk, t):
                if coverage[ts_index[s]] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                for s in itertools.combinations(blk, t):
                    coverage[ts_index[s]] -= 1
                blocks.pop(i)
                improved = True
                break

        # Try swapping a point in a block to improve coverage of uncovered subsets
        if not improved and _remaining() > 2.0:
            # Find uncovered subsets
            uncovered = [i for i in range(n_ts) if coverage[i] == 0]
            if uncovered:
                # Pick a random uncovered subset
                target_idx = rng.choice(uncovered)
                target_ts = all_ts[target_idx]

                # Find a block that shares at least (t-1) elements with target
                for i, blk in enumerate(blocks):
                    blk_set = set(blk)
                    overlap = len(blk_set.intersection(target_ts))
                    if overlap >= t - 1:
                        # Try swapping a non-overlapping point in blk with a missing point in target
                        missing = [x for x in target_ts if x not in blk_set]
                        extra = [x for x in blk if x not in target_ts]
                        if missing and extra:
                            # Try each swap
                            for x_out in extra:
                                for x_in in missing:
                                    new_blk = sorted([x if x != x_out else x_in for x in blk])
                                    if len(set(new_blk)) != k:
                                        continue
                                    # Compute gain: new t-subsets covered
                                    new_subs = set(itertools.combinations(new_blk, t))
                                    old_subs = set(itertools.combinations(blk, t))
                                    gain = sum(1 for s in new_subs if coverage[ts_index[s]] == 0)
                                    # Check we don't lose any coverage
                                    loss = sum(1 for s in old_subs if coverage[ts_index[s]] == 1 and s not in new_subs)
                                    if gain > loss:
                                        # Apply swap
                                        for s in old_subs:
                                            coverage[ts_index[s]] -= 1
                                        for s in new_subs:
                                            coverage[ts_index[s]] += 1
                                        blocks[i] = new_blk
                                        improved = True
                                        break
                                if improved:
                                    break
                        if improved:
                            break

    return blocks


def construct(v, k, t):
    global _start_time
    _start_time = time.time()

    rng = random.Random(42)

    all_ts = _all_t_subsets(v, t)
    n_ts = len(all_ts)

    # Strategy: multiple randomized greedy restarts, keep best, then local search
    best_blocks = None
    best_count = float('inf')

    # Number of restarts based on time budget
    # Each restart is O(n_ts * v * C(k-1, t-1)) roughly
    max_restarts = 5
    min_time_per_restart = 1.0

    for restart in range(max_restarts):
        if _remaining() < min_time_per_restart + 2.0:
            break

        # Randomized greedy
        blocks, remaining_unc = _greedy_build(v, k, t, rng, set(all_ts), n_ts)

        if not blocks:
            continue

        # Local search improvement
        blocks = _local_search_improve(v, k, t, blocks, rng)

        # Verify coverage
        cov = [0] * n_ts
        ts_index = {ts: i for i, ts in enumerate(all_ts)}
        for blk in blocks:
            for s in itertools.combinations(blk, t):
                cov[ts_index[s]] += 1
        if all(c > 0 for c in cov):
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks

        if _remaining() < 3.0:
            break

    if best_blocks is None:
        # Fallback: simple greedy
        best_blocks, _ = _greedy_build(v, k, t, rng, set(all_ts), n_ts)
        best_blocks = _local_search_improve(v, k, t, best_blocks, rng)

    return best_blocks
# EVOLVE-BLOCK-END
