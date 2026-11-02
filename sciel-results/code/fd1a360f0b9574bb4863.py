"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import math
from collections import Counter

# Deadline guard: evaluator kills after 150s; return best valid result with margin.
_DEADLINE = 150.0
_START = time.monotonic()


def _remaining():
    return _DEADLINE - (time.monotonic() - _START)


def construct(v, k, t):
    global _START
    _START = time.monotonic()
    _DEADLINE_LOCAL = 150.0

    # Precompute all t-subsets and their bitmasks for fast membership via integer sets.
    # For v=19, k=8, t=5: C(19,5)=11628 subsets, each fits in 19 bits.
    # We represent each t-subset as a frozenset of ints for set operations,
    # but for speed we also maintain a coverage count array.

    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)

    # coverage[i] = number of blocks covering t-subset i
    coverage = [0] * n_tsubs
    uncovered_count = n_tsubs

    blocks = []
    rng = random.Random(42)

    def add_block(block):
        nonlocal uncovered_count
        block = sorted(block)
        for ts in itertools.combinations(block, t):
            idx = tsub_index[ts]
            if coverage[idx] == 0:
                uncovered_count -= 1
            coverage[idx] += 1
        return block

    def gain_of_point(block_set, point):
        # Number of currently-uncovered t-subsets that would be covered by adding `point`
        # to the current block (block already has some points; we need t-1 more from block_set
        # plus this point to form a t-subset).
        # A t-subset S is covered by block_set ∪ {point} iff S ⊆ block_set ∪ {point} and
        # point ∈ S. So S must contain `point` and S \\ {point} ⊆ block_set.
        gain = 0
        for rest in itertools.combinations(block_set, t - 1):
            ts = tuple(sorted(rest + (point,)))
            idx = tsub_index.get(ts)
            if idx is not None and coverage[idx] == 0:
                gain += 1
        return gain

    def greedy_block(start_point=None, randomize=False):
        """Build one block greedily. Returns a sorted list of k points."""
        if start_point is None:
            # Pick the most uncovered t-subset's first element as a hint, or random
            if randomize:
                start_point = rng.randint(0, v - 1)
            else:
                # Find the t-subset with highest 'need' (all uncovered) and pick its min
                best_ts = None
                for ts in all_tsubs:
                    idx = tsub_index[ts]
                    if coverage[idx] == 0:
                        best_ts = ts
                        break
                start_point = best_ts[0] if best_ts else rng.randint(0, v - 1)

        block = [start_point]
        block_set = {start_point}

        while len(block) < k:
            best_point = None
            best_gain = -1
            candidates = [x for x in range(v) if x not in block_set]
            if randomize and rng.random() < 0.3:
                # Sometimes pick randomly among top-3 to diversify
                scored = sorted(candidates, key=lambda x: gain_of_point(block_set, x), reverse=True)
                top3 = scored[:3]
                best_point = rng.choice(top3)
            else:
                for x in candidates:
                    g = gain_of_point(block_set, x)
                    if g > best_gain:
                        best_gain = g
                        best_point = x
            block.append(best_point)
            block_set.add(best_point)

        return sorted(block)

    # Phase 1: Pure greedy until covered or deadline
    while uncovered_count > 0 and _remaining() > 10.0:
        b = greedy_block(randomize=False)
        add_block(b)
        blocks.append(b)

    if uncovered_count == 0:
        return blocks

    # Phase 2: Local improvement — try to replace some blocks with better ones
    # Strategy: for each block, try swapping one point to see if it covers more uncovered
    # t-subsets without losing coverage of any t-subset that was only covered by this block.
    # This is complex; instead, do a simpler hill-climb: remove a block, see if coverage
    # breaks, if so re-add or find a better replacement.

    # Simpler: restart with randomization and keep the best (fewest blocks) valid solution.
    best_blocks = list(blocks)
    best_count = len(blocks)

    # Reset for restarts
    def reset_state():
        nonlocal coverage, uncovered_count, blocks
        coverage = [0] * n_tsubs
        uncovered_count = n_tsubs
        blocks = []

    restarts = 0
    while _remaining() > 5.0 and restarts < 50:
        restarts += 1
        reset_state()
        # Greedy with randomization
        while uncovered_count > 0 and _remaining() > 5.0:
            b = greedy_block(randomize=True)
            add_block(b)
            blocks.append(b)

        if uncovered_count == 0 and len(blocks) < best_count:
            best_blocks = list(blocks)
            best_count = len(blocks)

    return best_blocks if len(best_blocks) < len(blocks) else blocks
# EVOLVE-BLOCK-END
