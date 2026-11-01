"""Covering design C(24,7,3): randomized greedy + local search."""
import random
import time
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START


def _block_triples(block, t):
    """Yield all t-subsets of a block."""
    return itertools.combinations(block, t)


def _is_valid(blocks, v, k, t):
    """Check that blocks form a valid covering of all t-subsets."""
    covered = set()
    for b in blocks:
        if len(b) != k or len(set(b)) != k:
            return False
        for x in b:
            if x < 0 or x >= v:
                return False
        covered.update(itertools.combinations(b, t))
    total = 1
    for i in range(v - t + 1, v + 1):
        total *= i
    for i in range(1, t + 1):
        total //= i
    return len(covered) == total


def _try_delete_block(blocks, v, k, t):
    """Try to remove each block if the covering remains valid. Returns new list or None."""
    if not blocks:
        return blocks
    # Count how many times each t-subset is covered
    cover_count = defaultdict(int)
    for b in blocks:
        for tr in itertools.combinations(b, t):
            cover_count[tr] += 1
    # Find removable blocks: all their t-subsets covered >= 2 times
    for i in range(len(blocks)):
        removable = True
        for tr in itertools.combinations(blocks[i], t):
            if cover_count[tr] < 2:
                removable = False
                break
        if removable:
            new_blocks = blocks[:i] + blocks[i + 1:]
            return new_blocks
    return None


def _try_swap(blocks, v, k, t, rng):
    """Try to swap a point in a block for another point to reduce redundancy.
    Returns new blocks list if an improving swap is found, else None."""
    if len(blocks) < 2:
        return None
    # Compute cover counts
    cover_count = defaultdict(int)
    for b in blocks:
        for tr in itertools.combinations(b, t):
            cover_count[tr] += 1

    # Try random point swaps: replace point p in block i with point q not in block i
    # A swap is "improving" if it creates a situation where some block becomes removable
    # or simply if it doesn't break coverage and we can then delete a block.
    for _ in range(50):  # limited attempts for speed
        bi = rng.randrange(len(blocks))
        block = list(blocks[bi])
        if len(block) < k:
            continue
        p_idx = rng.randrange(k)
        p = block[p_idx]
        # Choose a random replacement point
        q = rng.randrange(v)
        if q == p or q in block:
            continue
        new_block = block[:p_idx] + [q] + block[p_idx + 1:]
        new_block.sort()
        # Check: does the new block still cover all t-subsets that were ONLY covered by old block?
        old_triples = set(itertools.combinations(block, t))
        new_triples = set(itertools.combinations(new_block, t))
        # Triples in old but not new must still be covered by other blocks
        lost = old_triples - new_triples
        can_swap = True
        for tr in lost:
            if cover_count[tr] < 2:
                can_swap = False
                break
        if not can_swap:
            continue
        # Perform swap and try to delete a block
        new_blocks = list(blocks)
        new_blocks[bi] = new_block
        # Recompute cover counts quickly
        new_cover_count = defaultdict(int)
        for b in new_blocks:
            for tr in itertools.combinations(b, t):
                new_cover_count[tr] += 1
        # Try to find a removable block
        for i in range(len(new_blocks)):
            removable = True
            for tr in itertools.combinations(new_blocks[i], t):
                if new_cover_count[tr] < 2:
                    removable = False
                    break
            if removable:
                return new_blocks[:i] + new_blocks[i + 1:]
    return None


def _randomized_greedy(v, k, t, rng, deadline):
    """Build a covering using randomized greedy with randomization in point selection."""
    all_points = list(range(v))
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []

    while uncovered and time.time() < deadline:
        # Pick a random uncovered t-subset to start the block
        start_triple = rng.choice(list(uncovered))
        block = list(start_triple)

        while len(block) < k:
            # For each candidate point, compute how many new t-subsets it covers
            best_gain = -1
            best_candidates = []
            block_set = set(block)
            for x in all_points:
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    tr = tuple(sorted(rest + (x,)))
                    if tr in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [x]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(x)
            if not best_candidates:
                # No point helps; pick random
                remaining = [x for x in all_points if x not in block_set]
                if not remaining:
                    break
                block.append(rng.choice(remaining))
            else:
                # Randomize among top candidates
                block.append(rng.choice(best_candidates))
            block_set = set(block)

        block.sort()
        blocks.append(block)
        # Update uncovered
        for tr in itertools.combinations(block, t):
            uncovered.discard(tr)

    return blocks


def construct(v, k, t):
    """Construct a covering design C(v,k,t) using randomized greedy + local search."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    rng = random.Random(42)

    # Phase 1: Generate multiple randomized greedy solutions
    best_blocks = None
    best_count = float('inf')

    # First, get a deterministic greedy solution as baseline
    uncovered = set(itertools.combinations(range(v), t))
    blocks_det = []
    while uncovered:
        block = list(min(uncovered))
        block_set = set(block)
        while len(block) < k:
            best_gain = -1
            best_x = None
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    tr = tuple(sorted(rest + (x,)))
                    if tr in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            block_set = set(block)
        block.sort()
        blocks_det.append(block)
        for tr in itertools.combinations(block, t):
            uncovered.discard(tr)

    if blocks_det:
        best_blocks = blocks_det
        best_count = len(blocks_det)

    # Phase 2: Randomized greedy with multiple seeds
    for seed in range(200):
        if time.time() > deadline:
            break
        rng = random.Random(seed * 7919 + 1)
        blocks = _randomized_greedy(v, k, t, rng, deadline)
        if blocks and len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        return blocks_det

    # Phase 3: Local search - try to delete blocks and swap points
    current = list(best_blocks)
    current_count = len(current)
    no_improve_count = 0
    max_no_improve = 30

    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False

        # Try block deletion
        new_blocks = _try_delete_block(current, v, k, t)
        if new_blocks is not None:
            current = new_blocks
            current_count = len(current)
            improved = True
            no_improve_count = 0
            continue

        # Try point swaps
        rng_ls = random.Random(int(time.time() * 1000000))
        new_blocks = _try_swap(current, v, k, t, rng_ls)
        if new_blocks is not None:
            current = new_blocks
            current_count = len(current)
            improved = True
            no_improve_count = 0
            continue

        # Try a fresh randomized greedy restart if stuck
        if no_improve_count % 5 == 0:
            rng_restart = random.Random(int(time.time() * 1000000) + 999)
            blocks_restart = _randomized_greedy(v, k, t, rng_restart, deadline)
            if blocks_restart and len(blocks_restart) < current_count:
                current = blocks_restart
                current_count = len(current)
                improved = True
                no_improve_count = 0
                continue

        no_improve_count += 1

    # Final validation
    if not _is_valid(current, v, k, t):
        return best_blocks

    return current
# EVOLVE-BLOCK-END
