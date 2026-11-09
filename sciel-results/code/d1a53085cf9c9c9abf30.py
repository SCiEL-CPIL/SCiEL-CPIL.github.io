# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations


def construct(v, k, t):
    """Covering design C(v,k,t) via fast bitmask greedy with lookahead."""
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    total_t = 1
    for i in range(t):
        total_t *= (v - i)
    total_t //= 1 << (t * 3)  # divide by t!
    # Actually compute C(v,t)
    from math import comb
    total_t = comb(v, t)

    # Precompute all t-subsets as frozensets and map to index
    all_tsubs = [frozenset(c) for c in combinations(range(v), t)]
    tsub_index = {s: i for i, s in enumerate(all_tsubs)}

    # uncovered: set of indices of uncovered t-subsets
    uncovered = set(range(total_t))
    uncovered_count = total_t

    # For fast lookup: for a given t-1 subset, which t-subsets contain it
    # Precompute: for each t-1 frozenset, list of t-subset indices containing it
    t1_index = {}
    for s in all_tsubs:
        # get all t-1 subsets of s
        pts = sorted(s)
        for j in range(t):
            t1 = frozenset(pts[:j] + pts[j+1:])
            if t1 not in t1_index:
                t1_index[t1] = []
            t1_index[t1].append(tsub_index[s])

    blocks = []
    rng = random.Random(42)

    while uncovered:
        if time.time() > deadline - 5.0:
            break

        # Pick a t-subset to cover: prefer one with low "flexibility"
        # (fewest possible blocks can contain it) - but all t-subsets have
        # the same flexibility in a uniform design, so pick the one that
        # appears in the fewest already-placed blocks' potential extensions.
        # Simpler: pick the uncovered t-subset with minimum "degree" =
        # number of uncovered t-subsets it shares a t-1 with.
        # For speed, just pick the first uncovered one (deterministic) but
        # try a few random ones and pick the best.
        sample = list(uncovered)[:50] if len(uncovered) > 50 else list(uncovered)
        rng.shuffle(sample)
        # Score each candidate by how many uncovered t-subsets it can help
        best_seed = None
        best_seed_score = -1
        for idx in sample:
            seed = all_tsubs[idx]
            # Count uncovered t-subsets that share a t-1 with seed
            cnt = 0
            pts = sorted(seed)
            for j in range(t):
                t1 = frozenset(pts[:j] + pts[j+1:])
                if t1 in t1_index:
                    for t2_idx in t1_index[t1]:
                        if t2_idx in uncovered:
                            cnt += 1
            if cnt > best_seed_score:
                best_seed_score = cnt
                best_seed = seed

        if best_seed is None:
            # Fallback: pick first uncovered
            best_seed = all_tsubs[next(iter(uncovered))]

        # Build block greedily: start with best_seed, add points one by one
        block = set(best_seed)
        remaining_slots = k - len(block)

        for _ in range(remaining_slots):
            if time.time() > deadline - 5.0:
                break
            if not uncovered:
                break

            # For each candidate point x not in block, compute gain:
            # number of currently-uncovered t-subsets that would be covered
            # by adding x to block (i.e., t-subsets of block ∪ {x} that contain x)
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # t-subsets of (block ∪ {x}) that contain x and are uncovered
                gain = 0
                new_block = block | {x}
                # Only t-subsets that include x and are subsets of new_block
                # = t-subsets of new_block that contain x
                # = choose t-1 from (new_block - {x}) = choose t-1 from block
                for rest in combinations(sorted(block), t - 1):
                    s = frozenset(rest + (x,))
                    if s in tsub_index and tsub_index[s] in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x

            if best_x is None:
                break
            block.add(best_x)

        # Finalize block
        if len(block) == k:
            block_sorted = sorted(block)
            blocks.append(block_sorted)
            # Remove all covered t-subsets
            for s in combinations(block_sorted, t):
                idx = tsub_index[frozenset(s)]
                uncovered.discard(idx)

    # If we didn't finish (deadline), try to salvage: keep going if time permits
    # but we already broke. Return what we have - but it must be valid!
    # If not all covered, we need to ensure validity. Let's check.
    if uncovered and time.time() < deadline - 2.0:
        # Emergency: add blocks to cover remaining (may exceed time but try)
        while uncovered and time.time() < deadline - 1.0:
            seed = all_tsubs[next(iter(uncovered))]
            block = set(seed)
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in combinations(sorted(block), t - 1):
                        s = frozenset(rest + (x,))
                        if s in tsub_index and tsub_index[s] in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    # Just pick any
                    for x in range(v):
                        if x not in block:
                            best_x = x
                            break
                block.add(best_x)
            block_sorted = sorted(block)
            if block_sorted not in blocks:
                blocks.append(block_sorted)
            for s in combinations(block_sorted, t):
                idx = tsub_index[frozenset(s)]
                uncovered.discard(idx)

    return blocks
# EVOLVE-BLOCK-END
