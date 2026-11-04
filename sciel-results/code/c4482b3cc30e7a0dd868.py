"""Improved covering design C(19,7,4) with randomized greedy restarts."""
import itertools
import random
import time


def _all_tsubsets(v, t):
    return [tuple(c) for c in itertools.combinations(range(v), t)]


def _build_greedy(v, k, t, all_tsubs, rng, deadline, start_mode="random"):
    """
    Build a covering using greedy block construction.
    start_mode: 'random' picks random uncovered t-subset as seed,
                'rarity' picks the uncovered t-subset with fewest candidate points
                (heuristic: points that appear in few existing blocks).
    """
    # uncovered: set of t-subsets not yet covered
    uncovered = set(all_tsubs)
    blocks = []

    # For speed, precompute for each point the list of t-subsets containing it
    # Actually, let's use a different approach: maintain uncovered as a set,
    # and for gain computation, check membership.

    while uncovered:
        if time.time() > deadline:
            break

        # Choose seed t-subset
        if start_mode == "rarity" and blocks:
            # Count how many existing blocks each point is in
            point_count = [0] * v
            for b in blocks:
                for p in b:
                    point_count[p] += 1
            # Pick uncovered t-subset with minimum sum of point_counts (rarest points)
            best_seed = None
            best_score = float('inf')
            # Sample a subset of uncovered to speed up
            sample = list(uncovered)
            if len(sample) > 50:
                sample = rng.sample(sample, 50)
            for sub in sample:
                s = sum(point_count[p] for p in sub)
                if s < best_score:
                    best_score = s
                    best_seed = sub
            if best_seed is None:
                best_seed = next(iter(uncovered))
        else:
            best_seed = rng.choice(list(uncovered)) if uncovered else None
            if best_seed is None:
                break

        block = list(best_seed)
        block_set = set(block)

        # Greedily add points to maximize coverage of uncovered t-subsets
        while len(block) < k:
            if time.time() > deadline:
                break

            best_gain = -1
            best_point = None
            # For each candidate point, compute how many uncovered t-subsets it would cover
            # A t-subset {a1,...,at} is covered by adding x if {a1,...,at-1} ⊂ block and x is the missing one
            # More precisely: adding x covers t-subset S if S ∩ block has size t-1 and x ∈ S
            # So we need: S \ {x} ⊂ block, |S|=t, x∉block

            # Build set of (t-1)-subsets of current block for fast lookup
            t_minus_1 = t - 1
            block_t1 = set(itertools.combinations(block, t_minus_1))

            for x in range(v):
                if x in block_set:
                    continue
                # Count uncovered t-subsets S where S\{x} is in block_t1
                gain = 0
                # Iterate over all (t-1)-subsets of block
                for comb in block_t1:
                    # The t-subset would be comb + (x,)
                    tsub = tuple(sorted(comb + (x,)))
                    if tsub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                elif gain == best_gain and best_point is not None:
                    # Tie-breaking: prefer points that appear less in existing blocks
                    pass

            if best_point is None:
                # No point adds coverage; pick random
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break
                best_point = rng.choice(candidates)

            block.append(best_point)
            block_set.add(best_point)

        block.sort()
        blocks.append(block)

        # Update uncovered
        for tsub in itertools.combinations(block, t):
            uncovered.discard(tuple(tsub))

    return blocks


def _validate_and_fix(v, k, t, blocks, deadline):
    """Check if all t-subsets are covered. If not, add blocks greedily."""
    all_tsubs = _all_tsubsets(v, t)
    covered = set()
    for b in blocks:
        for tsub in itertools.combinations(b, t):
            covered.add(tuple(tsub))

    uncovered = [s for s in all_tsubs if s not in covered]
    if not uncovered:
        return blocks

    # Add more blocks to cover remaining
    rng = random.Random(42)
    uncovered_set = set(uncovered)
    while uncovered_set:
        if time.time() > deadline:
            break
        # Greedy: pick a random uncovered t-subset, build a block around it
        seed = rng.choice(list(uncovered_set))
        block = list(seed)
        block_set = set(block)

        while len(block) < k:
            if time.time() > deadline:
                break
            best_gain = -1
            best_point = None
            t_minus_1 = t - 1
            block_t1 = set(itertools.combinations(block, t_minus_1))
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for comb in block_t1:
                    tsub = tuple(sorted(comb + (x,)))
                    if tsub in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point is None:
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break
                best_point = rng.choice(candidates)
            block.append(best_point)
            block_set.add(best_point)

        block.sort()
        # Avoid duplicate blocks
        block_tuple = tuple(block)
        if block_tuple not in {tuple(b) for b in blocks}:
            blocks.append(block)
            for tsub in itertools.combinations(block, t):
                uncovered_set.discard(tuple(tsub))

    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    all_tsubs = _all_tsubsets(v, t)
    num_tsubs = len(all_tsubs)

    # Try multiple randomized restarts with different strategies
    best_blocks = None
    best_len = float('inf')

    # Strategy 1: Deterministic greedy (original, but optimized)
    rng_det = random.Random(0)
    blocks_det = _build_greedy(v, k, t, all_tsubs, rng_det, deadline, start_mode="random")
    if len(blocks_det) < best_len:
        best_blocks = blocks_det
        best_len = len(blocks_det)

    if time.time() > deadline:
        return _validate_and_fix(v, k, t, best_blocks, deadline)

    # Strategy 2: Multiple random restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        rng = random.Random(1000 + i)
        mode = "rarity" if i % 2 == 0 else "random"
        blocks_i = _build_greedy(v, k, t, all_tsubs, rng, deadline, start_mode=mode)
        if len(blocks_i) < best_len:
            best_blocks = blocks_i
            best_len = len(blocks_i)

    if time.time() > deadline:
        return _validate_and_fix(v, k, t, best_blocks, deadline)

    # Ensure validity
    result = _validate_and_fix(v, k, t, best_blocks, deadline)
    return result
