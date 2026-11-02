"""
Covering design C(24,7,3) - greedy construction + local search minimization.
"""
import itertools
import time
import random
import sys

def construct(v, k, t):
    """
    Construct a covering: every t-subset of {0,...,v-1} lies in at least one block.
    Returns a list of blocks (each a sorted list of k distinct integers).
    """
    start_time = time.time()
    deadline = 115.0  # seconds, safety margin before 120s kill

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    # Map each t-subset to an index
    tsubset_index = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Greedy construction
    uncovered = [True] * n_tsubsets
    n_uncovered = n_tsubsets
    blocks = []

    # For speed, precompute for each point x, the list of t-subsets containing x
    # Actually, we need: for a partial block B, and candidate x, how many uncovered t-subsets
    # are completed by adding x to B?
    # A t-subset S is "covered by B ∪ {x}" if S ⊆ B ∪ {x} and x ∈ S (i.e., S \ B ⊆ {x} and x ∈ S)
    # More precisely: S ⊆ B ∪ {x} means S \ B ⊆ {x}. Since |S|=t and |B|=b,
    # S \ B has size t - |S ∩ B|. For S ⊆ B ∪ {x}, we need S \ B ⊆ {x},
    # which means S \ B is either empty (S ⊆ B, already covered) or {x} (x ∈ S, S \ B = {x}).
    # So the new t-subsets covered by adding x to B are those S with S ⊆ B ∪ {x} and x ∈ S and S ⊄ B.
    # Equivalently: S = R ∪ {x} where R ⊆ B and |R| = t-1.

    # Greedy: repeatedly pick the best block
    # For each partial block, compute gain of adding each candidate point
    # We'll use a simpler greedy: pick the t-subset with max "potential", build a block around it

    # Simpler greedy approach:
    # While there are uncovered t-subsets:
    #   Pick the uncovered t-subset with the most "synergy" (or just the first)
    #   Build a block of size k containing it that covers the most uncovered t-subsets

    # For efficiency, let's use the standard greedy:
    # For each candidate block (containing a specific uncovered t-subset), compute how many
    # uncovered t-subsets it covers. Pick the best.

    # But C(24,7) is huge. Let's use the incremental approach:
    # Start with an uncovered t-subset, greedily add points that maximize coverage.

    while n_uncovered > 0:
        if time.time() - start_time > deadline - 30:
            break  # Leave time for local search

        # Find an uncovered t-subset (first one)
        ts_idx = -1
        for i in range(n_tsubsets):
            if uncovered[i]:
                ts_idx = i
                break
        if ts_idx < 0:
            break

        seed_ts = all_tsubsets[ts_idx]
        block = list(seed_ts)
        block_set = set(block)

        # Greedily add points to the block
        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Count uncovered t-subsets covered by adding x
                # S = R ∪ {x} where R is a (t-1)-subset of block
                gain = 0
                for R in itertools.combinations(block, t - 1):
                    S = tuple(sorted(R + (x,)))
                    if S in tsubset_index:
                        idx = tsubset_index[S]
                        if uncovered[idx]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x < 0 or best_gain <= 0:
                # No gain, pick a random point
                candidates = [x for x in range(v) if x not in block_set]
                if candidates:
                    best_x = random.choice(candidates)
                else:
                    break
            block.append(best_x)
            block_set.add(best_x)

        block.sort()
        blocks.append(block)

        # Update uncovered
        for ts in itertools.combinations(block, t):
            idx = tsubset_index[ts]
            if uncovered[idx]:
                uncovered[idx] = False
                n_uncovered -= 1

    if n_uncovered > 0:
        # Fallback: brute force remaining (shouldn't happen often)
        while n_uncovered > 0:
            if time.time() - start_time > deadline - 5:
                break
            ts_idx = -1
            for i in range(n_tsubsets):
                if uncovered[i]:
                    ts_idx = i
                    break
            if ts_idx < 0:
                break
            seed_ts = all_tsubsets[ts_idx]
            block = list(seed_ts)
            block_set = set(block)
            while len(block) < k:
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break
                best_x = max(candidates, key=lambda x: sum(
                    1 for R in itertools.combinations(block, t-1)
                    if tuple(sorted(R + (x,))) in tsubset_index
                       and uncovered[tsubset_index[tuple(sorted(R + (x,)))]]
                ))
                block.append(best_x)
                block_set.add(best_x)
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                idx = tsubset_index[ts]
                if uncovered[idx]:
                    uncovered[idx] = False
                    n_uncovered -= 1

    if not blocks:
        # Should not happen, but return something valid
        # Cover with all k-subsets (too many, but valid)
        return [list(b) for b in itertools.combinations(range(v), k)]

    # Local search: try to remove redundant blocks and swap
    # Build coverage count: for each t-subset, how many blocks cover it
    cov_count = [0] * n_tsubsets
    for block in blocks:
        for ts in itertools.combinations(block, t):
            cov_count[tsubset_index[ts]] += 1

    # Try removing blocks that don't make any t-subset uncovered
    improved = True
    while improved and time.time() - start_time < deadline - 10:
        improved = False
        for i in range(len(blocks)):
            if time.time() - start_time > deadline - 10:
                break
            # Check if removing block i leaves all t-subsets covered
            block_i = blocks[i]
            removable = True
            for ts in itertools.combinations(block_i, t):
                idx = tsubset_index[ts]
                if cov_count[idx] == 1:
                    removable = False
                    break
            if removable:
                # Remove block i
                for ts in itertools.combinations(block_i, t):
                    cov_count[tsubset_index[ts]] -= 1
                blocks.pop(i)
                improved = True
                break  # Restart scan after removal

    # Try swapping: replace a block with a better one that covers more
    # Actually, try: for each block, see if we can find a replacement that covers
    # at least the same t-subsets and enables further removals
    # Simpler: try to replace each block with a greedily-constructed block that
    # covers all t-subsets currently only covered by that block, plus extras

    for _ in range(3):  # A few rounds of improvement
        if time.time() - start_time > deadline - 5:
            break
        # Rebuild cov_count
        cov_count = [0] * n_tsubsets
        for block in blocks:
            for ts in itertools.combinations(block, t):
                cov_count[tsubset_index[ts]] += 1

        # Try to remove redundant blocks again
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() - start_time > deadline - 5:
                break
            block_i = blocks[i]
            removable = True
            for ts in itertools.combinations(block_i, t):
                idx = tsubset_index[ts]
                if cov_count[idx] == 1:
                    removable = False
                    break
            if removable:
                for ts in itertools.combinations(block_i, t):
                    cov_count[tsubset_index[ts]] -= 1
                blocks.pop(i)

    # Simulated annealing / random restarts for block swaps
    # Try replacing random blocks with better ones
    rng = random.Random(42)
    for _ in range(5000):
        if time.time() - start_time > deadline - 2:
            break
        if len(blocks) < 2:
            break

        # Rebuild cov_count periodically
        if _ % 100 == 0:
            cov_count = [0] * n_tsubsets
            for block in blocks:
                for ts in itertools.combinations(block, t):
                    cov_count[tsubset_index[ts]] += 1

        # Pick a random block to try replacing
        i = rng.randrange(len(blocks))
        old_block = blocks[i]

        # Find t-subsets only covered by this block
        exclusive = []
        for ts in itertools.combinations(old_block, t):
            idx = tsubset_index[ts]
            if cov_count[idx] == 1:
                exclusive.append(ts)

        if not exclusive:
            # Can just remove it
            for ts in itertools.combinations(old_block, t):
                cov_count[tsubset_index[ts]] -= 1
            blocks.pop(i)
            continue

        # Try to build a new block that covers all exclusive t-subsets
        # The new block must contain the union of all exclusive t-subsets
        must_include = set()
        for ts in exclusive:
            must_include.update(ts)

        if len(must_include) > k:
            continue  # Can't fit in one block

        # Build a block containing must_include, greedily adding points to cover more
        new_block = list(must_include)
        new_block_set = set(new_block)

        # Compute current coverage without old_block
        # For simplicity, just greedily add points
        for _add in range(k - len(new_block)):
            if time.time() - start_time > deadline - 2:
                break
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in new_block_set:
                    continue
                gain = 0
                for R in itertools.combinations(new_block, t - 1):
                    S = tuple(sorted(R + (x,)))
                    if S in tsubset_index:
                        idx = tsubset_index[S]
                        # Count if this helps (covers something not yet covered by other blocks)
                        # Approximate: count uncovered or low-coverage
                        if cov_count[idx] <= 1:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x < 0:
                break
            new_block.append(best_x)
            new_block_set.add(best_x)

        new_block.sort()
        if new_block == old_block:
            continue

        # Check if new block covers all exclusive t-subsets
        valid = True
        for ts in exclusive:
            if not all(x in new_block_set for x in ts):
                valid = False
                break
        if not valid:
            continue

        # Replace old block with new block
        for ts in itertools.combinations(old_block, t):
            cov_count[tsubset_index[ts]] -= 1
        blocks[i] = new_block
        for ts in itertools.combinations(new_block, t):
            cov_count[tsubset_index[ts]] += 1

    # Final cleanup: remove any redundant blocks
    cov_count = [0] * n_tsubsets
    for block in blocks:
        for ts in itertools.combinations(block, t):
            cov_count[tsubset_index[ts]] += 1

    for i in range(len(blocks) - 1, -1, -1):
        if time.time() - start_time > deadline - 1:
            break
        block_i = blocks[i]
        removable = True
        for ts in itertools.combinations(block_i, t):
            idx = tsubset_index[ts]
            if cov_count[idx] == 1:
                removable = False
                break
        if removable:
            for ts in itertools.combinations(block_i, t):
                cov_count[tsubset_index[ts]] -= 1
            blocks.pop(i)

    return blocks
