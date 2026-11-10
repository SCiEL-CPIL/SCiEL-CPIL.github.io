"""Covering design C(19,8,5) - multi-strategy greedy with local search."""
import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using:
    1. Fast greedy with numpy-based uncovered tracking
    2. Multi-seed exploration
    3. Local search to remove redundant blocks
    4. Block replacement / perturbation
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsub = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Precompute: for each point x, list of t-subset indices that contain x
    point_to_tsubsets = [[] for _ in range(v)]
    for idx, ts in enumerate(all_tsubsets):
        for x in ts:
            point_to_tsubsets[x].append(idx)

    # Precompute (t-1)-subset -> list of t-subset indices formed by adding each point
    # For gain computation: for a (t-1)-subset S and point x not in S,
    # the t-subset is S ∪ {x}. We precompute a mapping.
    # Key: (frozen (t-1)-subset, point x) -> t-subset index or -1
    t1_to_tsub = {}
    for idx, ts in enumerate(all_tsubsets):
        for i in range(t):
            rest = tuple(sorted(ts[:i] + ts[i+1:]))
            x = ts[i]
            key = (rest, x)
            t1_to_tsub[key] = idx

    def greedy_build(seed):
        """Greedy covering with random tie-breaking and numpy uncovered tracking."""
        rng = random.Random(seed)
        uncovered = np.ones(n_tsub, dtype=bool)
        uncovered_count = n_tsub
        blocks = []

        while uncovered_count > 0:
            if time.time() > deadline - 2:
                break

            # Pick a random uncovered t-subset to seed the block
            uncovered_indices = np.nonzero(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            chosen = uncovered_indices[rng.randint(0, len(uncovered_indices)-1)]
            seed_ts = all_tsubsets[chosen]

            # Build block starting from seed_ts
            block = list(seed_ts)
            rng.shuffle(block)

            while len(block) < k:
                if time.time() > deadline - 2:
                    break

                # Find point x not in block that covers most uncovered t-subsets
                best_gain = -1
                best_x = None
                block_set = set(block)
                block_t1_subsets = list(itertools.combinations(block, t - 1))

                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in block_t1_subsets:
                        idx = t1_to_tsub.get((rest, x), -1)
                        if idx >= 0 and uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x

                if best_x is None or best_gain <= 0:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_x = rng.choice(remaining)
                    else:
                        break

                block.append(best_x)

            block.sort()
            blocks.append(block)

            # Update uncovered
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(tuple(sorted(ts)))
                if idx is not None and uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1

        if uncovered_count == 0:
            return blocks
        return None

    def local_search(blocks, max_time=8):
        """Remove redundant blocks iteratively with precomputed t-subset indices."""
        start_ls = time.time()

        # Build cover counts using indices
        cover_count = np.zeros(n_tsub, dtype=np.int32)
        for block in blocks:
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx[tuple(sorted(ts))]
                cover_count[idx] += 1

        changed = True
        while changed and (time.time() - start_ls) < max_time:
            changed = False
            new_blocks = []
            for block in blocks:
                is_critical = False
                for ts in itertools.combinations(block, t):
                    idx = tsubset_to_idx[tuple(sorted(ts))]
                    if cover_count[idx] == 1:
                        is_critical = True
                        break
                if not is_critical:
                    for ts in itertools.combinations(block, t):
                        idx = tsubset_to_idx[tuple(sorted(ts))]
                        cover_count[idx] -= 1
                    changed = True
                else:
                    new_blocks.append(block)
            blocks = new_blocks

        return blocks

    def validate(blocks):
        """Check covering validity using numpy."""
        covered = np.zeros(n_tsub, dtype=bool)
        for block in blocks:
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx[tuple(sorted(ts))]
                covered[idx] = True
        return np.all(covered)

    def try_replace(blocks, seed, max_time=5):
        """Try to replace a block with a better one or remove it."""
        start = time.time()
        rng = random.Random(seed)
        n = len(blocks)
        if n == 0:
            return blocks

        # Try removing each block
        for i in range(n):
            if time.time() > start + max_time:
                break
            candidate = blocks[:i] + blocks[i+1:]
            if validate(candidate):
                return candidate

        return blocks

    # Main search loop
    best_result = None
    best_len = float('inf')

    # Phase 1: Multi-seed greedy + local search
    for seed in range(50):
        if time.time() > deadline - 15:
            break
        result = greedy_build(seed)
        if result is not None:
            result = local_search(result, max_time=5)
            if validate(result):
                if len(result) < best_len:
                    best_len = len(result)
                    best_result = result

    # Phase 2: Try to improve best_result with replacement
    if best_result is not None:
        for attempt in range(15):
            if time.time() > deadline - 5:
                break
            improved = try_replace(best_result, seed=100 + attempt, max_time=3)
            if improved is not None and len(improved) < best_len:
                best_len = len(improved)
                best_result = improved
                best_result = local_search(best_result, max_time=4)
                if validate(best_result) and len(best_result) < best_len:
                    best_len = len(best_result)

    # Phase 3: More greedy seeds if time permits
    for seed in range(50, 80):
        if time.time() > deadline - 5:
            break
        result = greedy_build(seed)
        if result is not None:
            result = local_search(result, max_time=3)
            if validate(result):
                if len(result) < best_len:
                    best_len = len(result)
                    best_result = result

    if best_result is not None:
        return best_result

    # Fallback
    return greedy_build(0) or []
