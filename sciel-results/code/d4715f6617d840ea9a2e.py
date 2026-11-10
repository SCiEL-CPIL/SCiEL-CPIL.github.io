# EVOLVE-BLOCK-START
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    total_t = len(all_t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Precompute for t=3: triple_index[pair][x] = index of triple
    if t == 3:
        triple_index = {}
        for a in range(v):
            for b in range(a + 1, v):
                pair = (a, b)
                triple_index[pair] = {}
                for x in range(v):
                    if x != a and x != b:
                        triple = tuple(sorted((a, b, x)))
                        triple_index[pair][x] = t_subset_index[triple]
    else:
        triple_index = None

    def get_gain(block, x, block_set):
        """Compute how many uncovered t-subsets are covered by adding x to block."""
        if t == 3:
            gain = 0
            for i in range(len(block)):
                for j in range(i + 1, len(block)):
                    a, b = block[i], block[j]
                    if a > b:
                        a, b = b, a
                    pair = (a, b)
                    idx = triple_index[pair][x]
                    if uncovered[idx]:
                        gain += 1
            return gain
        else:
            gain = 0
            for rest in itertools.combinations(block, t - 1):
                ts = tuple(sorted(rest + (x,)))
                idx = t_subset_index.get(ts, -1)
                if idx >= 0 and uncovered[idx]:
                    gain += 1
            return gain

    def cover_block(block):
        """Update coverage and uncovered arrays for the given block."""
        nonlocal uncovered_count
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            if uncovered[idx]:
                uncovered[idx] = False
                uncovered_count -= 1

    # Try multiple restarts with different random seeds
    best_blocks = None
    best_count = float('inf')

    # Determine number of restarts based on available time
    # First run is deterministic (seed 0), subsequent ones are randomized
    num_restarts = 10  # Will be limited by time

    for seed in range(num_restarts):
        if time.time() > deadline - 3:
            break

        random.seed(seed)

        # Reset state
        uncovered = [True] * total_t
        uncovered_count = total_t
        blocks = []
        block_set_cache = set()

        while uncovered_count > 0:
            if time.time() > deadline - 1:
                break

            # Find an uncovered t-subset to start with
            # For seed 0, use first uncovered; for others, randomize
            if seed == 0:
                start_idx = -1
                for i in range(total_t):
                    if uncovered[i]:
                        start_idx = i
                        break
                if start_idx < 0:
                    break
            else:
                # Collect uncovered indices and pick randomly
                uncovered_indices = [i for i in range(total_t) if uncovered[i]]
                if not uncovered_indices:
                    break
                start_idx = random.choice(uncovered_indices)

            start_ts = all_t_subsets[start_idx]
            block = list(start_ts)
            block_set = set(block)

            # Greedily add points to maximize gain
            while len(block) < k:
                if time.time() > deadline - 0.5:
                    break

                best_x = None
                best_gain = -1
                candidates = []

                for x in range(v):
                    if x not in block_set:
                        g = get_gain(block, x, block_set)
                        if g > best_gain:
                            best_gain = g
                            best_x = x
                            candidates = [(g, x)]
                        elif g == best_gain:
                            candidates.append((g, x))

                if best_x is None:
                    break

                # Add randomization for diversity (except seed 0)
                if seed > 0 and len(candidates) > 1:
                    # Pick randomly among top candidates
                    top_candidates = [c for c in candidates if c[0] == best_gain]
                    if len(top_candidates) > 1:
                        best_x = random.choice(top_candidates)[1]
                    elif random.random() < 0.1:
                        # Occasionally pick a non-best to explore
                        available = [x for x in range(v) if x not in block_set]
                        if available:
                            best_x = random.choice(available)

                block.append(best_x)
                block_set.add(best_x)

            # Only add complete blocks
            if len(block) == k:
                block.sort()
                b_tuple = tuple(block)
                if b_tuple not in block_set_cache:
                    blocks.append(block)
                    block_set_cache.add(b_tuple)
                    cover_block(block)

        # Check if this is a valid complete covering
        if uncovered_count == 0 and len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
            # If we found a good solution, we can stop early
            if best_count <= 77:  # Known best is 76, so <=77 is very good
                break

    # Return best complete covering found
    if best_blocks is not None:
        return best_blocks

    # Fallback: run one more deterministic pass if we ran out of time
    if time.time() < deadline - 2:
        uncovered = [True] * total_t
        uncovered_count = total_t
        blocks = []
        block_set_cache = set()

        while uncovered_count > 0:
            if time.time() > deadline - 0.5:
                break

            start_idx = -1
            for i in range(total_t):
                if uncovered[i]:
                    start_idx = i
                    break
            if start_idx < 0:
                break

            start_ts = all_t_subsets[start_idx]
            block = list(start_ts)
            block_set = set(block)

            while len(block) < k:
                if time.time() > deadline - 0.3:
                    break

                best_x = None
                best_gain = -1
                for x in range(v):
                    if x not in block_set:
                        g = get_gain(block, x, block_set)
                        if g > best_gain:
                            best_gain = g
                            best_x = x

                if best_x is None:
                    break

                block.append(best_x)
                block_set.add(best_x)

            if len(block) == k:
                block.sort()
                b_tuple = tuple(block)
                if b_tuple not in block_set_cache:
                    blocks.append(block)
                    block_set_cache.add(b_tuple)
                    cover_block(block)

        if uncovered_count == 0:
            return blocks

    # If we couldn't find a complete covering, return empty (invalid)
    # This should not happen if we have enough time
    return []
# EVOLVE-BLOCK-END
