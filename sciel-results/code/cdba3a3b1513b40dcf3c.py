"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    total_t = len(all_t_subsets)

    # For each point, precompute which t-subsets contain it
    # t_subsets_by_point[x] = list of indices into all_t_subsets that contain x
    t_subsets_by_point = [[] for _ in range(v)]
    for idx, combo in enumerate(all_t_subsets):
        for x in combo:
            t_subsets_by_point[x].append(idx)

    # uncovered[i] = True if t-subset i is not yet covered
    uncovered = bytearray(b'\x01') * total_t
    # count of uncovered subsets for quick checks
    remaining = total_t

    blocks = []

    def time_left():
        return deadline - time.time()

    def gain_for_point(x, block):
        """Count uncovered t-subsets that would be covered by adding x to block."""
        if t == 1:
            # Adding x covers t-subsets containing x
            return sum(1 for idx in t_subsets_by_point[x] if uncovered[idx])
        elif t == 2:
            # Need pairs (x, y) where y in block
            cnt = 0
            for y in block:
                # Find t-subsets containing both x and y
                # These are in t_subsets_by_point[x] and t_subsets_by_point[y]
                # For t=2, the subset is just (min(x,y), max(x,y))
                a, b = (x, y) if x < y else (y, x)
                # Check if this specific pair is uncovered
                # We need to find its index in all_t_subsets
                # For efficiency, precompute a mapping
                # Actually, let's just check directly
                pass
            # This approach is too slow for t=2 with large v
            # Let's use a different strategy
            return 0  # placeholder, will be replaced
        else:
            # General case: for each (t-1)-subset of block, count uncovered t-subsets
            # that extend it with x
            cnt = 0
            for rest in itertools.combinations(block, t - 1):
                # Find all t-subsets that contain all elements of rest and x
                # These are subsets of size t that include rest (size t-1) and x
                # So they are exactly rest + (x,) if x not in rest
                if x in rest:
                    continue
                # The t-subset is tuple(sorted(rest + (x,)))
                # We need to find if it's uncovered
                # For efficiency, we could precompute, but let's just check
                # Actually, for t=3, rest has 2 elements, so the t-subset is rest+(x,)
                # We need to find its index
                pass
            return 0  # placeholder

    # Better approach: use a direct counting method
    # For a given block B and candidate point x not in B:
    # gain(x) = number of uncovered t-subsets S such that S \ {x} ⊆ B
    # i.e., S contains x and S \ {x} is a (t-1)-subset of B

    # Precompute for each (t-1)-subset, the list of t-subsets that extend it
    # (t-1)-subset -> list of (point, t-subset_index) pairs
    t1_extend = {}
    for idx, combo in enumerate(all_t_subsets):
        for i in range(t):
            rest = tuple(combo[:i] + combo[i+1:])
            x = combo[i]
            if rest not in t1_extend:
                t1_extend[rest] = []
            t1_extend[rest].append((x, idx))

    def compute_gain(x, block):
        """Count uncovered t-subsets covered by adding x to block."""
        cnt = 0
        for rest in itertools.combinations(block, t - 1):
            rest_tuple = tuple(rest)
            if rest_tuple in t1_extend:
                for (px, idx) in t1_extend[rest_tuple]:
                    if px == x and uncovered[idx]:
                        cnt += 1
        return cnt

    # Greedy set cover with randomized restarts
    best_blocks = None
    best_len = float('inf')

    def greedy_cover(seed):
        nonlocal best_blocks, best_len
        rng = random.Random(seed)

        unc = bytearray(b'\x01') * total_t
        rem = total
        blks = []

        while rem > 0 and time_left() > 2.0:
            # Find an uncovered t-subset to start with
            # Randomly pick from uncovered for diversity
            unc_indices = [i for i in range(total_t) if unc[i]]
            if not unc_indices:
                break
            start_idx = unc_indices[rng.randint(0, len(unc_indices) - 1)]
            start_combo = list(all_t_subsets[start_idx])
            block = list(start_combo)

            # Fill the block greedily
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    g = compute_gain(x, block)
                    if g > best_gain:
                        best_gain = g
                        best_x = x
                    elif g == best_gain and best_x is not None:
                        # Tie-break randomly for diversity
                        if rng.random() < 0.5:
                            best_x = x
                if best_x is None or best_gain <= 0:
                    # No point adds coverage; just pick a random point
                    candidates = [x for x in range(v) if x not in block]
                    if not candidates:
                        break
                    best_x = candidates[rng.randint(0, len(candidates) - 1)]
                block.append(best_x)

            block.sort()
            blks.append(block)

            # Update uncovered
            for combo in itertools.combinations(block, t):
                # Find index of this combo in all_t_subsets
                # For efficiency, we can use a dictionary
                pass

        return blks

    # This is getting complex. Let me use a simpler but effective approach.
    # Use the original greedy but with better tie-breaking and randomization.

    # Reset and use a cleaner implementation
    uncovered = bytearray(b'\x01') * total_t
    remaining = total_t
    blocks = []

    # Build a map from t-subset to its index for O(1) lookup
    t_subset_to_idx = {combo: idx for idx, combo in enumerate(all_t_subsets)}

    def cover_block_tsubsets(block):
        """Return list of indices of t-subsets covered by this block."""
        covered = []
        for combo in itertools.combinations(block, t):
            idx = t_subset_to_idx.get(tuple(combo))
            if idx is not None:
                covered.append(idx)
        return covered

    def do_greedy(seed):
        nonlocal remaining
        rng = random.Random(seed)
        unc = bytearray(b'\x01') * total_t
        rem = total
        blks = []

        while rem > 0 and time_left() > 2.0:
            # Pick an uncovered t-subset
            # Try a few random ones, pick the one that leads to best gain
            unc_list = [i for i in range(total_t) if unc[i]]
            if not unc_list:
                break

            # Start with a random uncovered subset
            start_idx = unc_list[rng.randint(0, len(unc_list) - 1)]
            block = list(all_t_subsets[start_idx])

            # Greedily fill the block
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    # Compute gain
                    g = 0
                    for rest in itertools.combinations(block, t - 1):
                        rest_t = tuple(rest)
                        if rest_t in t1_extend:
                            for (px, idx) in t1_extend[rest_t]:
                                if px == x and unc[idx]:
                                    g += 1
                    if g > best_gain:
                        best_gain = g
                        best_x = x
                    elif g == best_gain and best_x is not None and rng.random() < 0.3:
                        best_x = x
                if best_x is None:
                    candidates = [x for x in range(v) if x not in block]
                    if not candidates:
                        break
                    best_x = candidates[rng.randint(0, len(candidates) - 1)]
                block.append(best_x)

            block.sort()
            blks.append(block)

            # Update uncovered
            for idx in cover_block_tsubsets(block):
                if unc[idx]:
                    unc[idx] = 0
                    rem -= 1

        return blks

    # Run greedy with different seeds and keep the best
    num_restarts = 50
    for seed in range(num_restarts):
        if time_left() < 3.0:
            break
        blks = do_greedy(seed)
        if len(blks) < best_len:
            best_len = len(blks)
            best_blocks = blks[:]

    if best_blocks is not None:
        return best_blocks

    # Fallback: simple deterministic greedy
    unc = bytearray(b'\x01') * total_t
    rem = total
    blks = []
    while rem > 0 and time_left() > 1.0:
        # Find first uncovered
        start_idx = next(i for i in range(total_t) if unc[i])
        block = list(all_t_subsets[start_idx])
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                g = 0
                for rest in itertools.combinations(block, t - 1):
                    rest_t = tuple(rest)
                    if rest_t in t1_extend:
                        for (px, idx) in t1_extend[rest_t]:
                            if px == x and unc[idx]:
                                g += 1
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x is None:
                candidates = [x for x in range(v) if x not in block]
                if not candidates:
                    break
                best_x = candidates[0]
            block.append(best_x)
        block.sort()
        blks.append(block)
        for idx in cover_block_tsubsets(block):
            if unc[idx]:
                unc[idx] = 0
                rem -= 1
    return blks
# EVOLVE-BLOCK-END
