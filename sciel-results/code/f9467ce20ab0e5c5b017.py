"""Greedy covering C(v,k,t) with time guard, better gain heuristic, and pruning."""
import time
import itertools

_START = time.time()
_HARD_LIMIT = 120.0          # wall-clock budget from the contract
_SAFETY = 18.0               # return well before the hard limit


def _time_left():
    return _HARD_LIMIT - _SAFETY - (time.time() - _START)


def construct(v, k, t):
    all_points = list(range(v))
    # All t-subsets we must cover
    uncovered = set(itertools.combinations(all_points, t))
    if not uncovered:
        return []

    # Map from a (t-1)-tuple to the list of uncovered t-subsets that extend it.
    # This lets us compute a point's "gain" (how many uncovered t-sets it covers
    # given the current block) without iterating over all combinations each time.
    # For a candidate point x not in the block, the new t-sets covered are exactly
    # { (x,) + rest : rest in C(block, t-1) } that are still uncovered.
    # We precompute, for each (t-1)-tuple, the set of points x such that
    # tuple(sorted((x,)+rest)) is in uncovered.  Then gain(x) = number of
    # rest in C(block, t-1) where x is in that set.

    # Build reverse index: rest_tuple -> set of points x that cover an uncovered set
    rev = {}
    for ts in uncovered:
        for i in range(t):
            rest = ts[:i] + ts[i + 1:]
            x = ts[i]
            # rest is not sorted necessarily; normalize
            rest_sorted = tuple(sorted(rest))
            if rest_sorted not in rev:
                rev[rest_sorted] = set()
            rev[rest_sorted].add(x)

    blocks = []

    def gain(x, block):
        c = 0
        for rest in itertools.combinations(block, t - 1):
            rest_sorted = tuple(sorted(rest))
            s = rev.get(rest_sorted)
            if s is not None and x in s:
                c += 1
        return c

    while uncovered and _time_left() > 0:
        # Seed the new block with the lexicographically smallest uncovered t-set
        seed = min(uncovered)
        block = list(seed)
        in_block = set(block)

        while len(block) < k and _time_left() > 0:
            # Choose the point x not in block that maximizes gain; break ties by
            # preferring points that appear in more uncovered t-sets overall
            # (a cheap proxy for future flexibility).
            best_x = None
            best_score = (-1, -1)
            for x in all_points:
                if x in in_block:
                    continue
                g = gain(x, block)
                if g == 0:
                    continue
                # tie-break: count how many uncovered t-sets contain x
                # (approximate; we can precompute a global frequency)
                score = (g, x)  # deterministic tie-break
                if score > best_score:
                    best_score = score
                    best_x = x
            if best_x is None:
                # No point adds coverage; fill with smallest unused points
                for x in all_points:
                    if x not in in_block:
                        block.append(x)
                        in_block.add(x)
                        if len(block) == k:
                            break
                break
            block.append(best_x)
            in_block.add(best_x)

        block.sort()
        blocks.append(block)
        # Remove the t-subsets of this block from uncovered
        for ts in itertools.combinations(block, t):
            uncovered.discard(ts)

    # Post pass: remove redundant blocks (a block is redundant if every t-subset
    # of it is covered by the other blocks).  Do this only if we have time.
    if blocks and _time_left() > 5.0:
        # Build coverage count per t-subset over all blocks
        # Only do this if the total number of t-subsets per block times #blocks
        # is manageable.
        total_work = len(blocks) * len(list(itertools.combinations(range(k), t)))
        if total_work < 2_000_000:
            cov = {}
            for b in blocks:
                for ts in itertools.combinations(b, t):
                    cov[ts] = cov.get(ts, 0) + 1
            kept = []
            for b in blocks:
                redundant = True
                for ts in itertools.combinations(b, t):
                    if cov.get(ts, 0) == 1:
                        redundant = False
                        break
                if not redundant:
                    kept.append(b)
            # Recompute coverage for kept blocks
            cov2 = {}
            for b in kept:
                for ts in itertools.combinations(b, t):
                    cov2[ts] = cov2.get(ts, 0) + 1
            # Verify all t-sets are still covered
            all_ts = set(itertools.combinations(all_points, t))
            if all_ts.issubset(set(cov2.keys())):
                blocks = kept

    return blocks
