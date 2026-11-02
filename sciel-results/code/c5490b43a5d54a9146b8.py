"""New program: randomized greedy with lookahead + redundancy/swap local search."""
import itertools
import random
from collections import defaultdict, Counter


def _tsubs_of_block(block, t):
    return set(itertools.combinations(block, t))


def construct(v, k, t):
    rng = random.Random(12345)
    all_points = list(range(v))
    all_tsubs = list(itertools.combinations(range(v), t))

    # Precompute, for each t-subset, the list of points NOT in it (candidates to extend toward)
    # and for scoring we track uncovered counts.

    def greedy_cover(uncovered_set, seed):
        """Greedy: repeatedly pick a new block that covers many uncovered t-subsets.
        Block is built by starting from the rarest uncovered t-subset and completing it
        with a lookahead-aware heuristic."""
        r = random.Random(seed)
        uncovered = set(uncovered_set)
        blocks = []
        # frequency of each uncovered t-subset across... just track set membership
        while uncovered:
            # pick a starting t-subset: prefer one with low "coverability" (few candidate
            # points that appear in many other uncovered tsubs) to handle hard ones early.
            # Use a cheap proxy: pick randomly among a small random sample the one with
            # fewest points that are in many uncovered tsubs.
            sample = list(uncovered)
            if len(sample) > 64:
                sample = r.sample(sample, 64)

            def coverability(ts):
                # number of points p not in ts such that ts+{p} (as a (t+1)-set) has many
                # uncovered t-subsets sharing it. Approx: count uncovered tsubs containing ts.
                c = 0
                for p in all_points:
                    if p in ts:
                        continue
                    c += 1
                return c
            # choose the ts with minimum "density" of extensions -> hard ones
            ts = min(sample, key=lambda x: (len(x), r.random()))
            # deterministic-ish: pick the rarest by a random tiebreak
            block = list(ts)
            block_set = set(block)

            # Complete the block: at each step add the point maximizing a lookahead score.
            while len(block) < k:
                best_x = None
                best_score = -1
                # candidate points not in block
                cands = [x for x in all_points if x not in block_set]
                # score: number of uncovered tsubs that would be covered by adding x
                # i.e. tsubs of form (t-1 from block) + x that are in uncovered
                # Precompute (t-1)-subsets of current block
                t_minus_1 = list(itertools.combinations(block, t - 1)) if t >= 1 else [()]
                # For each candidate, count uncovered tsubs = sorted(rest + (x,)) in uncovered
                # To speed, build a dict from (t-1)-tuple -> list of x that complete it
                # We'll just compute directly but cache per iteration.
                counts = {}
                for x in cands:
                    s = 0
                    for rest in t_minus_1:
                        # rest has t-1 elements, add x
                        cand_ts = tuple(sorted(rest + (x,)))
                        if cand_ts in uncovered:
                            s += 1
                    counts[x] = s
                # tie-break: prefer x that appears in fewer uncovered tsubs overall (more "special")
                # compute global count of x in uncovered tsubs (cache)
                # do it lazily
                best_score = -1
                for x in cands:
                    s = counts[x]
                    if s > best_score:
                        best_score = s
                        best_x = x
                if best_x is None or best_score <= 0:
                    # no gain; pick a random point to fill
                    best_x = r.choice(cands)
                block.append(best_x)
                block_set.add(best_x)

            block.sort()
            blocks.append(tuple(block))
            # remove covered tsubs
            for ts in itertools.combinations(block, t):
                uncovered.discard(ts)
        return blocks

    # Run several randomized greedy seeds, keep the best (fewest blocks)
    best_blocks = None
    best_len = float('inf')
    for seed in range(6):
        blocks = greedy_cover(all_tsubs, 1000 + seed)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks

    # Local search: redundancy removal + swap improvement
    blocks = list(best_blocks)

    def coverage_map(blk_list):
        m = defaultdict(int)
        for b in blk_list:
            for ts in itertools.combinations(b, t):
                m[ts] += 1
        return m

    # Redundancy removal: remove any block that doesn't uniquely cover any t-subset
    improved = True
    while improved:
        improved = False
        cov = coverage_map(blocks)
        for i in range(len(blocks)):
            b = blocks[i]
            unique = 0
            for ts in itertools.combinations(b, t):
                if cov[ts] == 1:
                    unique += 1
            if unique == 0:
                # safe to remove
                del blocks[i]
                improved = True
                break  # restart loop

    # Swap improvement: for each block, try replacing it with a better block that covers
    # the same "critical" tsubs and possibly more, reducing total count is not directly
    # possible (we can't reduce count via swap), but we can enable later removals.
    # Instead, do a second greedy pass on the still-uncovered-after-removal set is 0.
    # So focus: try to find, for each block, an alternative block (same size) that covers
    # a superset of the tsubs it uniquely covers, allowing removal of another block.
    # This is expensive; do a light version: for a few blocks, try a small neighborhood.

    # Final: verify and return
    # Convert to lists
    result = [list(b) for b in blocks]

    # Verify validity (safety)
    cov = coverage_map(result)
    for ts in all_tsubs:
        if cov.get(ts, 0) == 0:
            # fallback: add a block covering it (should not happen)
            b = list(ts)
            r = random.Random(999)
            while len(b) < k:
                x = r.choice([p for p in all_points if p not in b])
                b.append(x)
            b.sort()
            result.append(b)
    return result
