"""Covering design C(24,7,3): randomized greedy + local search."""
import itertools
import time
import random

def _triple_index(v, t):
    # map each t-subset to a dense int id
    idx = {}
    i = 0
    for c in itertools.combinations(range(v), t):
        idx[c] = i
        i += 1
    return idx, i

def construct(v, k, t):
    start = time.time()
    DEADLINE = 140.0  # safety margin under 150s kill

    triple_id, ntriples = _triple_index(v, t)
    all_triples = list(itertools.combinations(range(v), t))

    # Precompute, for each point x and each (t-1)-subset, the list of triple-ids
    # that x completes. This lets us compute gain of adding x to a block quickly.
    # gain of adding x to block B = number of uncovered triples of form (t-1 subset of B) + x
    # We'll compute on the fly but with precomputed adjacency.

    # adj[x][tuple of (t-1)-subset] -> list of triple ids
    adj = [dict() for _ in range(v)]
    for x in range(v):
        d = adj[x]
        for rest in itertools.combinations([y for y in range(v) if y != x], t - 1):
            tr = tuple(sorted(rest + (x,)))
            d[rest] = triple_id[tr]

    best_blocks = None

    def greedy(seed):
        rng = random.Random(seed)
        # uncovered as a set of triple ids
        uncovered = set(range(ntriples))
        blocks = []
        while uncovered:
            # pick a random uncovered triple as the seed of the new block
            seed_triple = rng.choice(list(uncovered))
            block = list(seed_triple)
            block_set = set(block)
            while len(block) < k:
                # choose x maximizing number of new triples covered
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        tid = adj[x][rest]
                        if tid in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        if best_gain == 0:
                            break
                block.append(best_x)
                block_set.add(best_x)
            block.sort()
            blocks.append(block)
            # mark covered
            for tr in itertools.combinations(block, t):
                uncovered.discard(triple_id[tr])
        return blocks

    def local_search(blocks, seed):
        rng = random.Random(seed)
        blocks = [list(b) for b in blocks]
        improved = True
        while improved and (time.time() - start) < DEADLINE:
            improved = False
            # build coverage count per triple
            cov = [0] * ntriples
            for b in blocks:
                for tr in itertools.combinations(b, t):
                    cov[triple_id[tr]] += 1
            # find triples with coverage 1 (removing a block that uniquely covers them is dangerous)
            # Try to drop a block if all its triples are still covered
            for i in range(len(blocks)):
                b = blocks[i]
                ok = True
                for tr in itertools.combinations(b, t):
                    if cov[triple_id[tr]] == 1:
                        ok = False
                        break
                if ok:
                    blocks.pop(i)
                    improved = True
                    break
            if improved:
                continue
            # Try to replace a block with one that covers more currently-uncovered-ish triples
            # Compute uncovered-ish: triples with cov==0 (shouldn't happen if valid) 
            # Instead, try to find a block replacement that reduces redundancy.
            # Heuristic: pick a block, remove it, see how many triples become uncovered,
            # then greedily build a replacement that covers those.
            for i in range(len(blocks)):
                if (time.time() - start) > DEADLINE:
                    return blocks
                b = blocks[i]
                # triples only covered by this block
                only = []
                for tr in itertools.combinations(b, t):
                    if cov[triple_id[tr]] == 1:
                        only.append(tr)
                if not only:
                    continue
                # build a replacement block covering all of 'only'
                # must include union? Not necessarily; need a k-set covering all triples in only.
                # Greedy: start from first triple, add points to cover remaining only-triples.
                rep = list(only[0])
                rep_set = set(rep)
                remaining = list(only)
                while remaining and len(rep) < k:
                    best_x = None
                    best_g = -1
                    for x in range(v):
                        if x in rep_set:
                            continue
                        g = 0
                        for tr in remaining:
                            if x in tr:
                                g += 1
                        if g > best_g:
                            best_g = g
                            best_x = x
                    if best_x is None:
                        break
                    rep.append(best_x)
                    rep_set.add(best_x)
                    remaining = [tr for tr in remaining if not all(pt in rep_set for pt in tr)]
                if not remaining:
                    # replacement covers all only-triples; also ensure it doesn't break others (it adds coverage)
                    rep.sort()
                    blocks[i] = rep
                    # recompute cov
                    cov = [0] * ntriples
                    for bb in blocks:
                        for tr in itertools.combinations(bb, t):
                            cov[triple_id[tr]] += 1
                    improved = True
                    break
        return blocks

    # Multiple restarts
    seed = 0
    while (time.time() - start) < DEADLINE:
        blocks = greedy(seed)
        # local search to try to reduce count
        blocks = local_search(blocks, seed)
        if best_blocks is None or len(blocks) < len(best_blocks):
            best_blocks = blocks
            # if we hit a great number, we could stop, but keep going for a bit
        seed += 1

    if best_blocks is None:
        best_blocks = greedy(0)

    return [sorted(b) for b in best_blocks]
