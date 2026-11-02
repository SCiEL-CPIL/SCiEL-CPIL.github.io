import itertools
import random
import time

_START_TIME = None


def _deadline_ok(budget):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return (time.time() - _START_TIME) < budget


def construct(v, k, t):
    global _START_TIME
    _START_TIME = time.time()
    budget = 140.0  # safety margin before 150s kill

    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    # map triple -> index
    triple_index = {tr: i for i, tr in enumerate(all_triples)}
    n_triples = len(all_triples)

    # Precompute, for each point, the list of triples it participates in
    triples_by_point = [[] for _ in range(v)]
    for tr in all_triples:
        for p in tr:
            triples_by_point[p].append(triple_index[tr])

    # Precompute for each pair of points, the triples containing both (for gain calc)
    # Actually for t=3, gain of adding x to block B = number of uncovered triples
    # {a,b,x} where {a,b} is a pair in B. We compute on the fly with a coverage array.

    rng = random.Random(12345)

    def greedy_cover(seed):
        r = random.Random(seed)
        # coverage count per triple (0 = uncovered)
        cov = [0] * n_triples
        uncovered_count = n_triples
        blocks = []

        while uncovered_count > 0 and _deadline_ok(budget):
            # pick a random uncovered triple to start (demand-aware: pick one with low cov, which is 0)
            # find an uncovered triple
            # We'll maintain a list of uncovered triple indices for speed
            # For simplicity, scan; but to be fast, keep a set
            # Use a list of uncovered indices, rebuild occasionally
            # Simpler: pick randomly from uncovered by scanning
            start_triple = None
            for _ in range(200):
                idx = r.randrange(n_triples)
                if cov[idx] == 0:
                    start_triple = all_triples[idx]
                    break
            if start_triple is None:
                # fallback: linear scan
                for idx in range(n_triples):
                    if cov[idx] == 0:
                        start_triple = all_triples[idx]
                        break
                if start_triple is None:
                    break

            block = list(start_triple)
            block_set = set(block)

            while len(block) < k:
                best_x = None
                best_gain = -1
                # For each candidate point, compute how many new triples it covers
                # A new triple = {a, b, x} where a,b in block, x not in block
                # Precompute pairs in block
                block_pairs = list(itertools.combinations(block, t - 1))
                for x in points:
                    if x in block_set:
                        continue
                    gain = 0
                    for pair in block_pairs:
                        # triple = sorted(pair + (x,))
                        tr = tuple(sorted(pair + (x,)))
                        idx = triple_index[tr]
                        if cov[idx] == 0:
                            gain += 1
                    # tie-break: prefer points that appear in many uncovered triples
                    if gain > best_gain or (gain == best_gain and best_x is not None and r.random() < 0.3):
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
                block_set.add(best_x)

            block.sort()
            blocks.append(block)

            # update coverage
            for tr in itertools.combinations(block, t):
                idx = triple_index[tr]
                if cov[idx] == 0:
                    uncovered_count -= 1
                cov[idx] += 1

        return blocks, cov, uncovered_count

    def is_valid(blocks):
        # check every triple is covered
        cov = [0] * n_triples
        for block in blocks:
            for tr in itertools.combinations(block, t):
                idx = triple_index[tr]
                cov[idx] += 1
        return all(c > 0 for c in cov)

    def local_search(blocks, cov, uncovered_count, max_time=20.0):
        """Simulated annealing / greedy local search to remove redundant blocks
        and swap points to improve coverage."""
        if not _deadline_ok(budget):
            return blocks, cov, uncovered_count

        ls_start = time.time()
        best_blocks = list(blocks)
        best_cov = list(cov)
        best_uncovered = uncovered_count
        current_blocks = list(blocks)
        current_cov = list(cov)
        current_uncovered = uncovered_count

        n_blocks = len(current_blocks)

        # Precompute for each block, the set of triples it covers (indices)
        def block_triples(block):
            return [triple_index[tr] for tr in itertools.combinations(block, t)]

        # For removal: a block is removable if every triple it covers is also covered by another block
        # i.e., cov[tr] >= 2 for all tr in block_triples(block)

        def try_remove(block_idx):
            nonlocal current_blocks, current_cov, current_uncovered
            bt = block_triples(current_blocks[block_idx])
            removable = True
            for idx in bt:
                if current_cov[idx] <= 1:
                    removable = False
                    break
            if not removable:
                return False
            # remove it
            for idx in bt:
                current_cov[idx] -= 1
            current_blocks.pop(block_idx)
            return True

        def try_swap(block_idx, old_point, new_point):
            nonlocal current_blocks, current_cov, current_uncovered
            # compute gain: how many triples become covered (cov 0->1) minus how many become uncovered (cov 1->0)
            block = current_blocks[block_idx]
            # triples removed: those containing old_point
            removed = []
            for tr in itertools.combinations(block, t):
                if old_point in tr:
                    removed.append(triple_index[tr])
            # triples added: those containing new_point (and other t-1 points from block)
            added = []
            others = [p for p in block if p != old_point]
            for tr in itertools.combinations(others, t - 1):
                tr = tuple(sorted(tr + (new_point,)))
                added.append(triple_index[tr])

            gain = 0
            for idx in removed:
                if current_cov[idx] == 1:
                    gain -= 1  # would become uncovered
            for idx in added:
                if current_cov[idx] == 0:
                    gain += 1  # would become covered
            if gain <= 0:
                return False, 0
            # apply
            for idx in removed:
                current_cov[idx] -= 1
            for idx in added:
                if current_cov[idx] == 0:
                    current_uncovered -= 1
                current_cov[idx] += 1
            current_blocks[block_idx] = sorted([p for p in block if p != old_point] + [new_point])
            return True, gain

        # Phase 1: greedy removal of redundant blocks
        progress = True
        while progress and _deadline_ok(budget) and (time.time() - ls_start) < max_time:
            progress = False
            for i in range(len(current_blocks) - 1, -1, -1):
                if try_remove(i):
                    progress = True
                    if len(current_blocks) < len(best_blocks):
                        best_blocks = list(current_blocks)
                        best_cov = list(current_cov)
                        best_uncovered = current_uncovered

        # Phase 2: swap-based local search
        for _ in range(5000):
            if not _deadline_ok(budget) or (time.time() - ls_start) > max_time:
                break
            if not current_blocks:
                break
            block_idx = rng.randrange(len(current_blocks))
            block = current_blocks[block_idx]
            old_point = rng.choice(block)
            new_point = rng.choice(points)
            if new_point == old_point or new_point in block:
                continue
            ok, gain = try_swap(block_idx, old_point, new_point)
            if ok:
                if len(current_blocks) < len(best_blocks):
                    best_blocks = list(current_blocks)
                    best_cov = list(current_cov)
                    best_uncovered = current_uncovered

        return best_blocks, best_cov, best_uncovered

    # Main: multiple restarts
    best_result = None
    for restart in range(200):
        if not _deadline_ok(budget):
            break
        blocks, cov, unc = greedy_cover(restart)
        if unc == 0:
            blocks, cov, unc = local_search(blocks, cov, unc)
            if best_result is None or len(blocks) < len(best_result):
                best_result = blocks
                if len(best_result) <= 76:
                    # good enough, do more local search
                    if _deadline_ok(budget - 5):
                        blocks2, cov2, unc2 = local_search(best_result, cov, unc, max_time=10.0)
                        if len(blocks2) < len(best_result):
                            best_result = blocks2
        else:
            # incomplete cover, try to fix by adding blocks
            # find uncovered triples
            while unc > 0 and _deadline_ok(budget):
                # find an uncovered triple
                target = None
                for idx in range(n_triples):
                    if cov[idx] == 0:
                        target = all_triples[idx]
                        break
                if target is None:
                    break
                # build a block around it
                block = list(target)
                block_set = set(block)
                while len(block) < k:
                    best_x = None
                    best_gain = -1
                    block_pairs = list(itertools.combinations(block, t - 1))
                    for x in points:
                        if x in block_set:
                            continue
                        gain = 0
                        for pair in block_pairs:
                            tr = tuple(sorted(pair + (x,)))
                            idx = triple_index[tr]
                            if cov[idx] == 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x is None:
                        break
                    block.append(best_x)
                    block_set.add(best_x)
                block.sort()
                # check if block already exists
                if block not in blocks:
                    blocks.append(block)
                for tr in itertools.combinations(block, t):
                    idx = triple_index[tr]
                    if cov[idx] == 0:
                        unc -= 1
                    cov[idx] += 1

            if unc == 0:
                blocks, cov, unc = local_search(blocks, cov, unc)
                if best_result is None or len(blocks) < len(best_result):
                    best_result = blocks

    if best_result is None:
        # fallback: return a simple cover (may be large)
        blocks, cov, unc = greedy_cover(0)
        best_result = blocks

    # Final validation
    if not is_valid(best_result):
        # emergency fix
        cov = [0] * n_triples
        for block in best_result:
            for tr in itertools.combinations(block, t):
                cov[triple_index[tr]] += 1
        while any(c == 0 for c in cov):
            target = None
            for idx in range(n_triples):
                if cov[idx] == 0:
                    target = all_triples[idx]
                    break
            if target is None:
                break
            block = list(target)
            block_set = set(block)
            while len(block) < k:
                best_x = None
                best_gain = -1
                block_pairs = list(itertools.combinations(block, t - 1))
                for x in points:
                    if x in block_set:
                        continue
                    gain = 0
                    for pair in block_pairs:
                        tr = tuple(sorted(pair + (x,)))
                        idx = triple_index[tr]
                        if cov[idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
                block_set.add(best_x)
            block.sort()
            if block not in best_result:
                best_result.append(block)
            for tr in itertools.combinations(block, t):
                cov[triple_index[tr]] += 1

    return best_result
