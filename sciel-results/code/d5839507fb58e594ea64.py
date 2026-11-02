"""Improved greedy covering with local search (block replacement / removal)."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def _cover_count(block, uncovered):
    """Number of uncovered t-subsets covered by block."""
    return sum(1 for c in itertools.combinations(block, t) if c in uncovered)


def _build_greedy(v, k, t, start_block=None, rng=None):
    """Greedy set cover: repeatedly add the block covering the most uncovered t-subsets."""
    all_tsubs = set(itertools.combinations(range(v), t))
    uncovered = set(all_tsubs)
    blocks = []
    if start_block is not None:
        blocks.append(list(start_block))
        uncovered.difference_update(itertools.combinations(start_block, t))
    while uncovered:
        best_gain = 0
        best_block = None
        # Iterate over candidate blocks. For speed, we use a heuristic:
        # pick the first uncovered t-subset and build a block around it greedily,
        # then also try a few random extensions to pick the best.
        first = min(uncovered)
        # Build a candidate block containing 'first'
        cand = set(first)
        while len(cand) < k:
            best_x = None
            best_g = -1
            for x in range(v):
                if x in cand:
                    continue
                # count how many uncovered t-subsets would be covered if x added
                g = 0
                for rest in itertools.combinations(cand, t - 1):
                    tt = tuple(sorted(rest + (x,)))
                    if tt in uncovered:
                        g += 1
                if g > best_g:
                    best_g = g
                    best_x = x
            if best_x is None:
                break
            cand.add(best_x)
        cand_block = sorted(cand)
        gain = _cover_count(cand_block, uncovered)
        if gain > best_gain:
            best_gain = gain
            best_block = cand_block
        # Also try a few random blocks containing 'first' to diversify
        if rng is not None:
            for _ in range(3):
                cand2 = set(first)
                remaining = [x for x in range(v) if x not in cand2]
                rng.shuffle(remaining)
                for x in remaining:
                    if len(cand2) >= k:
                        break
                    cand2.add(x)
                cand_block2 = sorted(cand2)
                gain2 = _cover_count(cand_block2, uncovered)
                if gain2 > best_gain:
                    best_gain = gain2
                    best_block = cand_block2
        if best_block is None or best_gain <= 0:
            # Fallback: just use the first uncovered t-subset padded arbitrarily
            if best_block is None:
                best_block = sorted(first + tuple(x for x in range(v) if x not in first)[:k - len(first)])
            if best_gain <= 0:
                best_gain = 0
        blocks.append(best_block)
        uncovered.difference_update(itertools.combinations(best_block, t))
    return blocks


def _try_remove_block(blocks, uncovered, v, k, t):
    """Try to remove a block that is redundant (all its t-subsets covered by other blocks)."""
    for i in range(len(blocks) - 1, -1, -1):
        b = blocks[i]
        # Check if all t-subsets of b are covered by other blocks
        redundant = True
        for c in itertools.combinations(b, t):
            covered = False
            for j, other in enumerate(blocks):
                if j == i:
                    continue
                if c in set(itertools.combinations(other, t)):
                    covered = True
                    break
            if not covered:
                redundant = False
                break
        if redundant:
            blocks.pop(i)
            return True
    return False


def _local_search(blocks, v, k, t, rng, time_limit):
    """Local search: try to replace a block with a better one, or remove redundant blocks."""
    start_time = time.time()
    all_tsubs = set(itertools.combinations(range(v), t))
    # Compute uncovered after current blocks
    covered = set()
    for b in blocks:
        covered.update(itertools.combinations(b, t))
    uncovered = all_tsubs - covered
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try removing a redundant block
        for i in range(len(blocks) - 1, -1, -1):
            if (time.time() - start_time) >= time_limit:
                return blocks
            b = blocks[i]
            # Check redundancy
            redundant = True
            for c in itertools.combinations(b, t):
                if c in uncovered:
                    # c is not covered by any other block (since uncovered means not covered at all)
                    # Actually, we need to check if c is covered by other blocks
                    covered_by_other = False
                    for j, other in enumerate(blocks):
                        if j == i:
                            continue
                        if c in set(itertools.combinations(other, t)):
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        redundant = False
                        break
            if redundant:
                blocks.pop(i)
                # Recompute uncovered
                covered = set()
                for b2 in blocks:
                    covered.update(itertools.combinations(b2, t))
                uncovered = all_tsubs - covered
                improved = True
                break
        if improved:
            continue
        # Try replacing a block with a better one
        for i in range(len(blocks)):
            if (time.time() - start_time) >= time_limit:
                return blocks
            old_block = blocks[i]
            # Try to find a new block that covers more uncovered t-subsets
            # while still covering all t-subsets that old_block uniquely covers
            # Simplified: just try random replacements and keep if it doesn't break coverage
            for _ in range(5):
                if (time.time() - start_time) >= time_limit:
                    return blocks
                # Generate a random block
                new_block = sorted(rng.sample(range(v), k))
                # Check if replacing old_block with new_block keeps all t-subsets covered
                # This is expensive, so we do a quick check
                old_cover = set(itertools.combinations(old_block, t))
                new_cover = set(itertools.combinations(new_block, t))
                # t-subsets only covered by old_block must also be covered by new_block or other blocks
                ok = True
                for c in old_cover:
                    if c not in new_cover:
                        # Check if c is covered by other blocks
                        covered = False
                        for j, other in enumerate(blocks):
                            if j == i:
                                continue
                            if c in set(itertools.combinations(other, t)):
                                covered = True
                                break
                        if not covered:
                            ok = False
                            break
                if ok:
                    # Check if new_block covers more uncovered t-subsets
                    gain = sum(1 for c in new_cover if c in uncovered)
                    loss = sum(1 for c in old_cover if c in uncovered and c not in new_cover)
                    if gain > loss:
                        blocks[i] = new_block
                        # Recompute uncovered
                        covered = set()
                        for b2 in blocks:
                            covered.update(itertools.combinations(b2, t))
                        uncovered = all_tsubs - covered
                        improved = True
                        break
    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = 140.0  # safety margin
    rng = random.Random(42)
    
    # Initial greedy construction
    blocks = _build_greedy(v, k, t, start_block=None, rng=rng)
    
    # Local search to improve
    if (time.time() - start_time) < deadline - 10:
        blocks = _local_search(blocks, v, k, t, rng, deadline - (time.time() - start_time))
    
    # Try a few more random restarts if time permits
    best = blocks
    while (time.time() - start_time) < deadline - 20:
        # Random restart with a random initial block
        initial = sorted(rng.sample(range(v), k))
        blocks2 = _build_greedy(v, k, t, start_block=initial, rng=rng)
        if len(blocks2) < len(best):
            best = blocks2
        if (time.time() - start_time) < deadline - 5:
            blocks2 = _local_search(blocks2, v, k, t, rng, deadline - (time.time() - start_time) - 5)
            if len(blocks2) < len(best):
                best = blocks2
    
    return best
# EVOLVE-BLOCK-END
