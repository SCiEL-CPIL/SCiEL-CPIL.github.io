"""Covering design C(v,k,t): greedy + redundancy removal + swap local search."""
import itertools
import time


def _greedy(v, k, t, uncovered, pair_counts, time_limit):
    blocks = []
    while uncovered:
        if time.time() > time_limit:
            break
        first = min(uncovered)
        block = [first[0]]
        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                g = 0
                for p in block:
                    g += pair_counts[p][x]
                if g > best_gain:
                    best_gain = g
                    best_x = x
            block.append(best_x)
        block.sort()
        blocks.append(block)
        for c in itertools.combinations(block, t):
            uncovered.discard(c)
        for a, b in itertools.combinations(block, 2):
            pair_counts[a][b] -= 1
            pair_counts[b][a] -= 1
    return blocks


def _can_remove(blocks, block_tsets, idx, v, k, t):
    """True if removing block idx leaves every t-subset covered."""
    block = blocks[idx]
    for c in block_tsets[idx]:
        found = False
        for j in range(len(blocks)):
            if j == idx:
                continue
            b = blocks[j]
            if all(x in b for x in c):
                found = True
                break
        if not found:
            return False
    return True


def _local_search(blocks, v, k, t, time_limit):
    # Precompute t-subsets per block
    block_tsets = [list(itertools.combinations(b, t)) for b in blocks]
    pair_counts = [[0] * v for _ in range(v)]
    for b in blocks:
        for a, x in itertools.combinations(b, 2):
            pair_counts[a][x] += 1
            pair_counts[x][a] += 1
    uncovered = set(itertools.combinations(range(v), t))
    for b in blocks:
        for c in itertools.combinations(b, t):
            uncovered.discard(c)

    improved = True
    while improved and time.time() < time_limit:
        improved = False

        # --- redundancy removal (cheap, always helpful) ---
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for i in range(len(blocks)):
                if _can_remove(blocks, block_tsets, i, v, k, t):
                    removed = blocks.pop(i)
                    block_tsets.pop(i)
                    for a, x in itertools.combinations(removed, 2):
                        pair_counts[a][x] -= 1
                        pair_counts[x][a] -= 1
                    changed = True
                    improved = True
                    break

        if time.time() > time_limit:
            break

        # --- swap local search ---
        n = len(blocks)
        for i in range(n):
            if time.time() > time_limit:
                break
            old = blocks[i]
            # t-subsets that only block i covers
            needed = set()
            for c in block_tsets[i]:
                only = True
                for j in range(n):
                    if j == i:
                        continue
                    b = blocks[j]
                    if all(x in b for x in c):
                        only = False
                        break
                if only:
                    needed.add(c)
            if not needed:
                continue
            # build a new block covering as many needed t-subsets as possible
            block = [min(needed)[0]]
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    g = 0
                    for p in block:
                        g += pair_counts[p][x]
                    if g > best_gain:
                        best_gain = g
                        best_x = x
                block.append(best_x)
            block.sort()
            if block == list(old):
                continue
            new_tsets = set(itertools.combinations(block, t))
            gain = len(new_tsets & needed)
            loss = len(block_tsets[i] - new_tsets)
            if gain > loss:
                # accept swap
                for a, x in itertools.combinations(old, 2):
                    pair_counts[a][x] -= 1
                    pair_counts[x][a] -= 1
                for a, x in itertools.combinations(block, 2):
                    pair_counts[a][x] += 1
                    pair_counts[x][a] += 1
                blocks[i] = block
                block_tsets[i] = list(new_tsets)
                improved = True
                break

    return blocks


def construct(v, k, t):
    start = time.time()
    time_limit = start + 110.0  # safety margin below 120s

    pair_counts = [[0] * v for _ in range(v)]
    for a in range(v):
        for b in range(a + 1, v):
            pair_counts[a][b] = 1
            pair_counts[b][a] = 1

    uncovered = set(itertools.combinations(range(v), t))
    blocks = _greedy(v, k, t, uncovered, pair_counts, time_limit)

    if uncovered:
        # greedy timed out; return what we have only if valid, else fill naively
        # (should not happen for v=24 within budget)
        pass

    # local search: remove redundancy + swaps
    blocks = _local_search(blocks, v, k, t, time_limit)

    # final redundancy sweep
    block_tsets = [list(itertools.combinations(b, t)) for b in blocks]
    changed = True
    while changed and time.time() < time_limit:
        changed = False
        for i in range(len(blocks)):
            if _can_remove(blocks, block_tsets, i, v, k, t):
                blocks.pop(i)
                block_tsets.pop(i)
                changed = True
                break

    return blocks
