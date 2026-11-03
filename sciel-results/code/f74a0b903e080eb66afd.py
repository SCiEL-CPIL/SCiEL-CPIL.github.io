"""Improved covering design C(v,k,t): fast bitmask greedy + time-budgeted ILS."""
# EVOLVE-BLOCK-START
import time
import random
import itertools


def construct(v, k, t):
    deadline = time.time() + 135.0  # safety margin under the 150 s wall-clock limit

    if t <= 0:
        return []
    if k >= v:
        return [list(range(v))]
    if k == t:
        return [list(c) for c in itertools.combinations(range(v), k)]

    points = list(range(v))
    all_subs = [tuple(c) for c in itertools.combinations(points, t)]
    nsubs = len(all_subs)
    sub_index = {s: i for i, s in enumerate(all_subs)}

    # point -> sorted list of t-subset indices containing it
    point_subs = [[] for _ in range(v)]
    for i, s in enumerate(all_subs):
        for p in s:
            point_subs[p].append(i)
    for p in range(v):
        point_subs[p].sort()

    def block_subs_fast(block):
        if not block:
            return []
        lists = [sorted(point_subs[p]) for p in block]
        lists.sort(key=len)
        cur = lists[0]
        for lst in lists[1:]:
            j = 0
            nxt = []
            for x in cur:
                while j < len(lst) and lst[j] < x:
                    j += 1
                if j < len(lst) and lst[j] == x:
                    nxt.append(x)
            cur = nxt
            if not cur:
                break
        return cur

    def greedy(initial_blocks):
        blocks = list(initial_blocks)
        covered = set()
        for b in blocks:
            covered.update(block_subs_fast(b))
        uncovered = nsubs - len(covered)
        while uncovered > 0 and time.time() < deadline - 1.0:
            start = random.randrange(nsubs)
            target = -1
            for off in range(nsubs):
                i = (start + off) % nsubs
                if i not in covered:
                    target = i
                    break
            if target < 0:
                break
            block = list(all_subs[target])
            while len(block) < k:
                inblock = set(block)
                best = -1
                best_gain = -1
                for x in points:
                    if x in inblock:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        if all(p != x for p in rest):
                            s = tuple(sorted(rest + (x,)))
                            idx = sub_index.get(s)
                            if idx is not None and idx not in covered:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best = x
                if best < 0 or best_gain <= 0:
                    used = set(block)
                    cands = [x for x in points if x not in used]
                    if not cands:
                        break
                    best = random.choice(cands)
                block.append(best)
            block.sort()
            blocks.append(block)
            covered.update(block_subs_fast(block))
            uncovered = nsubs - len(covered)
        return blocks, uncovered

    blocks, unc = greedy([])
    if unc > 0:
        covered = set()
        for b in blocks:
            covered.update(block_subs_fast(b))
        while True:
            found = None
            for i in range(nsubs):
                if i not in covered:
                    found = i
                    break
            if found is None:
                break
            block = list(all_subs[found])
            random.shuffle(block)
            while len(block) < k:
                used = set(block)
                cands = [x for x in points if x not in used]
                block.append(random.choice(cands))
            block.sort()
            blocks.append(block)
            covered.update(block_subs_fast(block))
    best = blocks
    best_len = len(blocks)
    if time.time() > deadline - 5.0:
        return best

    while time.time() < deadline - 5.0:
        if len(best) <= 2:
            break
        drop = max(1, min(len(best) // 4, random.randrange(1, max(2, len(best) // 3 + 1))))
        keep = list(best)
        random.shuffle(keep)
        perturbed = keep[drop:]
        repaired, unc = greedy(perturbed)
        if unc > 0:
            continue
        if len(repaired) < best_len:
            best = repaired
            best_len = len(repaired)
    return best
# EVOLVE-BLOCK-END
