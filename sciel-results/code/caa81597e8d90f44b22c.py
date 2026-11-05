import time
import random
import itertools

def _classic_construct(n):
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    return word

def _overlap(a, b):
    # max k such that suffix of a (len k) == prefix of b (len k)
    lim = min(len(a), len(b))
    for k in range(lim, -1, -1):
        if a[len(a) - k:] == b[:k]:
            return k
    return 0

def _reorder_blocks(base_perms, new_symbol):
    # base_perms: list of tuples (length n-1); new_symbol: the symbol m
    # order blocks so that consecutive overlap (suffix of one block == prefix of next)
    # is maximized, then build the full word.
    if len(base_perms) <= 2:
        return None
    blocks = [list(p) + [new_symbol] + list(p) for p in base_perms]
    idx = list(range(len(blocks)))
    # nearest-neighbor construction
    order = []
    used = set()
    cur = 0
    order.append(0)
    used.add(0)
    while len(order) < len(idx):
        best, best_score = -1, -1
        for j in idx:
            if j in used:
                continue
            sc = _overlap(blocks[cur], blocks[j])
            if sc > best_score:
                best_score, best = sc, j
        order.append(best)
        used.add(best)
        cur = best
    # assemble
    new_word = []
    for bi in order:
        block = blocks[bi]
        mo = min(len(new_word), len(block))
        ov = 0
        for o in range(mo, -1, -1):
            if new_word[len(new_word) - o:] == block[:o]:
                ov = o
                break
        new_word += block[ov:]
    return new_word

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin for 120s limit

    if n == 1:
        return "1"

    # Standard recursive construction (guaranteed valid, length sum k!)
    base = _classic_construct(n)

    if n < 7:
        return "".join(map(str, base))

    # ---- Try to improve for n >= 7 ----
    candidates = [base]

    # Permutation-block reordering heuristic (top level)
    if n >= 6 and time.time() < deadline - 20:
        m = n
        base_perms, seen = [], set()
        for i in range(len(base) - m + 2):
            window = tuple(base[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                base_perms.append(window)
        reordered = _reorder_blocks(base_perms, m)
        if reordered is not None and len(reordered) < len(base):
            candidates.append(reordered)

    # ---- Local search: greedy removal of redundant characters ----
    def remove_pass(w, deadline):
        n = len(set(w))  # symbols are 1..n
        w = list(w)
        N = len(w)
        if N <= n:
            return w
        # count each permutation
        counts = {}
        for i in range(N - n + 1):
            p = tuple(w[i:i + n])
            counts[p] = counts.get(p, 0) + 1
        unique = set(p for p, c in counts.items() if c == 1)

        i = 0
        while i < len(w) - n + 1:
            if time.time() > deadline:
                break
            # windows that contain w[i]: start in [i-n+1, i]
            lo = max(0, i - n + 1)
            hi = min(i, len(w) - n + 1)
            removable = True
            for s in range(lo, hi + 1):
                if tuple(w[s:s + n]) in unique:
                    removable = False
                    break
            if not removable:
                i += 1
                continue

            # safe to remove w[i]; update counts
            for s in range(lo, hi + 1):
                p = tuple(w[s:s + n])
                c = counts[p]
                if c == 1:
                    del counts[p]
                    unique.discard(p)
                elif c == 2:
                    counts[p] = 1
                    unique.add(p)
                else:
                    counts[p] = c - 1

            w.pop(i)

            # new windows at boundary: starts in [i-n+1, i-1]
            for s in range(max(0, i - n + 1), i):
                if s < len(w) - n + 1:
                    p = tuple(w[s:s + n])
                    c = counts.get(p, 0)
                    if c == 0:
                        counts[p] = 1
                        unique.add(p)
                    elif c == 1:
                        counts[p] = 2
                        unique.discard(p)
                    else:
                        counts[p] = c + 1
        return w

    best = None
    for cand in candidates:
        if time.time() > deadline - 5:
            break
        # verify validity of candidate
        allp = set(itertools.permutations(range(1, n + 1)))
        present = set()
        for i in range(len(cand) - n + 1):
            present.add(tuple(cand[i:i + n]))
        if not allp.issubset(present):
            continue
        if best is None or len(cand) < len(best):
            best = cand

    # run removal pass on the best candidate
    if best is not None and time.time() < deadline - 3:
        improved = remove_pass(best, deadline)
        if len(improved) < len(best):
            best = improved

    if best is None:
        best = base

    # final safety verification
    try:
        allp = set(itertools.permutations(range(1, n + 1)))
        present = set()
        for i in range(len(best) - n + 1):
            present.add(tuple(best[i:i + n]))
        if not allp.issubset(present):
            best = base
    except Exception:
        best = base

    return "".join(map(str, best))
