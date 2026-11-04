import time
import itertools
import numpy as np

def _all_words(q, n):
    total = q ** n
    a = np.empty(total, dtype=np.uint8)
    v = 0
    for i in range(total):
        x = i
        for j in range(n):
            a[i, j] = x % q
            x //= q
        v = i
    return a

def _ball_indices(q, n, radius):
    # ball[c] = list of indices within Hamming distance <= radius of word c
    if radius == 1:
        total = q ** n
        step = q ** np.arange(n - 1, -1, -1, dtype=np.int64)
        b = [None] * total
        for c in range(total):
            lst = [c]
            for j in range(n):
                base = c - (c % step[j])
                for d in range(q):
                    idx = base + d * step[j]
                    if idx != c:
                        lst.append(idx)
            b[c] = lst
        return b
    # generic fallback (only used if radius != 1)
    words = _all_words(q, n)
    b = []
    for c in range(q ** n):
        wc = words[c]
        lst = [c]
        for i in range(q ** n):
            if i == c:
                continue
            if int(np.sum(words[i] != wc)) <= radius:
                lst.append(i)
        b.append(lst)
    return b

def _prune(chosen, balls):
    if not chosen:
        return []
    total = len(balls)
    cc = np.zeros(total, dtype=np.int32)
    for c in chosen:
        for j in balls[c]:
            cc[j] += 1
    keep = [True] * len(chosen)
    changed = True
    while changed:
        changed = False
        for i, c in enumerate(chosen):
            if not keep[i]:
                continue
            redundant = True
            for j in balls[c]:
                if cc[j] <= 1:
                    redundant = False
                    break
            if redundant:
                keep[i] = False
                changed = True
                for j in balls[c]:
                    cc[j] -= 1
    return [chosen[i] for i in range(len(chosen)) if keep[i]]

def _valid(chosen, balls, total):
    cov = np.zeros(total, dtype=bool)
    for c in chosen:
        for j in balls[c]:
            cov[j] = True
    return bool(np.all(cov))

def _greedy_cover(balls, total, deadline):
    uncovered = np.ones(total, dtype=bool)
    chosen = []
    while time.time() < deadline and np.any(uncovered):
        best = -1
        best_g = 0
        for c in range(total):
            g = 0
            for j in balls[c]:
                if uncovered[j]:
                    g += 1
            if g > best_g:
                best_g = g
                best = c
                if g >= len(balls[c]):
                    break
        if best < 0 or best_g == 0:
            break
        chosen.append(best)
        for j in balls[best]:
            if uncovered[j]:
                uncovered[j] = False
    return chosen

def _local_search(chosen, balls, total, deadline):
    chosen = _prune(chosen, balls)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        cc = np.zeros(total, dtype=np.int32)
        for c in chosen:
            for j in balls[c]:
                cc[j] += 1
        n = len(chosen)
        # Delete move
        for i in range(n):
            c = chosen[i]
            if all(cc[j] > 1 for j in balls[c]):
                for j in balls[c]:
                    cc[j] -= 1
                chosen.pop(i)
                improved = True
                break
        if improved:
            continue
        # 1-for-1 swap
        for i in range(len(chosen)):
            c = chosen[i]
            uniq = [j for j in balls[c] if cc[j] == 1]
            if not uniq:
                continue
            cand = set(balls[uniq[0]])
            for u in uniq[1:]:
                cand &= set(balls[u])
            cand.discard(c)
            cand -= set(chosen)
            for cp in cand:
                # validity: every point that c uniquely covered must be in ball[cp]
                ok = True
                for u in uniq:
                    if not any(u == x for x in balls[cp]):
                        ok = False
                        break
                if not ok:
                    continue
                # apply
                for j in balls[c]:
                    cc[j] -= 1
                for j in balls[cp]:
                    cc[j] += 1
                chosen[i] = cp
                improved = True
                break
            if improved:
                break
        if improved:
            continue
        # 2-for-1 move: replace c with two words
        for i in range(len(chosen)):
            c = chosen[i]
            uniq = [j for j in balls[c] if cc[j] == 1]
            if not uniq:
                continue
            base = set(balls[uniq[0]])
            for u in uniq[1:]:
                base &= set(balls[u])
            base.discard(c)
            base -= set(chosen)
            base = list(base)
            if not base:
                continue
            for cp in base:
                for cq in base:
                    if cq <= cp:
                        continue
                    # both must cover all uniq
                    if any(u not in balls[cp] for u in uniq):
                        continue
                    if any(u not in balls[cq] for u in uniq):
                        continue
                    # apply
                    for j in balls[c]:
                        cc[j] -= 1
                    for j in balls[cp]:
                        cc[j] += 1
                    for j in balls[cq]:
                        cc[j] += 1
                    chosen[i] = cp
                    chosen.append(cq)
                    improved = True
                    break
                if improved:
                    break
            if improved:
                break
        if improved:
            chosen = _prune(chosen, balls)
    return chosen

def construct(q, n, radius):
    start = time.time()
    deadline = start + 112.0
    total = q ** n
    try:
        balls = _ball_indices(q, n, radius)
        best = None
        best_size = total
        rng = np.random.default_rng(0)
        # First: greedy from empty
        chosen = _greedy_cover(balls, total, deadline)
        if _valid(chosen, balls, total):
            chosen = _local_search(chosen, balls, total, deadline)
            if _valid(chosen, balls, total):
                best = chosen
                best_size = len(chosen)
        # Multi-restart: seed from random subset + greedy
        while time.time() < deadline - 5.0:
            seed_size = int(total * 0.05)
            seed = rng.choice(total, size=min(seed_size, total), replace=False).tolist()
            chosen = list(seed)
            chosen = _greedy_cover(balls, total, deadline)  # greedy from current uncovered
            if _valid(chosen, balls, total):
                chosen = _local_search(chosen, balls, total, deadline)
                if _valid(chosen, balls, total):
                    if len(chosen) < best_size:
                        best_size = len(chosen)
                        best = chosen
        if best is None:
            # fallback: tetracode direct sum
            _TETRA = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
            if n < 4:
                return [list(t) for t in itertools.product(range(q), repeat=n)]
            return [h + list(tail) for h in _TETRA
                    for tail in itertools.product(range(q), repeat=n - 4)]
        words = _all_words(q, n)
        return [words[i].tolist() for i in best]
    except Exception:
        _TETRA = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        if n < 4:
            return [list(t) for t in itertools.product(range(q), repeat=n)]
        return [h + list(tail) for h in _TETRA
                for tail in itertools.product(range(q), repeat=n - 4)]
