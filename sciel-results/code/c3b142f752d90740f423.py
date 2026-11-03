"""Ternary covering code K_3(7,1): minimise number of length-7 words over {0,1,2}
so every word of the space is within Hamming distance <= 1 of some chosen word.

Strategy:
  1. Fast greedy construction -> a valid (smallish) covering.
  2. Exact set-cover reduction -> provably minimal within the chosen family.
  3. Time-boxed multi-start local search (add / 2-for-1 swap / 3-for-2 swap)
     to escape local minima, using vectorised ball coverage over the full space.

All heavy work is guarded by an explicit wall-clock deadline; the best valid
solution found is always returned.
"""
import time
import itertools
import numpy as np

# ---------- vectorised ball coverage over the full space ----------
# space: int array of shape (S, n) listing every word of {0..q-1}^n
# balls: int array of shape (S, ball_size) where row i lists the space-indices
#        of all words within Hamming distance <= radius of space[i]
def _build_space_balls(q, n, radius):
    S = q ** n
    space = np.zeros((S, n), dtype=np.int64)
    for i in range(S):
        v = i
        for j in range(n):
            space[i, j] = v % q
            v //= q
    base = np.arange(S)
    cols = [base]
    for pos in range(n):
        old = space[:, pos]
        for a in range(q):
            m = (old != a)
            if not np.any(m):
                continue
            new = old.copy()
            new[m] = a
            vals = (new * q).sum(axis=1)
            cols.append(vals)
    balls = np.concatenate(cols, axis=1)
    return space, balls


def _is_covered(balls, idx):
    return bool(np.any(balls[idx], axis=0).all())


def _unique_of(balls, idx):
    """space-indices covered by balls[idx] and by no other chosen codeword."""
    if len(idx) == 0:
        return np.zeros(0, dtype=np.int64)
    cnt = np.zeros(balls.shape[0], dtype=np.int32)
    for i in idx:
        cnt[balls[i]] += 1
    u = []
    for i in idx:
        b = balls[i]
        m = (cnt[b] == 1)
        if np.any(m):
            u.append(b[m])
    return np.concatenate(u) if u else np.zeros(0, dtype=np.int64)


# ---------- initial greedy construction ----------
def _greedy(q, n, radius, balls, space):
    S = balls.shape[0]
    covered = np.zeros(S, dtype=bool)
    chosen = []
    order = np.arange(S)
    rng = np.random.default_rng(12345)
    while not covered.all():
        # score each candidate = number of newly covered words
        gain = np.zeros(S, dtype=np.int32)
        for i in range(S):
            b = balls[i]
            gain[i] = int(np.sum(~covered[b]))
        # pick a candidate with maximal gain; tie-break randomly for diversity
        gmax = int(gain.max())
        cands = order[gain == gmax]
        pick = int(cands[rng.integers(len(cands))])
        chosen.append(pick)
        covered[balls[pick]] = True
    return chosen


# ---------- exact set-cover reduction ----------
def _reduce(balls, chosen):
    S = balls.shape[0]
    cover = np.zeros(S, dtype=np.int32)
    for i in chosen:
        cover[balls[i]] += 1
    kept = []
    for i in chosen:
        b = balls[i]
        if np.any(cover[b] == 1):   # some word only this codeword covers
            kept.append(i)
        else:
            cover[b] -= 1
    return kept


# ---------- local search ----------
def _local_search(q, n, radius, balls, space, chosen, deadline, seed):
    rng = np.random.default_rng(seed)
    S = balls.shape[0]
    order = np.arange(S)
    chosen = list(chosen)
    best = len(chosen)
    best_set = set(chosen)

    def _covered(chosen_list):
        c = np.zeros(S, dtype=np.int32)
        for i in chosen_list:
            c[balls[i]] += 1
        return c

    # --- Phase 1: add to restore coverage, then reduce (escape a stuck state) ---
    def _add_then_reduce():
        nonlocal best, best_set
        if time.time() > deadline:
            return
        cover = _covered(chosen)
        while True:
            if time.time() > deadline:
                return
            unc = np.where(cover == 0)[0]
            if len(unc) == 0:
                break
            gain = np.zeros(S, dtype=np.int32)
            for i in range(S):
                b = balls[i]
                gain[i] = int(np.sum(cover[b] == 0))
            gmax = int(gain.max())
            cands = order[gain == gmax]
            pick = int(cands[rng.integers(len(cands))])
            chosen.append(pick)
            cover[balls[pick]] += 1
        reduced = _reduce(balls, chosen)
        chosen = reduced
        if len(chosen) < best:
            best = len(chosen)
            best_set = set(chosen)

    # --- Phase 2: 2-for-1 swap (remove one, add up to two) ---
    def _swap21():
        nonlocal best, best_set
        if time.time() > deadline:
            return
        cover = _covered(chosen)
        for _ in range(400):
            if time.time() > deadline:
                return
            improved = False
            # order removals randomly
            idx = list(range(len(chosen)))
            rng.shuffle(idx)
            for pos in idx:
                if time.time() > deadline:
                    return
                if pos >= len(chosen):
                    continue
                i = chosen[pos]
                cover[balls[i]] -= 1
                u = np.where(cover == 0)[0]
                if len(u) == 0:
                    # removal is safe
                    chosen.pop(pos)
                    improved = True
                    continue
                # need to cover u; try adding up to 2 codewords
                need = set(u.tolist())
                # score candidates by how many of `need` they cover
                gain = np.zeros(S, dtype=np.int32)
                uarr = np.array(u, dtype=np.int64)
                for k in range(S):
                    b = balls[k]
                    gain[k] = int(np.sum(np.isin(b, uarr)))
                gmax = int(gain.max())
                cands = order[gain == gmax]
                if len(cands) == 0:
                    cover[balls[i]] += 1
                    continue
                a = int(cands[rng.integers(len(cands))])
                cover[balls[a]] += 1
                need2 = [x for x in need if cover[x] == 0]
                if need2:
                    # try one more
                    uarr2 = np.array(need2, dtype=np.int64)
                    gain2 = np.zeros(S, dtype=np.int32)
                    for k in range(S):
                        if k == a:
                            continue
                        b = balls[k]
                        gain2[k] = int(np.sum(np.isin(b, uarr2)))
                    gmax2 = int(gain2.max())
                    if gmax2 > 0:
                        cands2 = order[gain2 == gmax2]
                        b2 = int(cands2[rng.integers(len(cands2))])
                        cover[balls[b2]] += 1
                        chosen.append(a)
                        chosen.append(b2)
                        improved = True
                    else:
                        # can't fix with one more; undo
                        cover[balls[a]] -= 1
                else:
                    chosen.append(a)
                    improved = True
            if not improved:
                break
        reduced = _reduce(balls, chosen)
        chosen = reduced
        if len(chosen) < best:
            best = len(chosen)
            best_set = set(chosen)

    # --- Phase 3: 3-for-2 swap (remove two, add up to three) ---
    def _swap32():
        nonlocal best, best_set
        if time.time() > deadline:
            return
        cover = _covered(chosen)
        for _ in range(200):
            if time.time() > deadline:
                return
            improved = False
            L = len(chosen)
            p = int(rng.integers(L))
            q_ = int(rng.integers(L - 1))
            if q_ >= p:
                q_ += 1
            i1 = chosen[p]
            i2 = chosen[q_]
            cover[balls[i1]] -= 1
            cover[balls[i2]] -= 1
            u = np.where(cover == 0)[0]
            if len(u) == 0:
                # safe removal of both
                for pos in sorted([p, q_], reverse=True):
                    chosen.pop(pos)
                improved = True
                continue
            # add up to 3 to cover u
            added = []
            for _a in range(3):
                if time.time() > deadline:
                    break
                u = np.where(cover == 0)[0]
                if len(u) == 0:
                    break
                uarr = np.array(u, dtype=np.int64)
                gain = np.zeros(S, dtype=np.int32)
                for k in range(S):
                    if k in added:
                        continue
                    b = balls[k]
                    gain[k] = int(np.sum(np.isin(b, uarr)))
                gmax = int(gain.max())
                if gmax == 0:
                    break
                cands = order[gain == gmax]
                pick = int(cands[rng.integers(len(cands))])
                added.append(pick)
                cover[balls[pick]] += 1
            if len(added) <= 1:
                # net removal of at least one
                for pos in sorted([p, q_], reverse=True):
                    chosen.pop(pos)
                chosen.extend(added)
                improved = True
            else:
                # undo (didn't shrink)
                cover[balls[i1]] += 1
                cover[balls[i2]] += 1
                for k in added:
                    cover[balls[k]] -= 1
        reduced = _reduce(balls, chosen)
        chosen = reduced
        if len(chosen) < best:
            best = len(chosen)
            best_set = set(chosen)

    # --- main loop: alternate moves, periodically add-then-reduce to escape ---
    for it in range(4000):
        if time.time() > deadline:
            break
        if it % 30 == 0 and it > 0:
            _add_then_reduce()
        _swap21()
        if it % 7 == 0:
            _swap32()
        if len(chosen) < best:
            best = len(chosen)
            best_set = set(chosen)

    return list(best_set), best


# ---------- entry point ----------
def construct(q, n, radius):
    start = time.time()
    deadline = start + 145.0   # safety margin under 150s wall-clock kill
    S = q ** n
    space, balls = _build_space_balls(q, n, radius)

    # fast greedy initial covering
    chosen = _greedy(q, n, radius, balls, space)
    chosen = _reduce(balls, chosen)
    best = chosen
    best_len = len(chosen)

    # verify validity of the initial solution
    assert _is_covered(balls, chosen), "initial covering invalid"

    # multi-start local search with time budget
    seed = 0
    while time.time() < deadline - 5.0:
        # restart from a perturbation of the best solution for diversity
        if seed % 4 == 0:
            start_set = list(best)
        else:
            start_set = list(best)
            rng = np.random.default_rng(seed + 1000)
            k = max(1, len(start_set) // 10)
            for _ in range(k):
                pos = int(rng.integers(len(start_set)))
                start_set.pop(pos)
                # add a random word to keep it a covering
                add = int(rng.integers(balls.shape[0]))
                start_set.append(add)
            start_set = _reduce(balls, start_set)
        res, res_len = _local_search(q, n, radius, balls, space, start_set,
                                     deadline, seed + 7)
        if res_len < best_len and _is_covered(balls, res):
            best = res
            best_len = res_len
        seed += 1

    # final safety: ensure we return a valid covering
    if not _is_covered(balls, best):
        # fall back to a fresh greedy (should not happen)
        best = _greedy(q, n, radius, balls, space)
        best = _reduce(balls, best)

    return [space[i].tolist() for i in best]
