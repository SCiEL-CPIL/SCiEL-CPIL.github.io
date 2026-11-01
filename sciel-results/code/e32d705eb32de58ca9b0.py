import time
import numpy as np
import random
import itertools

# ---- module-level precomputed structure for the (3,8,1) case ----
_N = 8
_Q = 3
_TOTAL = _Q ** _N            # 6561
_BS = 1 + 2 * _N             # 17
_WEIGHTS = [3 ** (_N - 1 - i) for i in range(_N)]

_BALLS = np.zeros((_TOTAL, _BS), dtype=np.int32)
for _w in range(_TOTAL):
    _BALLS[_w, 0] = _w
    _idx = 1
    for _pos in range(_N):
        _wt = _WEIGHTS[_pos]
        _dg = (_w // _wt) % _Q
        for _nd in range(_Q):
            if _nd == _dg:
                continue
            _BALLS[_w, _idx] = _w + (_nd - _dg) * _wt
            _idx += 1


def _int_to_base3(x, n):
    digits = [0] * n
    for i in range(n - 1, -1, -1):
        digits[i] = x % 3
        x //= 3
    return digits


def _ball_min(cov, c):
    return int(np.min(cov[_BALLS[c]]))


def _redundancy_removal(code_set, cov, deadline):
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        for c in list(code_set):
            if _ball_min(cov, c) >= 2:
                cov[_BALLS[c]] -= 1
                code_set.discard(c)
                changed = True
    return code_set, cov


def _plain_greedy(deadline):
    remaining = np.ones(_TOTAL, dtype=bool)
    cov = np.zeros(_TOTAL, dtype=np.int32)
    code_set = set()
    while remaining.any():
        if time.time() > deadline:
            break
        counts = remaining[_BALLS].sum(axis=1)
        best = int(np.argmax(counts))
        if counts[best] == 0:
            break
        code_set.add(best)
        remaining[_BALLS[best]] = False
        cov[_BALLS[best]] += 1
    return code_set, cov


def _dual_greedy(deadline, seed):
    remaining = np.ones(_TOTAL, dtype=bool)
    cov = np.zeros(_TOTAL, dtype=np.int32)
    code_set = set()
    price = np.full(_TOTAL, 1.0 / _BS, dtype=np.float64)
    while remaining.any():
        if time.time() > deadline:
            break
        scores = (price[_BALLS] * remaining[_BALLS].astype(np.float64)).sum(axis=1)
        best = int(np.argmax(scores))
        if scores[best] <= 0:
            break
        code_set.add(best)
        remaining[_BALLS[best]] = False
        cov[_BALLS[best]] += 1
    return code_set, cov


def _random_greedy(deadline, seed):
    rng = random.Random(seed)
    remaining = np.ones(_TOTAL, dtype=bool)
    cov = np.zeros(_TOTAL, dtype=np.int32)
    code_set = set()
    while remaining.any():
        if time.time() > deadline:
            break
        counts = remaining[_BALLS].sum(axis=1)
        nz = counts > 0
        if not nz.any():
            break
        idx = np.where(nz)[0]
        c = counts[idx]
        k = min(8, len(idx))
        top = idx[np.argsort(c)[-k:]]
        best = int(rng.choice(top))
        code_set.add(best)
        remaining[_BALLS[best]] = False
        cov[_BALLS[best]] += 1
    return code_set, cov


def _local_search(code_set, cov, deadline):
    improved = True
    it = 0
    max_iter = 60
    while improved and time.time() < deadline and it < max_iter:
        improved = False
        it += 1
        c_list = list(code_set)
        random.shuffle(c_list)

        # ---- 1-swap ----
        for c in c_list:
            if time.time() > deadline:
                break
            if c not in code_set:
                continue
            b = _BALLS[c]
            crit = b[cov[b] == 1]
            if len(crit) == 0:
                cov[b] -= 1
                code_set.discard(c)
                improved = True
                continue
            if len(crit) > 3:
                continue
            cand = set(_BALLS[crit[0]])
            for w in crit[1:]:
                cand &= set(_BALLS[w])
            for d in cand:
                if d in code_set:
                    continue
                ok = True
                db = _BALLS[d]
                for w in crit:
                    if w not in set(db):
                        ok = False
                        break
                if ok:
                    cov[b] -= 1
                    cov[db] += 1
                    code_set.discard(c)
                    code_set.add(d)
                    improved = True
                    break

        # ---- 2-opt: remove c1,c2, add single d ----
        if not improved and time.time() < deadline:
            c_list = list(code_set)
            random.shuffle(c_list)
            for i in range(min(len(c_list), 25)):
                if time.time() > deadline:
                    break
                c1 = c_list[i]
                if c1 not in code_set:
                    continue
                b1 = _BALLS[c1]
                crit1 = b1[cov[b1] == 1]
                if len(crit1) > 2:
                    continue
                for j in range(i + 1, min(len(c_list), i + 25)):
                    if time.time() > deadline:
                        break
                    c2 = c_list[j]
                    if c2 not in code_set:
                        continue
                    b2 = _BALLS[c2]
                    union = np.unique(np.concatenate([b1, b2]))
                    temp = cov[union].copy()
                    m1 = np.isin(union, b1)
                    m2 = np.isin(union, b2)
                    temp[m1] -= 1
                    temp[m2] -= 1
                    crit_all = union[temp <= 0]
                    if len(crit_all) > 4:
                        continue
                    if len(crit_all) == 0:
                        cov[b1] -= 1
                        cov[b2] -= 1
                        code_set.discard(c1)
                        code_set.discard(c2)
                        improved = True
                        break
                    cand = set(_BALLS[crit_all[0]])
                    for w in crit_all[1:]:
                        cand &= set(_BALLS[w])
                    for d in cand:
                        if d in code_set:
                            continue
                        ok = True
                        db = _BALLS[d]
                        for w in crit_all:
                            if w not in set(db):
                                ok = False
                                break
                        if ok:
                            cov[b1] -= 1
                            cov[b2] -= 1
                            cov[db] += 1
                            code_set.discard(c1)
                            code_set.discard(c2)
                            code_set.add(d)
                            improved = True
                            break
                    if improved:
                        break

        # ---- 2-swap: remove c1,c2, add d1,d2 ----
        if not improved and time.time() < deadline:
            c_list = list(code_set)
            random.shuffle(c_list)
            for i in range(min(len(c_list), 15)):
                if time.time() > deadline:
                    break
                c1 = c_list[i]
                if c1 not in code_set:
                    continue
                b1 = _BALLS[c1]
                crit1 = b1[cov[b1] == 1]
                if len(crit1) > 2:
                    continue
                for j in range(i + 1, min(len(c_list), i + 15)):
                    if time.time() > deadline:
                        break
                    c2 = c_list[j]
                    if c2 not in code_set:
                        continue
                    b2 = _BALLS[c2]
                    union = np.unique(np.concatenate([b1, b2]))
                    temp = cov[union].copy()
                    m1 = np.isin(union, b1)
                    m2 = np.isin(union, b2)
                    temp[m1] -= 1
                    temp[m2] -= 1
                    crit_all = union[temp <= 0]
                    if len(crit_all) > 4:
                        continue
                    if len(crit_all) == 0:
                        cov[b1] -= 1
                        cov[b2] -= 1
                        code_set.discard(c1)
                        code_set.discard(c2)
                        improved = True
                        break
                    cand = set(_BALLS[crit_all[0]])
                    for w in crit_all[1:]:
                        cand &= set(_BALLS[w])
                    cand_list = list(cand)
                    for d1 in cand_list[:12]:
                        if time.time() > deadline:
                            break
                        if d1 in code_set:
                            continue
                        db1 = _BALLS[d1]
                        rem = [w for w in crit_all if w not in set(db1)]
                        if not rem:
                            cov[b1] -= 1
                            cov[b2] -= 1
                            cov[db1] += 1
                            code_set.discard(c1)
                            code_set.discard(c2)
                            code_set.add(d1)
                            improved = True
                            break
                        if len(rem) > 3:
                            continue
                        cand2 = set(_BALLS[rem[0]])
                        for w in rem[1:]:
                            cand2 &= set(_BALLS[w])
                        for d2 in cand2:
                            if d2 in code_set or d2 == d1:
                                continue
                            ok = True
                            db2 = _BALLS[d2]
                            for w in rem:
                                if w not in set(db2):
                                    ok = False
                                    break
                            if ok:
                                cov[b1] -= 1
                                cov[b2] -= 1
                                cov[db1] += 1
                                cov[db2] += 1
                                code_set.discard(c1)
                                code_set.discard(c2)
                                code_set.add(d1)
                                code_set.add(d2)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break

    return code_set, cov


def _cover_is_valid(code_set):
    cov = np.zeros(_TOTAL, dtype=np.int32)
    for c in code_set:
        cov[_BALLS[c]] += 1
    return bool(np.all(cov > 0))


def _construct_ternary_8_1(deadline):
    search_deadline = deadline - 4.0

    inits = [
        ("dual", _dual_greedy, 101),
        ("plain", _plain_greedy, None),
        ("rand", _random_greedy, 202),
    ]

    best_set = None
    best_size = _TOTAL
    best_cov = None

    for name, fn, seed in inits:
        if time.time() > search_deadline:
            break
        t_init = min(time.time() + 25.0, search_deadline - 15.0)
        code_set, cov = fn(t_init, seed)
        code_set, cov = _redundancy_removal(code_set, cov, search_deadline - 8.0)
        t_ls = min(time.time() + 40.0, search_deadline - 6.0)
        code_set, cov = _local_search(code_set, cov, t_ls)
        code_set, cov = _redundancy_removal(code_set, cov, search_deadline - 4.0)

        if not _cover_is_valid(code_set):
            cov = np.zeros(_TOTAL, dtype=np.int32)
            for c in code_set:
                cov[_BALLS[c]] += 1
            for w in np.where(cov == 0)[0]:
                code_set.add(w)
                cov[_BALLS[w]] += 1
            code_set, cov = _redundancy_removal(code_set, cov, deadline - 2.0)

        sz = len(code_set)
        if sz < best_size:
            best_size = sz
            best_set = set(code_set)
            best_cov = cov.copy()

    if best_set is not None and time.time() < deadline - 3.0:
        cov = best_cov if best_cov is not None else np.zeros(_TOTAL, dtype=np.int32)
        if best_cov is None:
            for c in best_set:
                cov[_BALLS[c]] += 1
        best_set, cov = _local_search(best_set, cov, deadline - 2.0)
        best_set, cov = _redundancy_removal(best_set, cov, deadline - 1.0)

    if best_set is None:
        best_set = set(range(_TOTAL))

    cov = np.zeros(_TOTAL, dtype=np.int32)
    for c in best_set:
        cov[_BALLS[c]] += 1
    if not np.all(cov > 0):
        for w in np.where(cov == 0)[0]:
            best_set.add(w)

    return [_int_to_base3(c, _N) for c in best_set]


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)

    return _general_greedy(q, n, radius, deadline - start_time)


def _general_greedy(q, n, radius, time_limit):
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    codewords = []

    def int_to_word(x):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = x % q
            x //= q
        return w

    def word_to_int(w):
        res = 0
        for d in w:
            res = res * q + d
        return res

    start = time.time()
    while covered.sum() < total:
        if time.time() - start > time_limit - 5:
            break
        best_c = -1
        best_count = -1
        for c in range(total):
            if covered[c]:
                continue
            w = int_to_word(c)
            count = 1
            for j in range(n):
                for v in range(q):
                    if v == w[j]:
                        continue
                    w2 = w[:]
                    w2[j] = v
                    idx = word_to_int(w2)
                    if not covered[idx]:
                        count += 1
            if count > best_count:
                best_count = count
                best_c = c
        if best_c == -1:
            break
        codewords.append(best_c)
        w = int_to_word(best_c)
        covered[best_c] = True
        for j in range(n):
            for v in range(q):
                if v == w[j]:
                    continue
                w2 = w[:]
                w2[j] = v
                covered[word_to_int(w2)] = True

    return [int_to_word(c) for c in codewords]
