"""SA-based ternary covering code K_3(8,1); EVOLVE-BLOCK rewritten."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np


def _neighbors_radius1(q, n):
    total = q ** n
    ncols = 1 + n * (q - 1)
    neighbors = np.zeros((total, ncols), dtype=np.int32)
    pow_q = [q ** (n - 1 - i) for i in range(n)]
    digits = [0] * n
    for idx in range(total):
        t = idx
        for i in range(n - 1, -1, -1):
            digits[i] = t % q
            t //= q
        col = 0
        neighbors[idx, col] = idx
        col += 1
        for i in range(n):
            p = pow_q[i]
            base = idx - digits[i] * p
            for s in range(q):
                if s == digits[i]:
                    continue
                neighbors[idx, col] = base + s * p
                col += 1
    return neighbors


def _neighbors_generic(q, n, radius):
    total = q ** n
    out = []
    digits = [0] * n
    for idx in range(total):
        t = idx
        for i in range(n - 1, -1, -1):
            digits[i] = t % q
            t //= q
        found = []
        cur = list(digits)

        def rec(pos, changes):
            if changes > radius:
                return
            if pos == n:
                v = 0
                for i in range(n):
                    v = v * q + cur[i]
                found.append(v)
                return
            rec(pos + 1, changes)
            for s in range(q):
                if s != cur[pos]:
                    cur[pos] = s
                    rec(pos + 1, changes + 1)
                    cur[pos] = digits[pos]

        rec(0, 0)
        out.append(sorted(set(found)))
    maxc = max(len(x) for x in out)
    neighbors = np.zeros((total, maxc), dtype=np.int32)
    for i, lst in enumerate(out):
        neighbors[i, :len(lst)] = lst
    return neighbors


def _idx_to_word(idx, q, n):
    digits = [0] * n
    for i in range(n - 1, -1, -1):
        digits[i] = idx % q
        idx //= q
    return digits


def construct(q, n, radius):
    start = time.time()
    deadline = 140.0
    total = q ** n
    if radius == 1:
        neighbors = _neighbors_radius1(q, n)
    else:
        neighbors = _neighbors_generic(q, n, radius)

    np.random.seed(12345)
    random.seed(12345)

    # ---- Greedy vectorized initialization ----
    uncovered = np.ones(total, dtype=bool)
    in_code = np.zeros(total, dtype=bool)
    code_list = []
    pos = np.full(total, -1, dtype=np.int32)
    while True:
        newly = uncovered[neighbors].sum(axis=1)
        newly[in_code] = -1
        best = int(np.argmax(newly))
        if newly[best] <= 0:
            break
        in_code[best] = True
        pos[best] = len(code_list)
        code_list.append(best)
        uncovered[neighbors[best]] = False

    size = len(code_list)
    counts = np.zeros(total, dtype=np.int32)
    for c in code_list:
        counts[neighbors[c]] += 1
    uncovered_count = int((counts == 0).sum())

    INF = 10 ** 9
    best_size = size if uncovered_count == 0 else INF
    best_in_code = in_code.copy()

    A = 10.0   # penalty per uncovered point
    B = 1.0    # penalty per size change
    T0 = 6.0
    Tmin = 0.05
    cool_time = deadline - start

    def accept(dU, dS, T):
        dE = A * dU + B * dS
        if dE <= 0.0:
            return True
        if T <= 1e-9:
            return False
        return random.random() < math.exp(-dE / T)

    def do_remove(c):
        nb = neighbors[c]
        before_zero = int((counts[nb] == 0).sum())
        counts[nb] -= 1
        after_zero = int((counts[nb] == 0).sum())
        return after_zero - before_zero, nb

    step = 0
    while time.time() - start < deadline:
        elapsed = time.time() - start
        frac = min(1.0, elapsed / cool_time)
        T = T0 * (Tmin / T0) ** frac
        if T < Tmin:
            T = Tmin

        r = random.random()
        if r < 0.50 and size > 0:
            # ---- remove ----
            c = code_list[random.randrange(size)]
            dU, nb = do_remove(c)
            dS = -1
            if accept(dU, dS, T):
                in_code[c] = False
                i = pos[c]
                last = code_list[-1]
                code_list[i] = last
                pos[last] = i
                code_list.pop()
                pos[c] = -1
                size -= 1
                uncovered_count += dU
                if uncovered_count == 0 and size < best_size:
                    best_size = size
                    best_in_code = in_code.copy()
            else:
                counts[nb] += 1
        elif r < 0.90:
            # ---- swap (net zero size, main escape move) ----
            if size == 0:
                c = random.randrange(total)
                nb = neighbors[c]
                before_zero = int((counts[nb] == 0).sum())
                counts[nb] += 1
                after_zero = int((counts[nb] == 0).sum())
                dU = after_zero - before_zero
                dS = 1
                if accept(dU, dS, T):
                    in_code[c] = True
                    pos[c] = len(code_list)
                    code_list.append(c)
                    size += 1
                    uncovered_count += dU
                else:
                    counts[nb] -= 1
            else:
                c_rem = code_list[random.randrange(size)]
                c_add = random.randrange(total)
                while in_code[c_add]:
                    c_add = random.randrange(total)
                nb_rem = neighbors[c_rem]
                nb_add = neighbors[c_add]
                affected = np.unique(np.concatenate((nb_rem, nb_add)))
                before_zero = int((counts[affected] == 0).sum())
                counts[nb_rem] -= 1
                counts[nb_add] += 1
                after_zero = int((counts[affected] == 0).sum())
                dU = after_zero - before_zero
                dS = 0
                if accept(dU, dS, T):
                    in_code[c_rem] = False
                    in_code[c_add] = True
                    i = pos[c_rem]
                    last = code_list[-1]
                    code_list[i] = last
                    pos[last] = i
                    code_list.pop()
                    pos[c_rem] = -1
                    pos[c_add] = len(code_list)
                    code_list.append(c_add)
                    uncovered_count += dU
                    if uncovered_count == 0 and size < best_size:
                        best_size = size
                        best_in_code = in_code.copy()
                else:
                    counts[nb_rem] += 1
                    counts[nb_add] -= 1
        else:
            # ---- add (rare, only if it helps validity or tiny prob) ----
            c = random.randrange(total)
            while in_code[c]:
                c = random.randrange(total)
            nb = neighbors[c]
            before_zero = int((counts[nb] == 0).sum())
            counts[nb] += 1
            after_zero = int((counts[nb] == 0).sum())
            dU = after_zero - before_zero
            dS = 1
            ok = (dU < 0) or (random.random() < 0.02)
            if ok:
                in_code[c] = True
                pos[c] = len(code_list)
                code_list.append(c)
                size += 1
                uncovered_count += dU
            else:
                counts[nb] -= 1
        step += 1

    # ---- take best valid and polish ----
    if best_size < INF:
        work_in_code = best_in_code.copy()
        work_counts = np.zeros(total, dtype=np.int32)
        for i in range(total):
            if work_in_code[i]:
                work_counts[neighbors[i]] += 1
        work_size = int(work_in_code.sum())
    else:
        work_in_code = in_code.copy()
        work_counts = counts.copy()
        work_size = size

    # deterministic greedy removal polish (a few random-order passes)
    rng = random.Random(777)
    plist = [i for i in range(total) if work_in_code[i]]
    ppos = np.full(total, -1, dtype=np.int32)
    for i, c in enumerate(plist):
        ppos[c] = i
    for _ in range(4):
        if not plist:
            break
        rng.shuffle(plist)
        for c in list(plist):
            nb = neighbors[c]
            if (work_counts[nb] > 1).all():
                work_counts[nb] -= 1
                work_in_code[c] = False
                i = ppos[c]
                last = plist[-1]
                plist[i] = last
                ppos[last] = i
                plist.pop()
                ppos[c] = -1
                work_size -= 1

    result_idx = [i for i in range(total) if work_in_code[i]]
    if not result_idx:
        # ultimate fallback: trivial full cover
        result_idx = list(range(total))
    return [_idx_to_word(idx, q, n) for idx in result_idx]
# EVOLVE-BLOCK-END
