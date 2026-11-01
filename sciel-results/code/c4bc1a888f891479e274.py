import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # explicit wall-clock guard, safety margin before 150s

    # Specialised fast path for the public task: ternary covering code K_3(7,1).
    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline)

    # General fallback: greedy set cover + count-driven pruning.
    return _construct_general(q, n, radius, deadline)


def _construct_3_7_1(deadline):
    N = 3 ** 7  # 2187
    n = 7

    # Powers of 3 for fast index <-> digits conversion (little-endian digits).
    p3 = [1] * n
    for i in range(1, n):
        p3[i] = p3[i - 1] * 3

    # digits[i] = list of n ternary digits of index i.
    digits = np.empty((N, n), dtype=np.int8)
    for i in range(N):
        tmp = i
        for pos in range(n):
            digits[i, pos] = tmp % 3
            tmp //= 3

    # ball[i] = indices of all words within Hamming distance 1 of index i.
    # Size = 1 + n*(q-1) = 15.
    ball_size = 1 + n * 2
    ball = np.empty((N, ball_size), dtype=np.int32)
    for i in range(N):
        d = digits[i]
        ball[i, 0] = i
        k = 1
        for pos in range(n):
            orig = d[pos]
            for val in range(3):
                if val == orig:
                    continue
                ball[i, k] = i + (val - orig) * p3[pos]
                k += 1

    # ---- Vectorised local search (deletion + exact swap) ----
    def local_search(code, seed):
        rng = random.Random(seed)
        code_set = set(code)
        cnt = np.zeros(N, dtype=np.int32)
        for c in code:
            cnt[ball[c]] += 1

        best_code = list(code)
        best_size = len(code)

        def try_deletions():
            nonlocal best_code, best_size
            changed = False
            order = list(range(len(code)))
            rng.shuffle(order)
            for idx in order:
                if idx >= len(code):
                    continue
                c = code[idx]
                # Safe to delete iff every word in ball(c) is covered >= 2.
                if cnt[ball[c]].min() >= 2:
                    cnt[ball[c]] -= 1
                    del code[idx]
                    code_set.discard(c)
                    changed = True
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = list(code)
            return changed

        def try_swaps():
            nonlocal best_code, best_size
            for _ in range(200):
                if time.time() > deadline - 4.0:
                    return
                if not code:
                    return
                c = rng.choice(code)
                bc = ball[c]
                # Words that would become uncovered if c is removed.
                U = bc[cnt[bc] == 1]
                if len(U) == 0:
                    # c is redundant; delete it directly.
                    cnt[bc] -= 1
                    code.remove(c)
                    code_set.discard(c)
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = list(code)
                    continue
                if len(U) > 4:
                    # Too many words to cover with one codeword; skip.
                    continue
                # Candidate c' must lie in ball(u) for every u in U.
                cand = np.intersect1d(ball[int(U[0])], ball[int(U[1])]) if len(U) >= 2 else ball[int(U[0])]
                for j in range(2, len(U)):
                    cand = np.intersect1d(cand, ball[int(U[j])])
                if cand.size == 0:
                    continue
                for c_new in cand:
                    c_new = int(c_new)
                    if c_new in code_set:
                        continue
                    # Check c_new covers all of U (guaranteed by construction, but verify).
                    if not np.all(np.isin(U, ball[c_new])):
                        continue
                    # Apply swap.
                    cnt[bc] -= 1
                    cnt[ball[c_new]] += 1
                    code.remove(c)
                    code_set.discard(c)
                    code.append(c_new)
                    code_set.add(c_new)
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = list(code)
                    break

        no_improve = 0
        while time.time() < deadline - 4.0 and no_improve < 30:
            changed = try_deletions()
            if time.time() > deadline - 4.0:
                break
            sw = try_swaps()
            if changed:
                no_improve = 0
            else:
                no_improve += 1
            if not changed and no_improve >= 30:
                break
        return best_code

    # ---- Greedy initialisation (randomised, fast) ----
    def greedy_init(seed):
        rng = random.Random(seed)
        uncovered = np.ones(N, dtype=bool)
        selected = []
        selected_set = set()
        while uncovered.any():
            if time.time() > deadline - 15.0:
                break
            best_idx = -1
            best_count = -1
            # Random order for tie diversity; full scan over N=2187 is cheap.
            for i in rng.sample(range(N), N):
                if i in selected_set:
                    continue
                cnt = uncovered[ball[i]].sum()
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_idx < 0 or best_count == 0:
                break
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered[ball[best_idx]] = False
        return selected

    best = None
    best_size = N + 1
    # Try several seeds; keep the best code found.
    for s in range(0, 400):
        if time.time() > deadline - 20.0:
            break
        code = greedy_init(s)
        if not code:
            continue
        code = local_search(code, s)
        if len(code) < best_size:
            best_size = len(code)
            best = code
            if best_size <= 186:  # best known upper bound; stop early.
                break

    if best is None:
        # Absolute fallback: full space (always valid).
        return [list(itertools.product(range(3), repeat=7))]

    # Convert indices back to codeword lists.
    return [digits[int(c)].tolist() for c in best]


def _construct_general(q, n, radius, deadline):
    N = q ** n
    p = [1] * n
    for i in range(1, n):
        p[i] = p[i - 1] * q

    # Precompute ball (words within Hamming distance radius) for every index.
    # For small N this is fine; for large N we cap the work.
    def index_to_digits(i):
        d = [0] * n
        tmp = i
        for pos in range(n):
            d[pos] = tmp % q
            tmp //= q
        return d

    def sphere(idx):
        w = index_to_digits(idx)
        result = {idx}
        # all positions changed in up to 'radius' spots
        from itertools import combinations
        for r in range(1, radius + 1):
            for combo in combinations(range(n), r):
                # enumerate all q^r value assignments differing in those positions
                for vals in itertools.product(range(q), repeat=r):
                    nw = list(w)
                    ok = True
                    for j, pos in enumerate(combo):
                        if vals[j] == w[pos]:
                            ok = False
                            break
                        nw[pos] = vals[j]
                    if not ok:
                        continue
                    new_i = 0
                    tmp = nw
                    for pos in range(n):
                        new_i += tmp[pos] * p[pos]
                    result.add(new_i)
        return list(result)

    ball = np.array([sphere(i) for i in range(N)], dtype=np.int32)

    rng = random.Random(12345)
    uncovered = np.ones(N, dtype=bool)
    selected = []
    selected_set = set()
    while uncovered.any():
        if time.time() > deadline - 10.0:
            break
        best_idx = -1
        best_count = -1
        for i in rng.sample(range(N), N):
            if i in selected_set:
                continue
            cnt = uncovered[ball[i]].sum()
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_idx < 0 or best_count == 0:
            break
        selected.append(best_idx)
        selected_set.add(best_idx)
        uncovered[ball[best_idx]] = False

    # Simple count-driven deletion pruning.
    cnt = np.zeros(N, dtype=np.int32)
    for c in selected:
        cnt[ball[c]] += 1
    code = list(selected)
    code_set = set(code)
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        for idx in range(len(code) - 1, -1, -1):
            c = code[idx]
            if cnt[ball[c]].min() >= 2:
                cnt[ball[c]] -= 1
                del code[idx]
                code_set.discard(c)
                changed = True

    # Convert indices to codeword lists.
    out = []
    for c in code:
        d = [0] * n
        tmp = c
        for pos in range(n):
            d[pos] = tmp % q
            tmp //= q
        out.append(d)
    return out
