import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline, start_time)

    # Fallbacks
    import itertools
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(c) for c in itertools.product(range(q), repeat=n)]


def _solve_ternary_8_1(deadline, start_time):
    n = 8
    q = 3
    N = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17

    weights = [3 ** (n - 1 - i) for i in range(n)]

    # Precompute each word's ball (17 neighbors) as an index array
    balls = np.zeros((N, ball_size), dtype=np.int32)
    for w in range(N):
        neighbors = [w]
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for nd in range(q):
                if nd == digit:
                    continue
                neighbors.append(w + (nd - digit) * weight)
        balls[w] = neighbors

    # Bit-parallel coverage: each word's ball as a bitset over 6561 words.
    # 6561 bits = 103 64-bit words.
    nbits = (N + 63) // 64
    ball_bits = np.zeros((N, nbits), dtype=np.uint64)
    for w in range(N):
        for nb in balls[w]:
            ball_bits[w, nb >> 6] |= (1 << (nb & 63))

    def idx_to_list(idx):
        digits = []
        t = idx
        for _ in range(n):
            digits.append(t % 3)
            t //= 3
        return digits[::-1]

    def tetracode_indices():
        res = []
        for a in range(3):
            for b in range(3):
                tc = [a, b, (a + b) % 3, (a + 2 * b) % 3]
                for tail in range(3 ** 4):
                    ds = []
                    t = tail
                    for _ in range(4):
                        ds.append(t % 3)
                        t //= 3
                    ds = ds[::-1]
                    word = tc + ds
                    idx = 0
                    for d in word:
                        idx = idx * 3 + d
                    res.append(idx)
        return res

    def make_cover_count(selected):
        cc = np.zeros(N, dtype=np.int32)
        for c in selected:
            cc[balls[c]] += 1
        return cc

    def prune(selected, cc):
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(selected):
                c = selected[i]
                if np.min(cc[balls[c]]) > 1:
                    cc[balls[c]] -= 1
                    selected.pop(i)
                    changed = True
                else:
                    i += 1
        return selected

    def greedy_fill(cc, selected, deadline):
        # Add words to cover everything; use bitsets for uncovered set.
        uncovered_bits = np.ones(nbits, dtype=np.uint64)
        # mask out last word's extra bits
        last_extra = (nbits * 64) - N
        if last_extra > 0:
            uncovered_bits[-1] &= ((1 << (64 - last_extra)) - 1)

        def popcount(x):
            return int(np.sum(np.unpackbits(x.astype(np.uint8), axis=1)))

        # maintain uncovered as list of indices for simplicity but use bits for speed of gain
        while True:
            if time.time() > deadline - 3.0:
                break
            if not np.any(uncovered_bits):
                break
            # gain for each candidate = number of uncovered words in its ball
            # compute via bitset AND
            gains = np.empty(N, dtype=np.int32)
            for w in range(N):
                a = ball_bits[w] & uncovered_bits
                gains[w] = int(np.sum(np.unpackbits(a.astype(np.uint8), axis=1)))
            gmax = int(gains.max())
            if gmax == 0:
                break
            cands = np.where(gains == gmax)[0]
            pick = int(cands[np.random.randint(len(cands))])
            selected.append(pick)
            cc[balls[pick]] += 1
            uncovered_bits &= ~ball_bits[pick]
        return selected

    def swap_local_search(selected, cc, deadline):
        improved = True
        iters = 0
        while improved and time.time() < deadline - 3.0:
            improved = False
            iters += 1
            if iters > 200000:
                break
            K = len(selected)
            if K == 0:
                break
            # Try random single swaps
            for _ in range(60):
                if time.time() > deadline - 3.0:
                    break
                i = np.random.randint(K)
                u = selected[i]
                v = np.random.randint(N)
                if v == u or v in selected:
                    continue
                # risk = ball(u) \ ball(v)
                risk_bits = ball_bits[u] & ~ball_bits[v]
                if not np.any(risk_bits):
                    # safe, always fine
                    pass
                else:
                    # need cc >= 2 on all risk words
                    # get risk word indices
                    rw = np.where(risk_bits != 0)[0]
                    # extract indices
                    risk_idx = []
                    for b in rw:
                        mask = risk_bits[b]
                        while mask:
                            lsb = mask & (-mask)
                            bit = (mask & (-mask)).bit_length() - 1
                            idx = (b << 6) + bit
                            if idx < N:
                                risk_idx.append(idx)
                            mask &= mask - 1
                    if risk_idx and np.min(cc[np.array(risk_idx, dtype=np.int32)]) < 2:
                        continue
                # perform swap
                bu = balls[u]
                bv = balls[v]
                cc[bu] -= 1
                cc[bv] += 1
                selected[i] = v
                before = len(selected)
                selected = prune(selected, cc)
                if len(selected) < before:
                    improved = True
                    break
                else:
                    # revert
                    cc[bv] -= 1
                    cc[bu] += 1
                    selected[i] = u
        return selected

    def two_for_one(selected, cc, deadline):
        improved = True
        iters = 0
        while improved and time.time() < deadline - 3.0:
            improved = False
            iters += 1
            if iters > 60000:
                break
            K = len(selected)
            if K < 2:
                break
            for _ in range(40):
                if time.time() > deadline - 3.0:
                    break
                i1 = np.random.randint(K)
                i2 = np.random.randint(K)
                if i1 == i2:
                    continue
                c1, c2 = selected[i1], selected[i2]
                # words uniquely covered by c1 or c2
                u1 = balls[c1][cc[balls[c1]] == 1]
                u2 = balls[c2][cc[balls[c2]] == 1]
                allu = np.unique(np.concatenate([u1, u2])) if len(u1) + len(u2) else np.array([], dtype=np.int32)
                if len(allu) == 0:
                    # can drop one
                    m1 = np.min(cc[balls[c1]])
                    m2 = np.min(cc[balls[c2]])
                    if m1 >= m2:
                        cc[balls[c1]] -= 1
                        selected.pop(i1)
                    else:
                        cc[balls[c2]] -= 1
                        selected.pop(i2)
                    selected = prune(selected, cc)
                    improved = True
                    break
                # find v covering allu
                # v valid if ball_bits[v] has all allu bits set
                # build allu bitset
                ab = np.zeros(nbits, dtype=np.uint64)
                for w in allu:
                    ab[w >> 6] |= (1 << (w & 63))
                # valid v: (ball_bits[v] & ab) == ab
                ok = True
                valid = []
                for v in range(N):
                    if np.array_equal(ball_bits[v] & ab, ab):
                        valid.append(v)
                if not valid:
                    continue
                for v in valid[:8]:
                    if time.time() > deadline - 3.0:
                        break
                    bv = balls[v]
                    # risk = (ball(c1) ∪ ball(c2)) \ ball(v)
                    combined = ball_bits[c1] | ball_bits[c2]
                    risk = combined & ~ball_bits[v]
                    if np.any(risk):
                        rw = np.where(risk != 0)[0]
                        risk_idx = []
                        for b in rw:
                            mask = risk[b]
                            while mask:
                                bit = (mask & (-mask)).bit_length() - 1
                                idx = (b << 6) + bit
                                if idx < N:
                                    risk_idx.append(idx)
                                mask &= mask - 1
                        if risk_idx:
                            cc_sim = cc.copy()
                            cc_sim[balls[c1]] -= 1
                            cc_sim[balls[c2]] -= 1
                            if np.min(cc_sim[np.array(risk_idx, dtype=np.int32)]) < 2:
                                continue
                    cc[balls[c1]] -= 1
                    cc[balls[c2]] -= 1
                    cc[bv] += 1
                    hi, lo = max(i1, i2), min(i1, i2)
                    selected.pop(hi)
                    selected.pop(lo)
                    selected.append(v)
                    selected = prune(selected, cc)
                    improved = True
                    break
                if improved:
                    break
        return selected

    best_size = N
    best_set = None

    while time.time() < deadline - 12.0:
        # Initialization
        if random.random() < 0.6:
            selected = tetracode_indices()
        else:
            # random subset
            selected = [int(x) for x in np.random.permutation(N)[:600]]

        cc = make_cover_count(selected)
        if np.any(cc == 0):
            selected = greedy_fill(cc, selected, deadline)
            if time.time() > deadline - 12.0:
                break
        selected = prune(selected, cc)
        selected = swap_local_search(selected, cc, deadline)
        selected = two_for_one(selected, cc, deadline)
        selected = prune(selected, cc)

        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            if best_size <= 486:
                break

    # Final polish on best
    if best_set is not None and time.time() < deadline - 4.0:
        cc = make_cover_count(best_set)
        best_set = prune(best_set, cc)
        best_set = swap_local_search(best_set, cc, deadline)
        best_set = two_for_one(best_set, cc, deadline)
        best_set = prune(best_set, cc)
        if len(best_set) < best_size:
            best_size = len(best_set)
            best_set = best_set.copy()

    if best_set is None:
        best_set = list(range(N))

    return [idx_to_list(w) for w in best_set]
