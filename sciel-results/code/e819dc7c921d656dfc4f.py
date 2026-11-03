"""Greedy set-covering with bitset uncovered tracker, sampled candidates and swaps."""
import time
import random
import numpy as np

_DEADLINE = 112.0
_TIME0 = time.perf_counter()


def _bits_to_ints(bits, n, q):
    out = np.zeros(n, dtype=np.int64)
    for d in range(n):
        v = int(bits[d])
        out[d] = v % q
        v //= q
    return out


def _ints_to_bits(w, q):
    return np.fromiter((int(v) * q ** d for d, v in enumerate(w)), dtype=np.int64, count=len(w))


def _neighborhood(w, q, n):
    """Yield (word, ball_bits) for w and each single-position variant."""
    base = _ints_to_bits(w, q)
    yield list(w), base
    for d in range(n):
        cur = w[d]
        for alt in range(q):
            if alt == cur:
                continue
            w2 = list(w)
            w2[d] = alt
            yield w2, base + (alt - cur) * q ** d


def construct(q, n, radius):
    global _TIME0
    _TIME0 = time.perf_counter()
    deadline = _TIME0 + _DEADLINE

    # --- Fast path: perfect ternary tetracode direct sum (radius 1, n>=4) ---
    if q == 3 and radius == 1 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode
                for tail in itertools.product(range(3), repeat=n - 4)]

    # --- General greedy set-covering ---
    full_mask = (1 << (q * n)) - 1
    rng = random.Random(12345)
    best = None
    best_len = 1 << 30

    max_restarts = 40
    for _r in range(max_restarts):
        if time.perf_counter() > deadline:
            break

        uncovered = np.full(q * n, 1, dtype=np.int64)
        uncovered[0] = 0
        codewords = [[0] * n]
        total = 1

        while total < q * n:
            if time.perf_counter() > deadline:
                break
            if best is not None and len(codewords) >= best_len:
                break

            # Sample candidate words from uncovered words' neighborhoods.
            nz = np.nonzero(uncovered)[0]
            if len(nz) == 0:
                break
            k = min(len(nz), 64)
            samp = np.array([nz[rng.randrange(len(nz))] for _ in range(k)])

            chosen = None
            chosen_ball = None
            best_gain = -1
            for s in samp:
                w = _bits_to_ints(s, n, q)
                for cand, ball in _neighborhood(w, q, n):
                    gain = int(np.sum(uncovered & ball))
                    if gain > best_gain:
                        best_gain = gain
                        chosen = cand
                        chosen_ball = ball
            if chosen is None:
                break

            uncovered &= ~chosen_ball
            codewords.append(chosen)
            total += best_gain

        if len(codewords) < best_len:
            best_len = len(codewords)
            best = [list(c) for c in codewords]

        # Swap-improvement pass (limited candidates) on the current solution.
        if best is None:
            continue
        improved = True
        while improved and time.perf_counter() < deadline:
            improved = False
            for i in range(len(codewords)):
                if time.perf_counter() > deadline:
                    break
                # Coverage if codeword i removed.
                cover_wo_i = np.zeros(q * n, dtype=np.int64)
                for j, cw in enumerate(codewords):
                    if j == i:
                        continue
                    cover_wo_i |= _ints_to_bits(cw, q)
                missing = full_mask & ~cover_wo_i
                if missing.sum() == 0:
                    codewords.pop(i)
                    improved = True
                    break
                # Try to find one replacement word covering all missing.
                mpos = np.nonzero(missing)[0]
                found = False
                for s in mpos[:64]:
                    w = _bits_to_ints(s, n, q)
                    for cand, ball in _neighborhood(w, q, n):
                        if np.all((missing & ball) == missing):
                            codewords[i] = cand
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

        if len(codewords) < best_len:
            best_len = len(codewords)
            best = [list(c) for c in codewords]

    if best is None:
        # Fallback: brute-force ball around 000..0 (valid only if it covers all).
        return [[0] * n]
    return best
