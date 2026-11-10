"""Ternary covering code K_3(7,1): greedy construction + local-optimization polish."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE = 150.0
_EPS = 0.6          # safety margin (seconds)
_RNG = random.Random(20240517)


def _all_words(q, n):
    # q**n words, each row a base-q vector, as an int32 matrix
    return np.array(list(_product(range(q), n)), dtype=np.int32)


def _product(iterable, n):
    if n == 0:
        yield ()
        return
    first, rest = iterable[:1], iterable[1:]
    for x in iterable:
        for tail in _product(rest, n - 1):
            yield (x,) + tail


def _encode(words, q, n):
    # base-q encode: word (v0..v_{n-1}) -> sum v_i * q^{n-1-i}
    pw = q ** np.arange(n - 1, -1, -1, dtype=np.int64)
    return (words.astype(np.int64) * pw).sum(axis=1)


def _decode(codes, q, n):
    out = np.empty((codes.size, n), dtype=np.int32)
    for i in range(n):
        out[:, i] = (codes // (q ** (n - 1 - i))) % q
    return out


def _covered_by(words, codes, q, n, radius):
    # boolean per word: within `radius` of some codeword
    covered = np.zeros(words.shape[0], dtype=bool)
    W = words.astype(np.int64)
    C = codes.astype(np.int64)
    for i in range(n):
        diff = np.abs(W[:, i:i+1] - C[:, i:i+1].T)
        covered |= (diff <= radius).any(axis=1)
        if covered.all():
            break
    return covered


def _build_greedy(q, n, radius, deadline):
    """Greedy set cover: repeatedly add the codeword covering the most uncovered words."""
    words = _all_words(q, n)
    m = words.shape[0]
    uncovered = np.ones(m, dtype=bool)
    codes = []

    while uncovered.any() and time.time() < deadline:
        # candidate center = one uncovered word (best-guess); pick the uncovered word
        # whose neighbourhood is largest among uncovered (here all equal) -> just first,
        # but we evaluate a few candidates for better coverage.
        cand_idx = np.nonzero(uncovered)[0]
        # sample up to 32 candidate centers from uncovered words
        if cand_idx.size > 32:
            cand_idx = _RNG.sample(list(cand_idx), 32)
        best_gain = -1
        best_code = None
        for ci in cand_idx:
            c = words[ci]
            mask = _covered_by(words, c.reshape(1, -1), q, n, radius)
            gain = int((mask & uncovered).sum())
            if gain > best_gain:
                best_gain = gain
                best_code = c.copy()
        if best_code is None:
            break
        codes.append(best_code)
        # update uncovered
        mask = _covered_by(words, np.array(best_code).reshape(1, -1), q, n, radius)
        uncovered &= ~mask
    return codes


def _remove_redundant(codes, words, q, n, radius, deadline):
    """Remove any codeword whose removal leaves the covering property intact."""
    codes = list(codes)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(codes) - 1, -1, -1):
            if time.time() > deadline:
                break
            trial = codes[:i] + codes[i+1:]
            if not trial:
                break
            if _covered_by(words, np.array(trial), q, n, radius).all():
                codes = trial
                changed = True
    return codes


def _swap_refine(codes, words, q, n, radius, deadline, max_rounds=4):
    """Try replacing a codeword with a better one; accept if coverage stays full
    and we can then drop another (net reduction)."""
    codes = list(codes)
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        # for a few codewords, try replacing with a random nearby word and see if
        # we can then remove a redundant one (net -1).
        order = list(range(len(codes)))
        _RNG.shuffle(order)
        for i in order[:40]:
            if time.time() > deadline:
                break
            base = codes[i]
            for _ in range(3):
                cand = base.copy()
                pos = _RNG.randrange(n)
                cand[pos] = _RNG.randrange(q)
                if any(np.array_equal(cand, c) for c in codes):
                    continue
                trial = codes[:i] + [cand] + codes[i+1:]
                if not _covered_by(words, np.array(trial), q, n, radius).all():
                    continue
                # try to drop a redundant codeword from the trial
                for j in range(len(trial) - 1, -1, -1):
                    if j == i:
                        continue
                    t2 = trial[:j] + trial[j+1:]
                    if _covered_by(words, np.array(t2), q, n, radius).all():
                        codes = t2
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return codes


def construct(q, n, radius):
    start = time.time()
    deadline = start + _DEADLINE - _EPS

    words = _all_words(q, n)

    # Phase 1: greedy construction
    codes = _build_greedy(q, n, radius, deadline)

    # Phase 2: redundant removal
    if time.time() < deadline:
        codes = _remove_redundant(codes, words, q, n, radius, deadline)

    # Phase 3: swap-based refinement
    if time.time() < deadline:
        codes = _swap_refine(codes, words, q, n, radius, deadline)

    # Final safety: ensure distinct + valid (drop duplicates, re-verify)
    seen = set()
    uniq = []
    for c in codes:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            uniq.append(list(c))
    codes = uniq

    # If somehow invalid (shouldn't be), fall back to a guaranteed valid direct sum.
    if not _covered_by(words, np.array(codes), q, n, radius).all():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                     for a in range(3) for b in range(3)]
        import itertools
        codes = [h + list(tail) for h in tetracode
                 for tail in itertools.product(range(3), repeat=n - 4)]

    return codes
# EVOLVE-BLOCK-END
