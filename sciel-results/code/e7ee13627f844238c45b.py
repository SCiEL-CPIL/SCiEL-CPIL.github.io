"""Linear covering codes over GF(3) + greedy reduction for K_q(n, radius)."""
import time
import itertools
import numpy as np

_DEADLINE = 150.0


def _gen_matrix(q, n, k, seed):
    """Random k x n generator matrix over GF(q), rows made linearly independent."""
    rng = np.random.default_rng(seed)
    M = rng.integers(0, q, size=(k, n)).astype(np.int64)
    rank = 0
    for r in range(k):
        for c in range(n):
            if M[r, c] != 0:
                break
        else:
            # zero row: fill randomly
            M[r, :] = rng.integers(0, q, size=n)
            for c in range(n):
                if M[r, c] != 0:
                    break
        # eliminate earlier pivots in this column
        for r2 in range(r):
            if M[r2, c] != 0:
                inv = _inv(M[r2, c], q)
                M[r] = (M[r] - inv * M[r2]) % q
                M[r, c] = 0
        # eliminate this pivot from later rows
        for r2 in range(r + 1, k):
            if M[r2, c] != 0:
                inv = _inv(M[r, c], q)
                M[r2] = (M[r2] - inv * M[r]) % q
        rank += 1
    return M


def _inv(a, q):
    for x in range(1, q):
        if (a * x) % q == 1:
            return x
    return 0


def _encode(M, q, n, k):
    """Return all 3**k codewords as an (3**k, n) int array."""
    total = q ** k
    words = np.zeros((total, n), dtype=np.int64)
    for i in range(total):
        # base-q representation of i gives the message
        msg = []
        t = i
        for _ in range(k):
            msg.append(t % q)
            t //= q
        msg = np.array(msg[::-1], dtype=np.int64)  # most significant first
        words[i] = (msg @ M) % q
    return words


def _all_words(q, n):
    total = q ** n
    words = np.zeros((total, n), dtype=np.int64)
    for i in range(total):
        t = i
        for c in range(n - 1, -1, -1):
            words[i, c] = t % q
            t //= q
    return words


def _covered_mask(codewords, q, n, radius):
    """Boolean array over all q**n words: True if within radius of some codeword."""
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    all_w = _all_words(q, n)
    # For each codeword, mark words within radius.
    # radius small (1): for each codeword mark itself + n*(q-1) neighbors.
    if radius == 1:
        for cw in codewords:
            base = int(cw[0])
            for c in range(1, n):
                base = base * q + int(cw[c])
            covered[base] = True
            for pos in range(n):
                # compute index offset for changing position pos
                # index = sum cw[j]*q**(n-1-j); changing pos by delta adds delta*q**(n-1-pos)
                offset = q ** (n - 1 - pos)
                cur = int(cw[pos])
                for v in range(q):
                    if v == cur:
                        continue
                    idx = base + (v - cur) * offset
                    covered[idx] = True
    else:
        # generic: compute distances (slower)
        for cw in codewords:
            d = np.sum(all_w != cw, axis=1)
            covered |= (d <= radius)
    return covered


def _greedy_reduce(codewords, q, n, radius, time_budget):
    """Remove redundant codewords greedily. Returns reduced list."""
    if len(codewords) < 2:
        return list(codewords)
    t0 = time.time()
    cw = [list(map(int, c)) for c in codewords]
    # index each codeword
    def idx_of(c):
        v = 0
        for x in c:
            v = v * q + x
        return v
    idx_map = {idx_of(c): c for c in cw}
    covered = _covered_mask(cw, q, n, radius)
    # coverage count per codeword: number of words it covers (for radius 1, count = 1 + n*(q-1) but with overlaps)
    # We'll compute, for each codeword, how many words would become uncovered if removed.
    # Efficient: maintain count of how many codewords cover each word.
    # Build cover_count array.
    def build_counts(cw_list):
        counts = np.zeros(q ** n, dtype=np.int32)
        if radius == 1:
            for c in cw_list:
                base = idx_of(c)
                counts[base] += 1
                for pos in range(n):
                    offset = q ** (n - 1 - pos)
                    cur = c[pos]
                    for v in range(q):
                        if v == cur:
                            continue
                        counts[base + (v - cur) * offset] += 1
        else:
            all_w = _all_words(q, n)
            for c in cw_list:
                d = np.sum(all_w != c, axis=1)
                counts[d <= radius] += 1
        return counts

    counts = build_counts(cw)
    improved = True
    while improved and (time.time() - t0) < time_budget:
        improved = False
        # find a codeword whose removal keeps all covered words covered
        # i.e., for every word it covers, count[word] >= 2
        for i, c in enumerate(cw):
            base = idx_of(c)
            if counts[base] < 2:
                continue
            ok = True
            if radius == 1:
                for pos in range(n):
                    offset = q ** (n - 1 - pos)
                    cur = c[pos]
                    for v in range(q):
                        if v == cur:
                            continue
                        if counts[base + (v - cur) * offset] < 2:
                            ok = False
                            break
                    if not ok:
                        break
            else:
                all_w = _all_words(q, n)
                d = np.sum(all_w != c, axis=1)
                if np.any(counts[d <= radius] < 2):
                    ok = False
            if ok:
                # remove it
                for pos in range(n):
                    offset = q ** (n - 1 - pos)
                    cur = c[pos]
                    counts[base] -= 1
                    for v in range(q):
                        if v == cur:
                            continue
                        counts[base + (v - cur) * offset] -= 1
                cw.pop(i)
                improved = True
                break
    return cw


def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + _DEADLINE - 5.0  # safety margin

    # k such that q**k is manageable. Start with k = n-1 (like direct sum) but explore other k.
    # Direct sum gives k = 4 for n=8 (9*3**4 = 729? no). Actually direct sum = 9 * 3**(n-4) = 3**2 * 3**(n-4) = 3**n / 3**(n-4-2)...
    # For n=8: 9 * 3**4 = 9*81 = 729? But task says 486. Let me recompute: 3**(n-4)=3**4=81, 9*81=729. Hmm but best known 486.
    # The seed returns 9*3**(n-4). For n=8 that's 729, not 486. So seed score 1.0 corresponds to 729? 
    # Regardless, we minimize codeword count.
    # We'll try linear codes with various k.

    best = None
    best_len = None

    # Candidate k values: from 1 up to n
    # For each k, generate a few random generator matrices, encode, greedy-reduce.
    # Budget: spend most time on the most promising k.

    # Direct-sum baseline for reference (guaranteed valid)
    def direct_sum():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n < 4:
            # fall back
            return [list(t) for t in itertools.product(range(q), repeat=n)]
        out = []
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=n - 4):
                out.append(h + list(tail))
        return out

    baseline = direct_sum()
    best = baseline
    best_len = len(baseline)

    # Try linear codes
    max_k = min(n, 6)  # keep 3**k <= 729 for speed
    for k in range(1, max_k + 1):
        if time.time() > deadline:
            break
        # number of random matrices to try for this k
        if q ** k > 2000:
            continue
        n_trials = 3
        for trial in range(n_trials):
            if time.time() > deadline:
                break
            seed = 1000 * k + trial
            M = _gen_matrix(q, n, k, seed)
            words = _encode(M, q, n, k)
            # greedy reduce
            reduced = _greedy_reduce(words, q, n, radius, time_budget=20.0)
            if len(reduced) < best_len:
                # verify coverage
                cov = _covered_mask(reduced, q, n, radius)
                if cov.all():
                    best = reduced
                    best_len = len(reduced)

    # Final safety: ensure best is valid
    if best is None:
        best = baseline
    cov = _covered_mask(best, q, n, radius)
    if not cov.all():
        best = baseline
    return [[int(x) for x in c] for c in best]
