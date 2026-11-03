"""Ternary covering code K_3(7,1): greedy set-cover + time-bounded local search."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

_DEADLINE = 145.0  # safety margin vs 150s wall-clock kill


def _all_words(q, n):
    return np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int64)


def construct(q, n, radius):
    t0 = time.time()
    total = q ** n
    if total > 4_000_000:
        # Fallback: direct sum of ternary tetracode (only valid/useful for small q)
        if q == 3 and n >= 4:
            tet = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tet for tail in itertools.product(range(q), repeat=n - 4)]
        return [list(range(n))]

    # ---- Precompute balls and cover sets ----
    base = _all_words(q, n)
    idx_of = np.zeros(total, dtype=np.int64)
    for i, w in enumerate(base):
        idx_of[w[0] * q ** (n - 1) + w[1] * q ** (n - 2) + w[2] * q ** (n - 3) + w[3] * q ** (n - 4) +
                  w[4] * q ** (n - 5) + w[5] * q ** (n - 6) + w[6]] = i
    # faster index via base dot powers
    powers = q ** np.arange(n - 1, -1, -1, dtype=np.int64)
    idx_of = (base * powers).sum(axis=1)

    # balls: for each word, the set of words within Hamming distance <= radius
    # build via one-position flips (radius 1) ; general radius handled by repeated
    if radius == 1:
        ball = np.zeros((total, 1 + n * (q - 1)), dtype=np.int64)
        ball[:, 0] = idx_of
        k = 1
        for pos in range(n):
            for delta in range(1, q):
                flipped = base.copy()
                flipped[:, pos] = (base[:, pos] + delta) % q
                ball[:, k] = (flipped * powers).sum(axis=1)
                k += 1
        balls = ball
    else:
        # generic BFS up to radius (small spaces)
        balls = []
        for i in range(total):
            seen = {i}
            frontier = [i]
            for _ in range(radius):
                nxt = []
                for widx in frontier:
                    w = base[widx]
                    for pos in range(n):
                        for v in range(q):
                            if v == w[pos]:
                                continue
                            nw = w.copy()
                            nw[pos] = v
                            j = int((nw * powers).sum())
                            if j not in seen:
                                seen.add(j)
                                nxt.append(j)
                frontier = nxt
            balls.append(seen)
        maxb = max(len(b) for b in balls)
        balls = np.array([[idx_of[x] for x in b] + [0] * (maxb - len(b)) for b in balls], dtype=np.int64)

    # cover_set[i] = ball[i] minus ball[j] for all j != i (words uniquely covered by i)
    # = ball[i] \ union_{j!=i} ball[j]
    # Compute: a word w is in cover_set[i] iff w in ball[i] and i is the ONLY codeword-index whose ball contains w.
    # Equivalent: for each w, count how many i have w in ball[i]; if exactly 1 (that i), then w in cover_set[i].
    # Build cover sets via membership counts.
    # Use: for each i, cover_set[i] = [w in ball[i] : count_of_w == 1 and the unique i is i]
    # Efficient: for each i, the words in ball[i]; for each such w, we need count. 
    # Precompute count[w] = number of i with w in ball[i].
    count = np.zeros(total, dtype=np.int32)
    for i in range(total):
        count[balls[i]] += 1
    # cover_set[i]: words w in ball[i] with count[w]==1 (these are covered only by i)
    cover_sets = []
    for i in range(total):
        b = balls[i]
        cs = b[count[b] == 1]
        cover_sets.append(cs)
    maxcs = max(len(c) for c in cover_sets)
    cover_sets = np.array([[x for x in c] + [0] * (maxcs - len(c)) for c in cover_sets], dtype=np.int64)

    # ---- Greedy set cover ----
    cov = np.zeros(total, dtype=np.int32)
    chosen = []
    in_code = np.zeros(total, dtype=bool)

    def time_left():
        return _DEADLINE - (time.time() - t0)

    while time_left() > 1.0:
        gains = np.zeros(total, dtype=np.int64)
        for i in range(total):
            cs = cover_sets[i]
            if cs.size:
                gains[i] = (cov[cs] == 0).sum()
        # pick max gain
        best = int(np.argmax(gains))
        if gains[best] == 0:
            break
        chosen.append(best)
        in_code[best] = True
        cov[cover_sets[best]] += 1
        # also: the ball words that were newly covered are exactly cover_set[best] (unique ones)
        # but a word could be newly covered by best even if not unique to best? No: if it's in another
        # ball it's already covered. So newly-covered == cover_set[best]. Correct.

    # ---- Local search improvement (time-bounded) ----
    # Try to drop codewords that are redundant (their cover_set all already covered by others).
    # A codeword i is redundant if for every w in cover_set[i], some other chosen codeword covers w.
    # Since cover_set[i] words are covered ONLY by i among ALL words, if i is removed they become uncovered.
    # Wait: cover_set[i] = words covered only by i (among all 2187). So removing i always uncovers them.
    # So a codeword is redundant only if its cover_set is empty.
    # Thus we can only drop codewords with empty cover_set.
    # For swaps: replace codeword i with word j (not in code) such that j's cover_set covers the
    # words i uniquely covered, and j is not worse. Net: we need j to cover cover_set[i] AND we free
    # j's own unique words. This is complex; do simple 1-swap: remove i (if its unique words get
    # covered by the rest after we also add j).
    # Simpler effective improvement: iterate, for each chosen i, compute how many words i uniquely
    # covers given CURRENT code = cover_set[i] minus words also in other chosen balls. 
    # Actually unique-to-i within current code = ball[i] \ union_{k in chosen, k!=i} ball[k].
    # If that's 0, i is redundant -> remove it.
    while time_left() > 2.0:
        improved = False
        # redundancy removal
        for i in list(chosen):
            if time_left() < 2.0:
                break
            # unique coverage of i within current code
            others = np.zeros(total, dtype=bool)
            for k in chosen:
                if k != i:
                    others[balls[k]] = True
            unique_i = balls[i][~others[balls[i]]]
            if unique_i.size == 0:
                # remove i
                cov[cover_sets[i]] -= 1
                in_code[i] = False
                chosen.remove(i)
                improved = True
        if not improved:
            break

    # ---- Final: ensure validity; return as lists ----
    # Verify coverage over whole space
    if time_left() > 0.5:
        covered = np.zeros(total, dtype=bool)
        for i in chosen:
            covered[balls[i]] = True
        if not covered.all():
            # fill gaps greedily
            while time_left() > 0.5:
                g = (~covered)
                if not g.any():
                    break
                best = -1
                bestg = 0
                for i in range(total):
                    if in_code[i]:
                        continue
                    cs = cover_sets[i]
                    ng = (g[cs] & (cov[cs] == 0)).sum() if cs.size else 0
                    if ng > bestg:
                        bestg = ng
                        best = i
                if best < 0 or bestg == 0:
                    # fallback: add word covering most uncovered
                    for i in range(total):
                        if in_code[i]:
                            continue
                        ng = g[balls[i]].sum()
                        if ng > bestg:
                            bestg = ng
                            best = i
                    if best < 0 or bestg == 0:
                        break
                chosen.append(best)
                in_code[best] = True
                cov[cover_sets[best]] += 1
                covered[balls[best]] = True

    return [base[i].tolist() for i in chosen]
# EVOLVE-BLOCK-END
