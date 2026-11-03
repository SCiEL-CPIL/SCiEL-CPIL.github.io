"""Ternary covering code K_3(8,1): build a 243-word covering via the 27-word
length-6 code plus free coordinates, then prune with a greedy set cover."""
import time
import itertools
import numpy as np


def _neighbors(w, q, n, radius):
    """Yield all words within Hamming distance <= radius of w."""
    if radius <= 0:
        yield tuple(w)
        return
    for i in range(1, radius + 1):
        for pos in itertools.combinations(range(n), i):
            for vals in itertools.product(range(q), repeat=i):
                v = list(w)
                for p, val in zip(pos, vals):
                    v[p] = val
                yield tuple(v)


def _base_243():
    """243-word covering of F_3^8 with radius 1 (cosets of a 27-word code C2)."""
    a = np.array([1, 2, 1, 2, 1, 2])
    b = np.array([0, 1, 2, 1, 2, 1])
    c = np.array([0, 0, 0, 1, 2, 1])
    C2 = []
    for x, y, z in itertools.product(range(3), repeat=3):
        C2.append((x * a + y * b + z * c) % 3)
    out = []
    for v in C2:
        for t in itertools.product(range(3), repeat=2):
            out.append(list(v) + list(t))
    return out


def construct(q, n, radius):
    start = time.time()
    deadline = start + 125.0  # safety margin vs 150s kill

    if n <= 0:
        return [[]]
    if n == 1:
        return [[i] for i in range(q)]

    # Base covering (valid for q=3, n=8, radius=1).
    base = _base_243()

    # Verify base is a valid covering (cheap-ish); if not, fall back to seed.
    def is_covering(codes):
        # For radius 1, check every word is within 1 of some code.
        # Use: a word is covered iff it equals a code, or differs in exactly one
        # position. Represent covered set via codes plus their 1-neighbors.
        # Efficient: build a set of encoded covered words.
        covered = set()
        for cw in codes:
            cw = tuple(cw)
            covered.add(cw)
            for i in range(n):
                for val in range(q):
                    if val == cw[i]:
                        continue
                    lst = list(cw)
                    lst[i] = val
                    covered.add(tuple(lst))
        # now check all words
        for word in itertools.product(range(q), repeat=n):
            if word not in covered:
                return False
        return True

    if not is_covering(base):
        # Fallback: seed direct-sum construction (guaranteed valid).
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                     for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode
                for tail in itertools.product(range(3), repeat=n - 4)]

    best = base
    codes = base

    # Greedy set-cover pruning: repeatedly drop a code if the covering still holds.
    # Use a fast coverage test via the union of 1-neighborhoods.
    # Represent the universe implicitly; test coverage by checking each word.
    # To make this fast, precompute for current code set the covered set as a
    # numpy boolean array over the 3**n space (n=8 -> 6561).
    def covered_mask(code_list):
        mask = np.zeros(3 ** n, dtype=bool)
        # encode a word as base-3 integer
        pow3 = 3 ** np.arange(n)  # [1,3,9,...]
        for cw in code_list:
            base_idx = int(np.dot(cw, pow3))
            mask[base_idx] = True
            for i in range(n):
                p = pow3[i]
                for val in range(q):
                    if val == cw[i]:
                        continue
                    idx = base_idx - cw[i] * p + val * p
                    mask[idx] = True
        return mask

    def all_covered(mask):
        return mask.all()

    # Precompute pow3 and word enumeration indices.
    pow3 = 3 ** np.arange(n)
    # All words as indices: 0..3**n-1; but we need to map index->word for
    # building code lists. We'll keep code list as list of lists (indices via dot).

    # Greedy: try removing each code; if still covered, keep removed.
    # Order: random-ish by scanning. Re-scan until no improvement.
    # Each removal test costs O(|codes| * n * q) to rebuild mask; with 243 codes
    # and n=8 that's ~ 243*8*2 = 3888 ops -> fast. Do multiple passes.
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        cur = codes
        m = len(cur)
        # Try removing each index.
        for i in range(m):
            if time.time() > deadline - 5.0:
                break
            cand = cur[:i] + cur[i + 1:]
            mask = covered_mask(cand)
            if all_covered(mask):
                codes = cand
                improved = True
                break  # restart scan with smaller list

    # If we have spare time, try a second, more aggressive pass with random order.
    if time.time() < deadline - 10.0:
        import random
        rng = random.Random(12345)
        improved2 = True
        while improved2 and time.time() < deadline - 5.0:
            improved2 = False
            cur = codes
            m = len(cur)
            order = list(range(m))
            rng.shuffle(order)
            for i in order:
                if time.time() > deadline - 5.0:
                    break
                cand = cur[:i] + cur[i + 1:]
                mask = covered_mask(cand)
                if all_covered(mask):
                    codes = cand
                    improved2 = True
                    break

    best = codes
    return best
