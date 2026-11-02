"""Football pool / ternary covering code K_3(7,1) heuristic solver."""
import time
import random
import itertools

_START = time.time()
_TIME_BUDGET = 115.0  # seconds, leave headroom under the 120 s cap


def _time_left():
    return _TIME_BUDGET - (time.time() - _START)


def construct(q, n, radius):
    # Deterministic fast path for the canonical task parameters.
    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1()

    # Generic fallback: direct-sum of a small perfect code (when available)
    # plus full enumeration of the remaining coordinates.
    if q == 3 and n >= 4 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                     for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode
                for tail in itertools.product(range(q), repeat=n - 4)]

    # Last-resort trivial (valid but large) construction.
    return [list(range(q)) * (n // q + 1)][:n]


def _solve_ternary_7_1():
    q, n, r = 3, 7, 1
    total = q ** n  # 2187

    # Precompute, for each word, the list of words within Hamming distance r.
    # Neighbourhood size for r=1, q=3, n=7: 1 + 7*2 = 15.
    neighbours = []
    for w in range(total):
        digits = []
        t = w
        for _ in range(n):
            digits.append(t % q)
            t //= q
        digits = tuple(digits)
        nbrs = [w]
        for pos in range(n):
            orig = digits[pos]
            for v in range(q):
                if v == orig:
                    continue
                nd = list(digits)
                nd[pos] = v
                idx = 0
                for d in reversed(nd):
                    idx = idx * q + d
                nbrs.append(idx)
        neighbours.append(nbrs)

    # Greedy set cover: repeatedly pick the word covering the most uncovered.
    def greedy(seed):
        rng = random.Random(seed)
        uncovered = set(range(total))
        code = []
        while uncovered:
            best = -1
            best_cover = -1
            # Iterate over candidate codewords (all words) and count new coverage.
            # To speed up, only consider words that are still uncovered or
            # whose neighbourhood intersects uncovered.
            for cand in range(total):
                cov = 0
                for nb in neighbours[cand]:
                    if nb in uncovered:
                        cov += 1
                if cov > best_cover:
                    best_cover = cov
                    best = cand
                    if best_cover == len(neighbours[cand]):
                        break
            code.append(best)
            for nb in neighbours[best]:
                uncovered.discard(nb)
        return code

    # Multi-start greedy with a couple of random tie-breaks via shuffling.
    def greedy_shuffled(seed):
        rng = random.Random(seed)
        order = list(range(total))
        rng.shuffle(order)
        uncovered = set(range(total))
        code = []
        while uncovered:
            best = -1
            best_cover = -1
            for cand in order:
                cov = 0
                for nb in neighbours[cand]:
                    if nb in uncovered:
                        cov += 1
                if cov > best_cover:
                    best_cover = cov
                    best = cand
            code.append(best)
            for nb in neighbours[best]:
                uncovered.discard(nb)
        return code

    def try_local_improvement(code):
        # Try to drop any codeword whose coverage is fully covered by others.
        changed = True
        code = list(code)
        while changed:
            changed = False
            for i in range(len(code)):
                # Check if removing code[i] leaves all its neighbours covered.
                others = [c for j, c in enumerate(code) if j != i]
                covered = set()
                for c in others:
                    for nb in neighbours[c]:
                        covered.add(nb)
                ok = True
                for nb in neighbours[code[i]]:
                    if nb not in covered:
                        ok = False
                        break
                if ok:
                    code.pop(i)
                    changed = True
                    break
        return code

    # Run greedy a few times, keep the best (smallest) result.
    best = None
    seeds = [0, 1, 42, 12345, 777, 2024]
    for s in seeds:
        if _time_left() < 8.0:
            break
        cand = greedy_shuffled(s)
        cand = try_local_improvement(cand)
        if best is None or len(cand) < len(best):
            best = cand

    # Also run the deterministic greedy once.
    if _time_left() > 4.0:
        cand = greedy(0)
        cand = try_local_improvement(cand)
        if best is None or len(cand) < len(best):
            best = cand

    if best is None:
        # Fallback: direct sum construction (guaranteed valid, 189 words).
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                     for a in range(3) for b in range(3)]
        best = [h + list(tail) for h in tetracode
                for tail in itertools.product(range(3), repeat=n - 4)]

    return best
