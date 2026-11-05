"""Football pool problem: ternary covering code K_3(7,1).
Greedy covering code construction with efficient coverage tracking.
"""
import time
import itertools


def construct(q, n, radius):
    deadline = time.time() + 145

    if q != 3 or n != 7 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

    total = 3 ** n  # 2187

    def int_to_word(x):
        return [x % 3, (x // 3) % 3, (x // 9) % 3, (x // 27) % 3,
                (x // 81) % 3, (x // 243) % 3, (x // 729) % 3]

    def word_to_int(w):
        return w[0] + 3*w[1] + 9*w[2] + 27*w[3] + 81*w[4] + 243*w[5] + 729*w[6]

    # Precompute neighbors (coverage sets) and reverse neighbors
    neighbors = []
    for i in range(total):
        w = int_to_word(i)
        cov = [i]
        for pos in range(n):
            orig = w[pos]
            for v in (0, 1, 2):
                if v != orig:
                    w[pos] = v
                    cov.append(word_to_int(w))
            w[pos] = orig
        neighbors.append(cov)

    rev = [[] for _ in range(total)]
    for i in range(total):
        for j in neighbors[i]:
            rev[j].append(i)

    covered = [False] * total
    uncov_count = [len(neighbors[i]) for i in range(total)]
    uncovered_total = total
    codewords = []

    # Use a simple max-heap-like structure by scanning
    # For 2187 words and ~186 iterations, this should be fast enough
    while uncovered_total > 0 and time.time() < deadline - 2:
        best = 0
        best_c = -1
        for i in range(total):
            if uncov_count[i] > best_c:
                best_c = uncov_count[i]
                best = i
        if best_c <= 0:
            break
        codewords.append(best)
        for j in neighbors[best]:
            if not covered[j]:
                covered[j] = True
                uncovered_total -= 1
                for i in rev[j]:
                    uncov_count[i] -= 1

    return [int_to_word(c) for c in codewords]
