import numpy as np
import itertools


def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return _greedy_cover_general(q, n, radius)

    num_words = 3 ** 7
    all_words = np.array(list(itertools.product(range(3), repeat=7)), dtype=np.int8)

    pow3 = [1, 3, 9, 27, 81, 243, 729]

    def to_idx(w):
        return (w[0] * pow3[6] + w[1] * pow3[5] + w[2] * pow3[4] +
                w[3] * pow3[3] + w[4] * pow3[2] + w[5] * pow3[1] + w[6] * pow3[0])

    # Precompute coverage for each candidate codeword
    coverage = []
    for i in range(num_words):
        c = all_words[i]
        c_idx = to_idx(c)
        covered = [c_idx]
        for pos in range(7):
            for delta in [1, 2]:
                nw = list(c)
                nw[pos] = (c[pos] + delta) % 3
                covered.append(to_idx(nw))
        coverage.append(covered)

    # Greedy set cover
    uncovered_set = set(range(num_words))
    selected = []

    while uncovered_set:
        best_i = -1
        best_cov = 0
        for i in range(num_words):
            if best_cov >= 15:
                break
            cov = sum(1 for w_idx in coverage[i] if w_idx in uncovered_set)
            if cov > best_cov:
                best_cov = cov
                best_i = i
                if best_cov >= 15:
                    break

        if best_i == -1:
            break

        selected.append(all_words[best_i].tolist())
        for w_idx in coverage[best_i]:
            uncovered_set.discard(w_idx)

    return selected


def _greedy_cover_general(q, n, radius):
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = q ** n

    covered_sets = []
    for w in all_words:
        covered = set()
        for d in range(radius + 1):
            for positions in itertools.combinations(range(n), d):
                for deltas in itertools.product(range(1, q), repeat=d):
                    new_word = list(w)
                    for pos, delta in zip(positions, deltas):
                        new_word[pos] = (w[pos] + delta) % q
                    covered.add(tuple(new_word))
        covered_sets.append(covered)

    uncovered = set(range(num_words))
    selected = []

    while uncovered:
        best_i = -1
        best_count = 0
        for i in range(num_words):
            count = len(covered_sets[i] & uncovered)
            if count > best_count:
                best_count = count
                best_i = i
        if best_i == -1:
            break
        selected.append(list(all_words[best_i]))
        uncovered -= covered_sets[best_i]

    return selected
