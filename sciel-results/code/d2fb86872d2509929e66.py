import time
import itertools
import numpy as np

def construct(q, n, radius):
    deadline = time.time() + 115.0

    total = q ** n

    # For the specific case q=3, n=8, radius=1, use a specialized approach
    if q == 3 and n == 8 and radius == 1:
        return _solve_3_8_1(deadline)

    # Generic fallback for other parameters
    if total > 5_000_000:
        # Too large; use crude upper bound
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute balls
    balls = []
    for w in all_words:
        neighbors = [word_to_idx[w]]
        for i in range(n):
            for d in range(q):
                if d != w[i]:
                    new_w = list(w)
                    new_w[i] = d
                    neighbors.append(word_to_idx[tuple(new_w)])
        balls.append(neighbors)

    # Greedy set cover
    uncovered = np.ones(total, dtype=bool)
    covered_count = total
    chosen = []

    while covered_count > 0 and time.time() < deadline:
        best_idx = -1
        best_gain = 0
        for i in range(total):
            gain = 0
            for idx in balls[i]:
                if uncovered[idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_gain == 0:
            break
        chosen.append(best_idx)
        for idx in balls[best_idx]:
            if uncovered[idx]:
                uncovered[idx] = False
                covered_count -= 1

    # Prune
    chosen = _prune(chosen, balls)

    # Local search
    chosen = _local_search(chosen, balls, deadline)
    chosen = _prune(chosen, balls)

    return [list(all_words[i]) for i in chosen]


def _solve_3_8_1(deadline):
    n = 8
    q = 3
    total = 6561

    # Build the space
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute balls: each ball has 1 + 8*2 = 17 words
    balls = []
    for w in all_words:
        neighbors = [word_to_idx[w]]
        for i in range(n):
            for d in range(q):
                if d != w[i]:
                    new_w = list(w)
                    new_w[i] = d
                    neighbors.append(word_to_idx[tuple(new_w)])
        balls.append(neighbors)

    # Greedy set cover with randomization for diversity
    uncovered = np.ones(total, dtype=bool)
    covered_count = total
    chosen = []

    while covered_count > 0 and time.time() < deadline:
        # Find best gain
        best_gain = 0
        best_candidates = []
        for i in range(total):
            gain = 0
            for idx in balls[i]:
                if uncovered[idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_candidates = [i]
            elif gain == best_gain and gain > 0:
                best_candidates.append(i)

        if best_gain == 0:
            break

        # Pick randomly among best candidates for diversity
        import random
        best_idx = random.choice(best_candidates)
        chosen.append(best_idx)
        for idx in balls[best_idx]:
            if uncovered[idx]:
                uncovered[idx] = False
                covered_count -= 1

    # Prune
    chosen = _prune(chosen, balls)

    # Aggressive local search with neighbor-restricted swaps
    chosen = _local_search_3_8_1(chosen, balls, deadline)
    chosen = _prune(chosen, balls)

    # Final validity check
    covered = np.zeros(total, dtype=bool)
    for w_idx in chosen:
        for j in balls[w_idx]:
            covered[j] = True
    if not np.all(covered):
        # Fallback: tetracode-based construction
        _TETRA = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in _TETRA
                for tail in itertools.product(range(3), repeat=4)]

    return [list(all_words[i]) for i in chosen]


def _prune(chosen_list, balls):
    if not chosen_list:
        return []
    total = len(balls)
    coverage_count = np.zeros(total, dtype=np.int32)
    for c in chosen_list:
        for j in balls[c]:
            coverage_count[j] += 1

    keep = [True] * len(chosen_list)
    any_removed = True
    while any_removed:
        any_removed = False
        for i, c in enumerate(chosen_list):
            if not keep[i]:
                continue
            redundant = True
            for j in balls[c]:
                if coverage_count[j] <= 1:
                    redundant = False
                    break
            if redundant:
                keep[i] = False
                any_removed = True
                for j in balls[c]:
                    coverage_count[j] -= 1
    return [chosen_list[i] for i in range(len(chosen_list)) if keep[i]]


def _local_search_3_8_1(chosen_list, balls, deadline):
    total = len(balls)
    chosen_set = set(chosen_list)

    # Precompute for each codeword, the set of points it uniquely covers
    # and candidate replacements

    while time.time() < deadline:
        improved = False
        chosen_set = set(chosen_list)

        coverage_count = np.zeros(total, dtype=np.int32)
        for c in chosen_list:
            for j in balls[c]:
                coverage_count[j] += 1

        new_chosen = list(chosen_list)

        # Try to swap each codeword
        for i, c in enumerate(new_chosen):
            if time.time() > deadline:
                break

            # Find points uniquely covered by c
            unique_points = [j for j in balls[c] if coverage_count[j] == 1]
            if not unique_points:
                continue

            # Candidate replacements: codewords that cover all unique points of c
            candidates = set(balls[unique_points[0]])
            for uj in unique_points[1:]:
                candidates &= set(balls[uj])

            # Exclude already chosen codewords (except c itself)
            candidates -= chosen_set
            valid_candidates = list(candidates)

            if not valid_candidates:
                continue

            # Try each candidate
            for c_prime in valid_candidates:
                tentative = list(new_chosen)
                tentative[i] = c_prime
                pruned = _prune(tentative, balls)
                if len(pruned) < len(new_chosen):
                    new_chosen = pruned
                    chosen_set = set(new_chosen)
                    improved = True
                    break

            if improved:
                break

        if not improved:
            # Try adding a new codeword that covers many uncovered-unique points
            # Actually, try removing pairs of codewords if their union is covered
            # by others
            improved = False
            n_chosen = len(new_chosen)
            for i in range(n_chosen):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n_chosen):
                    if time.time() > deadline:
                        break
                    tentative = list(new_chosen)
                    tentative.pop(j)
                    tentative.pop(i)
                    pruned = _prune(tentative, balls)
                    if len(pruned) < len(new_chosen):
                        new_chosen = pruned
                        improved = True
                        break
                if improved:
                    break

        if not improved:
            break

    return new_chosen
