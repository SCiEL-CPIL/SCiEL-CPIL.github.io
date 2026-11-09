# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
import random


def _base_cover(q, n, radius):
    # Direct sum of the ternary tetracode (9 words of length 4, perfect covering radius 1)
    # with every word on the remaining n - 4 coordinates: 9 * 3**(n - 4) words.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(q) for b in range(q)]
    if n <= 4:
        return [list(h[:n]) for h in tetracode]
    return [h + list(tail) for h in tetracode
            for tail in itertools.product(range(q), repeat=n - 4)]


def _is_cover(codewords, q, n, radius):
    if not codewords:
        return n == 0
    C = np.array(codewords, dtype=np.int32).reshape(-1, n)
    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    diff = (words[:, None, :] != C[None, :, :])
    dist = diff.sum(axis=2)
    return bool(np.all(dist.min(axis=1) <= radius))


def _words_from_indices(indices, q, n):
    """Convert integer indices back to word tuples."""
    result = []
    for idx in indices:
        word = []
        rem = int(idx)
        for _ in range(n):
            word.append(rem % q)
            rem //= q
        result.append(word)
    return result


def _milp_cover(q, n, radius, deadline):
    """Try to solve the covering problem as a MILP using SciPy."""
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except ImportError:
        return None

    M = q ** n
    if M > 50000:
        return None

    all_words = np.array([_words_from_indices([i], q, n)[0] for i in range(M)], dtype=np.int32)
    diff = (all_words[:, None, :] != all_words[None, :, :])
    dist = diff.sum(axis=2)
    # A[i, j] = 1 if word i is within radius of codeword j
    A = (dist <= radius).astype(np.int8)
    del diff, dist, all_words

    # Variables: x_j in {0,1} for j in range(M)
    n_vars = M
    c = np.ones(n_vars)

    # Constraint: for each word i, sum_j A[i,j] * x_j >= 1
    # A is M x M, each row i gives a constraint
    constraint = LinearConstraint(A, lb=1, ub=np.inf)

    integrality = np.ones(n_vars)
    bounds = Bounds(0, 1)

    result = milp(c, constraints=constraint, integrality=integrality, bounds=bounds)

    if not result.success:
        return None

    x = np.round(result.x).astype(int)
    selected = np.where(x == 1)[0].tolist()
    return _words_from_indices(selected, q, n)


def _greedy_cover(q, n, radius, deadline):
    """Greedy set cover with local improvement."""
    M = q ** n
    all_words = np.array([_words_from_indices([i], q, n)[0] for i in range(M)], dtype=np.int32)
    diff = (all_words[:, None, :] != all_words[None, :, :])
    dist = diff.sum(axis=2)
    balls = [np.where(dist[i] <= radius)[0] for i in range(M)]
    del diff, dist, all_words

    selected = []
    covered = np.zeros(M, dtype=bool)
    remaining = M

    start = time.time()
    while remaining > 0 and time.time() < deadline - 20:
        # Find codeword that covers the most uncovered words
        best_c = -1
        best_gain = -1
        for c in range(M):
            if any(selected[i] == c for i in range(len(selected))):
                continue
            gain = np.sum(~covered[balls[c]])
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_c == -1 or best_gain == 0:
            break
        selected.append(best_c)
        covered[balls[best_c]] = True
        remaining -= best_gain

    # Local polish: try removing redundant codewords
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        for i in range(len(selected) - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            c = selected[i]
            # Check if removing c leaves all words covered
            temp_selected = selected[:i] + selected[i+1:]
            if not temp_selected:
                continue
            # Recompute coverage
            cov = np.zeros(M, dtype=bool)
            for sc in temp_selected:
                cov[balls[sc]] = True
            if np.all(cov):
                selected.pop(i)
                changed = True
                break

    return _words_from_indices(selected, q, n)


def _simulated_annealing(q, n, radius, deadline):
    M = q ** n
    if M > 100000:
        return _base_cover(q, n, radius)

    all_words = np.array([_words_from_indices([i], q, n)[0] for i in range(M)], dtype=np.int32)
    diff = (all_words[:, None, :] != all_words[None, :, :])
    dist = diff.sum(axis=2)
    balls = [np.where(dist[i] <= radius)[0] for i in range(M)]
    del diff, dist, all_words

    init_code = _base_cover(q, n, radius)
    if not init_code:
        return [[]]

    def word_to_idx(word):
        idx = 0
        for d in range(n):
            idx = idx * q + word[d]
        return idx

    code_indices = [word_to_idx(w) for w in init_code]
    code_indices = list(set(code_indices))

    coverage_count = np.zeros(M, dtype=np.int32)
    for c in code_indices:
        for w in balls[c]:
            coverage_count[w] += 1

    P = M + 10
    current_uncovered = np.sum(coverage_count == 0)
    current_cost = len(code_indices) + P * current_uncovered

    best_code = list(code_indices)
    best_cost = current_cost

    T = 1.0
    cooling = 0.9995
    min_T = 1e-6

    rng = random.Random(42)
    start_time = time.time()

    while time.time() < deadline - 5:
        if T < min_T:
            break

        r = rng.random()
        if r < 0.5:
            # Remove
            if not code_indices:
                continue
            c = code_indices[rng.randrange(len(code_indices))]
            newly_uncovered = 0
            affected_words = balls[c]
            for w in affected_words:
                if coverage_count[w] == 1:
                    newly_uncovered += 1
            delta_uncovered = newly_uncovered
            delta_cost = 1 + P * delta_uncovered

            if delta_cost < 0 or (T > 0 and rng.random() < np.exp(-delta_cost / T)):
                code_indices.remove(c)
                for w in affected_words:
                    coverage_count[w] -= 1
                current_uncovered += newly_uncovered
                current_cost += delta_cost

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_code = list(code_indices)
        elif r < 0.9:
            # Swap
            if not code_indices:
                continue
            c = code_indices[rng.randrange(len(code_indices))]
            c_new = rng.randrange(M)
            if c_new == c:
                continue
            if c_new in code_indices:
                continue

            affected = set(balls[c]) | set(balls[c_new])
            affected_list = list(affected)

            old_counts = {w: coverage_count[w] for w in affected_list}

            for w in balls[c]:
                coverage_count[w] -= 1
            for w in balls[c_new]:
                coverage_count[w] += 1

            new_uncovered = 0
            for w in affected_list:
                if coverage_count[w] == 0:
                    new_uncovered += 1

            old_uncovered = 0
            for w in affected_list:
                if old_counts[w] == 0:
                    old_uncovered += 1

            delta_uncovered = new_uncovered - old_uncovered
            delta_cost = P * delta_uncovered

            if delta_cost < 0 or (T > 0 and rng.random() < np.exp(-delta_cost / T)):
                code_indices[code_indices.index(c)] = c_new
                current_uncovered += delta_uncovered
                current_cost += delta_cost

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_code = list(code_indices)
            else:
                for w in affected_list:
                    coverage_count[w] = old_counts[w]

        T *= cooling

    # Final polish
    coverage_count = np.zeros(M, dtype=np.int32)
    for c in best_code:
        for w in balls[c]:
            coverage_count[w] += 1

    changed = True
    while changed and time.time() < deadline - 1:
        changed = False
        for i in range(len(best_code)):
            if time.time() > deadline - 1:
                break
            c = best_code[i]
            is_redundant = True
            for w in balls[c]:
                if coverage_count[w] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                best_code.pop(i)
                for w in balls[c]:
                    coverage_count[w] -= 1
                changed = True
                break

    return _words_from_indices(best_code, q, n)


def construct(q, n, radius):
    start = time.time()
    deadline = start + 115.0

    if n == 0:
        return [[]]

    base = _base_cover(q, n, radius)
    if not _is_cover(base, q, n, radius):
        base = [list(w) for w in itertools.product(range(q), repeat=n)]
    best = base

    # Try MILP first (exact)
    if time.time() < deadline - 30:
        try:
            milp_result = _milp_cover(q, n, radius, deadline)
            if milp_result is not None and len(milp_result) < len(best):
                if _is_cover(milp_result, q, n, radius):
                    best = milp_result
        except Exception:
            pass

    # Try greedy
    if time.time() < deadline - 25 and len(best) > 70:
        try:
            greedy_result = _greedy_cover(q, n, radius, deadline)
            if greedy_result is not None and len(greedy_result) < len(best):
                if _is_cover(greedy_result, q, n, radius):
                    best = greedy_result
        except Exception:
            pass

    # Try simulated annealing
    if time.time() < deadline - 10:
        try:
            sa_result = _simulated_annealing(q, n, radius, deadline)
            if len(sa_result) < len(best) and _is_cover(sa_result, q, n, radius):
                best = sa_result
        except Exception:
            pass

    return best
# EVOLVE-BLOCK-END
