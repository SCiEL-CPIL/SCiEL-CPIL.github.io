"""Spectral local search for sumset/difference-set exponent lower bound."""
import time
import math
import random
import numpy as np

_DEADLINE = 120.0
_TIME_BUDGET = 115.0  # safety margin before the 120s kill


def _score_from_counts(diff_count, sum_count, max_u):
    if sum_count <= 0 or diff_count <= 0 or max_u <= 0:
        return 0.0
    return 1.0 + math.log(diff_count / sum_count) / math.log(2 * max_u + 1)


def _golomb_ruler(n):
    # Small optimal Golomb rulers (distinct pairwise differences)
    tables = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 18, 23, 28],
        8: [0, 1, 4, 10, 18, 23, 29, 36],
        9: [0, 1, 4, 10, 18, 23, 29, 37, 44],
        10: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52],
    }
    if n in tables:
        return list(tables[n])
    # Fallback: greedy construction
    elems = [0]
    diffs = set()
    while len(elems) < n:
        x = 1
        while True:
            ok = True
            for e in elems:
                if abs(x - e) in diffs:
                    ok = False
                    break
            if ok:
                for e in elems:
                    diffs.add(abs(x - e))
                elems.append(x)
                break
            x += 1
    return elems


def _counts_from_set(U):
    """Return (diff_count, sum_count, max_u) using FFT."""
    U = sorted(set(int(x) for x in U))
    if not U:
        return 0, 0, 0
    max_u = U[-1]
    ind = np.zeros(max_u + 1, dtype=np.float64)
    for x in U:
        ind[x] = 1.0

    # Sumset: convolve(ind, ind); support is [0, 2*max_u]
    n_conv = 2 * max_u + 1
    conv = np.fft.irfft(np.fft.rfft(ind, n=n_conv) ** 2, n=n_conv)
    sum_count = int(np.sum(conv > 1e-6))

    # Difference set: convolve(ind, reversed(ind)); support [-(max_u), max_u]
    ind_rev = ind[::-1]
    conv_d = np.fft.irfft(np.fft.rfft(ind, n=n_conv) * np.fft.rfft(ind_rev, n=n_conv), n=n_conv)
    diff_count = int(np.sum(conv_d > 1e-6))

    return diff_count, sum_count, max_u


def _score_of(U):
    diff_count, sum_count, max_u = _counts_from_set(U)
    return _score_from_counts(diff_count, sum_count, max_u)


def construct():
    start = time.time()
    deadline = start + _TIME_BUDGET

    random.seed(12345)

    # Initial candidate: Golomb ruler of moderate size
    best_U = _golomb_ruler(10)
    best_score = _score_of(best_U)

    # Also try a few different starting sizes
    for n0 in (8, 12, 14, 16, 18, 20):
        if time.time() > deadline - 5:
            break
        cand = _golomb_ruler(n0)
        s = _score_of(cand)
        if s > best_score:
            best_score = s
            best_U = cand

    # Local search: add, remove, and move elements
    # Neighborhood: small shifts and additions/removals
    def try_move(U_set, current_score):
        # Returns (new_U, new_score, accepted) or (None, 0, False)
        U = sorted(U_set)
        max_u = U[-1] if U else 0
        move_type = random.random()
        if move_type < 0.35:
            # Add a new element
            # Try positions: small increments, or near existing elements
            candidates = []
            for x in U:
                for d in (1, 2, 3, 5, 8):
                    candidates.append(x + d)
            candidates.append(max_u + 1)
            candidates.append(max_u + 2)
            candidates.append(max_u + 3)
            # Also try a random position
            candidates.append(random.randint(1, max_u + 10))
            random.shuffle(candidates)
            for x in candidates[:6]:
                if x in U_set:
                    continue
                new_U = U_set | {x}
                s = _score_of(new_U)
                if s > current_score:
                    return new_U, s, True
            return None, 0, False
        elif move_type < 0.65:
            # Remove an element (not 0)
            removable = [x for x in U if x != 0]
            if not removable:
                return None, 0, False
            random.shuffle(removable)
            for x in removable[:4]:
                new_U = U_set - {x}
                s = _score_of(new_U)
                if s > current_score:
                    return new_U, s, True
            return None, 0, False
        else:
            # Move an element: remove x, add x+d
            movable = [x for x in U if x != 0]
            if not movable:
                return None, 0, False
            random.shuffle(movable)
            for x in movable[:3]:
                for d in (-3, -2, -1, 1, 2, 3, 5, 8):
                    y = x + d
                    if y < 0 or y in U_set:
                        continue
                    new_U = (U_set - {x}) | {y}
                    s = _score_of(new_U)
                    if s > current_score:
                        return new_U, s, True
            return None, 0, False

    U_set = set(best_U)
    current_score = best_score

    # Simulated-annealing-like schedule: occasionally accept worse moves
    temperature = 0.05
    iterations = 0
    while time.time() < deadline - 2:
        new_U, new_score, accepted = try_move(U_set, current_score)
        iterations += 1
        if accepted:
            U_set = new_U
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_U = sorted(U_set)
        else:
            # Occasionally do a random perturbation to escape local optima
            if iterations % 200 == 0 and temperature > 0.005:
                # Randomly add a few elements
                for _ in range(2):
                    max_u = max(U_set) if U_set else 0
                    x = random.randint(1, max_u + 15)
                    if x not in U_set:
                        U_set.add(x)
                s = _score_of(U_set)
                if s > current_score:
                    current_score = s
                else:
                    # Revert
                    U_set = set(best_U)
                    current_score = best_score
            temperature *= 0.999

        # Periodically try a larger Golomb ruler as a fresh start if stuck
        if iterations % 500 == 0:
            for n0 in (22, 24, 26, 28, 30):
                if time.time() > deadline - 5:
                    break
                cand = _golomb_ruler(n0)
                s = _score_of(cand)
                if s > best_score:
                    best_score = s
                    best_U = cand
                    U_set = set(best_U)
                    current_score = best_score
                    break

    return sorted(best_U)


if __name__ == "__main__":
    result = construct()
    print("Set size:", len(result))
    print("Max element:", max(result))
    print("Score:", _score_of(result))
    print("Set:", result)
