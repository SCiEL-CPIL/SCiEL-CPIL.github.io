"""Rewritten program for small nonzero signed harmonic subsums."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random
import numpy as np


def _compute_lcm_1_to_30():
    result = 1
    for i in range(1, 31):
        result = result * i // gcd(result, i)
    return result


_LCM = _compute_lcm_1_to_30()


def _weights():
    return [_LCM // i for i in range(1, 31)]


def _evaluate(coeffs, w):
    s = 0
    for i, c in enumerate(coeffs):
        s += c * w[i]
    return s


def _simulated_annealing(deadline, seed=42):
    """
    Simulated annealing over the 3^30 space.
    State: 30 coefficients in {-1, 0, 1}.
    Move: pick a random index, change its coefficient to a different value in {-1,0,1}.
    Objective: minimize |sum|, but sum must be nonzero.
    """
    w = _weights()
    n = 30
    rng = random.Random(seed)

    # Initial state: random
    coeffs = [rng.choice([-1, 0, 1]) for _ in range(n)]
    current_sum = _evaluate(coeffs, w)
    if current_sum == 0:
        # Perturb to nonzero
        idx = rng.randrange(n)
        old = coeffs[idx]
        for new in [-1, 0, 1]:
            if new != old:
                coeffs[idx] = new
                current_sum += (new - old) * w[idx]
                break

    best_abs = abs(current_sum)
    best_coeffs = list(coeffs)

    # Annealing schedule
    t0 = time.time()
    total_time = deadline - t0
    if total_time <= 0:
        return best_coeffs

    # Start temperature
    max_weight = max(w)
    T_start = max_weight * 10
    T_end = 1.0
    steps = int(total_time * 5000)  # ~5000 steps per second

    for step in range(steps):
        if time.time() > deadline:
            break

        # Temperature
        alpha = step / max(steps, 1)
        T = T_start * (T_end / T_start) ** alpha

        # Pick random index
        i = rng.randrange(n)
        old_c = coeffs[i]
        # Pick new coefficient different from old
        choices = [c for c in (-1, 0, 1) if c != old_c]
        new_c = rng.choice(choices)

        delta = (new_c - old_c) * w[i]
        new_sum = current_sum + delta

        # If new_sum is 0, we can't accept (must be nonzero)
        if new_sum == 0:
            continue

        new_abs = abs(new_sum)
        old_abs = abs(current_sum)

        if new_abs < old_abs:
            # Always accept improvement
            coeffs[i] = new_c
            current_sum = new_sum
            if new_abs < best_abs:
                best_abs = new_abs
                best_coeffs = list(coeffs)
        else:
            # Accept with probability exp(-(new_abs - old_abs) / T)
            if T > 0:
                prob = np.exp(-(new_abs - old_abs) / T)
                if rng.random() < prob:
                    coeffs[i] = new_c
                    current_sum = new_sum

    return best_coeffs


def _local_search_from(coeffs, w, deadline):
    """
    Hill-climbing local search from a given starting point.
    """
    n = 30
    current_sum = _evaluate(coeffs, w)
    if current_sum == 0:
        # Perturb
        for i in range(n):
            old = coeffs[i]
            for new in [-1, 0, 1]:
                if new == old:
                    continue
                coeffs[i] = new
                current_sum += (new - old) * w[i]
                if current_sum != 0:
                    break
            if current_sum != 0:
                break
        if current_sum == 0:
            return None

    current_abs = abs(current_sum)
    best_abs = current_abs
    best_coeffs = list(coeffs)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            old = coeffs[i]
            for new in [-1, 0, 1]:
                if new == old:
                    continue
                new_sum = current_sum + (new - old) * w[i]
                if new_sum == 0:
                    continue
                new_abs = abs(new_sum)
                if new_abs < current_abs:
                    coeffs[i] = new
                    current_sum = new_sum
                    current_abs = new_abs
                    improved = True
                    if current_abs < best_abs:
                        best_abs = current_abs
                        best_coeffs = list(coeffs)
                    break
    return best_coeffs


def _greedy_init(w):
    """
    Greedy initialization: process weights in descending order,
    pick coefficient to minimize running sum.
    """
    n = 30
    coeffs = [0] * n
    current_sum = 0
    order = sorted(range(n), key=lambda i: -w[i])
    for idx in order:
        best_c = 0
        best_abs = abs(current_sum)
        for c in [-1, 1]:
            new_abs = abs(current_sum + c * w[idx])
            if new_abs < best_abs:
                best_abs = new_abs
                best_c = c
        coeffs[idx] = best_c
        current_sum += best_c * w[idx]
    return coeffs


def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    w = _weights()

    # Strategy:
    # 1. Greedy init + local search
    # 2. Simulated annealing with multiple restarts
    # 3. Random restarts + local search

    best_abs = float('inf')
    best_coeffs = None

    # Phase 1: Greedy init + local search
    coeffs = _greedy_init(w)
    result = _local_search_from(coeffs, w, deadline)
    if result is not None:
        s = _evaluate(result, w)
        if s != 0 and abs(s) < best_abs:
            best_abs = abs(s)
            best_coeffs = list(result)

    # Phase 2: Simulated annealing (multiple runs)
    if time.time() < deadline:
        # Run SA for a portion of remaining time
        sa_deadline = min(deadline, start_time + 40)
        for seed in [42, 123, 777]:
            if time.time() > sa_deadline:
                break
            coeffs = _simulated_annealing(sa_deadline, seed=seed)
            s = _evaluate(coeffs, w)
            if s != 0 and abs(s) < best_abs:
                best_abs = abs(s)
                best_coeffs = list(coeffs)

    # Phase 3: Random restarts + local search
    if time.time() < deadline:
        random.seed(999)
        for r in range(200):
            if time.time() > deadline:
                break
            coeffs = [random.choice([-1, 0, 1]) for _ in range(30)]
            result = _local_search_from(coeffs, w, deadline)
            if result is not None:
                s = _evaluate(result, w)
                if s != 0 and abs(s) < best_abs:
                    best_abs = abs(s)
                    best_coeffs = list(result)

    # Final validation
    if best_coeffs is None:
        # Fallback: ensure nonzero
        best_coeffs = [0] * 28 + [1, -1]
        s = _evaluate(best_coeffs, w)
        if s == 0:
            best_coeffs = [1] + [0] * 29
            s = _evaluate(best_coeffs, w)

    s = _evaluate(best_coeffs, w)
    if s == 0:
        # Should not happen, but fallback
        best_coeffs = [0] * 28 + [1, -1]

    return best_coeffs


# EVOLVE-BLOCK-END
