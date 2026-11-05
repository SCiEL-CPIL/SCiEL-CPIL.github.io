# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

_BASE = 51975  # 3^3 * 5^2 * 7 * 11

def _all_odd_divisors(n):
    divs = []
    for d in range(3, n + 1, 2):
        if n % d == 0:
            divs.append(d)
    return divs

def _score_from_counts(counts):
    return int(np.count_nonzero(counts == 0))

def _best_residue_for(m, counts, rng=None):
    """Return the residue r in [0, m) maximizing coverage of currently-uncovered residues."""
    best_r = 0
    best_gain = -1
    for r in range(m):
        gain = np.count_nonzero(counts[r::m] == 0)
        if gain > best_gain:
            best_gain = gain
            best_r = r
    return best_r, best_gain

def _greedy_build(moduli, order, rng):
    counts = np.zeros(_BASE, dtype=np.int8)
    residues = [0] * len(moduli)
    for idx in order:
        m = moduli[idx]
        r, _ = _best_residue_for(m, counts, rng)
        residues[idx] = r
        counts[r::m] += 1
    return residues, counts, _score_from_counts(counts)

def _exact_polish(residues, counts, moduli, rng, deadline, min_temp=0.05, cooling=0.997):
    """Hill-climb then simulated-annealing polish using exact O(m) delta evaluation."""
    num = len(moduli)
    current_score = _score_from_counts(counts)
    best_score = current_score
    best_res = residues[:]
    temperature = 3.0
    # Track per-modulus best gain to bias moves
    while time.time() < deadline and temperature > min_temp:
        i = rng.randint(0, num - 1)
        m = moduli[i]
        old_r = residues[i]
        new_r = rng.randint(0, m - 1)
        if new_r == old_r:
            continue
        # Exact delta
        counts[old_r::m] -= 1
        counts[new_r::m] += 1
        new_score = _score_from_counts(counts)
        delta = new_score - current_score
        if delta < 0 or (delta == 0 and rng.random() < 0.3) or rng.random() < math.exp(-max(delta,0) / temperature):
            residues[i] = new_r
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_res = residues[:]
        else:
            counts[new_r::m] -= 1
            counts[old_r::m] += 1
        temperature *= cooling
    # Final greedy re-sweep: for each modulus, try all residues and keep the best
    for i in range(num):
        m = moduli[i]
        old_r = residues[i]
        best_r = old_r
        best_s = current_score
        for r in range(m):
            if r == old_r:
                continue
            counts[old_r::m] -= 1
            counts[r::m] += 1
            s = _score_from_counts(counts)
            if s < best_s:
                best_s = s
                best_r = r
            else:
                counts[r::m] -= 1
                counts[old_r::m] += 1
        if best_r != old_r:
            counts[old_r::m] -= 1
            counts[best_r::m] += 1
            residues[i] = best_r
            current_score = best_s
            if current_score < best_score:
                best_score = current_score
                best_res = residues[:]
    return best_res, best_score

def _softmin_surrogate(counts, beta=20.0):
    """Smooth surrogate for the count of zero-covered residues.
    For each residue, indicator(counts==0) is approximated by exp(-beta*counts) (soft-min).
    Sum over residues gives a smooth objective to guide search."""
    # counts are non-negative integers; exp(-beta*counts) is 1 when 0, decays otherwise
    return float(np.sum(np.exp(-beta * counts)))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    moduli = _all_odd_odd = _all_odd_divisors(_BASE)
    num = len(moduli)
    rng = random.Random(12345)

    # Sort moduli: small moduli first for greedy (they cover more)
    order_small = sorted(range(num), key=lambda i: moduli[i])
    order_large = sorted(range(num), key=lambda i: -moduli[i])

    best_score = _BASE
    best_res = None

    # ---- Phase 1: multiple greedy restarts with different orderings ----
    orders = [order_small, order_large]
    # A few random orders
    for _ in range(4):
        o = list(range(num))
        rng.shuffle(o)
        orders.append(o)

    for order in orders:
        if time.time() > deadline:
            break
        res, counts, score = _greedy_build(moduli, order, rng)
        if score < best_score:
            best_score = score
            best_res = res[:]

    # ---- Phase 2: exact polish on the best greedy solution ----
    if best_res is not None:
        counts = np.zeros(_BASE, dtype=np.int8)
        for i, m in enumerate(moduli):
            counts[best_res[i]::m] += 1
        polished_res, polished_score = _exact_polish(
            best_res[:], counts, moduli, rng, deadline - 5.0)
        if polished_score < best_score:
            best_score = polished_score
            best_res = polished_res

    # ---- Phase 3: additional random restarts + polish while time remains ----
    while time.time() < deadline - 10.0:
        # Random perturbation of current best or fresh greedy
        if rng.random() < 0.5 and best_res is not None:
            res = best_res[:]
            # Perturb a few random moduli
            for _ in range(rng.randint(1, 4)):
                i = rng.randint(0, num - 1)
                m = moduli[i]
                res[i] = rng.randint(0, m - 1)
        else:
            o = list(range(num))
            rng.shuffle(o)
            res, counts, score = _greedy_build(moduli, o, rng)
            continue  # already evaluated; polish below

        counts = np.zeros(_BASE, dtype=np.int8)
        for i, m in enumerate(moduli):
            counts[res[i]::m] += 1
        score = _score_from_counts(counts)
        if score < best_score:
            best_score = score
            best_res = res[:]

        # Polish this candidate
        pr, ps = _exact_polish(res[:], counts, moduli, rng, deadline - 5.0)
        if ps < best_score:
            best_score = ps
            best_res = pr

    # ---- Final exact greedy sweep polish ----
    if best_res is not None and time.time() < deadline - 2.0:
        counts = np.zeros(_BASE, dtype=np.int8)
        for i, m in enumerate(moduli):
            counts[best_res[i]::m] += 1
        final_res, final_score = _exact_polish(
            best_res[:], counts, moduli, rng, deadline - 1.0)
        if final_score < best_score:
            best_score = final_score
            best_res = final_res

    if best_res is None:
        # Fallback: all residues 0
        best_res = [0] * num

    return [[moduli[i], best_res[i]] for i in range(num)]
# EVOLVE-BLOCK-END
