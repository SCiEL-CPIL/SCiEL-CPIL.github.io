"""Hypergraph Ramsey R(4,5;3) via simulated annealing over the binary triple-colouring vector."""
import time
import random
import numpy as np
from itertools import combinations

_START = time.time()
_DEADLINE = 150.0  # seconds; return with margin


def _t_idx(a, b, c, n):
    """Lexicographic index of triple (a,b,c) among triples of n."""
    cnt = 0
    for x in range(a):
        cnt += (n - x - 1) * (n - x - 2) // 2
    for y in range(a + 1, b):
        cnt += (n - y - 1)
    cnt += (c - b - 1)
    return cnt


def _build_data(n):
    """Build triple list, quad/pent incidence, and per-triple affected lists."""
    triples = list(combinations(range(n), 3))
    m = len(triples)

    # Build quads
    quads = []
    for q in combinations(range(n), 4):
        a, b, c, d = q
        idxs = [_t_idx(a, b, c, n), _t_idx(a, b, d, n),
                _t_idx(a, c, d, n), _t_idx(b, c, d, n)]
        quads.append(idxs)
    n_quads = len(quads)

    # Build pents
    pents = []
    for p in combinations(range(n), 5):
        idxs = []
        for t in combinations(p, 3):
            idxs.append(_t_idx(t[0], t[1], t[2], n))
        pents.append(idxs)
    n_pents = len(pents)

    # Per-triple affected quads and pents
    tri_quads = [[] for _ in range(m)]
    for qi, idxs in enumerate(quads):
        for i in idxs:
            tri_quads[i].append(qi)

    tri_pents = [[] for _ in range(m)]
    for pi, idxs in enumerate(pents):
        for i in idxs:
            tri_pents[i].append(pi)

    return triples, m, quads, pents, tri_quads, tri_pents


def _initial_energy(colors, quads, pents):
    """Compute initial quad_red and pent_blue counts, plus total violations."""
    m = len(colors)
    quad_red = np.zeros(len(quads), dtype=np.int8)
    for qi, idxs in enumerate(quads):
        for i in idxs:
            if colors[i] == 0:
                quad_red[qi] += 1
    pent_blue = np.zeros(len(pents), dtype=np.int8)
    for pi, idxs in enumerate(pents):
        for i in idxs:
            if colors[i] == 1:
                pent_blue[pi] += 1
    red_k4 = int(np.sum(quad_red == 4))
    blue_k5 = int(np.sum(pent_blue == 10))
    return quad_red, pent_blue, red_k4, blue_k5


def _simulated_annealing(colors, quads, pents, tri_quads, tri_pents, time_budget, seed=0):
    """
    Simulated annealing to minimize energy = weighted sum of red K4 and blue K5 violations.
    Energy = sum over quads of max(0, quad_red - 3) * WQ + sum over pents of max(0, pent_blue - 9) * WP
    where WQ and WP are weights to balance the two constraints.
    """
    rng = np.random.RandomState(seed)
    colors = np.array(colors, dtype=np.int8)
    m = len(colors)
    n_quads = len(quads)
    n_pents = len(pents)

    WQ = 10  # weight for red K4 violations
    WP = 1   # weight for blue K5 violations

    quad_red, pent_blue, red_k4, blue_k5 = _initial_energy(colors, quads, pents)
    energy = WQ * red_k4 + WP * blue_k5

    # Annealing schedule
    T_start = 5.0
    T_end = 0.01
    t_end = time.time() + time_budget

    # Precompute affected lists as numpy arrays for speed
    tri_quads_np = [np.array(qs, dtype=np.int32) if qs else np.array([], dtype=np.int32) for qs in tri_quads]
    tri_pents_np = [np.array(ps, dtype=np.int32) if ps else np.array([], dtype=np.int32) for ps in tri_pents]

    # Estimate steps per second to set total steps
    # Do a quick timing
    t0 = time.time()
    for _ in range(1000):
        pass
    dt = (time.time() - t0) / 1000
    # Rough estimate: each step takes ~10-50 microseconds depending on n
    # We'll just run until time budget is exhausted

    step = 0
    # Use geometric cooling: T = T_start * (T_end/T_start)^(step/total_steps)
    # But we don't know total_steps, so use time-based cooling
    # T(t) = T_start * exp(-cooling_rate * elapsed)
    cooling_rate = 0.05  # per second

    while time.time() < t_end:
        elapsed = time.time() - (t_end - time_budget)
        T = T_start * np.exp(-cooling_rate * elapsed)
        if T < T_end:
            T = T_end

        # Pick a random triple to flip
        t = rng.randint(m)
        cur = colors[t]
        new = 1 - cur
        delta_red = 1 if new == 0 else -1
        delta_blue = 1 if new == 1 else -1

        # Compute delta energy
        dq = 0
        aq = tri_quads_np[t]
        if len(aq) > 0:
            old_q = quad_red[aq]
            new_q = old_q + delta_red
            # Contribution to energy from these quads
            old_contrib = np.maximum(old_q - 3, 0)
            new_contrib = np.maximum(new_q - 3, 0)
            dq = int(np.sum(new_contrib - old_contrib))

        dp = 0
        ap = tri_pents_np[t]
        if len(ap) > 0:
            old_p = pent_blue[ap]
            new_p = old_p + delta_blue
            old_contrib = np.maximum(old_p - 9, 0)
            new_contrib = np.maximum(new_p - 9, 0)
            dp = int(np.sum(new_contrib - old_contrib))

        delta_E = WQ * dq + WP * dp

        # Accept or reject
        if delta_E <= 0 or rng.random() < np.exp(-delta_E / max(T, 1e-9)):
            colors[t] = new
            if len(aq) > 0:
                quad_red[aq] += delta_red
            if len(ap) > 0:
                pent_blue[ap] += delta_blue
            energy += delta_E
            # Update violation counts
            # Redo counts only occasionally for efficiency, or track incrementally
            # For simplicity, recompute at end or every N steps
        step += 1

        # Recompute exact violation counts every 10000 steps to check for validity
        if step % 10000 == 0:
            red_k4 = int(np.sum(quad_red == 4))
            blue_k5 = int(np.sum(pent_blue == 10))
            if red_k4 == 0 and blue_k5 == 0:
                return colors.tolist(), 0, 0

    red_k4 = int(np.sum(quad_red == 4))
    blue_k5 = int(np.sum(pent_blue == 10))
    return colors.tolist(), red_k4, blue_k5


def _random_valid(n, time_budget, seed=0):
    """Try to find a valid colouring on n vertices via simulated annealing."""
    triples, m, quads, pents, tri_quads, tri_pents = _build_data(n)

    t_end = _START + time_budget
    attempts = 0
    while time.time() < t_end:
        attempts += 1
        remaining = t_end - time.time()
        if remaining < 1.0:
            break
        # Scale SA time with remaining budget
        sa_time = min(remaining * 0.4, 5.0)

        # Random init with varying bias
        p_red = 0.3 + 0.2 * random.random()
        colors = (np.random.rand(m) < p_red).astype(np.int8).tolist()
        colors, r4, b5 = _simulated_annealing(colors, quads, pents, tri_quads, tri_pents, sa_time, seed=seed + attempts)
        if r4 == 0 and b5 == 0:
            return colors
    return None


def construct():
    deadline = _START + _DEADLINE - 5.0  # safety margin
    best_n = 9
    best_colors = None

    # Seed with the known small valid construction
    n0 = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n0), 3)]
    best_n = n0

    # Try to grow n from 10 upward, using time budget
    for n in range(10, 41):
        if time.time() >= deadline:
            break
        remaining = deadline - time.time()
        if n <= 15:
            budget = min(remaining, 12.0)
        elif n <= 20:
            budget = min(remaining, 18.0)
        elif n <= 25:
            budget = min(remaining, 25.0)
        elif n <= 30:
            budget = min(remaining, 35.0)
        else:
            budget = min(remaining, 45.0)
        if budget < 2.0:
            break
        colors = _random_valid(n, budget, seed=n)
        if colors is not None:
            best_n = n
            best_colors = colors
        if time.time() >= deadline:
            break

    return {'n': best_n, 'colors': best_colors}
