"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

N = 256
DEADLINE = 115.0  # safety margin before the 150s kill (evaluator uses 120s)


def _count_mono4(c):
    """Exact count of monochromatic 4-term APs in a 0/1 numpy array (0-indexed)."""
    count = 0
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        if max_a <= 0:
            break
        c0 = c[0:max_a]
        c1 = c[d:d + max_a]
        c2 = c[2 * d:2 * d + max_a]
        c3 = c[3 * d:3 * d + max_a]
        mask = (c0 == c1) & (c0 == c2) & (c0 == c3)
        count += int(np.count_nonzero(mask))
    return count


def _thue_morse():
    """Thue-Morse sequence for 0..255."""
    c = np.zeros(N, dtype=np.int8)
    for i in range(N):
        c[i] = bin(i).count('1') % 2
    return c


def _build_ap_lists():
    """For each index i, list of (d, a) for APs a, a+d, a+2d, a+3d containing i."""
    aps = [[] for _ in range(N)]
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        if max_a <= 0:
            break
        for a in range(0, max_a):
            for k in range(4):
                i = a + k * d
                aps[i].append((d, a))
    return aps


def _delta_flip(c, aps, i):
    """Change in mono-4AP count if we flip position i (computed incrementally)."""
    delta = 0
    seen = set()
    ci = c[i]
    for (d, a) in aps[i]:
        key = (d, a)
        if key in seen:
            continue
        seen.add(key)
        v0, v1, v2, v3 = c[a], c[a + d], c[a + 2 * d], c[a + 3 * d]
        before = 1 if (v0 == v1 == v2 == v3) else 0
        nv0, nv1, nv2, nv3 = v0, v1, v2, v3
        if a == i:
            nv0 = 1 - ci
        elif a + d == i:
            nv1 = 1 - ci
        elif a + 2 * d == i:
            nv2 = 1 - ci
        elif a + 3 * d == i:
            nv3 = 1 - ci
        after = 1 if (nv0 == nv1 == nv2 == nv3) else 0
        delta += (after - before)
    return delta


def _local_search_delta(c, aps, rng, deadline, T0, T1, iters_per_phase, phase_iters):
    """Simulated annealing + final descent, using incremental delta. Returns (c, count)."""
    count = _count_mono4(c)
    best_c = c.copy()
    best_count = count
    for phase in range(iters_per_phase):
        T = T0 * (T1 / T0) ** (phase / max(1, iters_per_phase - 1))
        for _ in range(phase_iters):
            if time.time() > deadline:
                return best_c, best_count
            i = int(rng.integers(N))
            d = _delta_flip(c, aps, i)
            if d <= 0:
                c[i] = 1 - c[i]
                count += d
                if count < best_count:
                    best_count = count
                    best_c = c.copy()
            else:
                if T > 0 and rng.random() < np.exp(-d / T):
                    c[i] = 1 - c[i]
                    count += d

    # final deterministic descent polish
    for _ in range(3):
        improved = True
        while improved:
            if time.time() > deadline:
                return best_c, best_count
            improved = False
            for i in range(N):
                d = _delta_flip(c, aps, i)
                if d < 0:
                    c[i] = 1 - c[i]
                    count += d
                    improved = True
                    if count < best_count:
                        best_count = count
                        best_c = c.copy()
    return best_c, best_count


def construct():
    start_time = time.time()
    aps = _build_ap_lists()
    rng = np.random.default_rng(12345)

    starts = []
    starts.append(_thue_morse())
    for _ in range(3):
        starts.append(rng.integers(0, 2, size=N).astype(np.int8))

    best_c = None
    best_count = float('inf')

    for s in starts:
        if time.time() > DEADLINE:
            break
        c, cnt = _local_search_delta(s, aps, rng, DEADLINE, T0=1.0, T1=0.001,
                                     iters_per_phase=20, phase_iters=4000)
        if cnt < best_count:
            best_count = cnt
            best_c = c

    if best_c is None:
        best_c = _thue_morse()
    return list(best_c)
# EVOLVE-BLOCK-END
