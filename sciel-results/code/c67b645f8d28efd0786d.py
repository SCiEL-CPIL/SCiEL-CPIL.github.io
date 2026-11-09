import time
import numpy as np

N = 51975  # 3^3 * 5^2 * 7 * 11, odd


def _divisors(n):
    ds = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            ds.append(i)
            if i != n // i:
                ds.append(n // i)
        i += 1
    return ds


# All odd divisors > 1 of N (N is odd, so every divisor is odd).
MODS = sorted(d for d in _divisors(N) if d > 1)
K = {m: N // m for m in MODS}  # fiber size for modulus m


def _cov(assign):
    """Boolean array: covered[x] = exists (m,r) with x == r (mod m)."""
    cov = np.zeros(N, dtype=bool)
    for m in MODS:
        cov[assign[m]::m] = True
    return cov


def _uncovered(assign):
    return int((~_cov(assign)).sum())


def _cd(assign, max_passes=200):
    """Coordinate descent: for each modulus pick residue minimizing uncovered."""
    for _ in range(max_passes):
        improved = False
        for m in MODS:
            # coverage contributed by every other modulus
            cov = np.zeros(N, dtype=bool)
            for m2 in MODS:
                if m2 != m:
                    cov[assign[m2]::m2] = True
            unc = ~cov
            # for each residue r mod m, count uncovered positions in that fiber
            cnt = unc.reshape(K[m], m).sum(axis=0)
            br = int(np.argmax(cnt))
            if br != assign[m]:
                assign[m] = br
                improved = True
        if not improved:
            break
    return assign


def _cd_limited(assign, max_passes=3):
    """Limited coordinate descent for faster SA moves."""
    for _ in range(max_passes):
        improved = False
        for m in MODS:
            cov = np.zeros(N, dtype=bool)
            for m2 in MODS:
                if m2 != m:
                    cov[assign[m2]::m2] = True
            unc = ~cov
            cnt = unc.reshape(K[m], m).sum(axis=0)
            br = int(np.argmax(cnt))
            if br != assign[m]:
                assign[m] = br
                improved = True
        if not improved:
            break
    return assign


def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin well under the 150s limit
    rng = np.random.default_rng(20240517)

    best_assign = None
    best_unc = N

    def consider(a):
        nonlocal best_assign, best_unc
        u = _uncovered(a)
        if u < best_unc:
            best_unc = u
            best_assign = dict(a)
        return u

    # Deterministic baseline: all residues 0, then coordinate descent.
    a0 = {m: 0 for m in MODS}
    _cd(a0)
    consider(a0)

    # Try greedy with different orderings
    for order_seed in [1, 2, 3, 4, 5]:
        rng_g = np.random.default_rng(order_seed)
        mods_shuffled = list(MODS)
        rng_g.shuffle(mods_shuffled)
        a_g = {m: 0 for m in MODS}
        # Greedy: process moduli in shuffled order, pick best residue given current coverage
        for m in mods_shuffled:
            cov = np.zeros(N, dtype=bool)
            for m2 in MODS:
                if m2 != m:
                    cov[a_g[m2]::m2] = True
            unc = ~cov
            cnt = unc.reshape(K[m], m).sum(axis=0)
            a_g[m] = int(np.argmax(cnt))
        _cd(a_g)
        consider(a_g)

    # Random restarts + coordinate descent.
    while time.time() < deadline - 40.0:
        a = {m: int(rng.integers(0, m)) for m in MODS}
        _cd(a)
        u = consider(a)
        if u == 0:
            break

    # Simulated annealing on the best solution to escape local minima.
    if best_assign is not None and time.time() < deadline - 5.0:
        a = dict(best_assign)
        cur_u = best_unc
        T = 5.0
        T_min = 0.01
        while time.time() < deadline - 3.0 and T > T_min:
            # Random single-residue perturbation
            m = MODS[int(rng.integers(0, len(MODS)))]
            a[m] = int(rng.integers(0, m))
            _cd_limited(a, max_passes=2)
            u = _uncovered(a)
            if u < cur_u:
                cur_u = u
                if u < best_unc:
                    best_unc = u
                    best_assign = dict(a)
                    if u == 0:
                        break
            else:
                delta = u - cur_u
                if rng.random() < np.exp(-delta / max(T, 1e-9)):
                    cur_u = u
            T *= 0.998

        # Final polish: full CD on best
        if best_assign is not None and time.time() < deadline - 1.0:
            a_final = dict(best_assign)
            _cd(a_final, max_passes=300)
            consider(a_final)

    return [[m, int(best_assign[m])] for m in MODS]
