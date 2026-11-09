"""Broadband antireflection coating design (400-800 nm) on glass.

Structural change:
  - Joint optimization of layer *sequence* (L/H labels) and thicknesses.
  - Coarse global search via a lightweight evolutionary / hill-climb over
    (pattern, thicknesses) pairs, followed by high-fidelity local polishing.
  - Fast exact characteristic-matrix model vectorized over 401 wavelengths.
"""

import time
import random
import numpy as np


def _mean_reflectance(materials, thicknesses, problem):
    """Exact mean reflectance (%) over the band using the characteristic-matrix method."""
    if not materials:
        n_s = problem["substrate_index"]
        n_0 = problem["ambient_index"]
        return 100.0 * ((n_s - n_0) / (n_s + n_0)) ** 2

    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    n_mat = problem["materials"]
    ns = np.array([n_mat[m] for m in materials], dtype=float)
    ds = np.asarray(thicknesses, dtype=float)

    # Optical thickness vector per layer
    n_ds = ns * ds
    delta = 2.0 * np.pi * n_ds[:, None] / wavs[None, :]
    c = np.cos(delta)
    s = np.sin(delta)

    A = np.ones_like(wavs)
    B = np.zeros_like(wavs)
    C = np.zeros_like(wavs)
    D = np.ones_like(wavs)

    i = 1.0j
    for j in range(len(materials)):
        nj = ns[j]
        cj = c[j]
        sj = s[j]
        A_new = A * cj + i * B * nj * sj
        B_new = i * A * sj / nj + B * cj
        C_new = C * cj + i * D * nj * sj
        D_new = i * C * sj / nj + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new

    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return float(100.0 * np.mean(R))


def _clamp(ths, max_total, max_layer):
    ths = np.clip(ths, 0.0, max_layer)
    tot = ths.sum()
    if tot > max_total:
        ths *= max_total / tot
        ths = np.clip(ths, 0.0, max_layer)
    return ths


def _obj_fn(problem, max_total, max_layer):
    wavs = np.linspace(problem["band_nm"][0], problem["band_nm"][1], 401)
    n_mat = problem["materials"]
    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]

    def objective(materials, thicknesses):
        if not materials:
            return 100.0 * ((n_s - n_0) / (n_s + n_0)) ** 2
        ns = np.array([n_mat[m] for m in materials], dtype=float)
        ds = np.asarray(thicknesses, dtype=float)
        n_ds = ns * ds
        delta = 2.0 * np.pi * n_ds[:, None] / wavs[None, :]
        c = np.cos(delta)
        s = np.sin(delta)

        A = np.ones_like(wavs)
        B = np.zeros_like(wavs)
        C = np.zeros_like(wavs)
        D = np.ones_like(wavs)
        i = 1.0j
        for j in range(len(materials)):
            nj = ns[j]
            cj = c[j]
            sj = s[j]
            A, B, C, D = (
                A * cj + i * B * nj * sj,
                i * A * sj / nj + B * cj,
                C * cj + i * D * nj * sj,
                i * C * sj / nj + D * cj,
            )
        num = n_0 * (A + B * n_s) - (C + D * n_s)
        den = n_0 * (A + B * n_s) + (C + D * n_s)
        r = num / den
        return float(100.0 * np.mean(np.abs(r) ** 2))

    return objective


def _nelder_mead_1d(objective, x0, lo, hi, max_iters=200, tol=1e-9):
    """1-D Nelder-Mead (golden-section-ish) for a single variable."""
    # Use scipy if available, else simple golden section
    try:
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(objective, bounds=(lo, hi), method='bounded',
                              options={'maxiter': max_iters, 'xatol': tol})
        return res.x, res.fun
    except Exception:
        # Golden section fallback
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        a, b = lo, hi
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc, fd = objective(c), objective(d)
        for _ in range(max_iters):
            if b - a < tol:
                break
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc = objective(c)
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd = objective(d)
        x = 0.5 * (a + b)
        return x, objective(x)


def _local_polish(problem, materials, thicknesses, max_total, max_layer, time_budget=2.0):
    """Polish thicknesses for a fixed pattern using coordinate descent + Nelder-Mead."""
    obj = _obj_fn(problem, max_total, max_layer)
    ths = np.array(thicknesses, dtype=float)
    ths = _clamp(ths, max_total, max_layer)
    n = len(materials)
    if n == 0:
        return materials, ths

    start = time.time()
    best_score = obj(materials, ths)
    best_ths = ths.copy()

    # Coordinate descent with a few passes
    for _ in range(3):
        if time.time() - start > time_budget:
            break
        for i in range(n):
            if time.time() - start > time_budget:
                break
            def f1(x):
                t = best_ths.copy()
                t[i] = x
                t = _clamp(t, max_total, max_layer)
                return obj(materials, t)
            x_new, f_new = _nelder_mead_1d(f1, best_ths[i], 0.0, max_layer, max_iters=50, tol=1e-6)
            t_new = best_ths.copy()
            t_new[i] = x_new
            t_new = _clamp(t_new, max_total, max_layer)
            f_new = obj(materials, t_new)
            if f_new < best_score:
                best_score = f_new
                best_ths = t_new

    # Multi-start Nelder-Mead on full vector (limited)
    if time.time() - start < time_budget * 0.5:
        rng = np.random.RandomState(0)
        for trial in range(5):
            if time.time() - start > time_budget:
                break
            init = best_ths + rng.uniform(-0.1, 0.1, size=n) * max_layer
            init = _clamp(init, max_total, max_layer)

            def full_obj(v):
                v = _clamp(v, max_total, max_layer)
                return obj(materials, v)

            try:
                from scipy.optimize import minimize
                res = minimize(full_obj, init, method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.fun < best_score:
                    best_score = res.fun
                    best_ths = _clamp(res.x, max_total, max_layer)
            except Exception:
                pass

    return materials, best_ths


def _mutate_pattern(pattern, max_layers, rng):
    """Randomly mutate a layer sequence: flip, insert, delete."""
    p = list(pattern)
    op = rng.random()
    if not p:
        return [rng.choice("LH")]
    if op < 0.5:
        # Flip a random layer
        i = rng.randint(0, len(p) - 1)
        p[i] = 'H' if p[i] == 'L' else 'L'
    elif op < 0.75 and len(p) < max_layers:
        # Insert
        i = rng.randint(0, len(p))
        p.insert(i, rng.choice("LH"))
    elif op < 0.9 and len(p) > 1:
        # Delete
        i = rng.randint(0, len(p) - 1)
        p.pop(i)
    else:
        # Swap two
        i, j = rng.sample(range(len(p)), 2)
        p[i], p[j] = p[j], p[i]
    return p


def _random_pattern(rng, max_layers):
    length = rng.randint(2, min(max_layers, 14))
    return [rng.choice("LH") for _ in range(length)]


def _quarter_wave_init(pattern, problem, max_total, max_layer):
    wl = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    n_mat = problem["materials"]
    ths = [wl / (4.0 * n_mat[m]) for m in pattern]
    return _clamp(np.array(ths), max_total, max_layer)


def design(problem):
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]

    obj = _obj_fn(problem, max_total, max_layer)
    rng = random.Random(12345)
    np_rng = np.random.RandomState(12345)

    # Initial population of diverse patterns
    population = []

    # Quarter-wave single layer
    for m in ["L", "H"]:
        population.append(([m], _quarter_wave_init([m], problem, max_total, max_layer)))

    # Symmetric patterns
    for length in range(2, min(max_layers, 12) + 1, 2):
        half = length // 2
        for _ in range(4):
            h = [rng.choice("LH") for _ in range(half)]
            pat = h + h[::-1]
            population.append((pat, _quarter_wave_init(pat, problem, max_total, max_layer)))

    # Graded index
    for length in range(3, min(max_layers, 11) + 1):
        for _ in range(3):
            pat = []
            for i in range(length):
                if i < length // 2:
                    pat.append('L' if rng.random() < 0.85 else 'H')
                else:
                    pat.append('H' if rng.random() < 0.85 else 'L')
            population.append((pat, _quarter_wave_init(pat, problem, max_total, max_layer)))

    # Alternating
    for period in [2, 3]:
        for length in range(period, min(max_layers, 14) + 1, period):
            for start in ['L', 'H']:
                pat = [start if (i // period) % 2 == 0 else ('H' if start == 'L' else 'L') for i in range(length)]
                population.append((pat, _quarter_wave_init(pat, problem, max_total, max_layer)))

    # Random
    for _ in range(60):
        pat = _random_pattern(rng, max_layers)
        population.append((pat, _quarter_wave_init(pat, problem, max_total, max_layer)))

    # Evaluate initial population
    scored = []
    for pat, ths in population:
        ths = _clamp(ths, max_total, max_layer)
        score = obj(pat, ths)
        scored.append((pat, ths, score))

    scored.sort(key=lambda x: x[2])
    best_pat, best_ths, best_score = scored[0]

    # Evolutionary search over (pattern, thicknesses)
    pop_size = min(30, len(scored))
    top = scored[:pop_size]

    gen = 0
    while time.time() < deadline - 40:
        gen += 1
        children = []

        for _ in range(pop_size):
            # Select parent (tournament)
            a, b = rng.sample(range(len(top)), 2)
            parent = top[a] if top[a][2] < top[b][2] else top[b]
            pat, ths, _ = parent

            # Mutate pattern
            new_pat = _mutate_pattern(pat, max_layers, rng)

            # Mutate thicknesses
            new_ths = ths + np_rng.uniform(-0.15, 0.15, size=len(new_pat)) * max_layer
            # Resize if pattern length changed
            if len(new_ths) != len(new_pat):
                new_ths = _quarter_wave_init(new_pat, problem, max_total, max_layer)
                new_ths += np_rng.uniform(-0.1, 0.1, size=len(new_ths)) * max_layer
            new_ths = _clamp(new_ths, max_total, max_layer)

            score = obj(new_pat, new_ths)
            children.append((new_pat, new_ths, score))

            if score < best_score:
                best_score = score
                best_pat = new_pat
                best_ths = new_ths.copy()

        # Combine and select next generation
        combined = top + children
        combined.sort(key=lambda x: x[2])
        top = combined[:pop_size]

        # Occasionally add a fresh random individual
        if gen % 5 == 0:
            pat = _random_pattern(rng, max_layers)
            ths = _quarter_wave_init(pat, problem, max_total, max_layer)
            score = obj(pat, ths)
            top.append((pat, ths, score))
            top.sort(key=lambda x: x[2])
            top = top[:pop_size]

    # Final polishing phase
    # Polish the best and a few top candidates
    candidates_to_polish = top[:min(8, len(top))]
    for pat, ths, _ in candidates_to_polish:
        if time.time() > deadline - 10:
            break
        polished_pat, polished_ths = _local_polish(
            problem, pat, ths, max_total, max_layer,
            time_budget=max(0.5, (deadline - 10 - time.time()) / max(1, len(candidates_to_polish)))
        )
        score = obj(polished_pat, polished_ths)
        if score < best_score:
            best_score = score
            best_pat = polished_pat
            best_ths = polished_ths

    # One more polish on the absolute best
    if time.time() < deadline - 5:
        polished_pat, polished_ths = _local_polish(
            problem, best_pat, best_ths, max_total, max_layer, time_budget=5.0
        )
        score = obj(polished_pat, polished_ths)
        if score < best_score:
            best_score = score
            best_pat = polished_pat
            best_ths = polished_ths

    best_ths = _clamp(best_ths, max_total, max_layer)
    return list(best_pat), list(best_ths)
