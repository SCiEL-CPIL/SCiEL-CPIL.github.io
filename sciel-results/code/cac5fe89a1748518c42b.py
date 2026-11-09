"""
Broadband antireflection coating design: fast kernel + differential evolution + local polish.

Strategy (structural change from the seed):
  * Very fast reflectance kernel: precompute the phase matrix once per wavelength
    set, then evaluate many candidate stacks cheaply (vectorized over the layer
    loop using complex exponentials).
  * Global search: differential evolution over a mixed representation
    (material pattern encoded as binary, thicknesses continuous), with multiple
    restarts and a deadline guard.
  * Local polish: Nelder-Mead on the best few individuals' thicknesses only
    (materials fixed) to squeeze out the last percent.
"""
import time
import numpy as np


def _make_kernel(problem):
    """Precompute wavelength-dependent quantities for fast reflectance eval."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    band = problem["band_nm"]
    n_wl = 401
    wl = np.linspace(band[0], band[1], n_wl)
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    n_mat_arr = np.array([nL, nH])  # index 0 -> L, 1 -> H
    # phase[i_layer][j_wl] = 2*pi*n_i*d_i/wl  -- d_i filled at eval time
    # Precompute 2*pi/wl
    inv_wl = 2.0 * np.pi / wl
    return {
        "n0": n0, "ns": ns, "wl": wl, "n_wl": n_wl,
        "nL": nL, "nH": nH, "inv_wl": inv_wl,
        "bare": 100.0 * ((ns - n0) / (ns + n0)) ** 2,
    }


def _reflectance_fast(mats_idx, thicknesses, K):
    """mats_idx: int array 0=L,1=H; thicknesses: float array same length.
    Returns mean R in percent."""
    N = len(mats_idx)
    if N == 0:
        return K["bare"]
    n_wl = K["n_wl"]
    n0 = K["n0"]; ns = K["ns"]
    nL = K["nL"]; nH = K["nH"]
    inv_wl = K["inv_wl"]

    # Per-layer refractive index
    n_i = np.where(mats_idx == 0, nL, nH)  # shape (N,)
    d = np.asarray(thicknesses, dtype=float)  # shape (N,)

    # delta[i,j] = 2*pi*n_i*d_i/wl_j
    delta = (n_i[:, None] * d[:, None]) * inv_wl[None, :]  # (N, n_wl)

    # Start with identity characteristic matrix (A=1,B=0,C=0,D=1) over wavelengths
    A = np.ones(n_wl, dtype=complex)
    B = np.zeros(n_wl, dtype=complex)
    C = np.zeros(n_wl, dtype=complex)
    D = np.ones(n_wl, dtype=complex)

    for i in range(N):
        c = np.cos(delta[i])
        s = np.sin(delta[i])
        n = n_i[i]
        A_new = A * c + 1j * B * n * s
        B_new = 1j * A * s / n + B * c
        C_new = C * c + 1j * D * n * s
        D_new = 1j * C * s / n + D * c
        A, B, C, D = A_new, B_new, C_new, D_new

    num = n0 * (A + B * ns) - (C + D * ns)
    den = n0 * (A + B * ns) + (C + D * ns)
    r = num / den
    R = np.abs(r) ** 2
    return float(np.mean(R)) * 100.0


def _valid(materials, thicknesses, problem):
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < 0.0 or t > max_layer_nm:
            return False
    if sum(thicknesses) > max_total_nm + 1e-9:
        return False
    return True


def _clamp_thicknesses(thicknesses, problem):
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    clamped = [max(0.0, min(max_layer_nm, float(t))) for t in thicknesses]
    total = sum(clamped)
    if total > max_total_nm:
        scale = max_total_nm / total
        clamped = [t * scale for t in clamped]
    return clamped


def _decode(x, problem):
    """Decode a candidate vector into (materials, thicknesses).
    x layout: [n_layers (int 0..max_layers)] then for each of first max_layers
    positions: [mat_bit (0/1), thickness (0..max_layer_nm)].
    Only the first n_layers positions are used.
    """
    max_layers = problem["max_layers"]
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]

    n_layers = int(round(x[0]))
    n_layers = max(0, min(max_layers, n_layers))

    materials = []
    thicknesses = []
    for i in range(n_layers):
        base = 1 + 2 * i
        bit = x[base]
        th = x[base + 1]
        mat = 'L' if bit < 0.5 else 'H'
        th = max(0.0, min(max_layer_nm, float(th)))
        materials.append(mat)
        thicknesses.append(th)

    # Enforce total thickness constraint by scaling
    total = sum(thicknesses)
    if total > max_total_nm and total > 0:
        scale = max_total_nm / total
        thicknesses = [t * scale for t in thicknesses]

    return materials, thicknesses


def _encode(materials, thicknesses, problem):
    max_layers = problem["max_layers"]
    max_layer_nm = problem["max_layer_nm"]
    x = np.zeros(1 + 2 * max_layers)
    x[0] = len(materials)
    for i, (m, t) in enumerate(zip(materials, thicknesses)):
        base = 1 + 2 * i
        x[base] = 0.0 if m == 'L' else 1.0
        x[base + 1] = t
    return x


def _objective(x, problem, K):
    materials, thicknesses = _decode(x, problem)
    if not _valid(materials, thicknesses, problem):
        return 1e6
    mats_idx = np.array([0 if m == 'L' else 1 for m in materials], dtype=int)
    return _reflectance_fast(mats_idx, thicknesses, K)


def _nelder_mead_thicknesses(materials, thicknesses, problem, K, deadline, maxiter=80):
    """Polish thicknesses for a fixed material pattern."""
    if not materials:
        return thicknesses, _reflectance_fast(np.array([], dtype=int), np.array([]), K)

    start = time.time()
    n = len(thicknesses)
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]

    def obj(t):
        t = _clamp_thicknesses(t, problem)
        if not _valid(materials, t, problem):
            return 1e6
        mats_idx = np.array([0 if m == 'L' else 1 for m in materials], dtype=int)
        return _reflectance_fast(mats_idx, t, K)

    x0 = np.array(thicknesses, dtype=float)
    x0 = _clamp_thicknesses(x0, problem)

    # Build simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.1 * max_layer_nm
        xi[i] += step
        xi[i] = np.clip(xi[i], 0.0, max_layer_nm)
        simplex.append(xi)

    fvals = np.array([obj(simplex[i]) for i in range(n + 1)])
    best_idx = int(np.argmin(fvals))
    fbest = fvals[best_idx]
    xbest = simplex[best_idx].copy()

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

    for _ in range(maxiter):
        if time.time() > deadline:
            break
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = fvals[order]
        if fvals[0] < fbest:
            fbest = fvals[0]
            xbest = simplex[0].copy()
        if np.allclose(fvals, fvals[0], rtol=1e-7, atol=1e-7):
            break
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        x_r = centroid + alpha * (centroid - worst)
        x_r = np.clip(x_r, 0.0, max_layer_nm)
        f_r = obj(x_r)
        if f_r < fvals[0]:
            x_e = centroid + gamma * (centroid - worst)
            x_e = np.clip(x_e, 0.0, max_layer_nm)
            f_e = obj(x_e)
            if f_e < f_r:
                simplex[-1] = x_e; fvals[-1] = f_e
            else:
                simplex[-1] = x_r; fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r; fvals[-1] = f_r
        else:
            x_c = centroid + rho * (worst - centroid)
            x_c = np.clip(x_c, 0.0, max_layer_nm)
            f_c = obj(x_c)
            if f_c < fvals[-1]:
                simplex[-1] = x_c; fvals[-1] = f_c
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], 0.0, max_layer_nm)
                    fvals[i] = obj(simplex[i])
    t_opt = _clamp_thicknesses(xbest, problem)
    f_opt = obj(t_opt)
    return t_opt, f_opt


def _differential_evolution(problem, K, deadline, pop_size=40, max_gen=60, F=0.8, CR=0.9, seed=0):
    """DE over the full design space. Returns best (materials, thicknesses, score)."""
    rng = np.random.default_rng(seed)
    max_layers = problem["max_layers"]
    max_layer_nm = problem["max_layer_nm"]
    dim = 1 + 2 * max_layers

    # Bounds
    lo = np.zeros(dim)
    hi = np.zeros(dim)
    hi[0] = max_layers
    for i in range(max_layers):
        base = 1 + 2 * i
        lo[base] = 0.0; hi[base] = 1.0
        lo[base + 1] = 0.0; hi[base + 1] = max_layer_nm

    # Initial population
    pop = np.zeros((pop_size, dim))
    for i in range(pop_size):
        n_layers = int(rng.integers(2, max_layers + 1))
        pop[i, 0] = n_layers
        for j in range(max_layers):
            base = 1 + 2 * j
            pop[i, base] = rng.random()  # material bit
            # thickness: start near quarter-wave for that material
            mat = 'L' if pop[i, base] < 0.5 else 'H'
            n = K["nL"] if mat == 'L' else K["nH"]
            centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
            qw = centre / (4.0 * n)
            pop[i, base + 1] = qw * rng.uniform(0.7, 1.3)
        pop[i] = np.clip(pop[i], lo, hi)

    fvals = np.array([_objective(pop[i], problem, K) for i in range(pop_size)])
    best_idx = int(np.argmin(fvals))
    best_x = pop[best_idx].copy()
    best_f = fvals[best_idx]

    for gen in range(max_gen):
        if time.time() > deadline:
            break
        new_pop = [pop[best_idx].copy()]
        for i in range(pop_size):
            if time.time() > deadline:
                break
            # pick three distinct indices
            a, b, c = rng.choice(pop_size, size=3, replace=False)
            mutant = pop[a] + F * (pop[b] - pop[c])
            mutant = np.clip(mutant, lo, hi)
            # crossover
            trial = pop[i].copy()
            jrand = int(rng.integers(0, dim))
            for j in range(dim):
                if rng.random() < CR or j == jrand:
                    trial[j] = mutant[j]
            trial = np.clip(trial, lo, hi)
            f_trial = _objective(trial, problem, K)
            if f_trial <= fvals[i]:
                new_pop.append(trial)
                fvals[i] = f_trial
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
            else:
                new_pop.append(pop[i].copy())
        pop = np.array(new_pop)
    # Re-evaluate best to be safe
    materials, thicknesses = _decode(best_x, problem)
    if not _valid(materials, thicknesses, problem):
        # fallback
        return None
    mats_idx = np.array([0 if m == 'L' else 1 for m in materials], dtype=int)
    f = _reflectance_fast(mats_idx, thicknesses, K)
    return materials, thicknesses, f


def _quarter_wave_seed(problem, K, n_layers, start_mat):
    centre = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])
    qwL = centre / (4.0 * K["nL"])
    qwH = centre / (4.0 * K["nH"])
    materials = []
    thicknesses = []
    for i in range(n_layers):
        mat = start_mat if i % 2 == 0 else ('H' if start_mat == 'L' else 'L')
        materials.append(mat)
        thicknesses.append(qwL if mat == 'L' else qwH)
    return materials, thicknesses


def design(problem):
    start_time = time.time()
    # Evaluator kills at 150s wall-clock. Use a safety margin.
    deadline = start_time + 130.0

    K = _make_kernel(problem)
    band = problem["band_nm"]
    centre = 0.5 * (band[0] + band[1])
    qwL = centre / (4.0 * K["nL"])
    qwH = centre / (4.0 * K["nH"])

    best_materials = ['L']
    best_thicknesses = [qwL]
    best_score = _reflectance_fast(np.array([0], dtype=int), np.array([qwL]), K)

    # 1) Quick baseline: a few quarter-wave alternating stacks, polished.
    for n_layers in [2, 4, 6, 8]:
        if time.time() > deadline - 30:
            break
        for start_mat in ['L', 'H']:
            mats, thicks = _quarter_wave_seed(problem, K, n_layers, start_mat)
            thicks = _clamp_thicknesses(thicks, problem)
            if not _valid(mats, thicks, problem):
                continue
            mats_idx = np.array([0 if m == 'L' else 1 for m in mats], dtype=int)
            f = _reflectance_fast(mats_idx, thicks, K)
            # local polish
            if time.time() < deadline - 10:
                thicks2, f2 = _nelder_mead_thicknesses(mats, thicks, problem, K, deadline - 5, maxiter=60)
                if f2 < f:
                    f = f2
                    thicks = thicks2
            if f < best_score:
                best_score = f
                best_materials = mats
                best_thicknesses = thicks

    # 2) Global differential evolution (multiple restarts).
    # Allocate the bulk of the remaining time to DE.
    de_budget_end = deadline - 20.0  # leave 20s for final polish + safety
    n_restarts = 3
    per_restart = max(1.0, (de_budget_end - time.time()) / n_restarts)

    for r in range(n_restarts):
        restart_deadline = min(de_budget_end, time.time() + per_restart)
        if time.time() > restart_deadline - 1.0:
            break
        try:
            res = _differential_evolution(
                problem, K, restart_deadline,
                pop_size=35, max_gen=50, F=0.8, CR=0.9, seed=100 + r
            )
        except Exception:
            res = None
        if res is not None:
            mats, thicks, f = res
            if f < best_score:
                best_score = f
                best_materials = mats
                best_thicknesses = thicks

    # 3) Final local polish on the global best (and a couple of quarter-wave seeds).
    if time.time() < deadline - 5:
        mats, thicks, f = _nelder_mead_thicknesses(
            best_materials, best_thicknesses, problem, K, deadline - 3, maxiter=120
        )
        if f < best_score:
            best_score = f
            best_materials = mats
            best_thicknesses = thicks

    # Also polish a couple of strong quarter-wave structures in case DE found a
    # material pattern that's good but thicknesses are only locally tuned.
    for n_layers in [4, 6, 8]:
        if time.time() > deadline - 3:
            break
        for start_mat in ['L', 'H']:
            mats, thicks = _quarter_wave_seed(problem, K, n_layers, start_mat)
            thicks = _clamp_thicknesses(thicks, problem)
            if not _valid(mats, thicks, problem):
                continue
            thicks2, f2 = _nelder_mead_thicknesses(mats, thicks, problem, K, deadline - 2, maxiter=80)
            if f2 < best_score:
                best_score = f2
                best_materials = mats
                best_thicknesses = thicks2

    # Final validation
    best_thicknesses = _clamp_thicknesses(best_thicknesses, problem)
    if not _valid(best_materials, best_thicknesses, problem):
        best_materials = ['L']
        best_thicknesses = [qwL]
        best_score = _reflectance_fast(np.array([0], dtype=int), np.array([qwL]), K)

    return best_materials, best_thicknesses
