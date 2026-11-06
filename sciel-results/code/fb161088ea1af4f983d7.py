"""Broadband antireflection coating design via global DE + SLSQP polish.

New strategy: fixed-length parameterization (20 slots) where each slot is a
(material, thickness) pair and unused slots are zero-thickness air layers
(n=1.0, physically inert).  A global differential-evolution search over the
continuous [0,1] slot variables (first 20 = thicknesses, next 20 = material
assignments in (0,1)) is followed by a scipy SLSQP interior-point polish on the
best candidates.  This is a different search family (population global +
interior-point) rather than a topology scan + Nelder-Mead.
"""
import time
import numpy as np

# EVOLVE-BLOCK-START

MAXL = 20
MAXT = 1000.0
MAXD = 400.0
NAMB = 1.0
NSUB = 1.52
NL = 1.38
NH = 2.35
LAMMIN = 400.0
LAMMAX = 800.0
NW = 401


def _reflectance(labels, thicknesses):
    """Mean normal-incidence reflectance (%) via characteristic-matrix method."""
    lam = np.linspace(LAMMIN, LAMMAX, NW)
    n = np.array([NL if l == 'L' else NH for l in labels], dtype=float)
    d = np.array(thicknesses, dtype=float)
    m = len(labels)
    beta = 2.0 * np.pi * n * d
    M00 = np.ones(NW, dtype=complex); M01 = np.zeros(NW, dtype=complex)
    M10 = np.zeros(NW, dtype=complex); M11 = np.ones(NW, dtype=complex)
    for i in range(m):
        delta = beta[i] / lam
        c = np.cos(delta); sn = np.sin(delta)
        b = 1j * sn / n[i]
        cc = 1j * n[i] * sn
        M00, M01, M10, M11 = (M00 * c + M01 * cc, M00 * b + M01 * c,
                              M10 * c + M11 * cc, M10 * b + M11 * c)
    Y = M10 / M00
    r = (NAMB - Y) / (NAMB + Y)
    return float(np.mean(np.abs(r) ** 2) * 100.0)


def _decode(v):
    """Decode a 40-dim vector -> (labels, thicknesses).  Air slots are inert."""
    d = np.clip(v[:MAXL], 0.0, MAXD)
    mat = v[MAXL:2 * MAXL]
    labels, th = [], []
    for i in range(MAXL):
        if d[i] > 1e-6:
            labels.append('H' if mat[i] > 0.5 else 'L')
            th.append(float(d[i]))
    return labels, th


def _score(v):
    labels, th = _decode(v)
    tot = sum(th)
    if tot > MAXT:
        s = MAXT / tot
        th = [t * s for t in th]
    return _reflectance(labels, th)


def _polish(v0, deadline, iters=200, nw=401):
    """SLSQP interior-point polish on continuous slot vector."""
    from scipy.optimize import minimize

    def obj(x):
        d = np.clip(x[:MAXL], 0.0, MAXD)
        mat = np.where(x[MAXL:2 * MAXL] > 0.5, NH, NL)
        labels = []
        for i in range(MAXL):
            if d[i] > 1e-6:
                labels.append('H' if x[MAXL + i] > 0.5 else 'L')
        d = d[d > 1e-6]
        th = list(d)
        if not th:
            return 4.26
        tot = sum(th)
        if tot > MAXT:
            s = MAXT / tot
            th = [t * s for t in th]
        lam = np.linspace(LAMMIN, LAMMAX, nw)
        n = np.array([mat[i] for i in range(MAXL) if d[i] > 1e-6], dtype=float)
        dd = np.array(th, dtype=float)
        beta = 2.0 * np.pi * n * dd
        M00 = np.ones(nw, dtype=complex); M01 = np.zeros(nw, dtype=complex)
        M10 = np.zeros(nw, dtype=complex); M11 = np.ones(nw, dtype=complex)
        for i in range(len(labels)):
            delta = beta[i] / lam
            c = np.cos(delta); sn = np.sin(delta)
            b = 1j * sn / n[i]
            cc = 1j * n[i] * sn
            M00, M01, M10, M11 = (M00 * c + M01 * cc, M00 * b + M01 * c,
                                  M10 * c + M11 * cc, M10 * b + M11 * c)
        Y = M10 / M00
        r = (NAMB - Y) / (NAMB + Y)
        return float(np.mean(np.abs(r) ** 2) * 100.0)

    def cons_total(x):
        d = np.clip(x[:MAXL], 0.0, MAXD)
        return MAXT - d.sum()

    cons = [{'type': 'ineq', 'fun': cons_total}]
    bounds = [(0.0, MAXD)] * MAXL + [(0.0, 1.0)] * MAXL

    try:
        res = minimize(obj, v0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': iters, 'ftol': 1e-8})
        if res.success or res.fun < obj(v0):
            return np.clip(res.x, 0.0, None), float(res.fun)
    except Exception:
        pass
    return v0, obj(v0)


def design(problem):
    start = time.time()
    deadline = start + 135.0  # safety margin under 150s wall-clock

    # ---- Global differential-evolution search over continuous slot vector ----
    rng = np.random.default_rng(2024)
    pop_size = 60
    pop = np.zeros((pop_size, 2 * MAXL))
    for i in range(pop_size):
        # thicknesses: random up to max, biased small; materials: random
        pop[i, :MAXL] = rng.uniform(0.0, MAXD, MAXL) * 0.6
        pop[i, MAXL:] = rng.uniform(0.0, 1.0, MAXL)
    # seed a few good structured individuals
    centre = 0.5 * (LAMMIN + LAMMAX)
    for k, (start_m, nlay) in enumerate([('L', 9), ('H', 11), ('L', 13)]):
        v = np.zeros(2 * MAXL)
        for i in range(nlay):
            m = start_m if i % 2 == 0 else ('H' if start_m == 'L' else 'L')
            v[i] = centre / (4.0 * (NL if m == 'L' else NH))
            v[MAXL + i] = 1.0 if m == 'H' else 0.0
        pop[k] = v

    def de_step(pop, it):
        F = 0.8
        CR = 0.9
        newpop = np.empty_like(pop)
        fvals = np.array([_score(p) for p in pop])
        for i in range(pop_size):
            # DE/rand/1
            a, b, c = rng.choice(pop_size, 3, replace=False)
            trial = pop[a] + F * (pop[b] - pop[c])
            mask = rng.random(2 * MAXL) < CR
            mask[rng.integers(2 * MAXL)] = True
            trial = np.where(mask, trial, pop[i])
            trial = np.clip(trial, 0.0, None)
            ft = _score(trial)
            if ft < fvals[i]:
                newpop[i] = trial
                fvals[i] = ft
            else:
                newpop[i] = pop[i]
        return newpop, fvals

    best_v, best_s = None, float('inf')
    for it in range(40):
        if time.time() > deadline - 40.0:
            break
        pop, fvals = de_step(pop, it)
        bi = int(np.argmin(fvals))
        if fvals[bi] < best_s:
            best_s = fvals[bi]
            best_v = pop[bi].copy()

    # ---- Polish top candidates with SLSQP ----
    order = np.argsort(fvals)
    polished = []
    for idx in order[:6]:
        if time.time() > deadline - 15.0:
            break
        v, s = _polish(pop[idx], deadline, iters=120, nw=NW)
        polished.append((s, v))
    if best_v is not None:
        v, s = _polish(best_v, deadline, iters=200, nw=NW)
        polished.append((s, v))

    polished.sort(key=lambda x: x[0])
    if polished:
        best_s, best_v = polished[0]

    # ---- Final decode + constraint enforcement ----
    labels, th = _decode(best_v)
    tot = sum(th)
    if tot > MAXT:
        s = MAXT / tot
        th = [t * s for t in th]
    th = [max(0.0, min(t, MAXD)) for t in th]
    if not labels:
        labels, th = ['L'], [centre / (4.0 * NL)]
    return labels, th

# EVOLVE-BLOCK-END
