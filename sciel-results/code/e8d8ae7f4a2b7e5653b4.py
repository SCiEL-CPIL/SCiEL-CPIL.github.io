"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _characteristic_matrix(n, d, w, n0):
    """Return the 2x2 characteristic matrix for a single layer, vectorized over w."""
    beta = 2.0 * np.pi * n * d / w
    c = np.cos(beta)
    s = np.sin(beta)
    # M = [[cos beta, i sin beta / n], [i n sin beta, cos beta]]
    i = 1j
    M = np.empty((2, 2) + w.shape, dtype=complex)
    M[0, 0] = c
    M[0, 1] = i * s / n
    M[1, 0] = i * n * s
    M[1, 1] = c
    return M

def _reflectance_stack(materials, thicknesses, problem):
    """Mean reflectance (fraction) over the band, computed exactly."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    mats = problem["materials"]
    band = problem["band_nm"]
    w = np.linspace(band[0], band[1], 401)
    # Build total matrix
    Mtot = np.ones((2, 2) + w.shape, dtype=complex)
    for m, d in zip(materials, thicknesses):
        if d <= 0:
            continue
        n = mats[m]
        Mtot = Mtot @ _characteristic_matrix(n, d, w, n0)
    m11, m12, m21, m22 = Mtot[0,0], Mtot[0,1], Mtot[1,0], Mtot[1,1]
    # At normal incidence, admittance Y = m21/m22, reflectance
    Y = m21 / m22
    r = (Y - n0) / (Y + n0)
    R = np.abs(r) ** 2
    return float(np.mean(R))

def _objective(materials, thicknesses, problem):
    if len(materials) == 0:
        # bare glass
        r = (problem["substrate_index"] - problem["ambient_index"]) / (problem["substrate_index"] + problem["ambient_index"])
        return float(r * r)
    return _reflectance_stack(materials, thicknesses, problem)

def _valid(materials, thicknesses, problem):
    if len(materials) > problem["max_layers"]:
        return False
    for d in thicknesses:
        if d < 0 or d > problem["max_layer_nm"]:
            return False
    if sum(thicknesses) > problem["max_total_nm"] + 1e-6:
        return False
    return True

def _quarter_stack(n_layers, lam0, problem):
    """Symmetric alternating H/L stack of n_layers quarter-wave layers at lam0."""
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    dL = lam0 / (4.0 * nL)
    dH = lam0 / (4.0 * nH)
    mats = []
    th = []
    for i in range(n_layers):
        # alternate starting with H
        if i % 2 == 0:
            mats.append("H"); th.append(dH)
        else:
            mats.append("L"); th.append(dL)
    return mats, th

def design(problem):
    t_start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    lam0 = 0.5 * (problem["band_nm"][0] + problem["band_nm"][1])

    # Initial candidate: a few symmetric quarter-wave stacks
    candidates = []
    for nl in [3, 5, 7, 9]:
        mats, th = _quarter_stack(nl, lam0, problem)
        if _valid(mats, th, problem):
            candidates.append((mats, th))

    if not candidates:
        mats, th = _quarter_stack(3, lam0, problem)
        candidates = [(mats, th)]

    # Evaluate and pick best
    best_score = None
    best_mats = candidates[0][0]
    best_th = candidates[0][1]
    for mats, th in candidates:
        s = _objective(mats, th, problem)
        if best_score is None or s < best_score:
            best_score = s
            best_mats = list(mats)
            best_th = list(th)

    # Local search: perturb thicknesses, swap materials, add/remove layers
    rng = np.random.default_rng(0)
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    max_layers = problem["max_layers"]

    def try_candidate(mats, th):
        nonlocal best_score, best_mats, best_th
        if not _valid(mats, th, problem):
            return
        s = _objective(mats, th, problem)
        if best_score is None or s < best_score:
            best_score = s
            best_mats = list(mats)
            best_th = list(th)

    iterations = 0
    while time.time() < DEADLINE:
        iterations += 1
        n = len(best_mats)

        # 1) Perturb each thickness
        for i in range(n):
            for scale in [0.8, 1.2, 0.6, 1.4, 0.5, 1.5]:
                th2 = list(best_th)
                th2[i] = best_th[i] * scale
                try_candidate(best_mats, th2)
            # random small perturb
            for _ in range(3):
                th2 = list(best_th)
                th2[i] = best_th[i] * (1.0 + rng.uniform(-0.15, 0.15))
                try_candidate(best_mats, th2)

        # 2) Swap materials at a position
        for i in range(n):
            mats2 = list(best_mats)
            mats2[i] = "L" if mats2[i] == "H" else "H"
            try_candidate(mats2, best_th)

        # 3) Add a layer (if room)
        if n < max_layers:
            for pos in range(n + 1):
                for m in ["L", "H"]:
                    mats2 = list(best_mats)
                    th2 = list(best_th)
                    d = lam0 / (4.0 * (nL if m == "L" else nH))
                    mats2.insert(pos, m)
                    th2.insert(pos, d)
                    try_candidate(mats2, th2)

        # 4) Remove a layer (if more than 1)
        if n > 1:
            for i in range(n):
                mats2 = list(best_mats)
                th2 = list(best_th)
                mats2.pop(i)
                th2.pop(i)
                try_candidate(mats2, th2)

        # 5) Random restart to a quarter-wave stack occasionally
        if iterations % 50 == 0:
            nl = int(rng.integers(3, min(11, max_layers + 1)))
            mats2, th2 = _quarter_stack(nl, lam0, problem)
            if _valid(mats2, th2, problem):
                # only adopt if better
                s = _objective(mats2, th2, problem)
                if best_score is None or s < best_score:
                    best_score = s
                    best_mats = list(mats2)
                    best_th = list(th2)

    # Final validation
    if not _valid(best_mats, best_th, problem):
        # fallback
        mats, th = _quarter_stack(3, lam0, problem)
        best_mats, best_th = mats, th

    return best_mats, best_th
# EVOLVE-BLOCK-END
