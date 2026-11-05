"""Rewritten program: global optimization (simulated annealing) of a multi-layer
alternating L/H antireflection coating, evaluated with the thin-film
characteristic-matrix method."""
import numpy as np

# EVOLVE-BLOCK-START
def _reflectance(thicknesses, labels, problem, wl_grid):
    """Mean reflectance (fraction) over wl_grid using the characteristic-matrix method."""
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    mats = problem["materials"]
    n_layers = len(thicknesses)
    total = 0.0
    nwl = len(wl_grid)
    for wl in wl_grid:
        # start from substrate admittance
        M = np.array([[1.0 + 0.0j, 0.0 + 0.0j],
                      [0.0 + 0.0j, 1.0 + 0.0j]])
        # multiply layers from air side to substrate
        for k in range(n_layers):
            n = mats[labels[k]]
            d = thicknesses[k]
            delta = 2.0 * np.pi * n * d / wl
            c = np.cos(delta)
            s = np.sin(delta)
            Mk = np.array([[c, 1j * s / n],
                           [1j * n * s, c]])
            M = M @ Mk
        Y = M[1, 0] / M[0, 0]
        r = (Y - ns) / (Y + ns)
        total += abs(r) ** 2
    return total / nwl


def design(problem):
    band = problem["band_nm"]
    wl0, wl1 = band[0], band[1]
    centre = 0.5 * (wl0 + wl1)
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    mats = problem["materials"]
    nL, nH = mats["L"], mats["H"]

    # 401 equally spaced wavelengths
    wl_grid = np.linspace(wl0, wl1, 401)

    # Choose a number of layers; alternating L/H starting and ending with L
    # (L on air side reduces the first interface step).
    # We'll pick an even number so we can start and end with L.
    N = 10
    labels = ["L" if (i % 2 == 0) else "H" for i in range(N)]

    # Initialize thicknesses near quarter-wave at centre wavelength.
    def qw(n):
        return centre / (4.0 * n)
    thicknesses = np.array([qw(nL) if (i % 2 == 0) else qw(nH) for i in range(N)], dtype=float)

    # Ensure feasibility
    def feasible(t):
        return np.all(t >= 0) and np.all(t <= max_layer) and float(np.sum(t)) <= max_total

    # Objective: mean reflectance (fraction); convert to percent for reporting not needed.
    def obj(t):
        if not feasible(t):
            return 1e6
        return _reflectance(t.tolist(), labels, problem, wl_grid)

    # Simulated annealing
    rng = np.random.default_rng(12345)
    cur = thicknesses.copy()
    cur_f = obj(cur)
    best = cur.copy()
    best_f = cur_f

    T = 0.002
    T_min = 1e-5
    cooling = 0.9995
    iters = 40000

    for it in range(iters):
        # propose a perturbation: pick a random layer, add Gaussian noise
        k = rng.integers(0, N)
        scale = 15.0 * (0.3 + 0.7 * (T / T))  # keep scale moderate
        cand = cur.copy()
        cand[k] += rng.normal(0.0, scale)
        # occasionally perturb two layers
        if rng.random() < 0.3:
            j = rng.integers(0, N)
            cand[j] += rng.normal(0.0, scale)

        # clip to layer bounds (keep total check in obj)
        cand = np.clip(cand, 0.0, max_layer)

        f_new = obj(cand)
        dE = f_new - cur_f
        if dE < 0 or rng.random() < np.exp(-dE / max(T, 1e-9)):
            cur = cand
            cur_f = f_new
            if cur_f < best_f:
                best = cur.copy()
                best_f = cur_f
        T *= cooling
        if T < T_min:
            break

    # Final polish: local refinement with small steps
    T = 1e-4
    for it in range(8000):
        k = rng.integers(0, N)
        cand = best.copy()
        cand[k] += rng.normal(0.0, 3.0)
        cand = np.clip(cand, 0.0, max_layer)
        f_new = obj(cand)
        if f_new < best_f:
            best = cand
            best_f = f_new

    # Return best feasible design
    best = np.clip(best, 0.0, max_layer)
    if float(np.sum(best)) > max_total:
        # scale down proportionally
        best = best * (max_total / float(np.sum(best)))
        best = np.clip(best, 0.0, max_layer)

    return labels, [float(x) for x in best]
# EVOLVE-BLOCK-END
