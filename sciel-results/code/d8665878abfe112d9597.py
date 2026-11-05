"""Broadband antireflection coating design via multi-start global + local optimization."""
import time
import numpy as np

_START = None

def _elapsed():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START

def _reflectance(materials, thicknesses, wavelengths, ambient, substrate, nL, nH):
    """Mean normal-incidence reflectance (fraction) via characteristic matrix method.
    materials: list of 'L'/'H'; thicknesses: list of nm; wavelengths: array nm."""
    w = np.asarray(wavelengths, dtype=float)
    N = len(materials)
    # phase per layer: delta = 2*pi*n*t / w
    idx = np.array([nL if m == 'L' else nH for m in materials], dtype=float)
    t = np.asarray(thicknesses, dtype=float)
    # M_j = [[cos d_j, (i/sin d_j)/n_j], [i*n_j*sin d_j, cos d_j]]
    # Build product M1...MN then Y = (n0*M[0,0] + i*n0*nS*M[1,0])/(n0*M[0,1] + i*nS*M[1,0])? 
    # Use standard: for stack, total matrix M = product. Then
    # B = M[0,0] + M[0,1]*nS ; C = n0*M[1,0] + n0*M[1,1]*nS
    # r = (n0*B - C)/(n0*B + C)  ... need consistent convention.
    # We'll use the well-tested formulation:
    # Let M be total. Y = (M00 + M01*nS)/(M10 + M11*nS). r = (n0 - Y)/(n0 + Y). R=|r|^2.
    cosd = np.cos(2.0 * np.pi * idx[:, None] * t[:, None] / w[None, :])
    sind = np.sin(2.0 * np.pi * idx[:, None] * t[:, None] / w[None, :])
    # per-layer 2x2 matrices as arrays shape (2,2,W)
    # M_j[0,0]=cosd, M_j[0,1]=1j/sind/n, M_j[1,0]=1j*n*sind, M_j[1,1]=cosd
    M00 = cosd.copy()
    M01 = (1j / sind) / idx[:, None]
    M10 = 1j * idx[:, None] * sind
    M11 = cosd.copy()
    # multiply matrices in order layer0 (air side) * layer1 * ...
    T00 = M00[0]
    T01 = M01[0]
    T10 = M10[0]
    T11 = M11[0]
    for j in range(1, N):
        a00, a01 = M00[j], M01[j]
        a10, a11 = M10[j], M11[j]
        nT00 = T00 * a00 + T01 * a10
        nT01 = T00 * a01 + T01 * a11
        nT10 = T10 * a00 + T11 * a10
        nT11 = T10 * a01 + T11 * a11
        T00, T01, T10, T11 = nT00, nT01, nT10, nT11
    Y = (T00 + T01 * substrate) / (T10 + T11 * substrate)
    r = (ambient - Y) / (ambient + Y)
    R = np.abs(r) ** 2
    return float(np.mean(R))

def _coarse_grid(problem):
    b0, b1 = problem["band_nm"]
    return np.linspace(b0, b1, 81)

def _fine_grid(problem):
    b0, b1 = problem["band_nm"]
    return np.linspace(b0, b1, 401)

def _clamp_thick(x, max_layer):
    return np.clip(x, 0.0, float(max_layer))

def _valid(materials, thicknesses, problem):
    nL = problem["materials"]["L"]; nH = problem["materials"]["H"]
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    max_layers = problem["max_layers"]
    if len(materials) != len(thicknesses):
        return False
    if len(materials) > max_layers:
        return False
    if len(materials) == 0:
        return False
    for m in materials:
        if m not in ("L", "H"):
            return False
    for t in thicknesses:
        if t < -1e-9 or t > max_layer + 1e-9:
            return False
    if sum(thicknesses) > max_total + 1e-9:
        return False
    return True

def _objective(x, materials, problem, grid, nL, nH, ambient, substrate):
    thicknesses = _clamp_thick(x, problem["max_layer_nm"])
    if sum(thicknesses) > problem["max_total_nm"] + 1e-9:
        return 10.0
    return _reflectance(materials, thicknesses, grid, ambient, substrate, nL, nH)

def _local_refine(x0, materials, problem, grid, nL, nH, ambient, substrate, deadline):
    from scipy.optimize import minimize
    nL_ = nL; nH_ = nH
    def f(x):
        return _objective(x, materials, problem, grid, nL_, nH_, ambient, substrate)
    best_x = np.array(x0, dtype=float)
    best_f = f(best_x)
    for method in ("Nelder-Mead", "Powell"):
        if _elapsed() > deadline:
            break
        try:
            res = minimize(f, best_x, method=method,
                           options={"maxiter": 2000, "xatol": 0.05, "fatol": 1e-6})
            if res.success or res.fun < best_f:
                cand = _clamp_thick(res.x, problem["max_layer_nm"])
                if sum(cand) <= problem["max_total_nm"] + 1e-9:
                    cf = f(cand)
                    if cf < best_f:
                        best_f = cf
                        best_x = cand
        except Exception:
            pass
    return best_x, best_f

def design(problem):
    global _START
    _START = time.time()
    deadline = 140.0  # safety margin under 150s kill

    ambient = problem["ambient_index"]
    substrate = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    max_layer = problem["max_layer_nm"]
    max_total = problem["max_total_nm"]
    max_layers = problem["max_layers"]
    b0, b1 = problem["band_nm"]
    centre = 0.5 * (b0 + b1)

    grid_coarse = _coarse_grid(problem)
    grid_fine = _fine_grid(problem)

    rng = np.random.default_rng(12345)

    best_materials = None
    best_thicknesses = None
    best_score = float("inf")

    def consider(materials, thicknesses):
        nonlocal best_materials, best_thicknesses, best_score
        if not _valid(materials, thicknesses, problem):
            return
        s = _reflectance(materials, thicknesses, grid_fine, ambient, substrate, nL, nH)
        if s < best_score:
            best_score = s
            best_materials = list(materials)
            best_thicknesses = list(thicknesses)

    # Seed with a few analytic / heuristic designs first
    # 1) single L quarter wave at centre
    consider(["L"], [centre / (4.0 * nL)])
    # 2) 2-layer H,L quarter waves at centre
    consider(["H", "L"], [centre / (4.0 * nH), centre / (4.0 * nL)])
    # 3) 4-layer H,L,L,H quarter waves at centre (classic broadband)
    consider(["H", "L", "L", "H"],
             [centre / (4.0 * nH), centre / (4.0 * nL), centre / (4.0 * nL), centre / (4.0 * nH)])
    # 4) 6-layer alternating starting H
    mats6 = ["H", "L", "H", "L", "H", "L"]
    consider(mats6, [centre / (4.0 * (nH if m == 'H' else nL)) for m in mats6])

    # Now global + local search over several layer counts
    layer_counts = [4, 6, 8, 10, 12]
    # Pattern generator: alternate starting with H or L
    def make_pattern(n, start='H'):
        return [start if i % 2 == 0 else ('L' if start == 'H' else 'H') for i in range(n)]

    n_restarts_per_count = 6
    for nc in layer_counts:
        if _elapsed() > deadline:
            break
        if nc > max_layers:
            continue
        for start in ('H', 'L'):
            if _elapsed() > deadline:
                break
            materials = make_pattern(nc, start)
            for r in range(n_restarts_per_count):
                if _elapsed() > deadline:
                    break
                # random init within [0, max_layer], then scale to respect total
                x = rng.uniform(0.1, max_layer, size=nc)
                # randomize a bit of the quarter-wave structure too
                qw = np.array([centre / (4.0 * (nH if m == 'H' else nL)) for m in materials])
                x = 0.5 * x + 0.5 * qw
                x = _clamp_thick(x, max_layer)
                if sum(x) > max_total:
                    x = x * (max_total / sum(x))
                x, f = _local_refine(x, materials, problem, grid_coarse, nL, nH, ambient, substrate, deadline)
                consider(materials, x)

    # Final polish on the best found with fine grid
    if best_materials is not None and _elapsed() < deadline:
        x = np.array(best_thicknesses, dtype=float)
        x, f = _local_refine(x, best_materials, problem, grid_fine, nL, nH, ambient, substrate, deadline)
        consider(best_materials, x)

    if best_materials is None:
        # fallback single layer
        return ["L"], [centre / (4.0 * nL)]

    # ensure strict validity and round
    mats = [m for m in best_materials]
    th = [float(min(max(t, 0.0), max_layer)) for t in best_thicknesses]
    if sum(th) > max_total:
        th = [t * (max_total / sum(th)) for t in th]
    return mats, th
