"""Broadband AR coating: graded-index init + layer-count search + multi-start Nelder-Mead."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _reflectance_spectrum(materials, thicknesses, n0, ns, band_nm, n_waves=401):
    """Exact mean reflectance (%) via characteristic-matrix method."""
    wl = np.linspace(band_nm[0], band_nm[1], n_waves)
    nL = materials['L']; nH = materials['H']
    n = np.array([n0] + [nL if m == 'L' else nH for m in materials] + [ns])
    d = np.array([0.0] + list(thicknesses) + [0.0])
    k = 2.0 * np.pi / wl
    # phase phi_i = k * n_i * d_i  (broadcast over wavelengths)
    phi = k[:, None] * n[None, :] * d[None, :]
    # build 2x2 characteristic matrix for each wavelength
    # M_i = [[cos phi, i sin phi / n_i], [i n_i sin phi, cos phi]]
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    # accumulate matrix product
    M = np.zeros((n_waves, 2, 2), dtype=complex)
    M[:, 0, 0] = 1.0
    M[:, 1, 1] = 1.0
    for i in range(1, len(n) - 1):
        ni = n[i]
        # left-multiply M = M @ M_i
        c = cos_p[:, i]; s = sin_p[:, i]
        # M_i = [[c, i s/ni], [i ni s, c]]
        M00 = M[:, 0, 0] * c + M[:, 0, 1] * (1j * s / ni)
        M01 = M[:, 0, 0] * (1j * s / ni) + M[:, 0, 1] * c
        M10 = M[:, 1, 0] * c + M[:, 1, 1] * (1j * ni * s)
        M11 = M[:, 1, 0] * (1j * ni * s) + M[:, 1, 1] * c
        M[:, 0, 0] = M00; M[:, 0, 1] = M01
        M[:, 1, 0] = M10; M[:, 1, 1] = M11
    # reflectance for s-polarization at normal incidence (identical for p)
    Y = (M[:, 0, 0] + M[:, 0, 1] * ns) / (M[:, 1, 0] / ns + M[:, 1, 1])
    r = (n0 - Y) / (n0 + Y)
    R = (r * r.conj()).real
    return 100.0 * np.mean(R)


def _clamp(x, lo, hi):
    return min(max(x, lo), hi)


def _graded_init(n_layers, nL, nH, ns, max_total_nm, max_layer_nm):
    """Logarithmic graded-index thickness profile from air to substrate."""
    # target index per layer: geometric interpolation from n0~1.0 to ns
    n0 = 1.0
    idx = np.geomspace(n0, ns, n_layers + 1)
    # assign L or H by which is closer
    mats = []
    for i in range(n_layers):
        target = idx[i + 1]
        if abs(target - nL) <= abs(target - nH):
            mats.append('L')
        else:
            mats.append('H')
    # thicknesses: quarter-wave at centre wavelength, scaled to fit budget
    centre = 600.0
    base = centre / (4.0 * nL)  # ~108 nm
    # use graded thickness: thicker in middle
    prof = np.linspace(0.6, 1.4, n_layers)
    thick = base * prof
    # clamp to max_layer_nm
    thick = np.clip(thick, 0.0, max_layer_nm)
    # scale down if over budget
    total = thick.sum()
    if total > max_total_nm:
        thick *= max_total_nm / total
        thick = np.clip(thick, 0.0, max_layer_nm)
    return mats, list(thick)


def _nelder_mead(f, x0, bounds, max_iter, tol=1e-8, max_step=40.0):
    """Nelder-Mead with bounds via reflection penalty."""
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        pt = list(x0)
        pt[i] = x0[i] + 0.1 * max_step
        simplex.append(pt)
    vals = [f(p) for p in simplex]
    for it in range(max_iter):
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        # centroid
        cen = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        # reflection
        alpha = 1.0
        xr = cen + alpha * (cen - np.array(worst))
        # bound
        xr_b = np.clip(xr, bounds[0], bounds[1])
        fr = f(xr_b)
        if vals[0] <= fr < vals[-2]:
            simplex[-1] = xr_b; vals[-1] = fr
        elif fr < vals[0]:
            # expansion
            beta = 2.0
            xe = cen + beta * (xr_b - cen)
            xe = np.clip(xe, bounds[0], bounds[1])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe; vals[-1] = fe
            else:
                simplex[-1] = xr_b; vals[-1] = fr
        else:
            # contraction
            gamma = 0.5
            if fr < vals[-1]:
                xc = cen + gamma * (xr_b - cen)
            else:
                xc = cen - gamma * (worst - cen)
            xc = np.clip(xc, bounds[0], bounds[1])
            fc = f(xc)
            if fc < max(fr, vals[-1]):
                simplex[-1] = xc; vals[-1] = fc
            else:
                # shrink
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                    simplex[i] = np.clip(simplex[i], bounds[0], bounds[1]).tolist()
                    vals[i] = f(simplex[i])
    order = np.argsort(vals)
    return simplex[order[0]], vals[order[0]]


def design(problem):
    t_start = time.time()
    deadline = t_start + 135.0  # 15s safety margin

    nL = problem['materials']['L']
    nH = problem['materials']['H']
    n0 = problem['ambient_index']
    ns = problem['substrate_index']
    band = problem['band_nm']
    max_layers = problem['max_layers']
    max_total = problem['max_total_nm']
    max_layer = problem['max_layer_nm']

    def objective(thick_vec, mats, nL, nH, n0, ns, band):
        # penalty for constraint violation
        if len(thick_vec) > max_layers:
            return 1e6
        if any(t < -1e-9 or t > max_layer + 1e-9 for t in thick_vec):
            return 1e6
        if sum(thick_vec) > max_total + 1e-9:
            return 1e6
        return _reflectance_spectrum(dict(zip(mats, [nL if m == 'L' else nH for m in mats])),
                                     list(thick_vec), n0, ns, band)

    best_score = 4.26  # bare glass
    best_mats = ['L']
    best_thick = [100.0]

    # try a range of layer counts
    for n_layers in [4, 6, 8, 10, 12, 14, 16, 18, 20]:
        if time.time() > deadline:
            break
        mats, thick = _graded_init(n_layers, nL, nH, ns, max_total, max_layer)
        x0 = np.array(thick)
        bounds = (np.zeros(n_layers), np.full(n_layers, max_layer))
        # multi-start
        rng = np.random.RandomState(n_layers)
        for start in range(3):
            if time.time() > deadline:
                break
            if start == 0:
                x = x0.copy()
            else:
                x = x0 + rng.uniform(-20, 20, n_layers)
                x = np.clip(x, 0.0, max_layer)
            # scale to fit total budget
            if x.sum() > max_total:
                x *= max_total / x.sum()
            def f(p):
                p = np.clip(p, 0.0, max_layer)
                if p.sum() > max_total:
                    p = p * (max_total / p.sum())
                return _reflectance_spectrum(
                    {m: (nL if m == 'L' else nH) for m in mats},
                    list(p), n0, ns, band)
            x_opt, val = _nelder_mead(f, x, bounds, max_iter=300)
            x_opt = np.clip(x_opt, 0.0, max_layer)
            if x_opt.sum() > max_total:
                x_opt *= max_total / x_opt.sum()
            if val < best_score:
                best_score = val
                best_mats = list(mats)
                best_thick = list(x_opt)

    return best_mats, best_thick

# EVOLVE-BLOCK-END
