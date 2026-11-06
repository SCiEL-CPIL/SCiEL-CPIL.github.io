"""Antireflection coating design via characteristic-matrix + Nelder-Mead local search."""
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def _build_stack(n, materials, thicknesses, ambient, substrate):
    n = int(n)
    n = max(0, min(n, 20))
    thicknesses = np.asarray(thicknesses, dtype=float)
    if thicknesses.shape[0] < n:
        pad = np.zeros(n - thicknesses.shape[0])
        thicknesses = np.concatenate([thicknesses, pad])
    else:
        thicknesses = thicknesses[:n]
    thicknesses = np.clip(thicknesses, 0.0, 400.0)

    # Per-wavelength matrices
    wl = np.arange(400.0, 801.0, 1.0)  # 401 wavelengths
    k = 2.0 * np.pi / wl  # (M,)
    n_mat = np.empty((n, len(wl)))
    for i in range(n):
        n_mat[i] = materials[i]
    n_mat = n_mat.T  # (M, n)

    t = thicknesses  # (n,)
    phi = k[None, :] * n_mat * t[:, None]  # (n, M)
    c = np.cos(phi)
    s = np.sin(phi)
    B = n_mat * c + 1j * s          # (n, M)
    C = 1.0 / (n_mat * c + 1j * s)  # (n, M)

    # Stack product M_total = prod over layers of [[B, C],[n*B, n*C]]
    # Use cumulative: M_i = M_{i-1} @ M_i_layer
    M = np.empty((2, 2, len(wl)), dtype=complex)
    M[:, :, :] = np.eye(2)[None, :, :]  # identity per wavelength
    for i in range(n):
        Mi = np.empty((2, 2, len(wl)), dtype=complex)
        Mi[0, 0] = B[i]
        Mi[0, 1] = C[i]
        Mi[1, 0] = n_mat[:, i] * B[i]
        Mi[1, 1] = n_mat[:, i] * C[i]
        # M = M @ Mi  (matrix multiply per wavelength)
        M00 = M[0, 0] * Mi[0, 0] + M[0, 1] * Mi[1, 0]
        M01 = M[0, 0] * Mi[0, 1] + M[0, 1] * Mi[1, 1]
        M10 = M[1, 0] * Mi[0, 0] + M[1, 1] * Mi[1, 0]
        M11 = M[1, 0] * Mi[0, 1] + M[1, 1] * Mi[1, 1]
        M[0, 0] = M00; M[0, 1] = M01; M[1, 0] = M10; M[1, 1] = M11

    # Effective admittance Y = (M10 + M11*n_sub) / (M00 + M01*n_sub)
    n_sub = substrate
    num = M[1, 0] + M[1, 1] * n_sub
    den = M[0, 0] + M[0, 1] * n_sub
    Y = num / den
    r = (ambient - Y) / (ambient + Y)
    R = np.abs(r) ** 2
    return R.mean() * 100.0


def _objective(x, materials, ambient, substrate):
    n = int(x[0])
    thicknesses = x[1:]
    return _build_stack(n, materials, thicknesses, ambient, substrate)


def design(problem):
    deadline = time.time() + 140.0  # safety margin under 150s wall-clock
    materials = problem["materials"]  # {'L':1.38,'H':2.35}
    ambient = problem.get("ambient_index", 1.0)
    substrate = problem.get("substrate_index", 1.52)
    band = problem["band_nm"]
    centre = 0.5 * (band[0] + band[1])

    # Seed: 5-layer quarter-wave L/H/L/H/L centered at 600 nm
    N = 5
    labels = ["L", "H", "L", "H", "L"]
    seed_th = [centre / (4.0 * materials[l]) for l in labels]

    best_x = np.concatenate([[float(N)], np.array(seed_th, dtype=float)])
    best_val = _objective(best_x, labels, ambient, substrate)

    # Nelder-Mead local search
    from scipy.optimize import minimize
    try:
        res = minimize(
            _objective,
            best_x,
            args=(labels, ambient, substrate),
            method="Nelder-Mead",
            options={"maxiter": 4000, "xatol": 1e-3, "fatol": 1e-5,
                     "adaptive": True, "maxfev": 6000},
        )
        if res.fun < best_val:
            best_val = float(res.fun)
            best_x = np.asarray(res.x, dtype=float)
    except Exception:
        pass

    # Restart phase: perturb the best and re-optimize with a smaller budget
    if time.time() < deadline - 20.0:
        rng = np.random.default_rng(0)
        for _ in range(4):
            if time.time() >= deadline - 15.0:
                break
            cand = best_x.copy()
            cand[1:] += rng.normal(0.0, 20.0, size=N)
            try:
                res = minimize(
                    _objective,
                    cand,
                    args=(labels, ambient, substrate),
                    method="Nelder-Mead",
                    options={"maxiter": 1500, "xatol": 1e-3, "fatol": 1e-5,
                             "adaptive": True, "maxfev": 2000},
                )
                if res.fun < best_val:
                    best_val = float(res.fun)
                    best_x = np.asarray(res.x, dtype=float)
            except Exception:
                pass

    # Finalize
    n_out = int(round(best_x[0]))
    n_out = max(0, min(n_out, 20))
    th = best_x[1:1 + n_out]
    th = np.clip(th, 0.0, 400.0)
    # Enforce total thickness cap
    total = th.sum()
    if total > 1000.0 and total > 0:
        th = th * (1000.0 / total)
    labels_out = labels[:n_out]
    thicknesses_out = [float(t) for t in th]
    return labels_out, thicknesses_out
# EVOLVE-BLOCK-END
