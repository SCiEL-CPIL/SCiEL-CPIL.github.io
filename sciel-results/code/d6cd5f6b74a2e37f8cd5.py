"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _mean_reflectance_vec(materials, thicknesses, problem):
    """Vectorized mean reflectance (percent) over 401 wavelengths via characteristic matrix."""
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    wls = np.linspace(band[0], band[1], 401)
    n_mat = np.array([n_L if m == 'L' else n_H for m in materials], dtype=float)
    d = np.array(thicknesses, dtype=float)
    if len(materials) == 0:
        r0 = (n_amb - n_sub) / (n_amb + n_sub)
        return float(np.abs(r0)**2 * 100.0)
    # phase[i, j] = 2*pi*n_i*d_i/wl_j
    phase = 2.0 * np.pi * np.outer(n_mat * d, 1.0 / wls)
    cos_p = np.cos(phase)
    sin_p = np.sin(phase)
    # Build matrices: B[i, j, a, b]
    B = np.zeros((len(materials), len(wls), 2, 2), dtype=complex)
    B[:, :, 0, 0] = cos_p
    B[:, :, 0, 1] = 1j * sin_p / n_mat[:, None]
    B[:, :, 1, 0] = 1j * n_mat[:, None] * sin_p
    B[:, :, 1, 1] = cos_p
    # Product over layers for each wavelength
    M = np.zeros((len(wls), 2, 2), dtype=complex)
    for j in range(len(wls)):
        M[j] = np.eye(2, dtype=complex)
    for i in range(len(materials)):
        M = M @ B[i]  # M[j] = M[j] @ B[i][j]
    Y = M[:, 0, 1] / M[:, 1, 1]
    r = (n_amb - Y) / (n_amb + Y)
    R = np.abs(r)**2 * 100.0
    return float(np.mean(R))

def _qwave_thicknesses(mats, problem, scale=1.0):
    """Quarter-wave thicknesses scaled by `scale` (dimensionless QWOT factor)."""
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    centre = 0.5 * (band[0] + band[1])
    thick = []
    for m in mats:
        n = n_L if m == 'L' else n_H
        thick.append(scale * centre / (4.0 * n))
    return thick

def _optimize_thicknesses(mats, thick0, problem, deadline, max_rounds=6):
    """Coordinate descent on thicknesses for fixed material sequence."""
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    best_thick = list(thick0)
    best_score = _mean_reflectance_vec(mats, best_thick, problem)
    n_layers = len(mats)
    if n_layers == 0:
        return best_thick, best_score
    for rnd in range(max_rounds):
        if time.time() > deadline - 35:
            break
        improved = False
        for i in range(n_layers):
            if time.time() > deadline - 35:
                break
            cur = best_thick[i]
            # multi-scale steps
            steps = [0.5, -0.5, 2.0, -2.0, 5.0, -5.0, 10.0, -10.0, 20.0, -20.0, 0.2, -0.2]
            for s in steps:
                nv = cur + s
                if nv < 0 or nv > max_layer_nm:
                    continue
                trial = best_thick[:]
                trial[i] = nv
                if sum(trial) > max_total_nm:
                    continue
                sc = _mean_reflectance_vec(mats, trial, problem)
                if sc < best_score:
                    best_score = sc
                    best_thick = trial
                    cur = nv
                    improved = True
        if not improved:
            break
    return best_thick, best_score

def design(problem):
    """Broadband AR coating: SA over layer sequences + QWOT scaling, then NM polish."""
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    centre = 0.5 * (band[0] + band[1])
    random.seed(1234)

    # Fallback: single quarter-wave layer
    best_mats = ["L"]
    best_thick = [centre / (4.0 * n_L)]
    best_score = _mean_reflectance_vec(best_mats, best_thick, problem)

    # Try alternating quarter-wave stacks of increasing length
    for n_layers in range(2, max_layers + 1):
        if time.time() > deadline - 50:
            break
        mats = []
        thick = []
        for i in range(n_layers):
            if i % 2 == 0:
                mats.append('L'); thick.append(centre / (4.0 * n_L))
            else:
                mats.append('H'); thick.append(centre / (4.0 * n_H))
        if sum(thick) > max_total_nm or any(t > max_layer_nm for t in thick):
            continue
        sc = _mean_reflectance_vec(mats, thick, problem)
        if sc < best_score:
            best_score = sc
            best_mats = mats[:]
            best_thick = thick[:]

    # Simulated annealing over discrete sequences with QWOT scaling
    if time.time() < deadline - 40:
        cur_mats = best_mats[:]
        cur_thick = best_thick[:]
        cur_score = best_score
        best_mats_sa = best_mats[:]
        best_thick_sa = best_thick[:]
        best_score_sa = best_score

        T = 1.5
        cooling = 0.99
        max_sa = 4000
        for it in range(max_sa):
            if time.time() > deadline - 40:
                break
            new_mats = cur_mats[:]
            r = random.random()
            if len(new_mats) < max_layers and r < 0.30:
                pos = random.randint(0, len(new_mats))
                new_mats.insert(pos, random.choice(['L', 'H']))
            elif len(new_mats) > 1 and r < 0.55:
                pos = random.randint(0, len(new_mats) - 1)
                new_mats.pop(pos)
            elif r < 0.75:
                pos = random.randint(0, len(new_mats) - 1)
                new_mats[pos] = 'L' if new_mats[pos] == 'H' else 'H'
            else:
                if len(new_mats) >= 2:
                    p1, p2 = random.sample(range(len(new_mats)), 2)
                    new_mats[p1], new_mats[p2] = new_mats[p2], new_mats[p1]

            # QWOT scaling: try a few scale factors
            best_local_mats = None
            best_local_thick = None
            best_local_score = float('inf')
            for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
                new_thick = _qwave_thicknesses(new_mats, problem, scale)
                if sum(new_thick) > max_total_nm or any(t > max_layer_nm for t in new_thick):
                    continue
                sc = _mean_reflectance_vec(new_mats, new_thick, problem)
                if sc < best_local_score:
                    best_local_score = sc
                    best_local_mats = new_mats[:]
                    best_local_thick = new_thick[:]
            if best_local_mats is None:
                continue

            delta = best_local_score - cur_score
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-6)):
                cur_mats = best_local_mats
                cur_thick = best_local_thick
                cur_score = best_local_score
                # local thickness optimization
                if time.time() < deadline - 40:
                    opt_thick, opt_score = _optimize_thicknesses(cur_mats, cur_thick, problem, deadline, max_rounds=3)
                    cur_thick = opt_thick
                    cur_score = opt_score
                if cur_score < best_score_sa:
                    best_score_sa = cur_score
                    best_mats_sa = cur_mats[:]
                    best_thick_sa = cur_thick[:]
            T *= cooling

        if best_score_sa < best_score:
            best_score = best_score_sa
            best_mats = best_mats_sa
            best_thick = best_thick_sa

    # Final high-precision Nelder-Mead polish
    if time.time() < deadline - 15 and len(best_mats) >= 2:
        try:
            from scipy.optimize import minimize
            x0 = np.array(best_thick, dtype=float)
            def obj(x):
                if np.any(x < 0) or np.any(x > max_layer_nm) or np.sum(x) > max_total_nm:
                    return 100.0
                return _mean_reflectance_vec(best_mats, list(x), problem)
            res = minimize(obj, x0, method='Nelder-Mead',
                           options={'maxiter': 600, 'xatol': 1e-5, 'fatol': 1e-9, 'adaptive': True})
            if res.success and res.fun < best_score:
                x_opt = np.clip(res.x, 0, max_layer_nm)
                if np.sum(x_opt) <= max_total_nm:
                    best_score = float(res.fun)
                    best_thick = list(x_opt)
        except Exception:
            pass

    # Final validation
    if sum(best_thick) > max_total_nm:
        # scale down proportionally
        f = max_total_nm / sum(best_thick)
        best_thick = [t * f for t in best_thick]
    for i, t in enumerate(best_thick):
        if t > max_layer_nm:
            best_thick[i] = max_layer_nm
        if t < 0:
            best_thick[i] = 0.0

    return best_mats, best_thick
# EVOLVE-BLOCK-END
