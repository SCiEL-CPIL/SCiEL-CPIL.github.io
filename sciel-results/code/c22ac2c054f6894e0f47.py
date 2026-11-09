import time
import math
import random
import numpy as np

try:
    from scipy.optimize import minimize as _scipy_minimize, differential_evolution as _de
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _compute_matrices(materials, thicknesses, problem, wavs):
    """Compute characteristic matrices and return (A, B, C, D) arrays over wavelengths."""
    if not materials:
        n = len(wavs)
        return (np.ones(n), np.zeros(n), np.zeros(n), np.ones(n))
    
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = [n_mat[m] for m in materials]
    ds = np.asarray(thicknesses, dtype=float)
    
    n_ds = np.array([n * d for n, d in zip(ns, ds)])
    delta = 2.0 * np.pi * n_ds[:, None] / wavs[None, :]
    c = np.cos(delta)
    s = np.sin(delta)
    
    A = np.ones_like(wavs)
    B = np.zeros_like(wavs)
    C = np.zeros_like(wavs)
    D = np.ones_like(wavs)
    
    i = 1.0j
    for j in range(len(materials)):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        A_new = A * cj + i * B * n_j * sj
        B_new = i * A * sj / n_j + B * cj
        C_new = C * cj + i * D * n_j * sj
        D_new = i * C * sj / n_j + D * cj
        A, B, C, D = A_new, B_new, C_new, D_new
    
    return A, B, C, D


def _mean_reflectance(materials, thicknesses, problem):
    """Mean reflectance (percent) over the band using characteristic matrices."""
    if not materials:
        return 100.0 * ((problem["substrate_index"] - problem["ambient_index"]) /
                        (problem["substrate_index"] + problem["ambient_index"])) ** 2
    band = problem["band_nm"]
    wavs = np.linspace(band[0], band[1], 401)
    A, B, C, D = _compute_matrices(materials, thicknesses, problem, wavs)
    
    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    num = n_0 * (A + B * n_s) - (C + D * n_s)
    den = n_0 * (A + B * n_s) + (C + D * n_s)
    r = num / den
    R = np.abs(r) ** 2
    return 100.0 * np.mean(R)


def _gradient_reflectance(materials, thicknesses, problem, wavs):
    """Compute analytical gradient of mean reflectance w.r.t. each thickness."""
    if not materials:
        return np.array([])
    
    n_mat = {k: v for k, v in problem["materials"].items()}
    ns = [n_mat[m] for m in materials]
    ds = np.asarray(thicknesses, dtype=float)
    n_layers = len(materials)
    
    n_s = problem["substrate_index"]
    n_0 = problem["ambient_index"]
    
    n_ds = np.array([n * d for n, d in zip(ns, ds)])
    delta = 2.0 * np.pi * n_ds[:, None] / wavs[None, :]
    c = np.cos(delta)
    s = np.sin(delta)
    i = 1.0j
    
    prefix = []
    A_pre = np.ones_like(wavs)
    B_pre = np.zeros_like(wavs)
    C_pre = np.zeros_like(wavs)
    D_pre = np.ones_like(wavs)
    
    for j in range(n_layers):
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        A_new = A_pre * cj + i * B_pre * n_j * sj
        B_new = i * A_pre * sj / n_j + B_pre * cj
        C_new = C_pre * cj + i * D_pre * n_j * sj
        D_new = i * C_pre * sj / n_j + D_pre * cj
        prefix.append((A_new.copy(), B_new.copy(), C_new.copy(), D_new.copy()))
        A_pre, B_pre, C_pre, D_pre = A_new, B_new, C_new, D_new
    
    suffix = []
    A_suf = np.ones_like(wavs)
    B_suf = np.zeros_like(wavs)
    C_suf = np.zeros_like(wavs)
    D_suf = np.ones_like(wavs)
    
    for j in range(n_layers - 1, -1, -1):
        suffix.append((A_suf.copy(), B_suf.copy(), C_suf.copy(), D_suf.copy()))
        n_j = ns[j]
        cj = c[j]
        sj = s[j]
        A_new = A_suf * cj + i * B_suf * n_j * sj
        B_new = i * A_suf * sj / n_j + B_suf * cj
        C_new = C_suf * cj + i * D_suf * n_j * sj
        D_new = i * C_suf * sj / n_j + D_suf * cj
        A_suf, B_suf, C_suf, D_suf = A_new, B_new, C_new, D_new
    
    A_tot, B_tot, C_tot, D_tot = prefix[-1]
    
    num = n_0 * (A_tot + B_tot * n_s) - (C_tot + D_tot * n_s)
    den = n_0 * (A_tot + B_tot * n_s) + (C_tot + D_tot * n_s)
    r = num / den
    
    grad = np.zeros(n_layers)
    
    for j in range(n_layers):
        if j == 0:
            A_l, B_l, C_l, D_l = np.ones_like(wavs), np.zeros_like(wavs), np.zeros_like(wavs), np.ones_like(wavs)
        else:
            A_l, B_l, C_l, D_l = prefix[j-1]
            
        A_r, B_r, C_r, D_r = suffix[j]
        
        n_j = ns[j]
        d_j = delta[j]
        sd = np.sin(d_j)
        cd = np.cos(d_j)
        k = 2.0 * np.pi / wavs
        
        dA = -sd * n_j * k
        dB = i * cd * k
        dC = i * n_j * cd * k
        dD = -sd * n_j * k
        
        tmp_A = A_l * dA + B_l * dC
        tmp_B = A_l * dB + B_l * dD
        tmp_C = C_l * dA + D_l * dC
        tmp_D = C_l * dB + D_l * dD
        
        dA_tot = tmp_A * A_r + tmp_B * C_r
        dB_tot = tmp_A * B_r + tmp_B * D_r
        dC_tot = tmp_C * A_r + tmp_D * C_r
        dD_tot = tmp_C * B_r + tmp_D * D_r
        
        dnum = n_0 * (dA_tot + dB_tot * n_s) - (dC_tot + dD_tot * n_s)
        dden = n_0 * (dA_tot + dB_tot * n_s) + (dC_tot + dD_tot * n_s)
        
        dr = (dnum * den - num * dden) / (den * den)
        dR = 2.0 * np.real(np.conj(r) * dr)
        
        grad[j] = np.mean(dR)
    
    return grad


def design(problem):
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin (limit is 150s)
    
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    random.seed(42)
    np.random.seed(42)
    
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    
    best_ths = None
    best_mats = None
    best_score = float("inf")
    
    def maybe_record(mats, ths):
        nonlocal best_ths, best_mats, best_score
        if time.time() > deadline:
            return
        # Clip and normalize
        ths = [max(0.0, min(max_layer, t)) for t in ths]
        total = sum(ths)
        if total > max_total + 1e-6:
            ths = [t * (max_total / total) for t in ths]
            ths = [max(0.0, min(max_layer, t)) for t in ths]
        s = _mean_reflectance(mats, ths, problem)
        if s < best_score:
            best_score = s
            best_ths = list(ths)
            best_mats = list(mats)
    
    def polish_l_bfgs(materials, t_init, problem, T_target):
        """Polish using L-BFGS-B with softmax parameterization to enforce sum=T_target."""
        if time.time() > deadline:
            return None
            
        n = len(materials)
        t_safe = np.maximum(t_init, 1.0)
        x0 = np.log(t_safe)
        
        band = problem["band_nm"]
        wavs = np.linspace(band[0], band[1], 401)
        n_mat = {k: v for k, v in problem["materials"].items()}
        ns = [n_mat[m] for m in materials]
        
        def obj_x(x):
            ex = np.exp(x - np.max(x))
            S = np.sum(ex)
            t = ex / S * T_target
            if np.any(t > max_layer):
                return _mean_reflectance(materials, list(t), problem) + 1000 * (np.max(t) - max_layer)
            return _mean_reflectance(materials, list(t), problem)
            
        def grad_x(x):
            ex = np.exp(x - np.max(x))
            S = np.sum(ex)
            t = ex / S * T_target
            
            g_phys = _gradient_reflectance(materials, list(t), problem, wavs)
            
            grad = np.zeros(n)
            sum_gt = np.sum(g_phys * t) / T_target
            for k in range(n):
                grad[k] = t[k] * (g_phys[k] - sum_gt)
                
            if np.any(t > max_layer):
                for k in range(n):
                    if t[k] > max_layer:
                        grad[k] += 1000
            return grad
        
        try:
            res = _scipy_minimize(
                obj_x, x0, jac=grad_x, method="L-BFGS-B",
                options={"maxiter": 200, "maxfun": 2000, "ftol": 1e-12},
            )
            ex = np.exp(res.x - np.max(res.x))
            S = np.sum(ex)
            t = ex / S * T_target
            t = np.clip(t, 0, max_layer)
            maybe_record(materials, list(t))
            return list(t)
        except Exception:
            return None

    def run_de(materials, problem, deadline_local):
        n = len(materials)
        bounds = [(1.0, max_layer) for _ in range(n)]
        
        def objective(x):
            t = list(x)
            total = sum(t)
            if total > max_total:
                return _mean_reflectance(materials, t, problem) + 100 * (total - max_total)
            return _mean_reflectance(materials, t, problem)
        
        try:
            res = _de(
                objective, bounds,
                popsize=10,
                maxiter=20,
                seed=42,
                polish=False,
                init='latinhypercube',
                tol=1e-6
            )
            if res.fun < float('inf'):
                t = list(res.x)
                total = sum(t)
                if total > max_total:
                    t = [ti * max_total / total for ti in t]
                maybe_record(materials, t)
                return list(t)
        except Exception:
            pass
        return None

    # Structural change: Use a fixed symmetric 20-layer structure to focus all compute on global optimization
    # Pattern: H L H L ... H L H (10 H, 10 L) alternating, starting with H
    # This is a known good structure for broadband AR coatings
    N = 20
    mats = ['H' if i % 2 == 0 else 'L' for i in range(N)]
    
    # Also try starting with L
    mats2 = ['L' if i % 2 == 0 else 'H' for i in range(N)]
    
    # Run DE for the main pattern
    t_best = run_de(mats, problem, deadline)
    if t_best:
        candidates = [(mats, t_best)]
    else:
        candidates = []
    
    # Run DE for the second pattern
    t_best2 = run_de(mats2, problem, deadline)
    if t_best2:
        candidates.append((mats2, t_best2))
    
    # If DE didn't produce good results, add random starts
    if not candidates or best_score > 2.0:
        for _ in range(3):
            t_rand = [random.uniform(20, 100) for _ in range(N)]
            total = sum(t_rand)
            if total > max_total:
                t_rand = [ti * max_total / total for ti in t_rand]
            candidates.append((mats, t_rand))
            candidates.append((mats2, t_rand))
    
    # Polish candidates with different total thickness targets
    T_targets = [1000, 900, 800, 700]
    for mats_c, t_init in candidates:
        for T in T_targets:
            if time.time() > deadline - 5:
                break
            polish_l_bfgs(mats_c, t_init, problem, T)
    
    # If still not good enough, try a 12-layer structure
    if best_score > 1.5:
        N2 = 12
        mats3 = ['H' if i % 2 == 0 else 'L' for i in range(N2)]
        t_best3 = run_de(mats3, problem, deadline)
        if t_best3:
            for T in [1000, 800]:
                if time.time() > deadline - 5:
                    break
                polish_l_bfgs(mats3, t_best3, problem, T)
    
    # Fallback: single quarter-wave L layer
    if best_ths is None:
        qw_L = 600 / (4.0 * nL)
        return ['L'], [max(0.0, min(max_layer, qw_L))]
        
    return list(best_mats), list(best_ths)
