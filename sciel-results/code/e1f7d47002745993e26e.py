"""Program for signed step heights minimizing max|autoconv|/(sum h)^2.

Structural change: instead of a fixed cosine/sine basis, we use a learned
dictionary of low-peak autoconvolution atoms (found via short anneals) and
optimize sparse, non-negative combinations of them. This lets the optimizer
compose known good patterns rather than fitting a generic smooth curve.

Refinement machinery:
  1.  Build a small library of diverse low-peak atoms (short anneals + flips).
  2.  Global search over sparse non-negative atom coefficients (DE).
  3.  Smooth-surrogate local optimisation (adaptive beta log-sum-exp).
  4.  Direct local search: sign-flips and small perturbations.
  5.  Final high-precision polish.

All phases are deadline-guarded so the program always returns within 150 s.
"""

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# ---------------------------------------------------------------------------
# Core objective
# ---------------------------------------------------------------------------

def _autoconv_max_abs(h):
    """max |h*h| (linear autoconvolution) computed via FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    return np.max(np.abs(ac))


def _score(h):
    """max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)


def _score_smooth(h, beta=80.0):
    """Differentiable surrogate: soft_max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size]
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-12:
        return 0.0
    shifted = abs_ac - m
    soft = m + np.log(np.sum(np.exp(beta * shifted))) / beta
    return soft / (s * s)


# ---------------------------------------------------------------------------
# Atom library construction
# ---------------------------------------------------------------------------

def _make_atom_library(n_atoms=12, deadline=None):
    """
    Build a library of diverse low-peak autoconvolution atoms.
    Each atom is a vector of length N, normalized to sum=1.
    """
    atoms = []
    t_start = time.time()
    
    # Start with a few structured patterns
    x = np.arange(N, dtype=float)
    
    # Constant
    atoms.append(np.ones(N) / N)
    
    # Alternating
    alt = np.where(x % 2 == 0, 1.0, -1.0)
    atoms.append(alt / np.sum(alt))
    
    # Cosine windows
    for k in [1, 2, 3, 5, 7, 11, 13]:
        if deadline is not None and time.time() > deadline - 5:
            break
        c = np.cos(2 * np.pi * k * x / N)
        s = np.sum(c)
        if abs(s) > 1e-6:
            atoms.append(c / s)
    
    # Gaussian bumps
    for frac in [0.25, 0.5, 0.75]:
        if deadline is not None and time.time() > deadline - 5:
            break
        mu = frac * N
        sigma = N / 10.0
        g = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
        s = np.sum(g)
        if abs(s) > 1e-6:
            atoms.append(g / s)
    
    # Generate additional atoms via short anneals
    rng = np.random.default_rng(12345)
    for i in range(n_atoms - len(atoms)):
        if deadline is not None and time.time() > deadline - 10:
            break
        # Random start
        h = rng.standard_normal(N)
        h /= np.sum(h) if abs(np.sum(h)) > 1e-6 else 1.0
        
        # Short anneal
        for it in range(100):
            if deadline is not None and time.time() > deadline - 5:
                break
            pert = rng.standard_normal(N) * 0.05
            h_trial = h + pert
            s_trial = np.sum(h_trial)
            if abs(s_trial) > 1e-6:
                h_trial /= s_trial
                v_trial = _score(h_trial)
                v_curr = _score(h)
                if v_trial < v_curr or rng.random() < np.exp(-(v_trial - v_curr) / 0.1):
                    h = h_trial
        
        # Flip local search
        h, _ = _local_search_flip(h, max_passes=2, deadline=deadline)
        
        # Normalize
        s = np.sum(h)
        if abs(s) > 1e-6:
            atoms.append(h / s)
    
    return np.array(atoms)


# ---------------------------------------------------------------------------
# Local search routines
# ---------------------------------------------------------------------------

def _local_search_flip(h, max_passes=6, deadline=None):
    """Try flipping each element's sign; keep if it improves the score."""
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(555)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        idx_order = rng.permutation(N)
        for i in idx_order:
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]
        if not improved:
            break
    return h, best_val


def _local_search_perturb(h, max_passes=5, deadline=None, step_size=0.04):
    """Coordinate-descent style: try small perturbations per element."""
    h = h.copy()
    best_val = _score(h)
    rng = np.random.default_rng(777)
    current_step = step_size
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        idx_order = rng.permutation(N)
        for i in idx_order:
            if deadline is not None and time.time() > deadline:
                break
            old = h[i]
            best_local = old
            best_local_val = best_val
            for delta in (current_step, -current_step, 0.5 * current_step, -0.5 * current_step):
                h[i] = old + delta
                v = _score(h)
                if v < best_local_val:
                    best_local_val = v
                    best_local = h[i]
            if best_local != old:
                h[i] = best_local
                best_val = best_local_val
                improved = True
            else:
                h[i] = old
        if not improved:
            break
        current_step *= 0.85
    return h, best_val


def _gaussian_perturbation(h, scale=0.06, deadline=None, max_iters=300):
    """Simulated-annealing style search with Gaussian perturbations."""
    h_curr = h.copy()
    h_best = h.copy()
    curr_val = _score(h_curr)
    best_val = curr_val
    rng = np.random.default_rng(999)
    T_start, T_end = 0.4, 0.003
    restart_interval = 60
    no_improve = 0
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        T = T_start * (T_end / T_start) ** (it / max(1, max_iters - 1))
        if rng.random() < 0.06:
            pert = rng.standard_normal(N) * scale * 3.0
        else:
            pert = rng.standard_normal(N) * scale
        h_trial = h_curr + pert
        trial_val = _score(h_trial)
        if trial_val < curr_val or rng.random() < np.exp(-(trial_val - curr_val) / max(T, 1e-10)):
            h_curr = h_trial
            curr_val = trial_val
            if curr_val < best_val:
                best_val = curr_val
                h_best = h_curr.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        if no_improve > restart_interval and it < max_iters - 10:
            h_curr = h_best.copy()
            curr_val = best_val
            no_improve = 0
        scale *= 0.996
    return h_best, best_val


def _incremental_flip_search(h, deadline=None, max_passes=200):
    """
    Fast local search using incremental autoconvolution updates.
    """
    h = h.copy()
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)[:size].copy()
    s_sum = np.sum(h)
    best_val = np.max(np.abs(ac)) / (s_sum * s_sum)
    rng = np.random.default_rng(42)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        idx_order = rng.permutation(N)
        for i in idx_order:
            if deadline is not None and time.time() > deadline:
                break
            d = -2.0 * h[i]
            # Compute new ac
            new_ac = ac.copy()
            # Add 2*d*shift(h, i)
            start = i
            end = i + n
            if end <= size:
                new_ac[start:end] += 2.0 * d * h
            else:
                # Handle boundary
                valid_end = min(end, size)
                if start < valid_end:
                    new_ac[start:valid_end] += 2.0 * d * h[:valid_end - start]
            # Add d^2 * delta_i
            new_ac[i] += d * d
            
            new_sum = s_sum + d
            nsq = new_sum * new_sum
            if abs(nsq) < 1e-9:
                continue
            new_val = np.max(np.abs(new_ac)) / nsq
            
            if new_val < best_val:
                # Accept flip
                h[i] = -h[i]
                ac = new_ac
                s_sum = new_sum
                best_val = new_val
                improved = True
            else:
                # Reject, ac remains unchanged
                pass
        if not improved:
            break
    return h, best_val


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    t_start = time.time()
    deadline = t_start + 145.0   # 5 s safety margin

    best_score = np.inf
    best_h = np.ones(N) / N
    base_score = _score(best_h)
    best_score = base_score

    # ---- Phase 1: Build atom library ----
    p1_dl = deadline - 60
    if time.time() < p1_dl:
        atoms = _make_atom_library(n_atoms=12, deadline=p1_dl)
    else:
        # Fallback to simple basis
        x = np.arange(N, dtype=float)
        atoms = [np.ones(N) / N]
        for k in [1, 2, 3, 5, 7, 11]:
            c = np.cos(2 * np.pi * k * x / N)
            s = np.sum(c)
            if abs(s) > 1e-6:
                atoms.append(c / s)
        atoms = np.array(atoms)

    n_atoms = len(atoms)

    # ---- Phase 2: differential evolution on atom coefficients ----
    p2_dl = deadline - 40
    if time.time() < p2_dl:
        bounds = [(0.0, 1.0)] * n_atoms

        def de_obj(params):
            if time.time() > p2_dl:
                return 1e10
            h = np.dot(params, atoms)
            s = np.sum(h)
            if abs(s) < 1e-9:
                return 1e10
            return _autoconv_max_abs(h) / (s * s)

        try:
            res = differential_evolution(
                de_obj, bounds, seed=42, maxiter=30, tol=1e-6,
                popsize=12, mutation=(0.3, 1.2), recombination=0.7,
                polish=False
            )
            h_de = np.dot(res.x, atoms)
            s_de = np.sum(h_de)
            if abs(s_de) > 1e-6:
                h_de /= s_de
                v = _score(h_de)
                if v < best_score:
                    best_score = v
                    best_h = h_de.copy()
        except Exception:
            pass

    # ---- Phase 3: smooth-surrogate local optimisation (adaptive beta) ----
    p3_dl = deadline - 25
    if time.time() < p3_dl:
        h_curr = best_h.copy()
        s_curr = np.sum(h_curr)
        if abs(s_curr) > 1e-6:
            h_curr /= s_curr
        params, _, _, _ = np.linalg.lstsq(atoms.T, h_curr, rcond=None)

        rng = np.random.default_rng(123)
        inits = [params.copy()]
        for _ in range(4):
            inits.append(np.abs(rng.standard_normal(n_atoms)) * 0.5)

        for p_init in inits[:5]:
            if time.time() > p3_dl - 4:
                break
            try:
                for beta in [60.0, 120.0, 250.0]:
                    if time.time() > p3_dl - 2:
                        break

                    def sobj(params, b=beta):
                        if time.time() > p3_dl - 1:
                            return 1e10
                        h = np.dot(params, atoms)
                        s = np.sum(h)
                        if abs(s) < 1e-9:
                            return 1e10
                        return _score_smooth(h, beta=b)

                    res_s = minimize(sobj, p_init, method='BFGS',
                                     options={'maxiter': 200, 'gtol': 1e-8})
                    h_s = np.dot(res_s.x, atoms)
                    s_s = np.sum(h_s)
                    if abs(s_s) > 1e-6:
                        h_s /= s_s
                        p_init = np.linalg.lstsq(atoms.T, h_s, rcond=None)[0]

                # Final refinement with true objective
                def tobj(params):
                    if time.time() > p3_dl - 0.5:
                        return 1e10
                    h = np.dot(params, atoms)
                    s = np.sum(h)
                    if abs(s) < 1e-9:
                        return 1e10
                    return _autoconv_max_abs(h) / (s * s)

                res_t = minimize(tobj, p_init, method='L-BFGS-B',
                                 options={'maxiter': 300, 'gtol': 1e-10})
                h_t = np.dot(res_t.x, atoms)
                s_t = np.sum(h_t)
                if abs(s_t) > 1e-6:
                    h_t /= s_t
                    v = _score(h_t)
                    if v < best_score:
                        best_score = v
                        best_h = h_t.copy()
            except Exception:
                pass

    # ---- Phase 4: incremental flip local search ----
    p4_dl = deadline - 12
    if time.time() < p4_dl:
        h_fl, v_fl = _incremental_flip_search(best_h, deadline=p4_dl, max_passes=150)
        if v_fl < best_score:
            best_score = v_fl
            best_h = h_fl.copy()

    # ---- Phase 5: perturbation local search ----
    p5_dl = deadline - 6
    if time.time() < p5_dl:
        h_pt, v_pt = _local_search_perturb(best_h, max_passes=5,
                                           deadline=p5_dl, step_size=0.03)
        if v_pt < best_score:
            best_score = v_pt
            best_h = h_pt.copy()

    # ---- Phase 6: simulated-annealing polish ----
    p6_dl = deadline - 2.5
    if time.time() < p6_dl:
        h_sa, v_sa = _gaussian_perturbation(best_h, scale=0.04,
                                            deadline=p6_dl, max_iters=300)
        if v_sa < best_score:
            best_score = v_sa
            best_h = h_sa.copy()

    # ---- Phase 7: final high-precision polish ----
    if time.time() < deadline - 0.5:
        try:
            h_curr = best_h.copy()
            s_curr = np.sum(h_curr)
            if abs(s_curr) > 1e-6:
                h_curr /= s_curr
            params, _, _, _ = np.linalg.lstsq(atoms.T, h_curr, rcond=None)

            def final_obj(params):
                if time.time() > deadline - 0.2:
                    return 1e10
                h = np.dot(params, atoms)
                s = np.sum(h)
                if abs(s) < 1e-9:
                    return 1e10
                return _autoconv_max_abs(h) / (s * s)

            res_f = minimize(final_obj, params, method='L-BFGS-B',
                             options={'maxiter': 500, 'gtol': 1e-12})
            h_f = np.dot(res_f.x, atoms)
            s_f = np.sum(h_f)
            if abs(s_f) > 1e-6:
                h_f /= s_f
                v_f = _score(h_f)
                if v_f < best_score:
                    best_score = v_f
                    best_h = h_f.copy()
        except Exception:
            pass

    # ---- Phase 8: final sign-flip polish ----
    if time.time() < deadline - 0.3:
        try:
            h_ff, v_ff = _local_search_flip(best_h, max_passes=2,
                                            deadline=deadline - 0.1)
            if v_ff < best_score:
                best_score = v_ff
                best_h = h_ff.copy()
        except Exception:
            pass

    # Safety: never worse than baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N

    return best_h

# EVOLVE-BLOCK-END
