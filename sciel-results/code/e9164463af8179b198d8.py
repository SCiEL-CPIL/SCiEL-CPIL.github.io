"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv(f):
    """Compute full autoconvolution using FFT."""
    n = len(f)
    nfft = 2 * n
    F = np.fft.rfft(f, n=nfft)
    ac = np.fft.irfft(F * np.conj(F), n=nfft)[:n]
    return ac

def _soft_max_abs_autoconv(f, temp=0.5):
    """Soft-max surrogate for max |autoconv(f)|."""
    ac = _autoconv(f)
    # Use log-sum-exp for soft max of |ac|
    x = np.abs(ac)
    m = np.max(x)
    if m == 0:
        return 0.0
    return m + temp * np.log(np.sum(np.exp((x - m) / temp)))

def _objective(x):
    """Objective: soft-max |autoconv| / (sum f)^2"""
    f = x
    s = np.sum(f)
    if abs(s) < 1e-12:
        return 1e10
    val = _soft_max_abs_autoconv(f)
    return val / (s * s)

def _sign_flip_improve(f, max_iter=200):
    """Local search: flip signs to reduce max |autoconv|/(sum f)^2."""
    best = f.copy()
    best_score = _autoconv_max_abs(best) / (np.sum(best)**2)
    for _ in range(max_iter):
        improved = False
        for i in range(len(best)):
            trial = best.copy()
            trial[i] = -trial[i]
            s = np.sum(trial)
            if abs(s) < 1e-12:
                continue
            ac = _autoconv(trial)
            score = np.max(np.abs(ac)) / (s * s)
            if score < best_score * 0.999:  # 0.1% improvement threshold
                best = trial
                best_score = score
                improved = True
        if not improved:
            break
    return best

def _autoconv_max_abs(f):
    ac = _autoconv(f)
    return np.max(np.abs(ac))

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # 140s safety margin before 150s kill
    
    # Start with a structured initial guess using cosine basis
    # Low-frequency cosine modes tend to have smoother autocorrelation
    x0 = np.zeros(N)
    for k in range(1, 6):  # first few cosine modes
        x0 += np.cos(2 * np.pi * k * np.arange(N) / N)
    # Normalize
    x0 = x0 / np.linalg.norm(x0) * np.sqrt(N)
    
    # Ensure non-trivial sum
    if abs(np.sum(x0)) < 1e-6:
        x0[0] += 1.0
    
    best_f = x0.copy()
    best_score = _autoconv_max_abs(best_f) / (np.sum(best_f)**2)
    
    # Phase 1: Continuous optimization with L-BFGS on soft-max surrogate
    try:
        res = minimize(
            _objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 300, 'maxfun': 5000, 'gtol': 1e-6}
        )
        if time.time() > deadline:
            raise TimeoutError()
        if res.x is not None:
            cand = res.x
            s = np.sum(cand)
            if abs(s) > 1e-6:
                score = _autoconv_max_abs(cand) / (s * s)
                if score < best_score:
                    best_f = cand
                    best_score = score
    except Exception:
        pass
    
    # Phase 2: Sign-flip local search
    if time.time() < deadline - 20:
        # Convert to sign-based: keep magnitudes, flip signs
        mags = np.abs(best_f)
        signs = np.sign(best_f)
        signs[signs == 0] = 1
        
        # Optimize signs via local search
        f_signs = mags * signs
        improved = True
        max_rounds = 50
        for _ in range(max_rounds):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 10:
                    break
                trial = f_signs.copy()
                trial[i] = -trial[i]
                s = np.sum(trial)
                if abs(s) < 1e-12:
                    continue
                ac = _autoconv(trial)
                score = np.max(np.abs(ac)) / (s * s)
                if score < best_score * 0.995:
                    f_signs = trial
                    best_score = score
                    best_f = trial
                    improved = True
            if not improved:
                break
    
    # Phase 3: Final continuous polish if time permits
    if time.time() < deadline - 30 and np.linalg.norm(best_f) > 1e-6:
        try:
            res2 = minimize(
                _objective,
                best_f,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 2000, 'gtol': 1e-8}
            )
            if time.time() > deadline:
                raise TimeoutError()
            if res2.x is not None:
                cand = res2.x
                s = np.sum(cand)
                if abs(s) > 1e-6:
                    score = _autoconv_max_abs(cand) / (s * s)
                    if score < best_score:
                        best_f = cand
                        best_score = score
        except Exception:
            pass
    
    # Ensure we return a valid sequence
    if np.all(np.abs(best_f) < 1e-12):
        best_f = np.ones(N)
    
    return best_f
# EVOLVE-BLOCK-END
