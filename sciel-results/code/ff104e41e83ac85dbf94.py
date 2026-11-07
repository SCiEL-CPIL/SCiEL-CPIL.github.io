import time
import numpy as np
from scipy.optimize import minimize

N = 50
_START = time.time()
_DEADLINE = _START + 140.0  # safety margin under 150s kill

def _eval(h):
    h = np.asarray(h, dtype=float)
    if h.size != N:
        return -np.inf
    if not np.all(np.isfinite(h)):
        return -np.inf
    if np.any(h < -1e-12):
        return -np.inf
    h = np.maximum(h, 0.0)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    l2 = (w / 3.0) * np.sum(y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2)
    n1 = np.sum(g) / (m + 1)
    ninf = np.max(g)
    if n1 * ninf < 1e-30:
        return -np.inf
    return l2 / (n1 * ninf)

def _neg_eval(h):
    h = np.maximum(np.asarray(h, dtype=float), 0.0)
    v = _eval(h)
    if v == -np.inf:
        return 1e10
    return -v

def _lbfgs(h0, max_iter=500):
    try:
        bounds = [(0, 20)] * N
        res = minimize(_neg_eval, h0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50})
        if res.x is not None:
            h = np.maximum(res.x, 0.0)
            v = _eval(h)
            if np.isfinite(v):
                return h, v
    except Exception:
        pass
    return h0, _eval(h0)

def _nelder_mead(h0, max_iter=300):
    try:
        res = minimize(_neg_eval, h0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-12})
        if res.x is not None:
            h = np.maximum(res.x, 0.0)
            v = _eval(h)
            if np.isfinite(v):
                return h, v
    except Exception:
        pass
    return h0, _eval(h0)

def _cd(h0, budget):
    h = np.array(h0, dtype=float)
    h = np.maximum(h, 0.0)
    best = _eval(h)
    step = 0.5
    while time.time() < budget:
        improved = False
        for i in range(N):
            for d in (step, -step):
                cand = h.copy()
                cand[i] = max(0.0, cand[i] + d)
                v = _eval(cand)
                if v > best + 1e-12:
                    h = cand
                    best = v
                    improved = True
        if not improved:
            step *= 0.6
            if step < 1e-7:
                break
    return h, best

def _make_initial_guesses():
    t = np.linspace(-0.5, 0.5, N)
    guesses = []
    
    # Uniform
    guesses.append(np.ones(N))
    
    # Gaussian bumps at center with various widths
    for sigma in [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        guesses.append(np.exp(-t**2 / (2 * sigma**2)))
    
    # Cosine bumps
    for width in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5]:
        guesses.append(np.maximum(np.cos(np.pi * t / width), 0.0))
    
    # Tapered / triangular
    for a in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]:
        guesses.append(np.maximum(1.0 - a * np.abs(t), 0.0))
        guesses.append(np.maximum(1.0 - a * t**2, 0.0))
    
    # Power law
    for p in [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0]:
        guesses.append(np.maximum(1.0 - (2.0 * np.abs(t))**p, 0.0))
    
    # Exponential
    for a in [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]:
        guesses.append(np.exp(-a * np.abs(t)))
        guesses.append(np.exp(-a * t**2))
    
    # Lorentzian
    for c in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]:
        guesses.append(1.0 / (1.0 + c * t**2))
    
    # Sine-based
    for a in [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0]:
        guesses.append(np.maximum(np.sin(np.pi * t) ** a, 0.0))
        guesses.append(np.maximum(np.cos(np.pi * t * 0.5) ** a, 0.0))
    
    # Two-bump shapes
    for d in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        for s in [0.08, 0.1, 0.15, 0.2]:
            guesses.append(np.exp(-t**2 / (2 * s**2)) + np.exp(-(t-d)**2 / (2 * s**2)))
            guesses.append(np.exp(-t**2 / (2 * s**2)) + np.exp(-(t+d)**2 / (2 * s**2)))
    
    # Three-bump shapes
    for d in [0.1, 0.15, 0.2]:
        for s in [0.08, 0.1, 0.15]:
            guesses.append(np.exp(-t**2 / (2 * s**2)) + np.exp(-(t-d)**2 / (2 * s**2)) + np.exp(-(t+d)**2 / (2 * s**2)))
    
    # Plateau shapes
    for w in [0.1, 0.15, 0.2, 0.25, 0.3]:
        guesses.append(np.where(np.abs(t) < w, 1.0, 0.0))
        guesses.append(np.where(np.abs(t) < w, 1.0, np.exp(-3.0 * (np.abs(t) - w))))
    
    # Smoothstep / sigmoid transitions
    for w in [0.1, 0.15, 0.2, 0.25, 0.3]:
        x = np.abs(t) / w
        guesses.append(np.where(x < 1, 1 - 3*x**2/2 + x**3/2, 0.0))
    
    # Random smooth guesses (Fourier basis)
    rng = np.random.RandomState(42)
    for _ in range(30):
        coeffs = rng.randn(6)
        phases = rng.rand(6) * 2 * np.pi
        guess = np.zeros(N)
        for k in range(6):
            guess += coeffs[k] * np.cos((k + 1) * np.pi * t + phases[k])
        guess = np.maximum(guess, 0.0)
        if np.max(guess) > 1e-6:
            guess /= np.max(guess)
        guesses.append(guess)
    
    # Random with center emphasis
    for _ in range(20):
        rng2 = np.random.RandomState(42 + _ + 100)
        weights = rng2.rand(N)
        center_boost = np.exp(-t**2 / (2 * 0.2**2))
        guess = weights * center_boost
        guess = np.maximum(guess, 0.0)
        if np.max(guess) > 1e-6:
            guess /= np.max(guess)
        guesses.append(guess)
    
    return guesses

def construct():
    # Phase 1: Evaluate all initial guesses
    guesses = _make_initial_guesses()
    candidates = []
    
    for g in guesses:
        g = np.asarray(g, dtype=float)
        if not np.all(np.isfinite(g)):
            continue
        g = np.maximum(g, 0.0)
        if np.max(g) < 1e-12:
            continue
        v = _eval(g)
        if np.isfinite(v):
            candidates.append((v, g))
            
    if not candidates:
        return np.ones(N).tolist()
        
    # Sort by value descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 8 for refinement
    top_k = candidates[:8]
    
    best_h = top_k[0][1].copy()
    best_v = top_k[0][0]
    
    # Phase 2: L-BFGS-B from top 8 guesses
    refined = []
    for v, h in top_k:
        if time.time() > _DEADLINE - 70:
            break
        h_new, v_new = _lbfgs(h, max_iter=500)
        if v_new > best_v:
            best_v = v_new
            best_h = h_new.copy()
        refined.append((v_new, h_new))
        
    # Sort refined
    refined.sort(key=lambda x: x[0], reverse=True)
    top_k_refined = refined[:5]
    
    # Phase 3: Nelder-Mead on top 5 refined
    for v, h in top_k_refined:
        if time.time() > _DEADLINE - 35:
            break
        h_new, v_new = _nelder_mead(h, max_iter=300)
        if v_new > best_v:
            best_v = v_new
            best_h = h_new.copy()
            
    # Phase 4: Another round of L-BFGS from the best Nelder-Mead result
    if time.time() < _DEADLINE - 20:
        h_new, v_new = _lbfgs(best_h, max_iter=500)
        if v_new > best_v:
            best_v = v_new
            best_h = h_new.copy()
    
    # Phase 5: Final Coordinate Descent
    budget = min(time.time() + 15.0, _DEADLINE - 3.0)
    if time.time() < budget:
        h_new, v_new = _cd(best_h, budget)
        if v_new > best_v:
            best_v = v_new
            best_h = h_new.copy()
    
    # Final sanity check
    final = np.maximum(best_h, 0.0)
    final = np.where(np.isfinite(final), final, 0.0)
    if not np.all(np.isfinite(final)) or np.any(final < 0):
        final = np.ones(N)
    
    return final.tolist()
