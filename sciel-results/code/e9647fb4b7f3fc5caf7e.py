"""
Strategy: Structural Search over Support and Parametric Shapes.
Instead of optimizing 50 independent heights, we search over:
1. Support structure: Which indices are non-zero? (e.g., contiguous block, sparse set)
2. Shape parameters: For a given support, fit a parametric curve (Gaussian, Plateau, etc.)
This reduces the effective dimensionality and allows sampling of "archetypal" shapes
that gradient-based methods on the full 50D space might struggle to find due to
the complex landscape of the L2/L1/Linf ratio.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1          # len(g) = 99
W = 1.0 / (M + 1)      # 1/100
TIME_LIMIT = 148.0     # safety margin before 150s kill
RNG = np.random.RandomState(50)

def _build_y(g):
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y

def compute_value(h):
    """Exact evaluator metric for given heights."""
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _gen_gaussian(center, width, amp):
    x = np.arange(N)
    return amp * np.exp(-0.5 * ((x - center) / width) ** 2)

def _gen_plateau(center, half_width, amp):
    x = np.arange(N)
    # Smooth plateau using cosine taper
    dist = np.abs(x - center)
    h = np.zeros(N)
    # Inside plateau
    inside = dist < half_width
    h[inside] = amp
    # Taper region
    taper = (dist >= half_width) & (dist < half_width + 5)
    if np.any(taper):
        t = (dist[taper] - half_width) / 5.0
        h[taper] = amp * 0.5 * (1 + np.cos(np.pi * t))
    return np.maximum(0, h)

def _gen_bump(center, width, amp, power=2):
    x = np.arange(N)
    t = np.abs(x - center) / width
    h = amp * np.maximum(0, 1 - t**power)
    return h

def _evaluate_and_refine(h0, label=""):
    """
    Take a coarse structural candidate, then refine it using L-BFGS-B
    on the full 50D space but starting from this good structural point.
    """
    if time.time() > TIME_LIMIT - 20:
        return h0, compute_value(h0)
    
    h0 = np.maximum(0, h0)
    v0 = compute_value(h0)
    
    # Use a smoothed objective for refinement to handle the max term
    def obj(x):
        x = np.maximum(0, x)
        g = np.convolve(x, x)
        m = len(g)
        y = _build_y(g)
        a = y[:m + 1]
        b = y[1:m + 2]
        l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
        norm_1 = np.sum(g) / (m + 1)
        # Smooth max
        g_max = np.max(g)
        if g_max <= 0:
            return 0.0
        beta = 100.0
        lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))
        if norm_1 <= 0 or lse <= 0:
            return 0.0
        return -l2 / (norm_1 * lse)
    
    try:
        res = opt.minimize(
            obj, h0, method='L-BFGS-B',
            bounds=[(0, None)] * N,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        h_ref = np.maximum(0, res.x)
        v_ref = compute_value(h_ref)
        if v_ref > v0:
            return h_ref, v_ref
    except Exception:
        pass
    return h0, v0

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_val = 0.0
    best_h = np.ones(N)
    
    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()
    
    # 1. Initial seed
    track(np.ones(N))
    
    # 2. Structural Search: Generate diverse parametric shapes
    # We sample a grid of (center, width, amp, type)
    # Centers: 0 to N-1
    # Widths: 1 to N
    # Amps: 0.5 to 2.0 (scaling doesn't matter for the ratio, but helps numerics)
    
    centers = np.linspace(0, N-1, 20)
    widths = np.linspace(1, N, 15)
    
    candidates = []
    
    # Type 1: Gaussians
    for c in centers:
        for w in widths:
            h = _gen_gaussian(c, w, 1.0)
            if np.sum(h) > 1e-6:
                candidates.append(h)
                
    # Type 2: Plateaus
    for c in centers:
        for hw in widths[:10]: # half-width
            h = _gen_plateau(c, hw, 1.0)
            if np.sum(h) > 1e-6:
                candidates.append(h)
                
    # Type 3: Power bumps
    for c in centers:
        for w in widths:
            for p in [2, 4, 8]:
                h = _gen_bump(c, w, 1.0, p)
                if np.sum(h) > 1e-6:
                    candidates.append(h)
                    
    # Type 4: Two peaks (interference patterns)
    # This is crucial for autocorrelation structures
    for c1 in np.linspace(0, N-1, 10):
        for c2 in np.linspace(0, N-1, 10):
            if abs(c1-c2) < 2: continue
            w = N / 4.0
            h1 = _gen_gaussian(c1, w, 1.0)
            h2 = _gen_gaussian(c2, w, 1.0)
            h = h1 + h2
            if np.sum(h) > 1e-6:
                candidates.append(h)
                
    # Type 5: Random smooth profiles (Fourier mix)
    for _ in range(50):
        freqs = RNG.randint(1, 10, size=8)
        phases = RNG.uniform(0, 2 * np.pi, size=8)
        amps = RNG.uniform(0.1, 1.0, size=8)
        x = np.arange(N)
        h = np.zeros(N)
        for f, p, a in zip(freqs, phases, amps):
            h += a * np.cos(2 * np.pi * f * x / N + p)
        h = np.maximum(0, h)
        if np.sum(h) > 1e-6:
            candidates.append(h)

    # Shuffle to ensure diverse early exploration
    RNG.shuffle(candidates)
    
    # 3. Evaluate and Refine
    # We have ~500 candidates. Refinement takes ~0.1-0.5s each.
    # Budget: ~120s for search/refine.
    
    for i, h_cand in enumerate(candidates):
        if time.time() > deadline - 25:
            break
        
        # Coarse evaluation first to filter out terrible ones?
        # The ratio is scale invariant, so we can just check if it's not zero.
        if np.sum(h_cand) < 1e-6:
            continue
            
        # Refine
        h_ref, v_ref = _evaluate_and_refine(h_cand)
        track(h_ref)
        
        # Progress check
        if i % 20 == 0:
            # print(f"Processed {i}/{len(candidates)}, best: {best_val:.6f}")
            pass

    # 4. Final Polish on the best found
    if time.time() < deadline - 5:
        # Try a few more random perturbations of the best
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            pert = best_h + RNG.normal(0, 0.1 * np.std(best_h), N)
            pert = np.maximum(0, pert)
            h_ref, v_ref = _evaluate_and_refine(pert)
            track(h_ref)
            
    # 5. Very tight final polish
    if time.time() < deadline - 1:
        try:
            def obj_tight(x):
                x = np.maximum(0, x)
                g = np.convolve(x, x)
                m = len(g)
                y = _build_y(g)
                a = y[:m + 1]
                b = y[1:m + 2]
                l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
                norm_1 = np.sum(g) / (m + 1)
                norm_inf = np.max(g)
                if norm_1 <= 0 or norm_inf <= 0:
                    return 0.0
                # Use a very high beta for smooth max to approximate exact
                beta = 500.0
                g_max = np.max(g)
                lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))
                return -l2 / (norm_1 * lse)
                
            res = opt.minimize(
                obj_tight, best_h, method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            track(res.x)
        except Exception:
            pass

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
