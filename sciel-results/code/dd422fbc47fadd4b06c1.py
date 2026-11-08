"""
Combined strategy: low-rank DCT parameterization to reduce dimensionality,
with multi-beta log-sum-exp smoothing, epigraph peak control, and
Nelder-Mead polish on the exact objective.

Key ideas:
1. Represent h as a nonnegative superposition of the first K DCT basis
   functions. Optimize only K amplitudes (K << N), which enforces global
   smoothness and prevents high-frequency spikes that hurt the L2/L1 ratio.
2. After converging in the low-dimensional space, do a full-N polish with
   L-BFGS-B (high-beta smooth surrogate) and Nelder-Mead on the exact value.
3. Epigraph formulation with an auxiliary variable t for peak control.
4. Diverse starts in the low-rank space mapped to full h.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1          # len(g) = 99
W = 1.0 / (M + 1)      # 1/100
TIME_LIMIT = 150.0
RNG = np.random.RandomState(50)

# DCT basis (Type-II DCT, orthonormal)
def _make_dct_basis(N, K):
    """Return K orthonormal DCT-II basis vectors of length N."""
    basis = np.zeros((K, N))
    for k in range(K):
        for n in range(N):
            if k == 0:
                basis[k, n] = np.sqrt(1.0 / N)
            else:
                basis[k, n] = np.sqrt(2.0 / N) * np.cos(np.pi * k * (2 * n + 1) / (2 * N))
    return basis

DCT_K = 20  # number of DCT basis functions (increased from 12)
DCT_BASIS = _make_dct_basis(N, DCT_K)

def _amps_to_h(amps):
    """Map amplitude vector (length DCT_K) to heights (length N), enforcing nonnegativity."""
    h = DCT_BASIS.T @ amps
    return np.maximum(0.0, h)

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

def _smooth_value_full(h, beta=50.0):
    """Smooth surrogate on full h using log-sum-exp for norm_inf."""
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)
    if norm_1 <= 0:
        return 0.0
    g_max = np.max(g)
    if g_max <= 0:
        return 0.0
    lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))
    if lse <= 0:
        return 0.0
    return -l2 / (norm_1 * lse)

def _neg_exact_value(h):
    h = np.maximum(0.0, h)
    return -compute_value(h)

def _smooth_value_lowrank(amps, beta=50.0):
    """Smooth surrogate operating on DCT amplitudes."""
    h = _amps_to_h(amps)
    return _smooth_value_full(h, beta)

def _epigraph_objective(x, beta=100.0):
    """
    Epigraph: minimize -l2/(norm_1 * t) + penalty * max(0, lse(g) - t)^2
    x = [h (N), t (1)]
    """
    h = x[:N]
    t = x[N]
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)
    if norm_1 <= 0 or t <= 0:
        return 1e6
    g_max = np.max(g)
    if g_max > 0:
        lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))
    else:
        lse = 0.0
    violation = np.maximum(0.0, lse - t)
    penalty = 1e4 * violation ** 2
    val = l2 / (norm_1 * t)
    return -val + penalty

def _generate_lowrank_starts():
    """Generate diverse starting points in DCT amplitude space."""
    starts = []
    # Uniform: project ones onto DCT basis
    ones = np.ones(N)
    amps_uniform = DCT_BASIS @ ones
    starts.append(amps_uniform.copy())

    # Gaussian bump projected
    x = np.linspace(0, N - 1, N)
    gauss = np.exp(-0.5 * ((x - N / 2) / (N / 4)) ** 2)
    starts.append(DCT_BASIS @ gauss)

    # Semicircle
    semi = np.sqrt(np.maximum(0.0, 1 - ((2 * x / (N - 1) - 1)) ** 2))
    starts.append(DCT_BASIS @ semi)

    # Cosine bump
    cosbump = 0.5 * (1 + np.cos(2 * np.pi * x / N))
    starts.append(DCT_BASIS @ cosbump)

    # Triangle
    tri = 1 - 2 * np.abs(x - (N - 1) / 2) / (N - 1)
    starts.append(DCT_BASIS @ tri)

    # Power-law profiles
    for p in [0.5, 1.0, 1.5, 2.0]:
        prof = np.power(np.abs(2 * x / (N - 1) - 1), p)
        starts.append(DCT_BASIS @ prof)

    # Two peaks
    p1 = np.exp(-0.5 * ((x - N * 0.25) / (N / 8)) ** 2)
    p2 = np.exp(-0.5 * ((x - N * 0.75) / (N / 8)) ** 2)
    starts.append(DCT_BASIS @ (p1 + p2))

    # Flat-top with taper
    for frac in [0.3, 0.5, 0.7]:
        width = int(N * frac)
        flat = np.zeros(N)
        si = (N - width) // 2
        flat[si:si + width] = 1.0
        taper = 3
        if si >= taper:
            flat[si - taper:si] = np.linspace(0, 1, taper)
        if si + width + taper <= N:
            flat[si + width:si + width + taper] = np.linspace(1, 0, taper)
        starts.append(DCT_BASIS @ flat)

    # Random smooth profiles projected
    for _ in range(20):
        freqs = RNG.randint(1, 10, size=6)
        phases = RNG.uniform(0, 2 * np.pi, size=6)
        amps_rand = RNG.uniform(0.3, 1.0, size=6)
        prof = np.zeros(N)
        for f_, p_, a_ in zip(freqs, phases, amps_rand):
            prof += a_ * np.cos(2 * np.pi * f_ * x / N + p_)
        prof = np.maximum(0.05, prof)
        starts.append(DCT_BASIS @ prof)

    # Random amplitude vectors (small)
    for _ in range(15):
        amps = RNG.normal(0, 0.8, DCT_K)
        # Ensure the resulting h is mostly positive
        h_test = DCT_BASIS.T @ amps
        if np.mean(h_test) > 0.1:
            starts.append(amps)

    # Plateau-like structures: K distinct levels
    for K_plateau in [2, 3, 4, 5]:
        for _ in range(5):
            # Random partition of [0, N) into K segments
            cuts = sorted(RNG.randint(1, N, size=K_plateau - 1))
            cuts = [0] + list(cuts) + [N]
            levels = RNG.uniform(0.2, 1.0, size=K_plateau)
            h_plateau = np.zeros(N)
            for i in range(K_plateau):
                h_plateau[cuts[i]:cuts[i+1]] = levels[i]
            starts.append(DCT_BASIS @ h_plateau)

    # Bimodal plateaus
    for _ in range(5):
        h_bi = np.zeros(N)
        center1 = RNG.randint(5, N//2)
        center2 = N - center1
        width1 = RNG.randint(3, N//4)
        width2 = RNG.randint(3, N//4)
        h_bi[max(0, center1-width1):center1+width1] = RNG.uniform(0.5, 1.0)
        h_bi[max(0, center2-width2):center2+width2] = RNG.uniform(0.5, 1.0)
        starts.append(DCT_BASIS @ h_bi)

    return starts

def _plateau_anneal(start_h, n_iter=5000, T0=0.1, T1=0.001):
    """
    Simulated annealing on plateau structure:
    Variables: K level values and their interval assignments.
    We fix K=4 plateaus and optimize level values + boundary positions.
    """
    K = 4  # number of plateaus
    # Initial: quantize start_h into K levels
    h = np.maximum(0.0, start_h.copy())
    # Simple quantization
    thresholds = np.quantile(h, np.linspace(0, 1, K+1)[1:-1])
    levels = [np.mean(h[(h >= thresholds[i-1]) & (h < thresholds[i])]) if i > 0 else np.mean(h[h < thresholds[0]]) for i in range(K)]
    levels = np.array(levels)
    # Initial boundaries
    boundaries = np.array([N//K * i for i in range(1, K)])
    
    def h_from_plateau(levels, boundaries):
        h_out = np.zeros(N)
        b = [0] + list(boundaries) + [N]
        for i in range(K):
            h_out[b[i]:b[i+1]] = max(0.0, levels[i])
        return h_out
    
    best_h_local = h_from_plateau(levels, boundaries)
    best_val_local = compute_value(best_h_local)
    current_h = best_h_local.copy()
    current_val = best_val_local
    
    for it in range(n_iter):
        T = T0 * (T1 / T0) ** (it / max(1, n_iter - 1))
        
        # Random move: either change a level or move a boundary
        if RNG.rand() < 0.5:
            # Change a level
            i = RNG.randint(K)
            old_level = levels[i].copy()
            levels[i] = max(0.0, levels[i] + RNG.normal(0, 0.1))
            new_h = h_from_plateau(levels, boundaries)
            new_val = compute_value(new_h)
            if new_val > current_val or RNG.rand() < np.exp((new_val - current_val) / max(T, 1e-10)):
                current_h = new_h
                current_val = new_val
                if new_val > best_val_local:
                    best_val_local = new_val
                    best_h_local = new_h.copy()
            else:
                levels[i] = old_level
        else:
            # Move a boundary
            i = RNG.randint(K - 1)
            old_b = boundaries[i].copy()
            delta = RNG.randint(-3, 4)
            new_b = boundaries[i] + delta
            if 1 <= new_b <= N - 1:
                boundaries[i] = new_b
                new_h = h_from_plateau(levels, boundaries)
                new_val = compute_value(new_h)
                if new_val > current_val or RNG.rand() < np.exp((new_val - current_val) / max(T, 1e-10)):
                    current_h = new_h
                    current_val = new_val
                    if new_val > best_val_local:
                        best_val_local = new_val
                        best_h_local = new_h.copy()
                else:
                    boundaries[i] = old_b
        
        if time.time() > deadline_global - 30:
            break
    
    return best_h_local

def construct():
    global deadline_global
    start_time = time.time()
    deadline_global = start_time + (TIME_LIMIT - 15.0)

    best_val = 0.0
    best_h = np.ones(N)

    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        h = np.nan_to_num(h, nan=0.0, posinf=1e6, neginf=0.0)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    # Phase 1: Low-rank DCT optimization
    lowrank_starts = _generate_lowrank_starts()

    # Evaluate starts
    for s in lowrank_starts:
        h_s = _amps_to_h(s)
        track(h_s)

    bounds_low = [(-10, 10)] * DCT_K
    betas = [10.0, 25.0, 50.0, 100.0]

    lowrank_candidates = []

    for s in lowrank_starts:
        if time.time() > deadline_global - 30:
            break
        for beta in betas:
            if time.time() > deadline_global - 30:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _smooth_value_lowrank(x, b),
                    s,
                    method='L-BFGS-B',
                    bounds=bounds_low,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res.fun < 0:
                    h_opt = _amps_to_h(res.x)
                    v_opt = compute_value(h_opt)
                    if v_opt > 0.1 * best_val:
                        lowrank_candidates.append(h_opt)
                        track(h_opt)
            except Exception:
                pass

    # Phase 2: Plateau-based simulated annealing from diverse starts
    plateau_starts = [best_h.copy()]
    for s in lowrank_starts[:10]:
        h_s = _amps_to_h(s)
        plateau_starts.append(h_s)
    
    for ps in plateau_starts:
        if time.time() > deadline_global - 25:
            break
        try:
            h_pa = _plateau_anneal(ps, n_iter=3000)
            track(h_pa)
        except Exception:
            pass

    # Phase 3: Full-N L-BFGS-B polish from low-rank candidates
    # Deduplicate
    unique_candidates = []
    for c in lowrank_candidates:
        if not any(np.allclose(c, uc, atol=1e-4) for uc in unique_candidates):
            unique_candidates.append(c)

    scored = [(compute_value(c), c) for c in unique_candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for _, c in scored[:12]]

    bounds_full = [(0, None)] * N

    for c in top:
        if time.time() > deadline_global - 18:
            break
        c = np.maximum(0.0, c)
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline_global - 18:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _smooth_value_full(x, b),
                    c,
                    method='L-BFGS-B',
                    bounds=bounds_full,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                track(res.x)
            except Exception:
                pass

    # Phase 4: Epigraph optimization on best
    for _ in range(10):
        if time.time() > deadline_global - 12:
            break
        pert = best_h + RNG.normal(0.0, 0.05, N)
        pert = np.maximum(0.0, pert)
        t0 = np.max(np.convolve(pert, pert)) + 0.1
        x0 = np.concatenate([pert, [t0]])
        try:
            res = opt.minimize(
                lambda x: _epigraph_objective(x, beta=100.0),
                x0,
                method='L-BFGS-B',
                bounds=[(0, None)] * N + [(0, None)],
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            track(res.x[:N])
        except Exception:
            pass

    # Phase 5: Nelder-Mead polish on exact objective
    for _ in range(25):
        if time.time() > deadline_global - 6:
            break
        scale = 0.03 * (1.0 + 0.5 * RNG.rand())
        pert = best_h + RNG.normal(0.0, scale, N)
        pert = np.maximum(0.0, pert)
        try:
            res = opt.minimize(
                _neg_exact_value,
                pert,
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            track(res.x)
        except Exception:
            pass

    # Phase 6: Final high-beta L-BFGS-B polish
    for _ in range(12):
        if time.time() > deadline_global - 3:
            break
        pert = best_h + RNG.normal(0.0, 0.02, N)
        pert = np.maximum(0.0, pert)
        try:
            res = opt.minimize(
                lambda x: _smooth_value_full(x, 300.0),
                pert,
                method='L-BFGS-B',
                bounds=bounds_full,
                options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            track(res.x)
        except Exception:
            pass

    # Final Nelder-Mead on best
    if time.time() < deadline_global - 1:
        try:
            res_final = opt.minimize(
                _neg_exact_value,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            track(res_final.x)
        except Exception:
            pass

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
