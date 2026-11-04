import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 600
DELTA = (1.0 / 2.0) / N  # interval width = 1/1200
BIN_CENTERS = -0.25 + (np.arange(N) + 0.5) * DELTA

# Fine grid for accurate autoconvolution evaluation
FINE_N = 12000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / DELTA).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()


def compute_autoconv(h):
    """Compute autoconvolution of step function with heights h on N equal intervals using fine grid."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    
    f_fine = h[CELL]
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX
    return conv


def compute_score(h):
    """Compute the normalized max autoconvolution score (L1-based normalization)."""
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    if np.any(h < -1e-12):
        return np.inf
    h = np.maximum(h, 0.0)
    
    autoconv = compute_autoconv(h)
    max_ac = np.max(autoconv)
    integral = DELTA * np.sum(h)
    if integral <= 0:
        return np.inf
    score = max_ac / (integral ** 2)
    return score


def params_to_h_gaussian(params):
    """Convert Gaussian superposition parameters to step heights."""
    K = len(params) // 3
    h = np.zeros(N)
    for i in range(K):
        a_raw = params[3 * i]
        x_c = params[3 * i + 1]
        s_raw = params[3 * i + 2]
        
        # Softplus for positive amplitude
        a = np.log1p(np.exp(np.clip(a_raw, -50, 50)))
        # Exp for positive width
        s = np.exp(np.clip(s_raw, -10, 10))
        s = max(s, 1e-4)
        # Clip center to domain
        x_c = np.clip(x_c, -0.25, 0.25)
        
        h += a * np.exp(-0.5 * ((BIN_CENTERS - x_c) / s) ** 2)
    
    return h


def params_to_h_block(params):
    """
    Convert block structure parameters to step heights.
    params: [b1_start, b1_end, w1, b2_start, b2_end, w2, ...]
    Each block is an indicator function on [start, end] with weight w.
    Blocks are disjoint and symmetric pairs are encouraged by initialization.
    """
    K = len(params) // 3
    h = np.zeros(N)
    for i in range(K):
        s_raw = params[3 * i]
        e_raw = params[3 * i + 1]
        w_raw = params[3 * i + 2]
        
        # Convert raw params to bin indices [0, N-1]
        s_idx = int(np.clip(np.round(s_raw), 0, N - 1))
        e_idx = int(np.clip(np.round(e_raw), 0, N - 1))
        
        if e_idx >= s_idx:
            # Weight is non-negative via softplus
            w = np.log1p(np.exp(np.clip(w_raw, -50, 50)))
            h[s_idx:e_idx + 1] += w
    
    return h


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    # Initial best: uniform
    best_h = np.ones(N, dtype=np.float64)
    best_val = compute_score(best_h)
    
    def update_best(h, val):
        nonlocal best_h, best_val
        if val < best_val and np.isfinite(val):
            best_val = val
            best_h = h.copy()
    
    rng = np.random.default_rng(42)
    
    # === Phase 1: Gaussian superposition with L-BFGS-B ===
    K_options = [3, 5, 7]
    
    for K in K_options:
        if time.time() > deadline - 40.0:
            break
        
        n_starts = 15
        for start in range(n_starts):
            if time.time() > deadline - 30.0:
                break
            
            # Generate diverse initializations
            p0 = np.zeros(K * 3)
            if start == 0:
                # Uniform-ish spread
                for i in range(K):
                    p0[3 * i] = np.log(1.0 / K)
                    p0[3 * i + 1] = float(rng.uniform(-0.2, 0.2))
                    p0[3 * i + 2] = np.log(0.1 + float(rng.uniform(0, 0.2)))
            elif start == 1:
                # Single dominant bump
                p0[0] = np.log(1.0)
                p0[1] = 0.0
                p0[2] = np.log(0.15)
                for i in range(1, K):
                    p0[3 * i] = np.log(0.01)
                    p0[3 * i + 1] = float(rng.uniform(-0.2, 0.2))
                    p0[3 * i + 2] = np.log(0.05)
            elif start == 2:
                # Asymmetric pair
                x1 = float(rng.uniform(-0.2, -0.05))
                x2 = float(rng.uniform(0.05, 0.2))
                p0[0] = float(rng.uniform(0, 1))
                p0[1] = x1
                p0[2] = float(rng.uniform(np.log(0.1), np.log(0.2)))
                if K >= 2:
                    p0[3] = float(rng.uniform(0, 1))
                    p0[4] = x2
                    p0[5] = float(rng.uniform(np.log(0.1), np.log(0.2)))
                for i in range(2, K):
                    p0[3 * i] = float(rng.uniform(-3, 0))
                    p0[3 * i + 1] = float(rng.uniform(-0.25, 0.25))
                    p0[3 * i + 2] = float(rng.uniform(np.log(0.05), np.log(0.2)))
            else:
                # Random smooth shapes
                for i in range(K):
                    p0[3 * i] = float(rng.uniform(-2, 1))
                    p0[3 * i + 1] = float(rng.uniform(-0.25, 0.25))
                    p0[3 * i + 2] = float(rng.uniform(np.log(0.05), np.log(0.3)))
            
            def objective_gauss(params):
                h = params_to_h_gaussian(params)
                val = compute_score(h)
                if not np.isfinite(val):
                    return 1e10
                return val
            
            try:
                res = minimize(
                    objective_gauss,
                    p0,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                if res.success or res.fun < best_val:
                    h_candidate = params_to_h_gaussian(res.x)
                    val = compute_score(h_candidate)
                    update_best(h_candidate, val)
            except Exception:
                pass
    
    # === Phase 2: Block structure optimization with Differential Evolution ===
    # Try different numbers of blocks
    n_blocks_options = [2, 3, 4, 5]
    
    for n_blocks in n_blocks_options:
        if time.time() > deadline - 30.0:
            break
        
        n_params = 3 * n_blocks
        bounds = []
        for _ in range(n_blocks):
            bounds.append((0.0, float(N - 1)))   # start bin
            bounds.append((0.0, float(N - 1)))   # end bin
            bounds.append((-5.0, 5.0))           # weight raw
        
        def objective_block(params):
            h = params_to_h_block(params)
            val = compute_score(h)
            if not np.isfinite(val):
                return 1e10
            return val
        
        # Initial population with symmetric block structures
        init_pop = []
        for _ in range(20):
            p = np.zeros(n_params)
            # Create symmetric pairs of blocks
            for i in range(n_blocks):
                # Center of domain is N//2
                center = N // 2
                # Random offset
                offset = int(rng.uniform(50, N // 2 - 50))
                width = int(rng.uniform(20, 100))
                
                if i % 2 == 0:
                    # Left block
                    s = max(0, center - offset - width // 2)
                    e = min(N - 1, center - offset + width // 2)
                else:
                    # Right block
                    s = max(0, center + offset - width // 2)
                    e = min(N - 1, center + offset + width // 2)
                
                p[3 * i] = float(s)
                p[3 * i + 1] = float(e)
                p[3 * i + 2] = float(rng.uniform(-1, 1))
            
            init_pop.append(p)
        
        try:
            result = differential_evolution(
                objective_block,
                bounds,
                maxiter=50,
                popsize=15,
                seed=42,
                polish=True,
                tol=1e-4,
                maxfev=1500,
                init=np.array(init_pop)
            )
            
            if result.fun < best_val:
                h_candidate = params_to_h_block(result.x)
                val = compute_score(h_candidate)
                update_best(h_candidate, val)
                
        except Exception:
            pass
    
    # === Phase 3: Direct bin-level optimization with L-BFGS-B ===
    if time.time() < deadline - 20.0:
        bounds = [(0.0, None)] * N
        
        def smooth_objective(h, beta=200.0):
            h = np.clip(h, 0.0, None)
            s = h.sum()
            if s < 1e-15:
                return 1e10
            integral_sq = (s * DELTA) ** 2
            
            f_fine = h[CELL]
            F = np.fft.rfft(f_fine, N_FFT)
            conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
            conv *= DX
            
            m = np.max(conv)
            if m < 1e-15:
                return 1e10
            soft_max = m + np.log(np.sum(np.exp(conv - m))) / beta
            return soft_max / integral_sq
        
        try:
            res_direct = minimize(
                smooth_objective,
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            h_candidate = np.clip(res_direct.x, 0.0, None)
            val = compute_score(h_candidate)
            update_best(h_candidate, val)
        except Exception:
            pass
    
    # === Phase 4: Local search polish ===
    if time.time() < deadline - 5.0:
        rng2 = np.random.default_rng(777)
        for it in range(3000):
            if time.time() > deadline - 2.0:
                break
            
            n_perturb = int(rng2.integers(5, 20))
            idx = rng2.choice(N, size=n_perturb, replace=False)
            h_try = best_h.copy()
            h_try[idx] += rng2.normal(0, 0.02, size=n_perturb)
            h_try = np.clip(h_try, 0, None)
            
            val = compute_score(h_try)
            update_best(h_try, val)
    
    return list(best_h)
