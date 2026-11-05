"""Autocorrelation inequality optimizer using CCP and spectral methods."""
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.ndimage import gaussian_filter1d
import time

N = 400
FFT_SIZE = 2 * N - 1  # 799, use next power of 2 for efficiency
FFT_SIZE_PAD = 1024  # next power of 2 >= 799


def _eval(h):
    """Compute the objective value. Returns (value, valid)."""
    s = np.sum(h)
    if abs(s) < 1e-14:
        return np.inf, False
    if not np.all(np.isfinite(h)):
        return np.inf, False
    # Use FFT for fast convolution
    h_fft = np.fft.rfft(h, FFT_SIZE_PAD)
    conv = np.fft.irfft(h_fft * h_fft, FFT_SIZE_PAD)[:FFT_SIZE]
    max_conv = np.max(conv)
    val = abs(2.0 * N * max_conv / (s * s))
    return val, True


def _eval_fast(h):
    """Fast evaluation using pre-allocated FFT buffer."""
    s = np.sum(h)
    if abs(s) < 1e-14:
        return np.inf, False
    h_fft = np.fft.rfft(h, FFT_SIZE_PAD)
    conv = np.fft.irfft(h_fft * h_fft, FFT_SIZE_PAD)[:FFT_SIZE]
    max_conv = np.max(conv)
    val = abs(2.0 * N * max_conv / (s * s))
    return val, True


def _soft_max_conv(h, beta=100.0):
    """Compute log-sum-exp approximation of max(conv(h,h))."""
    h_fft = np.fft.rfft(h, FFT_SIZE_PAD)
    conv = np.fft.irfft(h_fft * h_fft, FFT_SIZE_PAD)[:FFT_SIZE]
    max_c = np.max(conv)
    if max_c < -700 / beta:
        return max_c  # avoid underflow
    shifted = conv - max_c
    lse = max_c + np.log(np.sum(np.exp(beta * shifted))) / beta
    return lse


def _soft_max_grad(h, beta=100.0):
    """Gradient of soft-max of conv(h,h) w.r.t. h."""
    h_fft = np.fft.rfft(h, FFT_SIZE_PAD)
    conv = np.fft.irfft(h_fft * h_fft, FFT_SIZE_PAD)[:FFT_SIZE]
    max_c = np.max(conv)
    shifted = conv - max_c
    weights = np.exp(beta * shifted)
    w_sum = np.sum(weights)
    if w_sum < 1e-300:
        return np.zeros(N)
    weights /= w_sum
    # grad[i] = 2 * sum_j weights[j+i] * h[j] for i=0..N-1
    # Use correlate: corr_full = correlate(weights, h, 'full')
    # For lag i>=0, index in full = i + (N-1)
    corr_full = np.correlate(weights, h, mode='full')
    grad_lse = 2.0 * corr_full[N-1:2*N-1]
    return grad_lse


def _make_initial_points(rng):
    """Generate diverse initial points."""
    points = []
    t = np.linspace(0, 1, N, endpoint=False)
    i_arr = np.arange(N)
    
    # Seed
    points.append(np.ones(N))
    
    # Spectral-null: remove DC and low frequencies from random
    for _ in range(10):
        h = rng.uniform(-1, 1, N)
        h_fft = np.fft.rfft(h)
        h_fft[:5] = 0  # remove DC and low freq
        h_null = np.fft.irfft(h_fft, N)
        s = np.sum(h_null)
        if abs(s) > 1e-6:
            points.append(h_null / s)
        else:
            points.append(h_null + 0.01)
    
    # High-frequency random (anti-correlated)
    for _ in range(10):
        h = rng.uniform(-1, 1, N)
        h_fft = np.fft.rfft(h)
        h_fft[1:N//4] = 0  # keep only high freq
        h_hf = np.fft.irfft(h_fft, N)
        s = np.sum(h_hf)
        if abs(s) > 1e-6:
            points.append(h_hf / s)
        else:
            points.append(h_hf + 0.01)
    
    # Multi-mode cosines with random phases
    for n_modes in [2, 3, 4, 5, 7]:
        for _ in range(5):
            freqs = sorted(rng.randint(1, 60, n_modes))
            while len(set(freqs)) < n_modes:
                freqs = sorted(rng.randint(1, 60, n_modes))
            phases = rng.uniform(0, 2 * np.pi, n_modes)
            amps = rng.uniform(0.5, 2.0, n_modes)
            h = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h += a * np.cos(2 * np.pi * f * t + p)
            s = np.sum(h)
            if abs(s) > 1e-6:
                points.append(h / s)
            else:
                points.append(h + 0.01)
    
    # Bimodal sparse
    for _ in range(10):
        h = np.zeros(N)
        k = rng.randint(50, 200)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.choice([-1, 1], size=k)
        s = np.sum(h)
        if abs(s) > 1e-6:
            points.append(h / s)
        else:
            points.append(h + 0.01)
    
    # Smooth Gaussian-filtered random
    for sigma in [3, 7, 15, 30]:
        for _ in range(3):
            h = rng.uniform(-1, 1, N)
            h = gaussian_filter1d(h, sigma=sigma)
            s = np.sum(h)
            if abs(s) > 1e-6:
                points.append(h / s)
            else:
                points.append(h + 0.01)
    
    # Block patterns
    for block in [10, 20, 50]:
        n_blocks = N // block
        for _ in range(3):
            h = np.zeros(N)
            for b in range(n_blocks):
                h[b*block:(b+1)*block] = rng.choice([-1, 1])
            s = np.sum(h)
            if abs(s) > 1e-6:
                points.append(h / s)
            else:
                points.append(h + 0.01)
    
    # Cosine single mode
    for freq in [1, 2, 3, 5, 7, 10, 15, 20, 30]:
        h = np.cos(2 * np.pi * freq * t)
        s = np.sum(h)
        if abs(s) > 1e-6:
            points.append(h / s)
        else:
            points.append(h + 0.01)
    
    return points


def _ccp_step(h, C_target, max_iter_lp=20):
    """
    One CCP iteration: fix product terms linearly around current h.
    Minimize C s.t. sum(h)=1 and for all k: sum_{i+j=k} h_i*h_j <= C
    Linearize: h_i*h_j ≈ h_i*h_j^0 + h_i^0*h_j + (h_i - h_i^0)*(h_j - h_j^0)
    Drop second-order, keep: h_i*h_j^0 + h_i^0*h_j <= C (linear in h)
    
    Actually the constraint sum_{i+j=k} h_i h_j <= C is quadratic.
    CCP: at iterate h^0, linearize each product h_i*h_j:
    h_i*h_j ≈ h_i*h_j^0 + h_i^0*(h_j - h_j^0)  (asymmetric)
    or better: h_i*h_j ≈ (h_i + h_i^0)*(h_j + h_j^0)/2 - h_i^0*h_j^0/2
    Simplest: h_i*h_j ≈ h_i^0*h_j + h_i*h_j^0 - h_i^0*h_j^0
    
    So constraint becomes: sum_{i+j=k} [h_i^0*h_j + h_i*h_j^0 - h_i^0*h_j^0] <= C
    = sum_j h_i^0 h_j (over i+j=k) + sum_i h_i h_j^0 (over i+j=k) - sum_{i+j=k} h_i^0 h_j^0 <= C
    
    The first term: for fixed k, sum_{i=0}^{k} h_i^0 * h_{k-i} (with bounds)
    This is a linear function of h.
    Similarly second term is linear in h.
    """
    h0 = h.copy()
    s0 = np.sum(h0)
    if abs(s0) < 1e-14:
        return h0, np.inf
    
    # Build the linearized constraints matrix A and vector b
    # Variable: h (length N), with constraint sum(h) = 1
    # For each lag k = -(N-1) .. (N-1), constraint:
    #   sum_{i+j=k} [h0_i * h_j + h_i * h0_j - h0_i * h0_j] <= C
    
    # We'll build A matrix where each row is a linear constraint on h
    # A[k, i] coefficient for h_i in constraint k
    
    n_lags = 2 * N - 1  # lags from -(N-1) to (N-1)
    
    # For lag k (where k ranges 0 to 2N-2, representing actual lag k - (N-1)):
    # pairs (i,j) with i+j = k, 0<=i<N, 0<=j<N
    # Linearized: h0[i]*h[j] + h[i]*h0[j] - h0[i]*h0[j]
    # Coefficient of h[j]: h0[i]
    # Coefficient of h[i]: h0[j]
    # Constant: -h0[i]*h0[j]
    
    A = np.zeros((n_lags, N))
    const = np.zeros(n_lags)
    
    for k in range(n_lags):
        # i ranges from max(0, k-N+1) to min(N-1, k)
        i_min = max(0, k - N + 1)
        i_max = min(N - 1, k)
        for i in range(i_min, i_max + 1):
            j = k - i
            if 0 <= j < N:
                A[k, j] += h0[i]
                A[k, i] += h0[j]
                const[k] += -h0[i] * h0[j]
    
    # Objective: minimize C (auxiliary variable)
    # Variables: [h_0, ..., h_{N-1}, C]  (length N+1)
    # Objective: [0, 0, ..., 0, 1]
    c_obj = np.zeros(N + 1)
    c_obj[N] = 1.0  # minimize C
    
    # Constraints:
    # 1) sum(h) = 1  ->  [1,1,...,1, 0] . x = 1
    # 2) For each lag: A[k,:] @ h - C <= -const[k]
    #    i.e., A[k,:] @ h - C + const[k] <= 0
    #    A_ub row: [A[k,0], ..., A[k,N-1], -1], b_ub = -const[k]
    
    # Equality constraint: sum(h) = 1
    A_eq = np.zeros((1, N + 1))
    A_eq[0, :N] = 1.0
    b_eq = np.array([1.0])
    
    # Inequality constraints
    A_ub = np.zeros((n_lags, N + 1))
    b_ub = np.zeros(n_lags)
    for k in range(n_lags):
        A_ub[k, :N] = A[k]
        A_ub[k, N] = -1.0
        b_ub[k] = -const[k]
    
    # Bounds: h_i free (can be negative), C free
    # But for numerical stability, bound h_i to [-10, 10] and C to [-1, 10]
    bounds = [(-10, 10)] * N + [(-1, 10)]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs', options={'presolve': True,
                      'dual_feasibility_tolerance': 1e-9,
                      'primal_feasibility_tolerance': 1e-9})
        if res.status == 0:
            h_new = res.x[:N]
            C_val = res.x[N]
            return h_new, C_val
    except Exception:
        pass
    
    return h0, np.inf


def construct():
    """
    Construct a sequence of N=400 step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    """
    start_time = time.time()
    DEADLINE = 145  # 5s safety margin before 150s kill
    
    rng = np.random.RandomState(42)
    
    best_value = 2.0
    best_h = np.ones(N).astype(float)
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # --- Phase 1: Evaluate initial points and pick top candidates ---
    initial_points = _make_initial_points(rng)
    
    init_results = []
    for h_init in initial_points:
        if time_left() < 30:
            break
        val, valid = _eval(h_init)
        if valid and val < best_value:
            best_value = val
            best_h = h_init.copy()
        if valid:
            init_results.append((val, h_init))
    
    # Sort and take top candidates
    init_results.sort(key=lambda x: x[0])
    
    # --- Phase 2: CCP iteration from best candidates ---
    n_ccp_starts = min(30, len(init_results))
    
    for idx in range(n_ccp_starts):
        if time_left() < 10:
            break
        
        h_curr = init_results[idx][1].copy()
        s = np.sum(h_curr)
        if abs(s) < 1e-10:
            h_curr += 0.01
            s = np.sum(h_curr)
        h_curr = h_curr / s  # normalize to sum=1
        
        val_curr, valid_curr = _eval(h_curr)
        if not valid_curr or val_curr >= best_value:
            # Try to improve via CCP
            pass
        
        # Run CCP iterations
        C_init = np.max(np.convolve(h_curr, h_curr))
        C = C_init
        for it in range(15):
            if time_left() < 5:
                break
            h_new, C_new = _ccp_step(h_curr, C)
            if C_new >= C:
                # No improvement, break
                break
            h_curr = h_new
            C = C_new
            # Evaluate actual value
            s_curr = np.sum(h_curr)
            if abs(s_curr) > 1e-14:
                val_act, valid_act = _eval(h_curr)
                if valid_act and val_act < best_value:
                    best_value = val_act
                    best_h = h_curr.copy()
        
        # After CCP, try L-BFGS-B polish on soft-max
        if time_left() < 3:
            break
        
        s_curr = np.sum(h_curr)
        if abs(s_curr) < 1e-10:
            h_curr += 0.01
        
        def obj_soft(x, beta=120.0):
            s = np.sum(x)
            if abs(s) < 1e-12:
                return 1e10
            lse = _soft_max_conv(x, beta)
            val = abs(2.0 * N * lse / (s * s))
            if not np.isfinite(val):
                return 1e10
            return val
        
        def grad_soft(x, beta=120.0):
            s = np.sum(x)
            if abs(s) < 1e-12:
                return np.zeros(N)
            lse = _soft_max_conv(x, beta)
            glse = _soft_max_grad(x, beta)
            # f = 2*N*lse / s^2
            # df/dx = 2*N*glse/s^2 - 4*N*lse/s^3
            df = 2.0 * N * glse / (s * s) - 4.0 * N * lse / (s ** 3)
            # Handle abs: if f>0 grad=df, else grad=-df
            f_val = 2.0 * N * lse / (s * s)
            if f_val >= 0:
                return df
            else:
                return -df
        
        try:
            result = minimize(
                lambda x: obj_soft(x, 120.0),
                h_curr,
                jac=lambda x: grad_soft(x, 120.0),
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 600, 'gtol': 1e-9}
            )
            h_opt = result.x
            val_opt, valid_opt = _eval(h_opt)
            if valid_opt and val_opt < best_value:
                best_value = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass
        
        # Try with higher beta
        if time_left() < 3:
            break
        try:
            result = minimize(
                lambda x: obj_soft(x, 250.0),
                h_curr,
                jac=lambda x: grad_soft(x, 250.0),
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 600, 'gtol': 1e-9}
            )
            h_opt = result.x
            val_opt, valid_opt = _eval(h_opt)
            if valid_opt and val_opt < best_value:
                best_value = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # --- Phase 3: Polish best solution with high-precision L-BFGS-B ---
    if time_left() > 5:
        s_best = np.sum(best_h)
        if abs(s_best) > 1e-14:
            for beta in [200.0, 350.0, 500.0]:
                if time_left() < 3:
                    break
                try:
                    result = minimize(
                        lambda x: obj_soft(x, beta),
                        best_h,
                        jac=lambda x: grad_soft(x, beta),
                        method='L-BFGS-B',
                        options={'maxiter': 500, 'maxfun': 1000, 'gtol': 1e-10,
                                 'ftol': 1e-10}
                    )
                    h_opt = result.x
                    val_opt, valid_opt = _eval(h_opt)
                    if valid_opt and val_opt < best_value:
                        best_value = val_opt
                        best_h = h_opt.copy()
                except Exception:
                    pass
    
    # --- Phase 4: Final random perturbations ---
    if time_left() > 3:
        for _ in range(50):
            if time_left() < 2:
                break
            scale = 0.01
            h_perturb = best_h + rng.normal(0, scale, N)
            val_p, valid_p = _eval(h_perturb)
            if valid_p and val_p < best_value:
                best_value = val_p
                best_h = h_perturb.copy()
                # Quick refinement
                if time_left() > 3:
                    try:
                        result = minimize(
                            lambda x: obj_soft(x, 300.0),
                            best_h,
                            jac=lambda x: grad_soft(x, 300.0),
                            method='L-BFGS-B',
                            options={'maxiter': 200, 'maxfun': 400, 'gtol': 1e-9}
                        )
                        h_opt = result.x
                        val_opt, valid_opt = _eval(h_opt)
                        if valid_opt and val_opt < best_value:
                            best_value = val_opt
                            best_h = h_opt.copy()
                    except Exception:
                        pass
    
    # Final validation
    val_final, valid_final = _eval(best_h)
    if not valid_final:
        best_h = np.ones(N).astype(float)
        best_value = 2.0
    
    return best_h
