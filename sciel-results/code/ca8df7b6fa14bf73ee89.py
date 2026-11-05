"""
Autocorrelation inequality optimizer.
Minimizes |2*N*max(conv(h,h))/sum(h)^2| over N=400 step heights.
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 400

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    """
    start_time = time.time()
    DEADLINE = 142  # 8s safety margin before 150s kill

    best_value = np.inf
    best_h = np.ones(N).astype(float)

    # Precompute Chebyshev basis for low-dimensional parameterization
    i_arr = np.arange(N, dtype=float)
    # Normalize to [-1, 1] for Chebyshev
    x_norm = 2.0 * i_arr / (N - 1) - 1.0
    # Compute Chebyshev polynomials up to order K-1
    K = 8  # Number of basis functions
    cheb_basis = np.zeros((K, N))
    for k in range(K):
        cheb_basis[k] = np.cos(k * np.arccos(np.clip(x_norm, -1.0, 1.0)))

    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid

    def coeffs_to_h(coeffs):
        """Convert low-dimensional coefficients to N-dimensional h."""
        h = np.zeros(N)
        for k in range(min(len(coeffs), K)):
            h += coeffs[k] * cheb_basis[k]
        return h

    def evaluate_coeffs(coeffs):
        """Evaluate the objective for a coefficient vector."""
        h = coeffs_to_h(coeffs)
        return evaluate(h)

    val_seed, _ = evaluate(best_h)
    best_value = val_seed

    # Precompute FFT sizes for fast convolution
    fft_size = 1  # smallest power of 2 >= 2*N-1
    while fft_size < 2 * N - 1:
        fft_size *= 2

    def fast_convolve(h):
        """Fast convolution via FFT."""
        H = np.fft.rfft(h, n=fft_size)
        conv = np.fft.irfft(H * H, n=fft_size)
        return conv[:2 * N - 1]

    def fast_cross_corr(weights, h):
        """
        Compute grad_lse[i] = sum_{m=0}^{N-1} weights[i+m] * h[m] for i=0..N-1
        using FFT-based cross-correlation.
        
        This is equivalent to: for each i, dot product of weights[i:i+N] with h.
        Using the identity: sum_m weights[i+m]*h[m] = cross_correlation at lag i
        where cross_correlation(a,b)[k] = sum_j a[j+k]*b[j]
        
        We can compute this via: ifft(fft(weights) * conj(fft(h))) appropriately shifted.
        Actually, let's use: the operation is a "sliding dot product".
        
        Alternative: use the fact that this is a convolution of weights with reversed h,
        or use np.correlate.
        
        For N=400, direct computation with stride tricks is fast enough:
        """
        # Use stride tricks for efficient sliding window dot product
        # weights has length 2N-1, we need N windows of length N
        # grad[i] = sum_{m=0}^{N-1} weights[i+m] * h[m]
        # This is a matrix-vector product where M[i,m] = weights[i+m]
        # For N=400, building the matrix is 400*400=160K floats = ~1.3MB, very fast
        idx = np.arange(N)
        M = weights[idx[:, None] + idx[None, :]]
        return M @ h

    BETA = 200.0  # Higher beta for tighter max approximation

    def smooth_objective_and_grad(x):
        """Smooth surrogate using log-sum-exp of conv entries, with analytic gradient."""
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10, np.zeros_like(h)
        conv = fast_convolve(h)
        max_c = np.max(conv)
        shifted = conv - max_c
        # Numerically stable log-sum-exp
        max_shifted = np.max(shifted)
        if max_shifted > 500:  # overflow protection
            return 1e10, np.zeros_like(h)
        
        beta = BETA
        scaled = beta * (shifted - max_shifted)
        exp_scaled = np.exp(scaled)
        sum_exp = np.sum(exp_scaled)
        if sum_exp < 1e-300:
            return 1e10, np.zeros_like(h)
        
        lse = max_c + max_shifted + np.log(sum_exp) / beta
        weights = exp_scaled / sum_exp  # normalized weights
        
        # Gradient of LSE w.r.t. h
        # d_lse/dh[i] = 2 * sum_m weights[i+m] * h[m]
        grad_lse = 2.0 * fast_cross_corr(weights, h)
        
        # f = 2*N*lse / s^2
        # df/dh = 2*N*grad_lse/s^2 - 4*N*lse/(s^3) * ones
        f_val = 2.0 * N * lse / (s * s)
        
        if not np.isfinite(f_val):
            return 1e10, np.zeros_like(h)
        
        df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
        
        # Objective is |f|, so gradient is sign(f) * df_dh
        if f_val >= 0:
            grad = df_dh
        else:
            grad = -df_dh
        
        return abs(f_val), grad

    def smooth_objective(x):
        val, _ = smooth_objective_and_grad(x)
        return val

    def smooth_grad(x):
        _, grad = smooth_objective_and_grad(x)
        return grad

    rng = np.random.RandomState(42)

    # --- Generate diverse initial points ---
    initial_points = []

    # Seed
    initial_points.append(np.ones(N).astype(float))

    # Random uniform
    for _ in range(8):
        initial_points.append(rng.uniform(-1, 1, N))

    # Bimodal: mostly zero with random spikes
    for _ in range(8):
        h = np.zeros(N)
        k = rng.randint(50, 300)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.choice([-1, 1], size=k)
        initial_points.append(h)

    # Alternating signs
    h_alt = np.ones(N)
    h_alt[1::2] = -1
    initial_points.append(h_alt)

    # Low-frequency cosines
    t = np.linspace(0, 1, N, endpoint=False)
    for freq in [1, 2, 3, 5, 7, 10]:
        initial_points.append(np.cos(2 * np.pi * freq * t))
        initial_points.append(np.cos(2 * np.pi * freq * t) + 0.1 * rng.normal(0, 1, N))

    # Parameterized bump profiles
    i_arr = np.arange(N)
    for center in [N // 4, N // 2, 3 * N // 4]:
        for width in [15, 25, 40, 60, 80, 120, 160, 200]:
            h_bump = np.exp(-((i_arr - center) ** 2) / (2 * width ** 2))
            initial_points.append(h_bump)
            initial_points.append(-h_bump)
            h_double = np.exp(-((i_arr - center) ** 2) / (2 * width ** 2)) - np.exp(-((i_arr - (N - 1 - center)) ** 2) / (2 * width ** 2))
            initial_points.append(h_double)
            h_off = np.exp(-((i_arr - center) ** 2) / (2 * width ** 2)) + 0.3 * np.ones(N)
            initial_points.append(h_off)

    # Random Fourier synthesis
    for n_comp in [3, 5, 7]:
        for _ in range(4):
            freqs = rng.randint(1, 30, n_comp)
            phases = rng.uniform(0, 2 * np.pi, n_comp)
            amps = rng.uniform(0.5, 2.0, n_comp)
            h_fourier = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h_fourier += a * np.cos(2 * np.pi * f * t + p)
            initial_points.append(h_fourier)

    # Sort by initial value (best first)
    def sort_key(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = fast_convolve(h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    initial_points.sort(key=sort_key)

    # --- Global search over Chebyshev coefficients using differential evolution ---
    if time.time() - start_time < DEADLINE - 30:
        try:
            # Bounds for each coefficient: [-5, 5]
            bounds = [(-5.0, 5.0)] * K
            
            def de_objective(coeffs):
                h = coeffs_to_h(coeffs)
                val, valid = evaluate(h)
                if not valid:
                    return 1e10
                return val

            # Run differential evolution with limited time
            de_result = differential_evolution(
                de_objective,
                bounds=bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                callback=lambda xk, convergence: (
                    time.time() - start_time > DEADLINE - 35
                )
            )
            
            h_de = coeffs_to_h(de_result.x)
            val_de, valid_de = evaluate(h_de)
            if valid_de and val_de < best_value:
                best_value = val_de
                best_h = h_de.copy()
                print(f"DE found: {val_de:.6f}")
        except Exception as e:
            print(f"DE failed: {e}")

    # --- Optimize from each initial point using L-BFGS-B ---
    for init_h in initial_points:
        if time.time() - start_time > DEADLINE:
            break

        s = np.sum(init_h)
        if abs(s) < 1e-10:
            init_h = init_h + 0.1

        try:
            result = minimize(
                smooth_objective,
                init_h,
                jac=smooth_grad,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 600, 'gtol': 1e-6}
            )

            h_opt = result.x
            val_opt, valid = evaluate(h_opt)
            if valid and val_opt < best_value:
                best_value = val_opt
                best_h = h_opt.copy()
                if best_value < 1.0:
                    break
        except Exception:
            pass

        val_init, valid = evaluate(init_h)
        if valid and val_init < best_value:
            best_value = val_init
            best_h = init_h.copy()

    # --- Final high-precision polish using exact objective ---
    if time.time() - start_time < DEADLINE - 3:
        h_curr = best_h.copy()
        
        # Use L-BFGS-B with the exact (non-smooth) objective via finite differences
        # This should refine the solution near the optimum
        def exact_objective(h):
            s = np.sum(h)
            if abs(s) < 1e-12:
                return 1e10
            conv = np.convolve(h, h)
            max_conv = np.max(conv)
            val = abs(2.0 * N * max_conv / (s * s))
            return val if np.isfinite(val) else 1e10

        try:
            result = minimize(
                exact_objective,
                h_curr,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 200, 'gtol': 1e-8}
            )
            h_opt = result.x
            val_opt, valid = evaluate(h_opt)
            if valid and val_opt < best_value:
                best_value = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass

        # --- Additional polish: local search in Chebyshev coefficient space ---
        if time.time() - start_time < DEADLINE - 1:
            # Project best_h back to Chebyshev space and refine
            # Use least squares to find coefficients that best represent best_h
            A = cheb_basis  # K x N
            coeffs_init = np.linalg.lstsq(A.T, best_h, rcond=None)[0]
            
            def cheb_objective(coeffs):
                h = coeffs_to_h(coeffs)
                val, valid = evaluate(h)
                if not valid:
                    return 1e10
                return val

            try:
                result_cheb = minimize(
                    cheb_objective,
                    coeffs_init,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 400, 'gtol': 1e-8}
                )
                h_cheb = coeffs_to_h(result_cheb.x)
                val_cheb, valid_cheb = evaluate(h_cheb)
                if valid_cheb and val_cheb < best_value:
                    best_value = val_cheb
                    best_h = h_cheb.copy()
            except Exception:
                pass

    # --- Perturbation phase ---
    if time.time() - start_time < DEADLINE - 2:
        for _ in range(8):
            if time.time() - start_time > DEADLINE - 2:
                break
            h_pert = best_h + 0.05 * rng.normal(0, 1, N)
            val_pert, valid = evaluate(h_pert)
            if valid and val_pert < best_value:
                best_value = val_pert
                best_h = h_pert
            else:
                try:
                    result = minimize(
                        smooth_objective,
                        h_pert,
                        jac=smooth_grad,
                        method='L-BFGS-B',
                        options={'maxiter': 150, 'maxfun': 300, 'gtol': 1e-6}
                    )
                    h_opt = result.x
                    val_opt, valid = evaluate(h_opt)
                    if valid and val_opt < best_value:
                        best_value = val_opt
                        best_h = h_opt.copy()
                except Exception:
                    pass

    return best_h
