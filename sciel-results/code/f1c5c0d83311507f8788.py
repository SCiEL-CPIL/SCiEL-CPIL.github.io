import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import warnings

N = 600

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 145.0

    dx = 1.0 / (2 * N)
    x = np.linspace(-1/4, 1/4, N)

    def compute_autoconv(h):
        h = np.asarray(h, dtype=np.float64)
        n_fft = 2 * N
        h_padded = np.zeros(n_fft)
        h_padded[:N] = h
        H = np.fft.fft(h_padded)
        autoconv_circ = np.fft.ifft(H * H).real
        return autoconv_circ[:2*N-1] * dx**2

    def compute_score(h):
        h = np.asarray(h, dtype=np.float64)
        if np.any(h < 0):
            h = np.maximum(h, 0)
        l1 = np.sum(h) * dx
        if l1 < 1e-15:
            return float('inf')
        autoconv_vals = compute_autoconv(h)
        max_autoconv = np.max(autoconv_vals)
        return max_autoconv / (l1 ** 2)

    def compute_score_l2(h):
        h = np.asarray(h, dtype=np.float64)
        if np.any(h < 0):
            h = np.maximum(h, 0)
        l2_sq = np.sum(h**2) * dx
        if l2_sq < 1e-15:
            return float('inf')
        autoconv_vals = compute_autoconv(h)
        max_autoconv = np.max(autoconv_vals)
        return max_autoconv / l2_sq

    def params_to_h_gaussian(params, n_gaussians=4):
        h = np.zeros(N)
        for k in range(n_gaussians):
            amp = params[3*k]
            center = params[3*k+1]
            sigma = max(1e-6, abs(params[3*k+2]))
            h += amp * np.exp(-(x - center)**2 / (2 * sigma**2))
        return np.maximum(h, 0)

    def params_to_h_block(params, n_blocks=3):
        h = np.zeros(N)
        for k in range(n_blocks):
            amp = params[3*k]
            left = params[3*k+1]
            right = params[3*k+2]
            if left > right:
                left, right = right, left
            mask = (x >= left) & (x <= right)
            h[mask] += amp
        return np.maximum(h, 0)

    def params_to_h_tent(params):
        """Single symmetric tent (triangular pulse) with variable base and peak.
        params = [amp, half_width] where tent is amp * (1 - |x|/half_width) for |x| < half_width
        """
        amp = max(0, params[0])
        half_width = max(1e-6, abs(params[1]))
        h = amp * np.maximum(0, 1 - np.abs(x) / half_width)
        return h

    def smooth_objective_gaussian(params, n_gaussians=4, temperature=0.005):
        h = params_to_h_gaussian(params, n_gaussians)
        l1 = np.sum(h) * dx
        if l1 < 1e-15:
            return 1e10
        autoconv_vals = compute_autoconv(h)
        max_val = np.max(autoconv_vals)
        if max_val <= 0:
            return 0.0
        log_sum_exp = max_val + np.log(np.sum(np.exp((autoconv_vals - max_val) / temperature)))
        smooth_max = temperature * log_sum_exp
        return smooth_max / (l1 ** 2)

    def objective_gaussian(params, n_gaussians=4):
        h = params_to_h_gaussian(params, n_gaussians)
        return compute_score(h)

    def objective_tent(params):
        h = params_to_h_tent(params)
        return compute_score(h)

    best_score = float('inf')
    best_h = None

    def update_best(h, score):
        nonlocal best_score, best_h
        if score < best_score:
            best_score = score
            best_h = np.asarray(h, dtype=np.float64).copy()

    # Phase 1: Evaluate simple candidate shapes
    for sigma_idx in [5, 10, 20, 50, 100, 200]:
        sigma = sigma_idx * dx
        h = np.exp(-(x)**2 / (2 * sigma**2))
        update_best(h, compute_score(h))
    
    for half_width in [20, 50, 100, 200, 300]:
        h = np.zeros(N)
        mask = np.abs(x) <= half_width * dx
        h[mask] = 1.0
        update_best(h, compute_score(h))
    
    for separation in [0.02, 0.05, 0.1, 0.15]:
        for sigma in [0.01, 0.02, 0.05]:
            h = np.exp(-(x - separation)**2 / (2*sigma**2)) + np.exp(-(x + separation)**2 / (2*sigma**2))
            update_best(h, compute_score(h))

    # Phase 1b: Evaluate tent shapes
    for hw in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12]:
        for amp in [0.5, 1.0, 2.0, 5.0]:
            h = amp * np.maximum(0, 1 - np.abs(x) / hw)
            update_best(h, compute_score(h))

    # Phase 2: Differential Evolution on Gaussian mixture parameters
    if time.time() < deadline - 80:
        n_gaussians = 4
        n_params = 3 * n_gaussians
        bounds_gauss = []
        for k in range(n_gaussians):
            bounds_gauss.append((0.0, 10.0))
            bounds_gauss.append((-0.25, 0.25))
            bounds_gauss.append((0.001, 0.25))
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = differential_evolution(
                    objective_gaussian,
                    bounds=bounds_gauss,
                    args=(n_gaussians,),
                    maxiter=300,
                    popsize=20,
                    tol=1e-8,
                    seed=42,
                    polish=False
                )
            h_de = params_to_h_gaussian(result.x, n_gaussians)
            score_de = compute_score(h_de)
            update_best(h_de, score_de)
        except Exception:
            pass

    # Phase 2b: Differential Evolution on tent parameters
    if time.time() < deadline - 60:
        bounds_tent = [(0.0, 10.0), (0.001, 0.25)]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result_tent = differential_evolution(
                    objective_tent,
                    bounds=bounds_tent,
                    maxiter=200,
                    popsize=30,
                    tol=1e-10,
                    seed=7,
                    polish=True
                )
            h_tent = params_to_h_tent(result_tent.x)
            score_tent = compute_score(h_tent)
            update_best(h_tent, score_tent)
        except Exception:
            pass

    # Phase 3: L-BFGS-B polish on Gaussian mixture
    if best_h is not None and time.time() < deadline - 40:
        n_gaussians = 4
        n_params = 3 * n_gaussians
        bounds_gauss = []
        for k in range(n_gaussians):
            bounds_gauss.append((0.0, 10.0))
            bounds_gauss.append((-0.25, 0.25))
            bounds_gauss.append((0.001, 0.25))
        
        h_init = best_h.copy()
        kernel = np.ones(10) / 10
        smoothed = np.convolve(h_init, kernel, mode='same')
        peak_threshold = np.max(smoothed) * 0.3
        peaks = []
        for i in range(5, N-5):
            if smoothed[i] > peak_threshold and smoothed[i] == max(smoothed[i-5:i+6]):
                peaks.append(i)
        peaks = peaks[:n_gaussians]
        while len(peaks) < n_gaussians:
            peaks.append(N // 2 + len(peaks) * 50)
        
        init_params = np.zeros(n_params)
        for k in range(n_gaussians):
            init_params[3*k] = 1.0
            init_params[3*k+1] = x[min(peaks[k], N-1)]
            init_params[3*k+2] = 0.02
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res1 = minimize(
                    smooth_objective_gaussian,
                    init_params,
                    args=(n_gaussians,),
                    method='L-BFGS-B',
                    bounds=bounds_gauss,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                res2 = minimize(
                    objective_gaussian,
                    res1.x,
                    args=(n_gaussians,),
                    method='L-BFGS-B',
                    bounds=bounds_gauss,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14}
                )
            h_lbfgs = params_to_h_gaussian(res2.x, n_gaussians)
            score_lbfgs = compute_score(h_lbfgs)
            update_best(h_lbfgs, score_lbfgs)
        except Exception:
            pass

    # Phase 4: Direct bin-level L-BFGS-B polish on the best h
    if best_h is not None and time.time() < deadline - 20:
        try:
            h_flat = best_h.copy()
            
            def bin_objective(h_flat):
                h = np.maximum(h_flat, 0)
                return compute_score(h)
            
            def bin_smooth_objective(h_flat, temp=0.002):
                h = np.maximum(h_flat, 0)
                l1 = np.sum(h) * dx
                if l1 < 1e-15:
                    return 1e10
                autoconv_vals = compute_autoconv(h)
                max_val = np.max(autoconv_vals)
                if max_val <= 0:
                    return 0.0
                log_sum_exp = max_val + np.log(np.sum(np.exp((autoconv_vals - max_val) / temp)))
                smooth_max = temp * log_sum_exp
                return smooth_max / (l1 ** 2)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res_smooth = minimize(
                    bin_smooth_objective,
                    h_flat,
                    method='L-BFGS-B',
                    bounds=[(0, 10)] * N,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                res_sharp = minimize(
                    bin_objective,
                    res_smooth.x,
                    method='L-BFGS-B',
                    bounds=[(0, 10)] * N,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-14}
                )
            
            h_polished = np.maximum(res_sharp.x, 0)
            score_polished = compute_score(h_polished)
            update_best(h_polished, score_polished)
        except Exception:
            pass

    # Phase 5: Block-structure DE search
    if time.time() < deadline - 30:
        n_blocks = 3
        n_params = 3 * n_blocks
        bounds_block = []
        for k in range(n_blocks):
            bounds_block.append((0.0, 10.0))
            bounds_block.append((-0.25, 0.25))
            bounds_block.append((-0.25, 0.25))
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result_block = differential_evolution(
                    lambda p: compute_score(params_to_h_block(p, n_blocks)),
                    bounds=bounds_block,
                    maxiter=200,
                    popsize=15,
                    tol=1e-8,
                    seed=123,
                    polish=False
                )
            h_block = params_to_h_block(result_block.x, n_blocks)
            score_block = compute_score(h_block)
            update_best(h_block, score_block)
        except Exception:
            pass

    # Final: coordinate-descent-like local polish
    if best_h is not None and time.time() < deadline - 5:
        h_final = best_h.copy()
        current_score = best_score
        step_size = 0.05
        for iteration in range(50):
            if time.time() > deadline - 2:
                break
            improved = False
            np.random.seed(iteration)
            indices = np.random.choice(N, size=150, replace=False)
            for i in indices:
                if time.time() > deadline - 2:
                    break
                for direction in [1, -1]:
                    h_trial = h_final.copy()
                    h_trial[i] = max(0, h_final[i] + direction * step_size)
                    trial_score = compute_score(h_trial)
                    if trial_score < current_score:
                        h_final = h_trial
                        current_score = trial_score
                        improved = True
                        break
            if not improved:
                step_size *= 0.8
                if step_size < 1e-4:
                    break
        update_best(h_final, current_score)

    if best_h is None:
        return [1.0] * N

    return list(best_h)

# EVOLVE-BLOCK-END
