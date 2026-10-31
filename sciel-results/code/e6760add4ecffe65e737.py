import numpy as np
import time
import random
from scipy.optimize import minimize, differential_evolution

N = 600
DEADLINE = 148

def _fft_convolve(a, b):
    """Fast linear convolution via FFT."""
    n = len(a) + len(b) - 1
    size = 1
    while size < n:
        size <<= 1
    A = np.fft.fft(a, n=size)
    B = np.fft.fft(b, n=size)
    return np.fft.ifft(A * B).real[:n]

def _compute_autoconv_max(h):
    """Compute max of autoconvolution of h (assumed sum=1)."""
    h = np.asarray(h, dtype=float)
    conv = _fft_convolve(h, h)
    return float(np.max(conv))

def _smooth_max_obj_grad(h, beta):
    """Smooth max autoconvolution with gradient, using temperature beta."""
    h = np.maximum(h, 1e-10)
    h = h / np.sum(h)
    
    size = 2 * N
    h_pad = np.zeros(size)
    h_pad[:N] = h
    H = np.fft.fft(h_pad, n=size)
    conv = np.fft.ifft(H * H).real

    m = np.max(conv)
    exp_conv = np.exp((conv - m) * beta)
    weights = exp_conv / np.sum(exp_conv)

    w_full = np.zeros(size)
    w_full[:len(conv)] = weights
    W = np.fft.fft(w_full, n=size)
    grad_h = 2.0 * np.fft.ifft(H * W).real[:N]

    val = m + np.log(np.sum(exp_conv)) / beta
    return val, grad_h

def _cosine_bump_basis(i, center, width):
    """Cosine bump function on [center-width, center+width]."""
    x = i - center
    mask = np.abs(x) <= width
    result = np.zeros_like(i, dtype=float)
    if np.any(mask):
        result[mask] = 0.5 * (1 + np.cos(np.pi * x[mask] / width))
    return result

def construct():
    start_time = time.time()

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def compute_autoconv_max(h):
        return _compute_autoconv_max(h)

    i = np.arange(N, dtype=float)
    center = (N - 1) / 2.0

    # --- Cosine Bump Mixture parameterization ---
    # Represent h as a mixture of K cosine bumps: h(x) = sum_k w_k * bump(x; mu_k, sigma_k)
    # where bump is a cosine bump on [mu-sigma, mu+sigma]
    # Parameters: [w_0, mu_0, sigma_0, w_1, mu_1, sigma_1, ...]

    K = 4  # Number of cosine bump components

    def bump_to_h(params):
        """Convert cosine bump mixture parameters to discrete heights on N intervals."""
        h = np.zeros(N)
        for k in range(K):
            w_k = max(params[3*k], 0.01)
            mu_k = params[3*k+1]
            sigma_k = max(params[3*k+2], 2.0)
            h += w_k * _cosine_bump_basis(i, mu_k, sigma_k)
        h = np.maximum(h, 1e-10)
        h /= np.sum(h)
        return h

    def bump_objective(params):
        """Objective: max autoconvolution of bump-generated h."""
        h = bump_to_h(params)
        return compute_autoconv_max(h)

    # DE bounds for bump parameters
    # w_k in [0.1, 5.0], mu_k in [0, N-1], sigma_k in [5, N/4]
    bounds = []
    for k in range(K):
        bounds.append((0.1, 5.0))   # weight
        bounds.append((0, N-1))      # mean
        bounds.append((5.0, N/4.0))  # sigma (half-width)

    # Initial best: uniform
    best_h = np.ones(N) / N
    best_val = compute_autoconv_max(best_h)

    # Stage 1: Differential Evolution on bump parameters
    if time_left() > 60:
        try:
            de_result = differential_evolution(
                bump_objective,
                bounds=bounds,
                seed=42,
                maxiter=30,
                popsize=15,
                tol=1e-6,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube',
                updating='deferred',
                callback=None
            )
            if de_result.fun < best_val:
                h_de = bump_to_h(de_result.x)
                best_val = de_result.fun
                best_h = h_de.copy()
        except Exception:
            pass

    # Stage 2: L-BFGS-B polish on discrete heights with annealing temperature
    if time_left() > 30:
        # Annealing schedule for beta - start moderate, end very sharp
        betas = [20.0, 100.0, 400.0, 1000.0]
        
        try:
            current_h = best_h.copy()
            current_val = best_val
            
            for beta in betas:
                if time_left() < 5:
                    break
                
                result = minimize(
                    lambda h: _smooth_max_obj_grad(h, beta)[0],
                    current_h,
                    jac=lambda h: _smooth_max_obj_grad(h, beta)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 500}
                )
                
                final_h = np.maximum(result.x, 0)
                if np.sum(final_h) > 0:
                    final_h = final_h / np.sum(final_h)
                    final_val = compute_autoconv_max(final_h)
                    if final_val < current_val:
                        current_val = final_val
                        current_h = final_h.copy()
            
            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
        except Exception:
            pass

    # Stage 3: Multi-restart with bump perturbations + Annealed L-BFGS-B polish
    if time_left() > 20:
        n_restarts = int(min(6, time_left() // 4))
        for r in range(n_restarts):
            if time_left() < 8:
                break

            # Perturb bump parameters and generate new initialization
            base_params = np.zeros(3*K)
            # Initialize with symmetric bimodal-like structure
            for k in range(K):
                base_params[3*k] = 1.0
                if k < K//2:
                    base_params[3*k+1] = N * (0.15 + 0.1*k)
                else:
                    base_params[3*k+1] = N * (0.85 - 0.1*(k-K//2))
                base_params[3*k+2] = N/8

            # Add random perturbation
            perturbation = np.random.normal(0, 0.3, size=3*K)
            params_perturbed = base_params + perturbation
            # Clip to bounds
            for k in range(K):
                params_perturbed[3*k] = max(0.1, min(5.0, params_perturbed[3*k]))
                params_perturbed[3*k+1] = max(0, min(N-1, params_perturbed[3*k+1]))
                params_perturbed[3*k+2] = max(5.0, min(N/4.0, params_perturbed[3*k+2]))

            h_init = bump_to_h(params_perturbed)

            try:
                # Use a shorter annealing schedule for restarts
                betas_restart = [50.0, 200.0, 800.0]
                current_h = h_init
                current_val = compute_autoconv_max(current_h)
                
                for beta in betas_restart:
                    if time_left() < 4:
                        break
                    result = minimize(
                        lambda h: _smooth_max_obj_grad(h, beta)[0],
                        current_h,
                        jac=lambda h: _smooth_max_obj_grad(h, beta)[1],
                        method='L-BFGS-B',
                        options={'maxiter': 150, 'ftol': 1e-13, 'gtol': 1e-9, 'maxfun': 300}
                    )
                    if result.success:
                        final_h = np.maximum(result.x, 0)
                        if np.sum(final_h) > 0:
                            final_h = final_h / np.sum(final_h)
                            final_val = compute_autoconv_max(final_h)
                            if final_val < current_val:
                                current_val = final_val
                                current_h = final_h.copy()

                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()
            except Exception:
                pass

    # Stage 4: Coordinate descent fine-tuning (faster version)
    if time_left() > 10:
        current_h = best_h.copy()
        current_val = best_val
        max_rounds = int(min(6, time_left() // 2))

        for round_num in range(max_rounds):
            if time_left() < 4:
                break

            improved = False
            order = np.random.permutation(N)
            for idx in order:
                if time_left() < 2:
                    break

                # Try both directions with fewer bisection steps
                best_step = 0.0
                best_new_val = current_val
                
                for direction in [1, -1]:
                    lo, hi = 0.0, 0.3
                    for _ in range(5):
                        mid = (lo + hi) / 2
                        new_h = current_h.copy()
                        new_h[idx] = max(0, current_h[idx] * (1 + direction * mid))
                        s = np.sum(new_h)
                        if s < 1e-10:
                            break
                        new_h /= s
                        new_val = compute_autoconv_max(new_h)
                        if new_val < best_new_val:
                            best_new_val = new_val
                            best_step = mid
                            lo = mid
                        else:
                            hi = mid

                if best_step > 0 and best_new_val < current_val - 1e-12:
                    new_h = current_h.copy()
                    new_h[idx] = max(0, current_h[idx] * (1 + best_step))
                    s = np.sum(new_h)
                    new_h /= s
                    current_h = new_h
                    current_val = best_new_val
                    improved = True
                    break

            if not improved:
                break

        if current_val < best_val:
            best_val = current_val
            best_h = current_h

    # Stage 5: Final high-precision polish with very tight smooth-max
    if time_left() > 8:
        try:
            current_h = best_h.copy()
            current_val = best_val

            # Use very high beta for near-exact max
            for beta in [2000.0, 5000.0]:
                if time_left() < 4:
                    break

                result = minimize(
                    lambda h: _smooth_max_obj_grad(h, beta)[0],
                    current_h,
                    jac=lambda h: _smooth_max_obj_grad(h, beta)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 1000}
                )

                final_h = np.maximum(result.x, 0)
                if np.sum(final_h) > 0:
                    final_h = final_h / np.sum(final_h)
                    final_val = compute_autoconv_max(final_h)
                    if final_val < current_val:
                        current_val = final_val
                        current_h = final_h.copy()

            if current_val < best_val:
                best_val = current_val
                best_h = current_h
        except Exception:
            pass

    return best_h.tolist()
