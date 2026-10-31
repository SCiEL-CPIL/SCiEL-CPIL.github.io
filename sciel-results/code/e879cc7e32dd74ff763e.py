import numpy as np
import time
import random
from scipy.optimize import differential_evolution, minimize

N = 600
DEADLINE = 140

def construct():
    start_time = time.time()

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def compute_autoconv_max(h):
        """Compute max of autoconvolution of h (assumed sum=1)."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return np.max(conv)

    def smooth_max_autoconv(h, beta=50.0):
        """Compute smooth max (log-sum-exp) of autoconvolution of h."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        m = np.max(conv)
        if m < 1e-15:
            return 0.0
        exp_conv = np.exp((conv - m) * beta)
        return m + np.log(np.sum(exp_conv)) / beta

    def compute_autoconv(h):
        """Compute full autoconvolution of h."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return conv

    # Parametric families for initialization
    i = np.arange(N, dtype=float)
    center = (N - 1) / 2.0

    def make_bimodal(sigma, alpha):
        """Bimodal: two Gaussians at positions alpha*N and (1-alpha)*N"""
        pos1 = alpha * N
        pos2 = (1 - alpha) * N
        g1 = np.exp(-((i - pos1) ** 2) / (2 * sigma ** 2))
        g2 = np.exp(-((i - pos2) ** 2) / (2 * sigma ** 2))
        return g1 + g2

    def make_poly(c, d, e):
        """Polynomial: c + d*(x-center)^2 + e*(x-center)^4, clipped to nonneg"""
        x = (i - center) / (N / 2)
        val = c + d * x**2 + e * x**4
        return np.maximum(val, 0.01)

    # Generate diverse initializations
    inits = []

    # Uniform
    inits.append(np.ones(N))

    # Bimodal with various parameters
    for sigma in [N/10, N/8, N/6, N/5]:
        for alpha in [0.1, 0.15, 0.2, 0.25, 0.3]:
            inits.append(make_bimodal(sigma, alpha))

    # Polynomial shapes
    for c, d, e in [(1, 1, 0), (1, 2, 0), (1, 1, 0.5), (1, 0.5, 1), (2, 1, 0), (1, 3, 0)]:
        inits.append(make_poly(c, d, e))

    # U-shaped
    u_shape = 1.0 + 2.0 * ((i - center) / (N/2))**2
    inits.append(u_shape)

    # W-shape (two bumps)
    w_shape = 1.0 + 1.5 * np.cos(4 * np.pi * i / N)
    inits.append(np.maximum(w_shape, 0.01))

    # Random smooth
    for _ in range(5):
        freqs = np.random.choice([1, 2, 3, 4, 5], size=3, replace=False)
        phases = np.random.uniform(0, 2*np.pi, size=3)
        amps = np.random.uniform(0.5, 1.5, size=3)
        random_shape = np.ones(N)
        for f, p, a in zip(freqs, phases, amps):
            random_shape += a * np.cos(2 * np.pi * f * i / N + p)
        inits.append(np.maximum(random_shape, 0.01))

    best_h = None
    best_val = np.inf

    # Stage 1: Evaluate all initializations
    for init in inits:
        if time_left() < 15:
            break
        h = init / np.sum(init)
        val = compute_autoconv_max(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()

    if best_h is None:
        best_h = np.ones(N) / N
        best_val = compute_autoconv_max(best_h)

    # Stage 2: Differential Evolution on Gaussian Mixture Parameters
    # Represent h as a sum of K Gaussians with parameters (amplitude, center, width)
    if time_left() > 40:
        K = 8  # Number of Gaussian components
        # Parameters: for each Gaussian, [log_amp, center (0..1), log_width]
        # Total parameters: 3*K
        # We map center to [0, N-1] and width to a reasonable range
        
        x_grid = np.arange(N, dtype=float)
        
        def params_to_h(params):
            """Convert Gaussian mixture parameters to step function heights."""
            h = np.zeros(N)
            for k in range(K):
                log_amp = params[3*k]
                center_frac = params[3*k + 1]
                log_width = params[3*k + 2]
                
                amp = np.exp(log_amp)
                center = center_frac * (N - 1)
                width = np.exp(log_width)
                
                h += amp * np.exp(-0.5 * ((x_grid - center) / width) ** 2)
            return np.maximum(h, 1e-10)

        def objective_gm(params):
            """Objective function for differential evolution."""
            h = params_to_h(params)
            s = np.sum(h)
            if s < 1e-10:
                return 1e10
            h_norm = h / s
            val = compute_autoconv_max(h_norm)
            return val

        # Define bounds for each parameter
        # log_amp: [-5, 2] (amp in [0.006, 7.4])
        # center_frac: [0, 1]
        # log_width: [-2, 2] (width in [0.13, 7.4])
        bounds = []
        for k in range(K):
            bounds.append((-5, 2))   # log_amp
            bounds.append((0, 1))    # center_frac
            bounds.append((-2, 2))   # log_width

        # Initial population for DE
        # We can seed DE with parameters derived from our best_h or random
        # Let's just use random initialization within bounds, but maybe bias towards known good structures
        
        # Create a few good initial guesses for the population
        # 1. Uniform-like: many small Gaussians spread out
        # 2. Bimodal: two large Gaussians at 1/4 and 3/4
        
        # DE will handle the search, so we just need to run it.
        
        # To speed up DE, we can limit maxiter and popsize
        # Time budget: ~30s
        # N_params = 24
        # popsize = 10 * N_params = 240
        # Each eval is ~1ms (FFT based)
        # 240 * 24 * 1ms = ~5.7s per generation
        # We can afford ~5 generations in 30s? Maybe less.
        # Let's try maxiter=10, popsize=15
        
        try:
            # Use a fixed seed for reproducibility in DE if needed, but random is fine
            # scipy's DE uses random internally
            result_de = differential_evolution(
                objective_gm,
                bounds,
                maxiter=10,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=False,  # Disable polish, we'll do our own refinement
                init='latinhypercube'
            )
            
            if result_de.success or result_de.fun < best_val:
                best_params = result_de.x
                h_de = params_to_h(best_params)
                h_de = h_de / np.sum(h_de)
                val_de = compute_autoconv_max(h_de)
                if val_de < best_val:
                    best_val = val_de
                    best_h = h_de.copy()
                    
        except Exception as e:
            pass

    # Stage 3: Legendre polynomial parameterization with smooth max optimization
    if time_left() > 30:
        # Map indices to [-1, 1] for Legendre polynomials
        x = 2.0 * i / N - 1.0
        
        # Precompute Legendre polynomials up to order M
        M = 20
        leg_matrix = np.zeros((N, M + 1))
        leg_matrix[:, 0] = 1.0
        leg_matrix[:, 1] = x
        for m in range(2, M + 1):
            leg_matrix[:, m] = ((2 * m - 1) * x * leg_matrix[:, m - 1] - (m - 1) * leg_matrix[:, m - 2]) / m
        
        # Initialize coefficients from best_h using least squares
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(leg_matrix, best_h, rcond=None)
        except:
            coeffs = np.zeros(M + 1)
            coeffs[0] = np.mean(best_h)
        
        def coeffs_to_h_leg(c):
            """Convert Legendre coefficients to heights."""
            h = leg_matrix @ c
            return h
        
        def compute_grad_leg(c, beta=50.0):
            """Compute gradient of smooth max autoconvolution w.r.t. Legendre coefficients."""
            h = coeffs_to_h_leg(c)
            h = np.maximum(h, 1e-10)
            s = np.sum(h)
            if s < 1e-10:
                return np.zeros(M + 1), 1e10
            h_norm = h / s

            size = 2 * N
            h_pad = np.zeros(size)
            h_pad[:N] = h_norm
            H = np.fft.fft(h_pad, n=size)
            conv = np.fft.ifft(H * H).real

            m = np.max(conv)
            exp_conv = np.exp((conv - m) * beta)
            weights = exp_conv / np.sum(exp_conv)

            w_full = np.zeros(size)
            w_full[:len(conv)] = weights
            W = np.fft.fft(w_full, n=size)
            grad_h = 2.0 * np.fft.ifft(H * W).real[:N]

            grad_c = leg_matrix.T @ grad_h

            return grad_c, m + np.log(np.sum(exp_conv)) / beta

        # Use L-BFGS-B for optimization with smooth max objective
        def objective_leg(c, beta=50.0):
            h = coeffs_to_h_leg(c)
            # Soft non-negativity penalty
            neg_part = np.maximum(-h, 0)
            penalty = 1000.0 * np.sum(neg_part ** 2)
            h_pos = np.maximum(h, 1e-10)
            s = np.sum(h_pos)
            if s < 1e-10:
                return 1e10 + penalty
            h_norm = h_pos / s
            val = smooth_max_autoconv(h_norm, beta=beta)
            return val + penalty
        
        # Optimize using L-BFGS-B with smooth max
        result_leg = minimize(
            objective_leg,
            coeffs,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result_leg.fun < best_val:
            h_leg = coeffs_to_h_leg(result_leg.x)
            h_leg = np.maximum(h_leg, 0)
            if np.sum(h_leg) > 0:
                h_leg = h_leg / np.sum(h_leg)
                val_leg = compute_autoconv_max(h_leg)
                if val_leg < best_val:
                    best_val = val_leg
                    best_h = h_leg.copy()
        
        # Additional gradient descent refinement with smooth max
        current_coeffs = result_leg.x.copy()
        eta = 0.01
        for k in range(150):
            if time_left() < 10:
                break
            
            grad_c, smooth_val = compute_grad_leg(current_coeffs, beta=50.0)
            
            h_curr = coeffs_to_h_leg(current_coeffs)
            h_curr = np.maximum(h_curr, 1e-10)
            h_curr = h_curr / np.sum(h_curr)
            curr_val = compute_autoconv_max(h_curr)
            
            if curr_val < best_val:
                best_val = curr_val
                best_h = h_curr.copy()
            
            grad_norm = np.linalg.norm(grad_c)
            if grad_norm > 1e-10:
                grad_c = grad_c / grad_norm
            
            new_coeffs = current_coeffs - eta * grad_c
            new_h = coeffs_to_h_leg(new_coeffs)
            
            if np.any(new_h < -1e-6):
                eta *= 0.5
                if eta < 1e-10:
                    break
                continue
            
            new_h = np.maximum(new_h, 0)
            new_h = new_h / np.sum(new_h)
            new_val = compute_autoconv_max(new_h)
            
            if new_val < curr_val:
                current_coeffs = new_coeffs
                eta = min(eta * 1.1, 0.1)
            else:
                eta *= 0.8

    # Stage 4: Smooth max gradient descent with adaptive step size
    if time_left() > 35:
        current_h = best_h.copy()
        current_val = compute_autoconv_max(current_h)

        def compute_smooth_max_grad(h, beta=30.0):
            """Compute gradient of smooth max autoconvolution."""
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
            return grad_h, val

        n_iter = int(min(400, time_left() * 4))
        eta = 0.02
        for k in range(n_iter):
            if time_left() < 15:
                break

            grad_h, smooth_val = compute_smooth_max_grad(current_h, beta=30.0)

            grad_norm = np.linalg.norm(grad_h)
            if grad_norm > 1e-10:
                grad_h = grad_h / grad_norm

            new_h = current_h - eta * grad_h
            new_h = np.maximum(new_h, 0)
            s = np.sum(new_h)
            if s < 1e-10:
                eta *= 0.5
                continue
            new_h = new_h / s

            new_val = compute_autoconv_max(new_h)
            if new_val < current_val:
                current_h = new_h
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()
                eta = min(eta * 1.05, 0.1)
            else:
                eta *= 0.7
                if eta < 1e-6:
                    break

    # Stage 5: Local search with smooth max guidance
    if time_left() > 25:
        current_h = best_h.copy()
        current_val = best_val
        best_h_ls = best_h.copy()
        best_val_ls = best_val

        n_iter = int(min(3000, time_left() * 40))
        for k in range(n_iter):
            if time_left() < 5:
                break

            # Pick a random block of indices to perturb
            block_size = random.randint(3, 15)
            start_idx = random.randint(0, N - block_size)
            indices = range(start_idx, start_idx + block_size)

            new_h = current_h.copy()
            noise_scale = 0.05
            for idx in indices:
                noise = random.gauss(0, noise_scale)
                new_h[idx] = max(0, current_h[idx] + noise)

            s = np.sum(new_h)
            if s < 1e-10:
                continue
            new_h /= s

            new_val = compute_autoconv_max(new_h)

            if new_val < current_val:
                current_h = new_h
                current_val = new_val

                if current_val < best_val_ls:
                    best_val_ls = current_val
                    best_h_ls = current_h.copy()
            else:
                # Reject and try smaller perturbation
                pass

        if best_val_ls < best_val:
            best_val = best_val_ls
            best_h = best_h_ls

    # Stage 6: Final high-precision polish with tight tolerance
    if time_left() > 10:
        current_h = best_h.copy()
        current_val = best_val

        # Use smooth max with very high beta for near-exact max
        def final_objective(h):
            return smooth_max_autoconv(h, beta=200.0)

        # Gradient descent with very small steps
        eta = 0.005
        for k in range(200):
            if time_left() < 3:
                break

            size = 2 * N
            h_pad = np.zeros(size)
            h_pad[:N] = current_h
            H = np.fft.fft(h_pad, n=size)
            conv = np.fft.ifft(H * H).real

            m = np.max(conv)
            exp_conv = np.exp((conv - m) * 200.0)
            weights = exp_conv / np.sum(exp_conv)

            w_full = np.zeros(size)
            w_full[:len(conv)] = weights
            W = np.fft.fft(w_full, n=size)
            grad_h = 2.0 * np.fft.ifft(H * W).real[:N]

            grad_norm = np.linalg.norm(grad_h)
            if grad_norm > 1e-10:
                grad_h = grad_h / grad_norm

            new_h = current_h - eta * grad_h
            new_h = np.maximum(new_h, 0)
            s = np.sum(new_h)
            if s < 1e-10:
                break
            new_h = new_h / s

            new_val = compute_autoconv_max(new_h)
            if new_val < current_val:
                current_h = new_h
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()
            else:
                eta *= 0.5
                if eta < 1e-8:
                    break

    return best_h.tolist()
