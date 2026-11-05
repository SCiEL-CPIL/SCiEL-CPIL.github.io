"""Model-and-optimize bootstrap: parameterize step heights and optimize the objective."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    
    Strategy:
    1. Use Chebyshev polynomial basis to reduce dimensionality.
    2. Optimize coefficients via differential evolution.
    3. Local polish with L-BFGS-B on full 400-dim space.
    """
    start_time = time.time()
    DEADLINE = 145  # 5s safety margin before 150s kill
    
    best_value = np.inf
    best_h = np.ones(N).astype(float)
    
    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid
    
    # First evaluate the seed
    val_seed, _ = evaluate(best_h)
    best_value = val_seed
    
    rng = np.random.RandomState(42)
    
    # Chebyshev basis approach
    # h[i] = sum_k c_k * T_k(2*i/(N-1) - 1)
    # This constrains h to smooth functions, reducing effective dimensionality
    
    def chebyshev_to_h(coeffs):
        """Convert Chebyshev coefficients to step heights."""
        x = 2.0 * np.arange(N) / (N - 1) - 1.0  # [-1, 1]
        h = np.zeros(N)
        for k, c in enumerate(coeffs):
            h += c * np.cos(k * np.arccos(np.clip(x, -1, 1)))
        return h
    
    def h_to_chebyshev(h, n_coeffs=15):
        """Approximate h with Chebyshev polynomial of given degree."""
        x = 2.0 * np.arange(N) / (N - 1) - 1.0
        # Use discrete Chebyshev transform (DCT-like)
        # For simplicity, use least squares
        T = np.zeros((N, n_coeffs))
        for k in range(n_coeffs):
            T[:, k] = np.cos(k * np.arccos(np.clip(x, -1, 1)))
        # Solve least squares: T @ coeffs ≈ h
        coeffs, _, _, _ = np.linalg.lstsq(T, h, rcond=None)
        return coeffs
    
    # Phase 1: Differential evolution on Chebyshev coefficients
    n_cheb = 12  # Number of Chebyshev coefficients
    
    def cheb_objective(coeffs):
        h = chebyshev_to_h(coeffs)
        val, valid = evaluate(h)
        if not valid:
            return 1e10
        return val
    
    # Bounds for coefficients
    bounds = [(-5.0, 5.0)] * n_cheb
    
    # Run differential evolution
    if time.time() - start_time < DEADLINE - 30:
        try:
            de_result = differential_evolution(
                cheb_objective,
                bounds,
                seed=42,
                maxiter=80,
                popsize=15,
                tol=1e-6,
                polish=True,
                init='sobol'
            )
            
            if de_result.fun < best_value:
                h_de = chebyshev_to_h(de_result.x)
                val_de, valid_de = evaluate(h_de)
                if valid_de and val_de < best_value:
                    best_value = val_de
                    best_h = h_de.copy()
        except Exception:
            pass
    
    # Phase 2: Local optimization on full space using smooth surrogate
    def smooth_objective_and_grad(h, beta):
        """
        Smooth surrogate for |2N * max(conv) / sum(h)^2|.
        Vectorized gradient computation using FFT.
        """
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10, np.zeros_like(h)
        
        conv = np.convolve(h, h)
        
        # Log-sum-exp for numerical stability
        max_conv = np.max(conv)
        shifted = beta * (conv - max_conv)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        
        if sum_exp == 0:
            soft_max = max_conv
            weights = np.zeros_like(conv)
            weights[np.argmax(conv)] = 1.0
        else:
            logsumexp = max_conv + np.log(sum_exp) / beta
            soft_max = logsumexp
            weights = exp_shifted / sum_exp
        
        # f = 2N * soft_max / s^2
        f = 2.0 * N * soft_max / (s * s)
        
        # Vectorized gradient of soft_max w.r.t. h using FFT
        n_fft = 1 << (2 * N - 1).bit_length()
        W = np.fft.rfft(weights, n=n_fft)
        H = np.fft.rfft(h, n=n_fft)
        corr = np.fft.irfft(W * np.conj(H), n=n_fft)
        grad_soft_max = 2.0 * corr[:N]
        
        # Gradient of f
        grad_f = (2.0 * N / (s * s)) * grad_soft_max - (4.0 * N * soft_max / (s ** 3))
        
        return f, grad_f
    
    # Generate diverse initial points for local optimization
    init_points = []
    
    # Seed
    init_points.append(np.ones(N).astype(float))
    
    # From Chebyshev optimization
    if time.time() - start_time < DEADLINE - 20:
        # Generate a few Chebyshev-based starting points
        for _ in range(20):
            coeffs = rng.uniform(-3, 3, n_cheb)
            h_cheb = chebyshev_to_h(coeffs)
            init_points.append(h_cheb)
    
    # Random with Chebyshev projection
    for _ in range(30):
        h_rand = rng.uniform(-1, 1, N)
        # Project to Chebyshev space and back
        coeffs = h_to_chebyshev(h_rand, n_cheb)
        h_proj = chebyshev_to_h(coeffs)
        init_points.append(h_proj)
    
    # Pure random
    for _ in range(30):
        init_points.append(rng.uniform(-1, 1, N))
    
    # Evaluate and sort
    evals = []
    for h in init_points:
        if time.time() - start_time > DEADLINE - 10:
            break
        val, valid = evaluate(h)
        if valid:
            evals.append((val, h))
    
    evals.sort(key=lambda x: x[0])
    
    # Include current best
    val_best, _ = evaluate(best_h)
    if val_best < (evals[0][0] if evals else np.inf):
        evals.insert(0, (val_best, best_h.copy()))
    
    # Local optimization on top candidates
    n_optimize = min(8, len(evals))
    
    for beta in [100, 300, 1000]:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        for idx in range(n_optimize):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            h0 = evals[idx][1].copy()
            
            try:
                result = minimize(
                    lambda h, b=beta: smooth_objective_and_grad(h, b)[0],
                    h0,
                    jac=lambda h, b=beta: smooth_objective_and_grad(h, b)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                
                if result.fun < 1e10:
                    h_opt = result.x
                    val, valid = evaluate(h_opt)
                    if valid and val < best_value:
                        best_value = val
                        best_h = h_opt.copy()
                        
            except Exception:
                pass
    
    # Final coordinate descent polish
    if time.time() - start_time < DEADLINE - 2:
        h_curr = best_h.copy()
        lr = 0.03
        stagnation = 0
        
        for iteration in range(200):
            if time.time() - start_time > DEADLINE - 2:
                break
            improved = False
            for coord in rng.permutation(N):
                if time.time() - start_time > DEADLINE - 2:
                    break
                for step_mult in [1.0, 0.5, 0.2]:
                    h_try = h_curr.copy()
                    h_try[coord] += lr * step_mult * rng.uniform(-1, 1)
                    val_try, valid = evaluate(h_try)
                    if valid and val_try < best_value:
                        best_value = val_try
                        best_h = h_try.copy()
                        h_curr = h_try
                        improved = True
                        break
                if best_value < 0.7:
                    break
            if best_value < 0.7:
                break
            if not improved:
                stagnation += 1
                lr *= 0.7
                if stagnation > 6:
                    break
            else:
                stagnation = 0
    
    return best_h
# EVOLVE-BLOCK-END
