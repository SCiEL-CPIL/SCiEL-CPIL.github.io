"""
Strategy: Multi-start optimization using a smooth log-space surrogate for the ratio objective.
The value function is scale-invariant (value = l2/(norm_1*norm_inf)), so only the shape of h matters.
We use a smooth log-sum-exp approximation for log(max(|g|)) during optimization to give L-BFGS-B
well-behaved gradients. Optimizing in log-space (maximizing log(value)) improves numerical stability
for the ratio objective compared to optimizing the raw ratio directly.

Key ideas:
1. Log-space objective: max log(l2) - log(norm_1) - log(smooth_norm_inf).
2. Smooth surrogate for norm_inf: logsumexp(beta * g) / beta approximates log(max(g)) smoothly.
3. Multiple diverse initializations + perturbation of best.
4. Time-budgeted search with safety margin.
5. Final high-precision polish pass on the best solution.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_value(h):
    """Compute the exact value metric for given heights."""
    h = np.asarray(h, dtype=float)
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')

    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return 0.0

    h = np.maximum(0, h)

    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)

    y_sq_terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq_terms)

    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))

    if norm_1 == 0 or norm_inf == 0:
        return 0.0

    value = l2 / (norm_1 * norm_inf)
    return value


def compute_log_value_smooth(h, beta=50.0):
    """
    Compute a smooth surrogate of the log value metric.
    log(value) = log(l2) - log(norm_1) - log(smooth_norm_inf)
    Using log-space improves stability for ratio objectives.
    """
    h = np.asarray(h, dtype=float)
    h = np.maximum(0, h)
    
    # Early exit for trivial cases to avoid log(0)
    if np.sum(h) < 1e-12:
        return -1e10

    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)

    y_sq_terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq_terms)

    if l2 < 1e-12:
        return -1e10

    # Since h is non-negative, g is non-negative, so abs(g) == g
    norm_1 = np.sum(g) / (m + 1)
    if norm_1 < 1e-12:
        return -1e10

    abs_g = g
    max_abs = np.max(abs_g)
    if max_abs < 1e-12:
        return -1e10

    # Smooth log-max: log(max(g)) approx logsumexp(beta * g) / beta
    # We want to approximate log(max(g)).
    # Let M = max(g). logsumexp(beta * g) = log(sum(exp(beta * g)))
    # sum(exp(beta * g)) = exp(beta * M) * sum(exp(beta * (g - M)))
    # log(sum(exp(beta * g))) = beta * M + log(sum(exp(beta * (g - M))))
    # So logsumexp(beta * g) / beta = M + (1/beta) * log(sum(exp(beta * (g - M))))
    # This is a smooth approximation of M.
    # We need log(M_smooth).
    # M_smooth is close to M. log(M_smooth) is close to log(M).
    # However, the gradient of log(M_smooth) is smoother than gradient of log(M).
    
    shifted = beta * (abs_g - max_abs)
    # shifted is <= 0, so exp is safe (max exp(0)=1)
    log_sum_exp_term = np.log(np.sum(np.exp(shifted)))
    
    # This calculates the smooth approximation of max(g), let's call it g_max_smooth
    g_max_smooth = max_abs + (1.0 / beta) * log_sum_exp_term
    
    # We need log(g_max_smooth) for the objective function
    if g_max_smooth < 1e-12:
        return -1e10
        
    log_smooth_max = np.log(g_max_smooth)
    
    # log(value) = log(l2) - log(norm_1) - log_smooth_max
    log_val = np.log(l2) - np.log(norm_1) - log_smooth_max
    
    return log_val


def _generate_initializations():
    """Generate diverse initial height profiles."""
    x = np.linspace(0, N - 1, N)
    inits = []

    # Uniform
    inits.append(np.ones(N))

    # Gaussian centered
    inits.append(np.exp(-0.5 * ((x - N/2) / (N/4))**2))

    # Bimodal: two peaks
    peak1 = np.exp(-0.5 * ((x - N*0.25) / (N/8))**2)
    peak2 = np.exp(-0.5 * ((x - N*0.75) / (N/8))**2)
    inits.append(peak1 + peak2)

    # Plateau
    plateau = np.zeros(N)
    plateau[N//4 : 3*N//4] = 1.0
    inits.append(plateau)

    # Cosine bump
    inits.append(0.5 * (1 + np.cos(2 * np.pi * x / N)))

    # Triangular
    inits.append(1 - 2 * np.abs(x - (N-1)/2) / (N-1))

    # Semicircle
    inits.append(np.sqrt(np.maximum(0, 1 - ((2*x/(N-1) - 1))**2)))

    # Sharp central peak
    inits.append(np.exp(-0.5 * ((x - N/2) / (N/10))**2))

    # Broad flat top with edges
    inits.append(np.maximum(0, 1 - np.abs(x - (N-1)/2) / (N/3)))

    # Step-like: high in center, low at edges
    step = np.ones(N)
    step[:N//5] = 0.2
    step[-N//5:] = 0.2
    inits.append(step)

    # Double step
    ds = np.ones(N)
    ds[:N//8] = 0.1
    ds[-N//8:] = 0.1
    ds[N//3 : 2*N//3] = 1.5
    inits.append(ds)

    # Spike: very narrow peak
    spike = np.zeros(N)
    spike[N//2] = 1.0
    inits.append(spike)

    # Double spike
    ds2 = np.zeros(N)
    ds2[N//4] = 1.0
    ds2[3*N//4] = 1.0
    inits.append(ds2)

    # Random smooth profiles via random Fourier coefficients
    rng = np.random.RandomState(42)
    for _ in range(10):
        freqs = rng.randint(1, 8, size=6)
        phases = rng.uniform(0, 2*np.pi, size=6)
        amps = rng.uniform(0.5, 1.0, size=6)
        profile = np.zeros(N)
        for f, p, a in zip(freqs, phases, amps):
            profile += a * np.cos(2 * np.pi * f * x / N + p)
        profile = np.maximum(0.1, profile)
        inits.append(profile)

    # More random smooth profiles
    for _ in range(10):
        points = np.sort(rng.uniform(0, N, size=5))
        heights = rng.uniform(0.5, 2.0, size=5)
        profile = np.interp(x, points, heights)
        inits.append(profile)

    return inits


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def objective(h):
        h = np.maximum(0, h)
        log_val = compute_log_value_smooth(h, beta=50.0)
        return -log_val

    best_val = 0.0
    best_h = np.ones(N)

    # Evaluate the seed
    seed_val = compute_value(np.ones(N))
    best_val = seed_val
    best_h = np.ones(N).copy()

    inits = _generate_initializations()

    # First pass: evaluate all initializations directly with exact value
    for init in inits:
        if time.time() > deadline - 10:
            break
        init = np.asarray(init, dtype=float)
        val = compute_value(init)
        if val > best_val:
            best_val = val
            best_h = init.copy()

    # Second pass: refine each initialization with L-BFGS-B using smooth log-surrogate
    init_vals = [(compute_value(np.asarray(init, dtype=float)), init) for init in inits]
    init_vals.sort(key=lambda x: -x[0])

    # Add perturbed versions of the best initializations
    extra_starts = []
    best_init = init_vals[0][1]
    rng = np.random.RandomState(123)
    for _ in range(15):
        perturbed = best_init + rng.normal(0, 0.3, N)
        perturbed = np.maximum(0, perturbed)
        extra_starts.append(perturbed)

    # Combine all starts for refinement
    all_starts = [iv[1] for iv in init_vals] + extra_starts

    for idx, init in enumerate(all_starts):
        if time.time() > deadline - 8:
            break

        init = np.asarray(init, dtype=float).copy()
        
        try:
            res = minimize(
                objective,
                init,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={
                    'maxiter': 300,
                    'ftol': 1e-12,
                    'gtol': 1e-8
                }
            )

            h_opt = np.maximum(0, res.x)
            val_opt = compute_value(h_opt)
            if val_opt > best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Third pass: refine the best solution further with higher precision
    if time.time() < deadline - 5:
        for attempt in range(30):
            if time.time() > deadline - 5:
                break

            perturbed = best_h + np.random.normal(0, 0.05, N)
            perturbed = np.maximum(0, perturbed)

            try:
                res = minimize(
                    objective,
                    perturbed,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={
                        'maxiter': 500,
                        'ftol': 1e-14,
                        'gtol': 1e-10
                    }
                )

                h_opt = np.maximum(0, res.x)
                val_opt = compute_value(h_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
            except Exception:
                pass

    # Final: ensure non-negativity and finiteness
    best_h = np.maximum(0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)

    return best_h.tolist()


if __name__ == "__main__":
    h = construct()
    print(f"Score: {compute_value(h)}")
