import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    """
    Use a smooth periodic representation with gradient-based optimization.
    h_i = 0.5 + sum_k a_k * cos(2*pi*k*i/N + phi_k)
    This ensures smoothness and can break the uniform saddle point.
    """
    start_time = time.time()
    time_limit = 130.0  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    # Parameters: amplitudes and phases for first few Fourier modes
    # h_i = 0.5 + sum_{k=1}^{K} a_k * cos(2*pi*k*i/N + phi_k)
    K = 5  # number of Fourier modes
    # params = [a_1, phi_1, a_2, phi_2, ..., a_K, phi_K]
    
    def params_to_h(params):
        """Convert Fourier parameters to step heights in [0,1]."""
        h = np.full(N, 0.5)
        for k in range(1, K + 1):
            a = params[2*(k-1)]
            phi = params[2*(k-1)+1]
            h += a * np.cos(2 * np.pi * k * np.arange(N) / N + phi)
        # Clip to [0,1] and renormalize to sum = N/2
        h = np.clip(h, 0.0, 1.0)
        # Adjust to satisfy sum constraint
        s = np.sum(h)
        if s > 1e-10:
            h = h * (N / 2.0) / s
        # Final clip
        h = np.clip(h, 0.0, 1.0)
        # One more adjustment for precision
        s2 = np.sum(h)
        if abs(s2 - N/2.0) > 1e-12 and s2 > 1e-10:
            h += (N/2.0 - s2) / N
        return np.clip(h, 0.0, 1.0)

    def compute_overlap(h):
        """Compute C(k) = sum_i h_i * (1 - h_{i+k}) for all shifts k."""
        C = np.zeros(N)
        for k in range(N):
            h_shifted = np.roll(h, -k)
            C[k] = np.sum(h * (1.0 - h_shifted))
        return C

    def objective(params):
        """Maximize overlap (we minimize this). Use smooth max approximation."""
        h = params_to_h(params)
        C = compute_overlap(h)
        # Smooth maximum using log-sum-exp
        beta = 50.0
        return np.log(np.sum(np.exp(beta * C))) / beta

    def objective_grad(params):
        """Numerical gradient of the smooth max objective."""
        eps = 1e-7
        grad = np.zeros_like(params)
        f0 = objective(params)
        for i in range(len(params)):
            params[i] += eps
            f1 = objective(params)
            params[i] -= eps
            grad[i] = (f1 - f0) / eps
        return grad

    # Multi-restart optimization
    best_h = np.full(N, 0.5)
    best_score = np.inf

    def evaluate_h(h):
        """Return the max overlap for a given h."""
        C = compute_overlap(h)
        return np.max(C)

    # Initial score
    best_score = evaluate_h(best_h)

    rng = np.random.default_rng(42)

    # Try multiple random initializations
    n_restarts = 15
    for r in range(n_restarts):
        if elapsed() > time_limit - 10.0:
            break

        # Random initialization
        params = np.zeros(2 * K)
        for k in range(1, K + 1):
            # Smaller amplitudes for higher frequencies
            params[2*(k-1)] = rng.uniform(-0.3, 0.3) / k
            params[2*(k-1)+1] = rng.uniform(0, 2 * np.pi)

        # Run L-BFGS-B
        try:
            result = minimize(
                objective,
                params,
                method='L-BFGS-B',
                jac=None,  # Use numerical gradient
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            h_opt = params_to_h(result.x)
            score = evaluate_h(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
        except Exception:
            pass

    # If no improvement found, try a structured pattern
    if best_score >= N / 4.0 - 1e-6:
        # Try alternating pattern with slight perturbation
        for amp in [0.1, 0.2, 0.3]:
            if elapsed() > time_limit - 5.0:
                break
            h_alt = np.full(N, 0.5)
            h_alt[::2] = 0.5 + amp
            h_alt[1::2] = 0.5 - amp
            h_alt = np.clip(h_alt, 0.0, 1.0)
            s = np.sum(h_alt)
            h_alt = h_alt * (N/2.0) / s
            h_alt = np.clip(h_alt, 0.0, 1.0)
            s2 = np.sum(h_alt)
            h_alt += (N/2.0 - s2) / N
            h_alt = np.clip(h_alt, 0.0, 1.0)
            score = evaluate_h(h_alt)
            if score < best_score:
                best_score = score
                best_h = h_alt.copy()

    # Final verification and cleanup
    h_final = best_h.copy()
    s = np.sum(h_final)
    if abs(s - N/2.0) > 1e-10:
        h_final += (N/2.0 - s) / N
    h_final = np.clip(h_final, 0.0, 1.0)
    # Ensure exact sum
    s2 = np.sum(h_final)
    if abs(s2 - N/2.0) > 1e-12:
        h_final += (N/2.0 - s2) / N
    h_final = np.clip(h_final, 0.0, 1.0)

    return h_final

# EVOLVE-BLOCK-END
