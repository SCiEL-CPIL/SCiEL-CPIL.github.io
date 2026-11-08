"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys
from scipy.optimize import minimize, linear_sum_assignment

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin

    h = 0.5 / N  # interval width

    def compute_value(x):
        """Compute V = max_disc / (s^2 * dx). Lower is better."""
        x = np.asarray(x, dtype=np.float64)
        x[x < 0] = 0
        s = np.sum(x)
        if s == 0:
            return float('inf')
        # Linear autoconvolution via FFT
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(x, n=n_fft)
        conv = np.fft.irfft(F * F, n=n_fft)[:2 * N - 1]
        max_disc = np.max(conv)
        return max_disc / (s * s * h)

    def objective(x):
        return compute_value(x)

    def grad(x):
        """Compute gradient of objective using finite differences or analytic."""
        x = np.asarray(x, dtype=np.float64)
        x[x < 0] = 0
        s = np.sum(x)
        if s == 0:
            return np.zeros_like(x)
        
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(x, n=n_fft)
        conv = np.fft.irfft(F * F, n=n_fft)[:2 * N - 1]
        
        # Find index of max
        idx_max = np.argmax(conv)
        
        # Gradient of max conv w.r.t. x[i]
        # conv[k] = sum_j x[j] * x[j+k] (for linear conv, need to be careful)
        # Actually conv = ifft(fft(x)^2)
        # d(conv[k])/dx[i] = x[i+k] + x[i-k] (with appropriate bounds)
        # But we need the gradient of the max, which is subgradient at the argmax
        
        # Let's use numerical gradient for safety, but it's expensive.
        # Instead, let's use a smarter approach:
        # The objective is max(conv) / (s^2 * h)
        # Let M = max(conv)
        # dM/dx[i] is non-zero only for i that contribute to conv[idx_max]
        
        # conv[k] = sum_j x[j] x[j+k] where j and j+k are in [0, N-1]
        # For a given k, the pairs (j, j+k) and (j+k, j) both contribute? No, autoconvolution is symmetric.
        # (f*f)(t) = int f(t-s)f(s) ds. For discrete: sum_i x[i] x[i+k] * h^2? 
        # Let's stick to the FFT definition.
        
        # Numerical gradient is too slow for 600 vars * many iterations.
        # Let's use L-BFGS-B with numerical gradient, but limit iterations.
        # Or better: use the fact that we can compute the subgradient.
        
        # For now, let's return zeros and rely on L-BFGS-B's numerical gradient.
        # Actually, let's implement a proper numerical gradient with central differences
        # but only for a subset of coordinates at a time? No, L-BFGS-B needs full gradient.
        
        # Let's use forward differences with a small step, but that's 600 evaluations.
        # Too slow. Let's use the analytic subgradient.
        
        # The max of conv is at idx_max. 
        # conv[k] = sum_{j} x[j] * x[j+k] for valid j
        # Actually, for linear convolution of length N with itself, result has length 2N-1.
        # conv[k] = sum_{j=max(0,k-N+1)}^{min(N-1,k)} x[j] * x[j+k-N+1] ? No.
        # Let's use the FFT definition: conv = ifft(fft(x)^2)
        # This gives circular convolution if we don't zero-pad. We did zero-pad, so it's linear.
        # conv[k] = sum_j x[j] * x[k-j] for k in [0, 2N-2]
        
        # d(conv[k])/dx[i] = x[k-i] + x[i-k] (if we consider both terms, but for autoconv it's symmetric)
        # Actually, conv[k] = sum_j x[j] x[k-j]
        # d(conv[k])/dx[i] = x[k-i] (from j=i) + x[i-k] (from k-j=i => j=k-i)
        # But if i = k-i, i.e., k=2i, then it's 2*x[i].
        
        # So the subgradient of M = max(conv) w.r.t. x[i] is:
        # if conv[idx_max] is the unique max, then grad[i] = d(conv[idx_max])/dx[i]
        # Otherwise, it's a convex combination.
        
        # Let's assume unique max for simplicity.
        k = idx_max
        grad_conv = np.zeros(N)
        for i in range(N):
            # j = i, term x[i] * x[k-i]
            if 0 <= k - i < N:
                grad_conv[i] += x[k - i]
            # j = k-i, term x[k-i] * x[i]
            if 0 <= k - i < N and (k - i) != i:
                grad_conv[i] += x[k - i]
            # If k-i == i, we've already added it once, but the derivative should be 2*x[i]?
            # No, d/dx[i] (x[i]*x[i]) = 2*x[i]
            # In our sum, when j=i and k-j=i, it's the same term.
            # So if k-i == i, we add x[i] once for j=i and once for j=k-i, total 2*x[i]. Correct.
            # But my loop adds x[k-i] for j=i, and x[k-i] for j=k-i (if different).
            # If k-i == i, we add x[i] once in first if, and skip second if (since (k-i)!=i is false).
            # That's wrong. We should add 2*x[i] in that case.
        
        # Let's redo:
        grad_conv = np.zeros(N)
        for i in range(N):
            j1 = i
            j2 = k - i
            if 0 <= j1 < N:
                grad_conv[i] += x[j2] if 0 <= j2 < N else 0
            if 0 <= j2 < N and j2 != j1:
                grad_conv[i] += x[j1]
            # If j1 == j2, we added x[j2] once, but should add 2*x[j1]
            # Actually, if j1 == j2, then k = 2*i, and term is x[i]*x[i]
            # d/dx[i] (x[i]^2) = 2*x[i]
            # My loop: first if adds x[j2]=x[i]. Second if is skipped. Total: x[i]. Wrong.
        
        # Correct approach:
        grad_conv = np.zeros(N)
        for i in range(N):
            # Contribution to conv[k] from x[i]:
            # x[i] appears in term j=i: x[i]*x[k-i] (if k-i in range)
            # x[i] appears in term j=k-i: x[k-i]*x[i] (if k-i in range and k-i != i)
            if 0 <= k - i < N:
                grad_conv[i] += x[k - i]
            if 0 <= k - i < N and (k - i) != i:
                grad_conv[i] += x[k - i]
            # If k-i == i, we added once, but need 2*x[i]. So add one more x[i].
            if k - i == i and 0 <= i < N:
                grad_conv[i] += x[i]
        
        # Now, M = max(conv). If idx_max is unique, dM/dx[i] = grad_conv[i]
        # Objective: M / (s^2 * h)
        # d/dx[i] [M / (s^2 * h)] = (dM/dx[i] * s^2 - M * 2*s * dx[i]/dx[i]) / (s^4 * h)
        # = (grad_conv[i] * s^2 - 2*M*s*x[i]) / (s^4 * h)
        # = (grad_conv[i] * s - 2*M*x[i]) / (s^3 * h)
        
        M = conv[idx_max]
        if s == 0:
            return np.zeros(N)
        
        grad_obj = (grad_conv * s - 2 * M * x) / (s ** 3 * h)
        return grad_obj

    # Initial guess: Gaussian with moderate sigma
    mu = (N - 1) / 2.0
    sigma = 15.0
    x0 = np.exp(-0.5 * ((np.arange(N) - mu) / sigma) ** 2)
    x0 = np.maximum(x0, 0)

    # Try multiple initializations
    best_x = x0.copy()
    best_val = compute_value(best_x)
    
    # L-BFGS-B optimization
    if time.time() < deadline - 20:
        try:
            res = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                jac=grad,
                options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-8}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Coordinate descent refinement
    if time.time() < deadline - 10:
        rng = np.random.RandomState(42)
        order = rng.permutation(N)
        n_passes = 10
        for p in range(n_passes):
            if time.time() > deadline - 10:
                break
            improved = False
            for idx in order:
                if time.time() > deadline - 10:
                    break
                old = best_x[idx]
                for mult in [0.5, 0.75, 1.25, 1.5, 0.25, 2.0, 0.1, 0.9, 1.1]:
                    candidate = best_x.copy()
                    candidate[idx] = old * mult
                    if candidate[idx] < 0:
                        continue
                    v = compute_value(candidate)
                    if v < best_val:
                        best_val = v
                        best_x = candidate
                        improved = True
            if not improved:
                break

    return list(best_x)
# EVOLVE-BLOCK-END
