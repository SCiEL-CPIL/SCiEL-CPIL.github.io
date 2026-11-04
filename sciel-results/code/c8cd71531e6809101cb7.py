"""
Strategy: Bisection search on the metric value lambda.
For a given lambda, check if there exists x such that max_k |conv[k]| <= lambda * (sum x)^2.
Since the problem is homogeneous, we can fix sum(x) = 1 and minimize max_k |conv[k]|.
This is a minimax problem: minimize t subject to -t <= x^T A_k x <= t for all k.
We solve the feasibility problem for a fixed t using scipy.optimize.minimize (L-BFGS-B)
on the objective f(x) = max_k (x^T A_k x)^2.
If the minimum value is <= t, the constraint is feasible.
We bisect on t to find the smallest feasible t.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400

# Precompute FFT size
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

# Precompute the quadratic form matrices A_k for each lag k
# conv[k] = sum_{j} x[j] x[j+k] = x^T A_k x
# A_k is a symmetric matrix with 0.5 on the (j, j+k) and (j+k, j) positions.
# Instead of storing full matrices, we can compute x^T A_k x via FFT or direct loops.
# Direct loops are O(N^2) per evaluation, which is 160,000 ops.
# With L-BFGS-B, we might need ~100-500 evaluations per feasibility check.
# Bisection takes ~30-40 iterations.
# Total evals: 500 * 40 = 20,000.
# 20,000 * 160,000 = 3.2e9 ops. This might be too slow in Python.
# We need to vectorize the computation of all conv[k] values.

def _compute_conv(x):
    """Compute all autoconvolution lags for x using FFT."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv_full = np.fft.irfft(fx * fx, _FFT_SIZE)
    # conv_full has length _FFT_SIZE. The valid lags are centered or 0-indexed?
    # np.fft.ifft(np.fft.fft(x) * np.fft.fft(x)) gives circular convolution.
    # For linear convolution of length N, we need to unwrap.
    # conv[k] for k=0..N-1 is at index k in the linear convolution.
    # The circular convolution of size _FFT_SIZE >= 2N-1 contains the linear convolution.
    # Specifically, lin_conv[k] = circ_conv[k] for k=0..N-1 and lin_conv[k] = circ_conv[k] for k=N..2N-2?
    # Actually, if we zero-pad to size M >= 2N-1, then circ_conv[k] = lin_conv[k] for k=0..M-1.
    # The linear autoconvolution of x (length N) has length 2N-1.
    # lags: -(N-1) .. (N-1).
    # Standard FFT autoconv:
    # Let y = ifft(fft(x, M) * fft(x, M)).
    # y[k] = sum_j x[j] x[(j+k) % M].
    # Since M >= 2N-1, there is no wrap-around for lags k in -(N-1)..(N-1).
    # So y[k] = sum_j x[j] x[j+k] for j+k < N and j+k >= 0.
    # For k >= 0: y[k] = sum_{j=0}^{N-1-k} x[j] x[j+k].
    # For k < 0: let k' = -k > 0. y[M-k'] = sum_j x[j] x[j-k'] = sum_{i=k'}^{N-1} x[i] x[i-k'] = sum_{i=0}^{N-1-k'} x[i+k'] x[i] = conv[k'].
    # So the lags are at indices 0..N-1 and M-(N-1)..M-1.
    
    conv_pos = conv_full[:N] # lags 0 to N-1
    conv_neg = conv_full[M-N+1:M+1] if False else conv_full[_FFT_SIZE - (N - 1):] # lags -(N-1) to -1
    
    # Wait, let's be precise.
    # y[k] for k=0..M-1.
    # Lag 0: y[0]
    # Lag k (1..N-1): y[k]
    # Lag -k (-(N-1)..-1): y[M-k]
    
    lags = np.concatenate([conv_pos, conv_neg])
    return lags

def _objective_and_grad(x, t_target=None):
    """
    Objective: max_k (conv[k])^2.
    We want to minimize this.
    For gradient, we use the subgradient of the max.
    """
    conv = _compute_conv(x)
    sq_conv = conv * conv
    max_idx = np.argmax(sq_conv)
    val = sq_conv[max_idx]
    
    # Gradient of conv[k] w.r.t. x
    # conv[k] = x^T A_k x
    # grad = 2 A_k x
    # We can compute A_k x efficiently.
    # A_k x[j] = 0.5 * (x[j+k] + x[j-k]) (with boundary handling)
    # Actually, conv[k] = sum_i x[i] x[i+k].
    # d/dx_j conv[k] = x[j+k] + x[j-k] (if in bounds)
    
    k_star = max_idx
    if k_star < N:
        lag = k_star
    else:
        lag = k_star - N # This mapping is tricky, let's just use the index in the concatenated array
        
    # Let's re-evaluate the lag index properly.
    # lags array: [conv[0], conv[1], ..., conv[N-1], conv[-(N-1)], ..., conv[-1]]
    # Index 0..N-1 corresponds to lag 0..N-1.
    # Index N..2N-2 corresponds to lag -(N-1)..-1.
    
    if max_idx < N:
        k = max_idx
    else:
        k = max_idx - N + (N - 1) # No, simpler:
        # Index N is lag -(N-1). Index 2N-2 is lag -1.
        # k = -(N-1) + (max_idx - N)
        k = -(N - 1) + (max_idx - N)
        
    # Compute grad of conv[k]
    grad_conv = np.zeros(N)
    if k >= 0:
        # d/dx_j = x[j+k] + x[j-k]
        # For j=0..N-1-k: x[j+k] exists.
        # For j=k..N-1: x[j-k] exists.
        grad_conv[:N-k] += x[k:]
        grad_conv[k:] += x[:N-k]
    else:
        kk = -k
        grad_conv[:N-kk] += x[kk:]
        grad_conv[kk:] += x[:N-kk]
        
    # Objective is (conv[k_star])^2
    # grad = 2 * conv[k_star] * grad_conv
    grad = 2.0 * conv[max_idx] * grad_conv
    
    return val, grad

def _minimize_max_conv(x0):
    """
    Minimize max_k conv[k]^2 subject to sum(x)=1.
    We use L-BFGS-B with a constraint sum(x)=1.
    """
    # Initial guess
    x0 = x0 / np.sum(x0)
    
    # Constraint: sum(x) - 1 = 0
    # L-BFGS-B doesn't support equality constraints directly in the standard interface easily 
    # unless we use SLSQP or handle it manually. 
    # Let's use SLSQP for robustness with equality constraints, or project onto the hyperplane.
    # SLSQP might be slower but more reliable for small N.
    
    def obj(x):
        val, _ = _objective_and_grad(x)
        return val
    
    def obj_grad(x):
        _, g = _objective_and_grad(x)
        return g
    
    # Equality constraint
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0, 'jac': lambda x: np.ones(N)})
    
    # Bounds: None (x can be negative)
    
    res = minimize(obj, x0, jac=obj_grad, method='SLSQP', constraints=cons, options={'maxiter': 100, 'ftol': 1e-8})
    
    return res.fun, res.x

def construct():
    deadline = time.time() + 145.0
    
    # Initial guess: uniform
    x_init = np.ones(N)
    
    # We want to find the minimum possible value of max|conv|/s^2.
    # Since it's homogeneous, fix s=1.
    # Let f(x) = max|conv(x)|. We want min f(x) s.t. sum(x)=1.
    # Let lambda* be the optimal value.
    # We can just run the minimization directly. The "bisection" is implicit in the optimizer.
    # However, SLSQP on a non-smooth max function might struggle.
    # Let's use a smooth approximation: log-sum-exp of conv^2.
    
    def smooth_obj(x, sigma=0.1):
        conv = _compute_conv(x)
        sq = conv * conv
        # Normalize by s^2? No, s is fixed to 1 in the constraint.
        # But the metric is max|conv|/s^2. If s=1, it's max|conv|.
        # Wait, the evaluator computes raw value.
        # If we return x, the evaluator computes metric(x).
        # We want to minimize metric(x).
        # By homogeneity, metric(x) = metric(x/sum(x)).
        # So we can just minimize max|conv(x)| subject to sum(x)=1.
        
        # Smooth max: sigma * log(sum(exp(sq/sigma)))
        # To avoid overflow, subtract max
        max_sq = np.max(sq)
        if max_sq > 1e6:
            return max_sq
        return max_sq + sigma * np.log(np.sum(np.exp((sq - max_sq) / sigma)))
        
    def smooth_grad(x, sigma=0.1):
        conv = _compute_conv(x)
        sq = conv * conv
        max_sq = np.max(sq)
        if max_sq > 1e6:
            # Gradient is just gradient of max term
            idx = np.argmax(sq)
            _, g = _objective_and_grad(x)
            return g
        exp_sq = np.exp((sq - max_sq) / sigma)
        weights = exp_sq / np.sum(exp_sq)
        
        # Gradient of sum(weights_i * sq_i) = sum(weights_i * grad(sq_i))
        # grad(sq_i) = 2 * conv[i] * grad_conv[i]
        grad = np.zeros(N)
        for i in range(len(conv)):
            if weights[i] < 1e-6:
                continue
            k = i if i < N else -(N - 1) + (i - N)
            # Compute grad_conv for lag k
            g_k = np.zeros(N)
            if k >= 0:
                g_k[:N-k] += x[k:]
                g_k[k:] += x[:N-k]
            else:
                kk = -k
                g_k[:N-kk] += x[kk:]
                g_k[kk:] += x[:N-kk]
            grad += weights[i] * 2.0 * conv[i] * g_k
            
        return grad

    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0, 'jac': lambda x: np.ones(N)})
    
    best_x = x_init
    best_val = float('inf')
    
    rng = np.random.default_rng(2024)
    
    # Multi-start
    starts = [
        np.ones(N),
        np.cos(np.linspace(0, 2*np.pi, N, endpoint=False)),
        np.sin(np.linspace(0, 2*np.pi, N, endpoint=False)),
        rng.normal(size=N)
    ]
    
    for start in starts:
        if time.time() > deadline - 5:
            break
            
        start = start / np.sum(start)
        
        try:
            res = minimize(smooth_obj, start, jac=smooth_grad, method='SLSQP', 
                           constraints=cons, options={'maxiter': 200, 'ftol': 1e-10})
            
            if res.success or res.fun < best_val:
                # Check true metric
                x_test = res.x
                s = np.sum(x_test)
                if abs(s) < 1e-12:
                    continue
                conv = _compute_conv(x_test)
                max_abs_conv = np.max(np.abs(conv))
                val = max_abs_conv / (s * s)
                
                if val < best_val:
                    best_val = val
                    best_x = x_test
        except Exception:
            pass
            
    # Final polish with coordinate descent if time permits
    if time.time() < deadline - 2:
        # Simple local search
        x = best_x.copy()
        for _ in range(100):
            if time.time() > deadline - 1:
                break
            improved = False
            for i in range(N):
                # Try small perturbation
                for delta in [0.01, -0.01, 0.001, -0.001]:
                    x_new = x.copy()
                    x_new[i] += delta
                    s_new = np.sum(x_new)
                    if abs(s_new) < 1e-12:
                        continue
                    conv_new = _compute_conv(x_new)
                    val_new = np.max(np.abs(conv_new)) / (s_new * s_new)
                    if val_new < best_val:
                        best_val = val_new
                        best_x = x_new
                        x = x_new
                        improved = True
                        break
            if not improved:
                break
                
    return best_x
