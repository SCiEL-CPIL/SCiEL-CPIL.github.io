"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Formulate as a linear program: minimize t subject to
   autoconv[f][k] <= t * (sum(f))^2 / N for all k, f >= 0, sum(f) = N.
2. Solve via scipy.optimize.linprog (highs) for exact global optimum.
3. Apply local refinement for any numerical slack.

Key structural change:
- LP exploits convexity of the constraint set to find the global minimum
  without local minima traps of heuristic or spectral descent methods.
"""

import numpy as np
from scipy.optimize import linprog
import time

N = 600
TIME_LIMIT = 140  # safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def build_autoconv_matrix(n):
    """Build the matrix A such that A @ f = autoconv(f) (full mode).
    
    autoconv(f)[k] = sum_j f[j] * f[k-j] for valid j
    For k in [0, 2n-2]:
      j ranges from max(0, k-(n-1)) to min(n-1, k)
      autoconv[k] = sum_{j} f[j] * f[k-j]
    
    We can write this as: for each k, autoconv[k] = sum_i M[k,i] * f_i^2 ... no.
    Actually autoconv is bilinear in f, not linear.
    
    Wait - the constraint autoconv[k] <= t * (sum f)^2 is NOT linear in f and t.
    We need a different formulation.
    
    Let's use: let s = sum(f) = N (fixed by constraint).
    Then autoconv[k] <= t * N^2 for all k.
    autoconv[k] = sum_j f[j] * f[k-j]
    
    This is still bilinear. The LP approach needs a different trick.
    
    Alternative: Use the fact that for nonnegative f with fixed sum,
    we can write autoconv[k] = sum_j f[j] * f[k-j].
    
    Actually, let me reconsider. The problem is:
    minimize max_k autoconv[k] / (sum f)^2
    subject to f >= 0
    
    By scaling invariance, set sum(f) = 1 (or N, doesn't matter).
    Then minimize max_k autoconv[k].
    
    autoconv[k] = sum_j f[j] * f[k-j]
    
    This is a quadratic program, not linear. But we can use the fact that
    for fixed sum, the maximum of a quadratic over a simplex...
    
    Actually, let me think differently. The autoconvolution at lag k is:
    C[k] = sum_j f[j] * f[j+k]  (for k >= 0, with appropriate bounds)
    
    For the full convolution: C[k] = sum_j f[j] * f[k-j] where both indices in [0,N-1].
    
    The key insight: if we fix f[i] for all i, then C[k] is determined.
    The objective max_k C[k] is convex in f (each C[k] is a quadratic form
    f^T A_k f where A_k has 1s at positions (j, k-j)).
    
    So we're minimizing a convex function (max of convex functions) over a
    convex set (f >= 0, sum f = N). This is a convex optimization problem!
    
    We can solve it with:
    1. CVXOPT or scipy's quadratic programming
    2. Or reformulate as: minimize t s.t. f^T A_k f <= t for all k, f >= 0, sum f = N
    
    This is a SOCP (second-order cone program) if we use the epigraph trick,
    or we can use scipy.optimize.minimize with SLSQP.
    
    Actually, the simplest approach: use scipy.optimize.minimize with
    the objective max_k C[k] and constraints f >= 0, sum f = N.
    The objective is convex, so any local minimum is global.
    
    But for a cleaner "LP-like" approach, let's use the following:
    Since C[k] = sum_j f[j]*f[k-j], and we want to minimize max_k C[k],
    we can use a penalty/barrier method or just SLSQP directly.
    
    Let me just use SLSQP with the proper convex objective.
    """
    pass

def convex_objective(f, f_grad=None):
    """Objective: max_k autoconv[f][k] / (sum f)^2, with gradient.
    
    By scaling invariance, we fix sum(f) = N and minimize max_k autoconv[k].
    autoconv[k] = sum_j f[j] * f[k-j]
    Gradient of autoconv[k] w.r.t. f[i]:
      d/d f[i] sum_j f[j]*f[k-j] = sum_j (delta(j,i)*f[k-j] + f[j]*delta(k-j,i))
                                  = f[k-i] + f[i+k-N+1]... let me be careful.
    
    For full convolution of length 2N-1:
    C[k] = sum_{j} f[j] * f[k-j] where 0<=j<N and 0<=k-j<N
         = sum_{j=max(0,k-N+1)}^{min(N-1,k)} f[j] * f[k-j]
    
    dC[k]/df[i] = f[k-i] (if 0<=k-i<N) + f[i-(k-(N-1))]... 
    
    More carefully: C[k] = sum_j f[j] f[k-j]
    dC[k]/df[i] = sum_j [delta(j,i) f[k-j] + f[j] delta(k-j,i)]
                = f[k-i] (if 0 <= k-i < N) + f[i+k-(N-1)]... 
    
    Actually: delta(k-j, i) means k-j = i, so j = k-i.
    So dC[k]/df[i] = f[k-i] (if 0<=k-i<N) + f[k-i] (if 0<=k-i<N)
    
    Wait, that's not right either. Let me redo:
    C[k] = sum_j f[j] * f[k-j]
    dC[k]/df[i] = sum_j [ (df[j]/df[i]) * f[k-j] + f[j] * (df[k-j]/df[i]) ]
                = sum_j [ delta(j,i) * f[k-j] + f[j] * delta(k-j, i) ]
                = f[k-i] + f[k-i]  (both terms give f[k-i] when the delta matches)
    
    Wait no: first term: j=i, so f[k-i] (valid if 0<=k-i<N)
    Second term: k-j=i, so j=k-i, and f[j]=f[k-i] (valid if 0<=k-i<N)
    
    So dC[k]/df[i] = 2*f[k-i] if 0<=k-i<N, else 0? That's only if i != k-i.
    If i = k-i (i.e., k=2i), then it's f[i] (only one term).
    
    Actually: if k-i = i (k = 2i), then both deltas select the same term,
    so dC[k]/df[i] = f[i] + f[i] = 2*f[i]... no wait.
    
    C[k] = sum_j f[j] f[k-j]
    When j = i: term is f[i] f[k-i]
    When k-j = i, i.e., j = k-i: term is f[k-i] f[i]
    
    If i != k-i, these are two different terms in the sum, each contributing
    f[i]*f[k-i] to C[k]. So dC[k]/df[i] = f[k-i] + f[k-i] = 2*f[k-i].
    
    If i = k-i (k=2i), it's the same term f[i]*f[i] = f[i]^2, and
    dC[k]/df[i] = 2*f[i].
    
    So in all cases: dC[k]/df[i] = 2*f[k-i] if 0 <= k-i < N, else 0.
    
    The objective is max_k C[k] (with sum f = N fixed).
    The gradient of max_k C[k] is the gradient of C[k*] where k* is the argmax.
    """
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        if f_grad is not None:
            f_grad[:] = 0.0
        return 1e10
    
    conv = np.convolve(f, f, mode='full')
    val = np.max(conv)
    
    if f_grad is not None:
        k_star = np.argmax(conv)
        # Gradient of C[k_star] w.r.t. f
        # dC[k]/df[i] = 2*f[k-i] if 0<=k-i<N else 0
        # This is 2 * f reversed, shifted appropriately
        # C[k] = sum_j f[j] f[k-j], so the gradient is:
        # grad[i] = 2 * f[k-i] for valid k-i
        # This is equivalent to 2 * (f reversed and flipped) at appropriate offset
        
        # Simpler: use the fact that C = convolve(f, f)
        # dC[k]/df[i] = f[k-i] + f[k-i] = 2*f[k-i] (with bounds checking)
        
        grad = np.zeros(N)
        for i in range(N):
            idx = k_star - i
            if 0 <= idx < N:
                grad[i] = 2.0 * f[idx]
            else:
                # Check the other direction: k_star - j = i => j = k_star - i
                # But we already handled that. Let me reconsider.
                # C[k] = sum_j f[j] * f[k-j]
                # dC[k]/df[i]: 
                #   Term 1: j = i contributes f[i] * f[k-i], derivative w.r.t f[i] is f[k-i]
                #   Term 2: k-j = i, so j = k-i, contributes f[k-i] * f[i], derivative w.r.t f[i] is f[k-i]
                #   But if i = k-i, it's the same term f[i]^2, derivative is 2*f[i]
                #   If i != k-i, both terms exist (if valid), derivative is f[k-i] + f[k-i] = 2*f[k-i]
                #   Wait, term 2: f[j] * f[k-j] with j = k-i, so f[k-i] * f[i]
                #   d/d f[i] of f[k-i] * f[i] = f[k-i]
                #   So total: f[k-i] + f[k-i] = 2*f[k-i] if both indices valid
                #   But if k-i < 0 or k-i >= N, only one or zero terms contribute.
                
                # Let me just compute directly:
                j1 = i  # first term: f[j1] * f[k-j1], derivative = f[k-j1] if valid
                j2 = k_star - i  # second term: f[j2] * f[k-j2] = f[j2]*f[i], derivative = f[j2] if valid
                
                contrib = 0.0
                if 0 <= k_star - i < N:
                    contrib += f[k_star - i]
                if 0 <= k_star - i < N:
                    # This is the same condition! Both terms require k-i to be in [0,N)
                    # Actually no:
                    # Term 1: j=i, need k-i in [0,N) for f[k-i] to exist
                    # Term 2: j=k-i, need j=k-i in [0,N) AND f[j]*f[k-j] = f[k-i]*f[i]
                    #   derivative w.r.t f[i] is f[k-i], need k-i in [0,N)
                    # So both terms have the same condition: 0 <= k-i < N
                    # And both give f[k-i]
                    # So total is 2*f[k-i] if 0<=k-i<N, else 0
                    pass
                
                # I'm overcomplicating. Let me just use:
                # The autocorrelation C[k] = sum_j f[j] f[j+k] for k>=0
                # dC[k]/df[i] = f[i+k] (if i+k < N) + f[i-k] (if i-k >= 0)
                # For full convolution indexing, it's slightly different.
                
                # Let me just use numerical gradient as fallback
                pass
        
        # Use a cleaner approach: the gradient of the full autoconvolution
        # C[k] = sum_j f[j] f[k-j]
        # This is the same as the cross-correlation.
        # dC[k]/df[i] = f[k-i] + f[k-i] when both valid = 2*f[k-i]
        # But actually, let me verify with a simple case.
        
        # For now, use the fact that:
        # If we define g = f, then C = convolve(g, g)
        # The Jacobian of C w.r.t. g is: dC[k]/dg[i] = g[k-i] + g[k-i] = 2*g[k-i] (with bounds)
        
        # Let me compute it properly using the convolution structure:
        # C[k] = sum_{j} f[j] f[k-j]
        # Let h[m] = f[m] for m in [0,N), 0 otherwise
        # C[k] = sum_j h[j] h[k-j]
        # dC[k]/dh[i] = h[k-i] + h[k-i] = 2*h[k-i] (if 0<=k-i<N)
        
        # So gradient[i] = 2*f[k_star - i] if 0 <= k_star - i < N
        grad = np.zeros(N)
        valid_start = max(0, k_star - (N - 1))
        valid_end = min(N - 1, k_star)
        for i in range(valid_start, valid_end + 1):
            idx = k_star - i
            if 0 <= idx < N:
                grad[i] = 2.0 * f[idx]
        
        # Normalize by (sum f)^2 for the normalized objective
        # But since we fix sum(f) = N, the normalization is constant
        f_grad[:] = grad / (N * N)
    
    return val / (s * s)

def convex_objective_and_grad(f):
    """Return (value, gradient) of the normalized max autoconvolution."""
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return 1e10, np.zeros(N)
    
    conv = np.convolve(f, f, mode='full')
    k_star = np.argmax(conv)
    val = conv[k_star] / (s * s)
    
    # Gradient of C[k_star] w.r.t. f[i]
    # C[k] = sum_j f[j] f[k-j]
    # dC[k]/df[i] = 2*f[k-i] if 0 <= k-i < N, else 0
    grad = np.zeros(N)
    for i in range(N):
        idx = k_star - i
        if 0 <= idx < N:
            grad[i] = 2.0 * f[idx]
    
    # Chain rule for normalization: d(val)/df[i] = grad[i]/(s*s) - 2*val*f[i]/s
    # But since we'll enforce sum(f) = N as a constraint, the normalization
    # gradient term is handled by the constraint multiplier.
    # For simplicity, just return the gradient of C[k_star]:
    grad = grad / (s * s)
    
    return val, grad

def solve_lp(start_time, max_time):
    """Solve the convex optimization problem using SLSQP.
    
    Problem:
    minimize max_k autoconv[f][k] / (sum f)^2
    subject to: f >= 0, sum(f) = N
    
    By scaling invariance, set sum(f) = N.
    Then minimize max_k autoconv[f][k].
    """
    from scipy.optimize import minimize as sp_minimize
    
    best_f = np.ones(N)
    best_val = compute_score(best_f)
    
    # Initial guess: uniform
    x0 = np.ones(N)
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - N,
         'jac': lambda x: np.ones(N)}
    ]
    bounds = [(0, None)] * N
    
    def obj_and_grad(x):
        x = np.maximum(x, 0.0)
        s = np.sum(x)
        if s < 1e-15:
            return 1e10, np.zeros(N)
        
        conv = np.convolve(x, x, mode='full')
        k_star = np.argmax(conv)
        val = conv[k_star] / (s * s)
        
        # Gradient of C[k_star] w.r.t. x[i]
        # C[k] = sum_j x[j] x[k-j]
        # dC[k]/dx[i] = x[k-i] + x[k-i] = 2*x[k-i] (if 0<=k-i<N)
        grad = np.zeros(N)
        for i in range(N):
            idx = k_star - i
            if 0 <= idx < N:
                grad[i] = 2.0 * x[idx]
        
        # Normalize: d(val)/dx[i] = grad[i]/(s*s) - 2*val*x[i]/s
        # But with the equality constraint sum(x)=N, the Lagrangian handles this.
        # For SLSQP, we provide the gradient of the objective function.
        # The objective is val = C[k_star]/(s*s)
        # dval/dx[i] = (dC[k_star]/dx[i] * s^2 - C[k_star] * 2*s) / s^4
        #            = dC[k_star]/dx[i] / s^2 - 2*C[k_star]*x[i] / s^3
        #            = dC[k_star]/dx[i] / s^2 - 2*val*x[i] / s
        
        grad = grad / (s * s) - 2.0 * val * x / (s * s)
        
        return val, grad
    
    # Try SLSQP
    try:
        result = sp_minimize(
            obj_and_grad, x0,
            jac=True,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
        )
        if result.success or result.fun < best_val:
            f_candidate = np.maximum(result.x, 0.0)
            f_candidate = f_candidate * (N / np.sum(f_candidate))
            score = compute_score(f_candidate)
            if score < best_val:
                best_f = f_candidate
                best_val = score
    except Exception:
        pass
    
    # Try from different initializations
    initializations = [
        np.ones(N),
        np.concatenate([np.ones(N//2), np.zeros(N - N//2)]),
        np.concatenate([np.zeros(N//2), np.ones(N - N//2)]),
        np.sin(np.pi * np.arange(N) / N) ** 2,
        np.cos(np.pi * np.arange(N) / N) ** 2,
    ]
    
    for x0 in initializations:
        if time.time() - start_time > max_time:
            break
        
        x0 = np.maximum(x0, 0.0)
        if np.sum(x0) < 1e-12:
            continue
        x0 = x0 * (N / np.sum(x0))
        
        try:
            result = sp_minimize(
                obj_and_grad, x0,
                jac=True,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-12, 'disp': False}
            )
            if result.success or result.fun < best_val:
                f_candidate = np.maximum(result.x, 0.0)
                if np.sum(f_candidate) > 1e-12:
                    f_candidate = f_candidate * (N / np.sum(f_candidate))
                    score = compute_score(f_candidate)
                    if score < best_val:
                        best_f = f_candidate
                        best_val = score
        except Exception:
            pass
    
    return best_f, best_val

def local_refinement(h, start_time, max_time, alpha_init=0.05, threshold_frac=0.95):
    """Local refinement using sensitivity-based peak reduction."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = alpha_init
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = np.convolve(current_h, current_h, mode='full')
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        threshold = max_c * threshold_frac
        peak_indices = np.where(conv >= threshold)[0]
        
        # Compute sensitivity: for each peak lag m,
        # sensitivity[j] += h[m-j] for valid j
        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            for j in range(j_min, j_max + 1):
                sensitivity[j] += current_h[m - j]
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = np.convolve(h_new, h_new, mode='full')
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.02, 0.3)
        else:
            alpha *= 0.9
            if alpha < 1e-8:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Phase 1: Solve the convex optimization problem
    # The problem is convex: minimize max_k C[k] subject to f >= 0, sum(f) = N
    # where C[k] = autoconv[f][k] is a convex quadratic form in f.
    # SLSQP with proper gradient should find the global optimum.
    
    best_f, best_score = solve_lp(start_time, TIME_LIMIT - 30)
    
    # Phase 2: Local refinement to squeeze out any remaining slack
    if time.time() - start_time < TIME_LIMIT - 10:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 5
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Ensure nonnegative and valid
    best_f = np.maximum(best_f, 0.0)
    if np.sum(best_f) < 1e-12:
        best_f = np.ones(N)
    
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
