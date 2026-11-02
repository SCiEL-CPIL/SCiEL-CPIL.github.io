"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _overlap_at_shift(h, shift):
    """Compute integral of h(x) * (1 - h(x + shift)) for shift in [0, 1]."""
    n = len(h)
    # h is step function on [0,2] with n steps of width 2/n
    # h[i] is value on [2*i/n, 2*(i+1)/n)
    # (1-h)(x+shift): for x in [0,2], x+shift in [shift, 2+shift]
    # We need to integrate h(x)*(1-h(x+shift)) over x in [0,2]
    # 
    # h(x) is piecewise constant. (1-h(x+shift)) is also piecewise constant
    # but shifted. The product is piecewise constant on the union of breakpoints.
    # 
    # Simpler: use the fact that h is a step function with n steps.
    # h(x) = h[i] for x in [2i/n, 2(i+1)/n)
    # h(x+shift) = h[j] where j = floor((x+shift)*n/2)
    # 
    # For each step i of h, x ranges over [2i/n, 2(i+1)/n)
    # x+shift ranges over [2i/n + shift, 2(i+1)/n + shift)
    # We need to split this interval at the breakpoints of h, which are at 2k/n.
    # 
    # Alternative approach: discretize more finely or use FFT-based convolution.
    # 
    # Let's use a direct computation:
    # The overlap is sum over pairs (i,j) of h[i]*(1-h[j]) * measure of x where 
    # h(x)=h[i] and h(x+shift)=h[j]
    # 
    # h(x) = h[i] when x in [2i/n, 2(i+1)/n)
    # h(x+shift) = h[j] when x+shift in [2j/n, 2(j+1)/n), i.e., x in [2j/n - shift, 2(j+1)/n - shift)
    # 
    # Intersection: [max(2i/n, 2j/n - shift), min(2(i+1)/n, 2(j+1)/n - shift))
    # 
    # This is O(n^2) per shift. With n=100 and many shifts, this could be slow.
    # Let's use a smarter approach.
    
    # Actually, for the optimization, we only need to evaluate at a few key shifts.
    # The maximum overlap over all shifts is what we want to minimize.
    # But evaluating all shifts is expensive.
    
    # Let's use a different approach: compute the overlap as a function of shift
    # using convolution-like operations.
    
    # h is defined on [0,2] with n steps. Let's think of h as a function.
    # The overlap at shift s is: ∫₀² h(x)(1-h(x+s)) dx
    # = ∫₀² h(x) dx - ∫₀² h(x)h(x+s) dx
    # = 1 - ∫₀² h(x)h(x+s) dx  (since ∫₀² h(x)dx = 1)
    
    # So we want to maximize the minimum of ∫₀² h(x)h(x+s) dx over s.
    # Equivalently, minimize the maximum of 1 - ∫₀² h(x)h(x+s) dx.
    
    # The cross-correlation R(s) = ∫₀² h(x)h(x+s) dx
    # We want to minimize max_s (1 - R(s)) = 1 - min_s R(s)
    
    # So we want to maximize min_s R(s).
    
    # R(s) for s in [0, 2] (shifts beyond 2 give 0 overlap for the support, 
    # but since h is on [0,2], h(x+s) is 0 for x+s > 2, i.e., x > 2-s)
    # Actually h is defined on [0,2], so for x+s > 2, h(x+s) is outside domain.
    # We should consider s in [0, 2] and h(x+s) = 0 when x+s > 2.
    
    # Let's compute R(s) = ∫₀^{2-s} h(x)h(x+s) dx for s in [0,2]
    # (for s > 2, R(s) = 0)
    
    # Using step functions:
    # h(x) = h[i] for x in [2i/n, 2(i+1)/n)
    # h(x+s) = h[j] for x+s in [2j/n, 2(j+1)/n), i.e., x in [2j/n - s, 2(j+1)/n - s)
    
    # R(s) = Σᵢ Σⱼ h[i]h[j] * length([max(2i/n, 2j/n - s, 0), min(2(i+1)/n, 2(j+1)/n - s, 2-s)))
    
    # This is still O(n²) per s. For n=100, that's 10000 ops per s.
    # If we evaluate at, say, 50 shifts, that's 500k ops. Manageable.
    
    # But for gradient-based optimization, we need this many times.
    # Let's precompute or use a faster method.
    
    # Alternative: use FFT. Pad h to length 2n (for [0,4] with step 2/n), 
    # then circular cross-correlation gives us R at discrete shifts.
    
    # Actually, let's just do direct computation for now and optimize later.
    
    # For the optimization, let's evaluate R(s) at a set of shifts and take min.
    # The shifts that matter are where the minimum occurs.
    
    # Let's compute R(s) for s = 0, 2/n, 4/n, ..., 2
    # That's n+1 shifts.
    
    # R(s) = ∫₀^{2-s} h(x)h(x+s) dx
    # With h as step function:
    # For shift s = 2k/n (k = 0, ..., n):
    # h(x+s) is h shifted by k steps.
    # R(2k/n) = Σᵢ h[i] * h[i+k] * (2/n)   for i+k < n
    #           = (2/n) * Σ_{i=0}^{n-k-1} h[i] * h[i+k]
    #           = (2/n) * (h · h_k)  where h_k is h shifted by k
    
    # So R(2k/n) = (2/n) * Σ_{i=0}^{n-k-1} h[i]*h[i+k]
    
    # The minimum over k of R(2k/n) gives us the minimum cross-correlation.
    # We want to maximize this minimum.
    
    # This is much faster! O(n) per shift, O(n²) total for all shifts.
    
    pass  # placeholder

def _compute_cross_correlation_min(h):
    """Compute min over shifts of the cross-correlation R(s) = ∫ h(x)h(x+s)dx.
    
    We evaluate at s = 2k/n for k = 0, ..., n-1 (shifts within [0,2)).
    R(2k/n) = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k]
    
    We want to maximize min_k R(2k/n).
    """
    n = len(h)
    min_corr = float('inf')
    for k in range(1, n):  # k=0 gives R(0)=∫h² which is large, skip
        # R(2k/n) = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k]
        corr = np.dot(h[:n-k], h[k:]) * (2.0 / n)
        if corr < min_corr:
            min_corr = corr
    return min_corr

def _objective_and_grad(x):
    """Objective: minimize 1 - min_corr, which equals max_overlap.
    
    x is the vector of h values.
    """
    h = x
    min_corr = _compute_cross_correlation_min(h)
    obj = 1.0 - min_corr
    # Gradient: d(obj)/d(h[i]) = -d(min_corr)/d(h[i])
    # min_corr = min_k (2/n) * sum_{j=0}^{n-k-1} h[j]*h[j+k]
    # The gradient of min is the gradient of the argmin term (subgradient).
    # For a smooth surrogate, we could use soft-min, but for now use hard min.
    
    # Find which k achieves the minimum
    n = len(h)
    corrs = []
    for k in range(1, n):
        corr = np.dot(h[:n-k], h[k:]) * (2.0 / n)
        corrs.append(corr)
    min_idx = np.argmin(corrs)
    k_star = min_idx + 1
    
    # Gradient of R(2k_star/n) = (2/n) * sum_{j=0}^{n-k_star-1} h[j]*h[j+k_star]
    grad_corr = np.zeros(n)
    for j in range(n - k_star):
        grad_corr[j] += (2.0 / n) * h[j + k_star]
        grad_corr[j + k_star] += (2.0 / n) * h[j]
    
    grad_obj = -grad_corr
    return obj, grad_obj

def _project_to_simplex(v, target_sum):
    """Project v onto the simplex {x : x >= 0, sum(x) = target_sum}."""
    # We want x in [0,1]^n with sum = n/2
    # First, project onto {x >= 0, sum = target_sum}
    # Then clip to [0,1] and re-project if needed.
    
    n = len(v)
    # Use the standard simplex projection
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - target_sum))[0][-1]
    theta = (cssv[rho] - target_sum) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    
    # Now clip to [0,1] and adjust
    w = np.clip(w, 0, 1)
    # Adjust to ensure sum = target_sum
    diff = target_sum - np.sum(w)
    if abs(diff) > 1e-12:
        # Distribute the difference
        if diff > 0:
            # Need to increase some values
            mask = w < 1.0
            if np.sum(mask) > 0:
                w[mask] += diff / np.sum(mask)
                w = np.clip(w, 0, 1)
        else:
            # Need to decrease some values
            mask = w > 0.0
            if np.sum(mask) > 0:
                w[mask] += diff / np.sum(mask)
                w = np.clip(w, 0, 1)
    
    return w

def construct():
    """Construct step heights minimizing maximum overlap."""
    n = N
    target_sum = n / 2.0
    
    # Initial guess: uniform 0.5
    x0 = np.full(n, 0.5)
    
    best_x = x0.copy()
    best_obj = 1.0 - _compute_cross_correlation_min(x0)
    
    # Try multiple restarts with different initializations
    np.random.seed(42)
    
    restarts = [
        np.full(n, 0.5),
        np.concatenate([np.full(n//2, 0.0), np.full(n - n//2, 1.0)]),
        np.concatenate([np.full(n//2, 1.0), np.full(n - n//2, 0.0)]),
        np.random.uniform(0, 1, n),
        np.random.uniform(0, 1, n),
        # Bang-bang type: alternating 0 and 1
        np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)]),
        # Smooth transition
        np.linspace(0, 1, n),
        np.linspace(1, 0, n),
    ]
    
    for x0 in restarts:
        if _time_remaining() < 5.0:
            break
        
        # Project to feasible set
        x = _project_to_simplex(x0, target_sum)
        
        # Local optimization
        try:
            res = minimize(
                _objective_and_grad,
                x,
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, 1)] * n,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.success or res.fun < best_obj:
                # Project result back to feasible set
                res_x = _project_to_simplex(res.x, target_sum)
                res_obj = 1.0 - _compute_cross_correlation_min(res_x)
                if res_obj < best_obj:
                    best_obj = res_obj
                    best_x = res_x
        except Exception:
            pass
    
    # Final polish with tighter tolerance if time permits
    if _time_remaining() > 10.0:
        try:
            res = minimize(
                _objective_and_grad,
                best_x,
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, 1)] * n,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < best_obj:
                res_x = _project_to_simplex(res.x, target_sum)
                res_obj = 1.0 - _compute_cross_correlation_min(res_x)
                if res_obj < best_obj:
                    best_obj = res_obj
                    best_x = res_x
        except Exception:
            pass
    
    # Ensure exact sum
    best_x = np.clip(best_x, 0, 1)
    diff = target_sum - np.sum(best_x)
    if abs(diff) > 1e-12:
        mask = (best_x > 0) & (best_x < 1)
        if np.sum(mask) > 0:
            best_x[mask] += diff / np.sum(mask)
            best_x = np.clip(best_x, 0, 1)
    
    return best_x
# EVOLVE-BLOCK-END
