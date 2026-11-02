"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 135.0  # safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_all_corrs_fft(h):
    """Compute all linear cross-correlations R_k = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k] for k=1,...,n-1
    using FFT for O(n log n) speed."""
    n = len(h)
    h_padded = np.zeros(2 * n)
    h_padded[:n] = h
    H = np.fft.rfft(h_padded, n=2*n)
    acf = np.fft.irfft(H * np.conj(H), n=2*n)
    corrs = acf[1:n] * (2.0 / n)
    return corrs

def _compute_cross_correlation_min(h):
    """Compute min over shifts of the cross-correlation R(s) = ∫ h(x)h(x+s)dx."""
    corrs = _compute_all_corrs_fft(h)
    return np.min(corrs)

def _compute_grad_from_weights(h, weights):
    """Compute gradient of sum_k w_k * R_k with respect to h, using FFT.
    weights has length n-1, corresponding to k=1,...,n-1.
    R_k = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k]
    dR_k/dh[j] = (2/n) * (h[j+k] + h[j-k]) where indices are valid.
    """
    n = len(h)
    # Build the kernel for correlation: g[k] = weights[k-1] for k=1..n-1, g[0]=0
    g = np.zeros(n)
    g[1:] = weights
    
    # The gradient is (2/n) * (h * g + flip(h) * flip(g)) but with linear (not circular) correlation
    # More precisely: grad[j] = (2/n) * sum_{k=1}^{n-1} w_k * (h[j+k] if j+k<n else 0 + h[j-k] if j-k>=0 else 0)
    # This is equivalent to (2/n) * (linear correlation of h with g, plus linear correlation of h with flip(g))
    # Use FFT for linear correlation
    size = 2 * n
    h_p = np.zeros(size)
    h_p[:n] = h
    g_p = np.zeros(size)
    g_p[:n] = g
    
    H = np.fft.rfft(h_p, n=size)
    G = np.fft.rfft(g_p, n=size)
    
    # Linear cross-correlation: ifft(H * conj(G)) gives sum_i h[i] g[i+k] for various k
    # We need sum_k w_k * (h[j+k] + h[j-k])
    # = sum_k g[k] * h[j+k] + sum_k g[k] * h[j-k]
    # First term: for each j, sum_{k=1}^{n-1} g[k] h[j+k] = (linear cross-correlation of h and g) evaluated at appropriate indices
    # Let's compute c1[j] = sum_{k} g[k] h[j+k] for j=0..n-1
    # This is the convolution/correlation. Using FFT:
    # c1 = ifft(conj(H) * G) gives circular correlation; for linear, we take first n elements after zero-padding
    
    # Actually, let me just do it directly with the structure:
    # grad[j] = (2/n) * sum_{k=1}^{min(j, n-1-j)} ... no, let's use the FFT approach properly.
    
    # c1[j] = sum_{k=1}^{n-1} g[k] * h[j+k] for j+k < n, else 0
    # c2[j] = sum_{k=1}^{n-1} g[k] * h[j-k] for j-k >= 0, else 0
    # grad = (2/n) * (c1 + c2)
    
    # c1 is the cross-correlation of h and g: c1[j] = sum_k g[k] h[j+k]
    # Using FFT: pad both to 2n, compute ifft(conj(H) * G), take [0:n]
    corr1 = np.fft.irfft(np.conj(H) * G, n=size)
    c1 = corr1[:n]
    
    # c2[j] = sum_k g[k] h[j-k] = sum_k g[k] h[(j-k)] = cross-correlation in the other direction
    # c2[j] = sum_k g[n-k] h[j+k] ... let me think differently
    # c2[j] = sum_{k=1}^{n-1} g[k] h[j-k] for j>=k
    # = sum_{m=0}^{j-1} g[j-m] h[m]  (letting m = j-k)
    # This is the convolution of g and h evaluated at j
    g_rev = g[::-1]
    g_rev_p = np.zeros(size)
    g_rev_p[:n] = g_rev
    G_rev = np.fft.rfft(g_rev_p, n=size)
    corr2 = np.fft.irfft(H * G_rev, n=size)
    c2 = corr2[:n]
    
    grad = (2.0 / n) * (c1 + c2)
    return grad

def _objective_and_grad_soft(x, beta=100.0):
    """Objective and gradient using soft-min approximation with FFT-based correlations."""
    h = x
    n = len(h)
    
    # Compute all correlations via FFT
    corrs = _compute_all_corrs_fft(h)
    
    # Clean soft-min formulation: soft_min = -(1/beta) * log(sum_k exp(-beta * corrs_k))
    min_corr = np.min(corrs)
    shifted = corrs - min_corr
    exp_neg_beta = np.exp(-beta * shifted)
    sum_exp_neg = np.sum(exp_neg_beta)
    soft_min = min_corr - np.log(sum_exp_neg) / beta
    obj = 1.0 - soft_min
    
    # d(soft_min)/d(corrs_k) = exp(-beta*corrs_k) / sum_j exp(-beta*corrs_j)
    weights = exp_neg_beta / sum_exp_neg
    
    # Efficient gradient using FFT
    grad = _compute_grad_from_weights(h, weights)
    
    grad_obj = -grad
    return obj, grad_obj

def _project_to_simplex(v, target_sum):
    """Project v onto {x : 0 <= x <= 1, sum(x) = target_sum}."""
    n = len(v)
    
    # First project onto {x >= 0, sum = target_sum}
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = np.nonzero(u * np.arange(1, n + 1) > (cssv - target_sum))[0]
    if len(rho_candidates) == 0:
        rho = n - 1
    else:
        rho = rho_candidates[-1]
    theta = (cssv[rho] - target_sum) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    
    # Clip to [0, 1]
    w = np.clip(w, 0, 1)
    
    # Adjust to ensure sum = target_sum
    for _ in range(30):  # Iterate more times for better convergence
        diff = target_sum - np.sum(w)
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = w < 1.0
            cnt = np.sum(mask)
            if cnt == 0:
                break
            w[mask] += diff / cnt
        else:
            mask = w > 0.0
            cnt = np.sum(mask)
            if cnt == 0:
                break
            w[mask] += diff / cnt
        w = np.clip(w, 0, 1)
    
    return w

def _spectral_init(n, target_sum, num_modes=5):
    """Generate spectral-based initializations by optimizing in frequency domain."""
    inits = []
    for m in range(1, num_modes + 1):
        # Try different phase offsets
        for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            x = np.linspace(0, 2*np.pi*m, n, endpoint=False)
            h = 0.5 + 0.3 * np.sin(x + phase)
            h = np.clip(h, 0, 1)
            h = _project_to_simplex(h, target_sum)
            inits.append(h)
    return inits

def construct():
    """Construct step heights minimizing maximum overlap."""
    n = N
    target_sum = n / 2.0
    
    best_x = np.full(n, 0.5)
    best_obj = 1.0 - _compute_cross_correlation_min(best_x)
    
    # Generate diverse initializations
    np.random.seed(42)
    
    restarts = []
    # Uniform
    restarts.append(np.full(n, 0.5))
    # Block structures
    for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
        k = int(n * frac)
        if k > 0 and n - k > 0:
            restarts.append(np.concatenate([np.full(k, 0.0), np.full(n-k, 2.0*target_sum/(n-k))]))
            restarts.append(np.concatenate([np.full(k, 2.0*target_sum/k), np.full(n-k, 0.0)]))
    # Alternating patterns
    restarts.append(np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)]))
    restarts.append(np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)]))
    # Random
    for _ in range(15):
        restarts.append(np.random.uniform(0, 1, n))
    # Smooth gradients
    restarts.append(np.linspace(0, 1, n))
    restarts.append(np.linspace(1, 0, n))
    # Sinusoidal
    x = np.linspace(0, 2*np.pi, n)
    restarts.append(0.5 + 0.3*np.sin(x))
    restarts.append(0.5 + 0.3*np.cos(x))
    restarts.append(0.5 + 0.4*np.sin(2*x))
    restarts.append(0.5 + 0.4*np.cos(2*x))
    # Spectral inits
    restarts.extend(_spectral_init(n, target_sum, num_modes=4))
    
    # Phase 1: Soft-min optimization with increasing beta per restart
    beta_schedule = [20.0, 50.0, 100.0, 200.0]
    
    for x0 in restarts:
        if _time_remaining() < 8.0:
            break
        
        x = _project_to_simplex(x0, target_sum)
        
        for beta in beta_schedule:
            if _time_remaining() < 4.0:
                break
            try:
                res = minimize(
                    _objective_and_grad_soft,
                    x,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * n,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                res_x = _project_to_simplex(res.x, target_sum)
                res_obj = 1.0 - _compute_cross_correlation_min(res_x)
                if res_obj < best_obj:
                    best_obj = res_obj
                    best_x = res_x.copy()
                x = res_x
            except Exception:
                pass
    
    # Phase 2: High-beta polish on best solution
    if _time_remaining() > 12.0:
        for beta in [500.0, 1000.0, 2000.0, 5000.0, 10000.0]:
            if _time_remaining() < 4.0:
                break
            try:
                res = minimize(
                    _objective_and_grad_soft,
                    best_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * n,
                    options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                res_x = _project_to_simplex(res.x, target_sum)
                res_obj = 1.0 - _compute_cross_correlation_min(res_x)
                if res_obj < best_obj:
                    best_obj = res_obj
                    best_x = res_x.copy()
            except Exception:
                pass
    
    # Phase 3: Local search with subgradient on the active constraint
    if _time_remaining() > 8.0:
        step_size = 0.01
        no_improve = 0
        for it in range(500):
            if _time_remaining() < 4.0:
                break
            h = best_x
            
            # Find the minimum correlation shift
            corrs = _compute_all_corrs_fft(h)
            k_star = np.argmin(corrs) + 1  # +1 because corrs[0] corresponds to k=1
            min_corr = corrs[k_star - 1]
            
            # Gradient for the active constraint
            grad = np.zeros(n)
            w_active = (2.0 / n)
            for j in range(n - k_star):
                grad[j] += w_active * h[j + k_star]
                grad[j + k_star] += w_active * h[j]
            
            h_new = h + step_size * grad
            h_new = _project_to_simplex(h_new, target_sum)
            
            new_obj = 1.0 - _compute_cross_correlation_min(h_new)
            if new_obj < best_obj - 1e-10:
                best_obj = new_obj
                best_x = h_new.copy()
                no_improve = 0
            else:
                no_improve += 1
                step_size *= 0.7
                if no_improve > 30:
                    break
    
    # Phase 4: Final coordinate descent polish
    if _time_remaining() > 6.0:
        for _ in range(50):
            if _time_remaining() < 3.0:
                break
            improved = False
            for j in range(n):
                if _time_remaining() < 3.0:
                    break
                # Try adjusting h[j] to minimize the max correlation
                orig = best_x[j]
                # Binary search for optimal h[j] given others fixed
                lo, hi = 0.0, 1.0
                for _ in range(20):
                    mid = (lo + hi) / 2
                    test = best_x.copy()
                    test[j] = mid
                    # Adjust others to maintain sum
                    diff = target_sum - np.sum(test)
                    if abs(diff) > 1e-15:
                        mask = np.ones(n, dtype=bool)
                        mask[j] = False
                        cnt = np.sum(mask)
                        if cnt > 0:
                            test[mask] += diff / cnt
                            test = np.clip(test, 0, 1)
                    obj_test = 1.0 - _compute_cross_correlation_min(test)
                    if obj_test < best_obj:
                        lo = mid
                        improved = True
                    else:
                        hi = mid
                # Apply the best found
                best_x[j] = (lo + hi) / 2
                diff = target_sum - np.sum(best_x)
                if abs(diff) > 1e-15:
                    mask = np.ones(n, dtype=bool)
                    cnt = np.sum(mask)
                    if cnt > 0:
                        best_x[mask] += diff / cnt
                        best_x = np.clip(best_x, 0, 1)
                best_obj = 1.0 - _compute_cross_correlation_min(best_x)
            if not improved:
                break
    
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
