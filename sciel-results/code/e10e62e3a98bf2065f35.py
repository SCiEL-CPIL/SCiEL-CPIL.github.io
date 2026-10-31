import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def local_search_polish(h, support, max_iter=200, tol=1e-10):
    """Local search: try small perturbations to improve the ratio with adaptive steps."""
    h = h.copy()
    k = len(support)
    if k < 2:
        return h
    
    best_score = compute_ratio(h)
    step_scale = 0.01
    
    for it in range(max_iter):
        improved = False
        # Try swapping weight between two indices
        for i in range(k):
            for j in range(i+1, k):
                # Try moving a small amount from i to j
                for delta in [step_scale, step_scale*0.1, -step_scale, -step_scale*0.1]:
                    h_new = h.copy()
                    hi = h[support[i]]
                    hj = h[support[j]]
                    new_hi = hi + delta
                    new_hj = hj - delta
                    if new_hi >= 0 and new_hj >= 0:
                        h_new[support[i]] = new_hi
                        h_new[support[j]] = new_hj
                        s = np.sum(h_new)
                        if s > 1e-15:
                            h_new = h_new / s
                            sc = compute_ratio(h_new)
                            if sc > best_score + tol:
                                h = h_new
                                best_score = sc
                                improved = True
                                break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try single index adjustments
            for i in range(k):
                for delta in [step_scale, step_scale*0.1, -step_scale, -step_scale*0.1]:
                    h_new = h.copy()
                    hi = h[support[i]]
                    new_hi = hi + delta
                    if new_hi >= 0:
                        h_new[support[i]] = new_hi
                        s = np.sum(h_new)
                        if s > 1e-15:
                            h_new = h_new / s
                            sc = compute_ratio(h_new)
                            if sc > best_score + tol:
                                h = h_new
                                best_score = sc
                                improved = True
                                break
                if improved:
                    break
            
            if not improved:
                # Reduce step size
                step_scale *= 0.5
                if step_scale < 1e-8:
                    break
    
    return h

def is_sidon_set(indices):
    """Check if a set of indices forms a B2 (Sidon) set: all pairwise sums are distinct."""
    indices = sorted(indices)
    sums = set()
    for i in range(len(indices)):
        for j in range(i, len(indices)):
            s = indices[i] + indices[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def generate_sidon_sets():
    """Generate several Sidon sets of various sizes within [0, N)."""
    sidon_sets = []
    base_sets = [
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30, 35, 40],
        [0, 1, 4, 9, 15, 21, 27, 33, 39, 44, 48],
        [0, 2, 5, 11, 14, 20, 26, 32, 38, 43, 47],
        [0, 1, 3, 7, 12, 18, 25, 31, 38, 44, 49],
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30],
        [0, 1, 4, 9, 15, 21, 27, 33, 39],
        [0, 1, 4, 10, 12, 17, 20, 24],
        [0, 1, 4, 9, 15, 21, 27],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 9, 15, 21],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 9, 15],
        [0, 1, 4, 10],
        [0, 1, 4, 9],
        [0, 1, 4],
        [0, 1, 3],
    ]
    
    for s in base_sets:
        if all(0 <= x < N for x in s) and is_sidon_set(s):
            sidon_sets.append(list(s))
    
    np.random.seed(123)
    for _ in range(5000):
        k = np.random.randint(4, 11)
        indices = sorted(np.random.choice(N, k, replace=False).tolist())
        if is_sidon_set(indices):
            found = False
            for existing in sidon_sets:
                if len(existing) == k and set(existing) == set(indices):
                    found = True
                    break
            if not found:
                sidon_sets.append(indices)
                if len(sidon_sets) > 50:
                    break
    
    return sidon_sets

def optimize_weights_for_support(support, beta=100.0, x_init=None):
    """
    Optimize weights h_i for i in support to minimize max convolution coefficient.
    Uses smooth max surrogate with proper gradient handling on the simplex.
    """
    k = len(support)
    if k == 0:
        return np.zeros(N), 0.0
    
    # Initial guess: uniform weights or provided initialization
    if x_init is not None:
        x0 = np.asarray(x_init, dtype=np.float64)
        x0 = np.maximum(x0, 0.0)
        s = np.sum(x0)
        if s > 1e-15:
            x0 = x0 / s
        else:
            x0 = np.ones(k) / k
    else:
        x0 = np.ones(k) / k
    
    def objective_and_grad(x):
        """Compute smooth max of convolution and its gradient w.r.t x (on simplex)."""
        h = np.zeros(N)
        h[support] = x
        s = np.sum(h)
        if s < 1e-15:
            return 1e10, np.zeros(k)
        h = h / s
        conv = fftconvolve(h, h, mode='full')
        
        # Smooth max using log-sum-exp
        max_c = np.max(conv)
        if max_c < 1e-15:
            return 0.0, np.zeros(k)
        
        shifted = conv - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        log_sum_exp = max_c + np.log(sum_exp)
        obj = log_sum_exp / beta
        
        # Gradient: d/dx_i [1/beta * log(sum_k exp(beta * conv_k))]
        # = sum_k weights_k * d conv_k / d x_i
        # conv_k = sum_{i,j} h_i h_j * [i+j==k] (approximately, with normalization)
        # d conv_k / d h_j = 2 * h_{k-j} (with h_m = 0 if m out of bounds)
        # So d conv_k / d x_i = 2 * h_{k-support[i]} (for the unnormalized h)
        # But we have normalization h = x/s, so we need chain rule.
        # For simplicity, let's compute gradient numerically or use the fact that
        # on the simplex, we can project the gradient.
        
        # Compute grad_h[j] = 2 * sum_k weights[k] * h[k-j]
        grad_h = np.zeros(N)
        for m in support:
            hm = h[m]
            if hm > 1e-15:
                # weights[m+j] for j in 0..N-1
                # This is weights[m : m+N]
                if m + N <= len(weights):
                    grad_h[:N] += 2.0 * hm * weights[m : m+N]
        
        # Project gradient onto simplex: subtract mean to ensure sum = 0
        grad_x = grad_h[support]
        grad_x -= np.mean(grad_x)
        
        return obj, grad_x
    
    # Use L-BFGS-B with simplex projection
    # We'll parameterize as x = softmax(z) to ensure x >= 0 and sum(x) = 1
    def softmax(z):
        z_shifted = z - np.max(z)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z)
    
    def objective_z(z):
        x = softmax(z)
        obj, _ = objective_and_grad(x)
        return obj
    
    def grad_z(z):
        x = softmax(z)
        obj, grad_x = objective_and_grad(x)
        # d softmax(z)_i / dz_j = x_i * (delta_ij - x_j)
        # So grad_z_j = sum_i grad_x_i * x_i * (delta_ij - x_j)
        # = x_j * grad_x_j - x_j * sum_i grad_x_i * x_i
        sum_grad_x_x = np.sum(grad_x * x)
        grad_z = x * (grad_x - sum_grad_x_x)
        return grad_z
    
    # Optimize in z-space (unconstrained)
    # Find z0 such that softmax(z0) = x0
    z0 = np.log(np.maximum(x0, 1e-15))
    z0 -= np.max(z0)  # for numerical stability
    try:
        res = minimize(objective_z, z0, jac=grad_z, method='L-BFGS-B',
                      options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        z_opt = res.x
        x_opt = softmax(z_opt)
        x_opt = np.maximum(x_opt, 0.0)
        x_opt = x_opt / np.sum(x_opt)
        
        h_opt = np.zeros(N)
        h_opt[support] = x_opt
        
        # Local search polish
        h_opt = local_search_polish(h_opt, support, max_iter=100)
        
        sc = compute_ratio(h_opt)
        return h_opt, sc
    except Exception as e:
        # Fallback to uniform
        h_opt = np.zeros(N)
        h_opt[support] = 1.0 / k
        h_opt = local_search_polish(h_opt, support, max_iter=50)
        sc = compute_ratio(h_opt)
        return h_opt, sc

def gradient_polish(h, support, max_iter=200, tol=1e-12, beta=500.0):
    """
    High-precision gradient-based polish using L-BFGS-B on smoothed objective.
    Uses softmax parameterization to stay on the simplex.
    """
    k = len(support)
    if k < 2:
        return h
    
    x = h[support].copy()
    x = np.maximum(x, 0.0)
    s = np.sum(x)
    if s < 1e-15:
        return h
    x = x / s
    
    def softmax(z):
        z_shifted = z - np.max(z)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z)
    
    def objective_and_grad_z(z):
        x = softmax(z)
        h_full = np.zeros(N)
        h_full[support] = x
        conv = fftconvolve(h_full, h_full, mode='full')
        max_c = np.max(conv)
        if max_c < 1e-15:
            return 0.0, np.zeros(k)
        shifted = conv - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        log_sum_exp = max_c + np.log(sum_exp)
        obj = log_sum_exp / beta
        
        # Gradient w.r.t. x
        grad_x = np.zeros(k)
        for idx, m in enumerate(support):
            if m + N <= len(weights):
                grad_x[idx] = 2.0 * x[idx] * np.sum(weights[m:m+N])
            else:
                # Handle boundary
                valid_end = min(m + N, len(weights))
                if m < len(weights):
                    grad_x[idx] = 2.0 * x[idx] * np.sum(weights[m:valid_end])
        
        # Project to simplex: subtract mean
        grad_x -= np.mean(grad_x)
        
        # Chain rule through softmax
        sum_grad_x_x = np.sum(grad_x * x)
        grad_z = x * (grad_x - sum_grad_x_x)
        
        return obj, grad_z
    
    z0 = np.log(np.maximum(x, 1e-15))
    z0 -= np.max(z0)
    
    try:
        res = minimize(objective_and_grad_z, z0, jac=None, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10})
        z_opt = res.x
        x_opt = softmax(z_opt)
        x_opt = np.maximum(x_opt, 0.0)
        x_opt = x_opt / np.sum(x_opt)
        
        h_new = h.copy()
        h_new[support] = x_opt
        return h_new
    except Exception:
        return h

def construct():
    t0 = time.time()
    deadline = t0 + 135.0  # Safety margin

    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    sidon_sets = generate_sidon_sets()
    
    # Sort by size descending - larger sets have more room for optimization
    sidon_sets.sort(key=lambda s: len(s), reverse=True)
    
    # Try optimizing weights for each Sidon set
    for s in sidon_sets:
        if time.time() > deadline - 15.0:
            break
        
        # Try multiple beta values
        for beta in [30.0, 50.0, 100.0, 200.0]:
            if time.time() > deadline - 15.0:
                break
            
            h_opt, sc = optimize_weights_for_support(s, beta=beta)
            if sc > best_score:
                best_score = sc
                best_h = h_opt.copy()
        
        # Also try with a few random initializations
        if time.time() > deadline - 15.0:
            break
        
        np.random.seed(hash(tuple(s)) % 2**32)
        for _ in range(3):
            if time.time() > deadline - 15.0:
                break
            k = len(s)
            x_init = np.random.dirichlet(np.ones(k))
            h_opt, sc = optimize_weights_for_support(s, beta=100.0, x_init=x_init)
            if sc > best_score:
                best_score = sc
                best_h = h_opt.copy()

    # Final polish on the best solution
    if time.time() < deadline - 5.0:
        # Find support of best_h
        support = [i for i in range(N) if best_h[i] > 1e-10]
        if len(support) >= 2:
            # Gradient-based high-precision polish
            best_h = gradient_polish(best_h, support, max_iter=300, tol=1e-14, beta=500.0)
            best_score = compute_ratio(best_h)
            
            # Additional local search polish
            if time.time() < deadline - 3.0:
                best_h = local_search_polish(best_h, support, max_iter=300, tol=1e-13)
                best_score = compute_ratio(best_h)

    return list(best_h)
