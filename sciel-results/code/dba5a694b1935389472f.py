import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.sparse import csr_matrix

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
    h = h.copy()
    k = len(support)
    if k < 2:
        return h
    best_score = compute_ratio(h)
    step_scale = 0.01
    for it in range(max_iter):
        improved = False
        for i in range(k):
            for j in range(i+1, k):
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
                step_scale *= 0.5
                if step_scale < 1e-8:
                    break
    return h

def is_sidon_set(indices):
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

def softmax(z):
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)

def optimize_weights_for_support(support, beta=100.0, x_init=None):
    k = len(support)
    if k == 0:
        return np.zeros(N), 0.0
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
        h = np.zeros(N)
        h[support] = x
        s = np.sum(h)
        if s < 1e-15:
            return 1e10, np.zeros(k)
        h = h / s
        conv = fftconvolve(h, h, mode='full')
        max_c = np.max(conv)
        if max_c < 1e-15:
            return 0.0, np.zeros(k)
        shifted = conv - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        log_sum_exp = max_c + np.log(sum_exp)
        obj = log_sum_exp / beta
        grad_h = np.zeros(N)
        for m in support:
            hm = h[m]
            if hm > 1e-15:
                if m + N <= len(weights):
                    grad_h[:N] += 2.0 * hm * weights[m : m+N]
        grad_x = grad_h[support]
        grad_x -= np.mean(grad_x)
        return obj, grad_x

    def objective_z(z):
        x = softmax(z)
        obj, _ = objective_and_grad(x)
        return obj

    def grad_z(z):
        x = softmax(z)
        obj, grad_x = objective_and_grad(x)
        sum_grad_x_x = np.sum(grad_x * x)
        grad_z = x * (grad_x - sum_grad_x_x)
        return grad_z

    z0 = np.log(np.maximum(x0, 1e-15))
    z0 -= np.max(z0)
    try:
        res = minimize(objective_z, z0, jac=grad_z, method='L-BFGS-B',
                      options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        z_opt = res.x
        x_opt = softmax(z_opt)
        x_opt = np.maximum(x_opt, 0.0)
        x_opt = x_opt / np.sum(x_opt)
        h_opt = np.zeros(N)
        h_opt[support] = x_opt
        h_opt = local_search_polish(h_opt, support, max_iter=100)
        sc = compute_ratio(h_opt)
        return h_opt, sc
    except Exception:
        h_opt = np.zeros(N)
        h_opt[support] = 1.0 / k
        h_opt = local_search_polish(h_opt, support, max_iter=50)
        sc = compute_ratio(h_opt)
        return h_opt, sc

def gradient_polish(h, support, max_iter=200, tol=1e-12, beta=500.0):
    k = len(support)
    if k < 2:
        return h
    x = h[support].copy()
    x = np.maximum(x, 0.0)
    s = np.sum(x)
    if s < 1e-15:
        return h
    x = x / s

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
        grad_x = np.zeros(k)
        for idx, m in enumerate(support):
            if m + N <= len(weights):
                grad_x[idx] = 2.0 * x[idx] * np.sum(weights[m:m+N])
            else:
                valid_end = min(m + N, len(weights))
                if m < len(weights):
                    grad_x[idx] = 2.0 * x[idx] * np.sum(weights[m:valid_end])
        grad_x -= np.mean(grad_x)
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

def refine_candidate(h, deadline):
    """Full refinement pipeline for a candidate: support selection + weight optimization + polish."""
    h = np.maximum(h, 0.0)
    s = np.sum(h)
    if s < 1e-15:
        return h, 0.0
    h = h / s
    # Select support: keep indices with meaningful mass
    support = [i for i in range(N) if h[i] > 1e-8]
    if len(support) < 2:
        support = list(np.argsort(h)[::-1][:5])
    # Cap support size for optimization tractability
    if len(support) > 12:
        support = sorted(support[:12])
    # Optimize weights on this support
    h_opt, sc = optimize_weights_for_support(support, beta=100.0)
    if sc > 0:
        # High-precision gradient polish
        if time.time() < deadline - 8.0:
            h_opt = gradient_polish(h_opt, support, max_iter=300, tol=1e-14, beta=500.0)
            sc = compute_ratio(h_opt)
        # Local search polish
        if time.time() < deadline - 5.0:
            h_opt = local_search_polish(h_opt, support, max_iter=300, tol=1e-13)
            sc = compute_ratio(h_opt)
    return h_opt, sc

def global_differential_search(deadline, seed=42):
    """
    Global search using Differential Evolution over a dense vector of length N.
    The vector v encodes both support (via sign/threshold) and weights (via magnitude).
    We maximize the ratio by minimizing the smooth-max surrogate of the convolution.
    """
    from scipy.optimize import differential_evolution
    np.random.seed(seed)
    
    # Search space: v in [-1, 1]^N. We map v to h via softplus + thresholding.
    # h_i = max(0, v_i) if we want nonneg, but to allow support selection we use:
    # h_i = exp(v_i) for all i (all positive), then the optimizer will push some to ~0.
    # Actually, to get sparsity, use: h_i = max(0, v_i) ** 2 (squared to emphasize large values)
    # Let's use: h_i = (max(0, v_i))^2, which is nonnegative and sparsifying.
    
    def map_v_to_h(v):
        # v in [-1, 1]^N
        # Use squared positive part to encourage sparsity
        h = np.maximum(v, 0.0) ** 2
        s = np.sum(h)
        if s < 1e-15:
            return np.ones(N) / N
        return h / s
    
    def objective(v):
        h = map_v_to_h(v)
        conv = fftconvolve(h, h, mode='full')
        max_c = np.max(conv)
        if max_c < 1e-15:
            return 0.0
        # Smooth max surrogate (lower is better)
        beta = 50.0
        shifted = conv - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        log_sum_exp = max_c + np.log(sum_exp)
        # We want to MINIMIZE the convolution peak, but MAXIMIZE h2_sq.
        # Ratio = h2_sq / conv_inf. To maximize ratio, minimize conv_inf / h2_sq.
        # Or: minimize (conv_inf - h2_sq * t) for some t. 
        # Simpler: minimize conv_inf (since h2_sq is also in play, but let's focus on conv_inf)
        # Actually, the ratio is h2_sq / conv_inf. Let's minimize -log(h2_sq / conv_inf) = log(conv_inf) - log(h2_sq)
        h2_sq = np.sum(h**2)
        if h2_sq < 1e-15:
            return 1e10
        return np.log(max_c) - np.log(h2_sq)
    
    bounds = [(-1.0, 1.0)] * N
    try:
        result = differential_evolution(
            objective, bounds,
            seed=seed,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            polish=False,
            update_interval=10,
            workers=1
        )
        if result.success or result.fun < 1e10:
            v_opt = result.x
            h_opt = map_v_to_h(v_opt)
            return h_opt
    except Exception:
        pass
    return None

def global_random_support_search(deadline, seed=99):
    """
    Global search: randomly select supports (not just Sidon), then optimize weights.
    This explores a wider space of support structures.
    """
    np.random.seed(seed)
    best_h = None
    best_score = -1.0
    
    for trial in range(200):
        if time.time() > deadline - 20.0:
            break
        # Random support size
        k = np.random.randint(3, 13)
        support = sorted(np.random.choice(N, k, replace=False).tolist())
        # Random initial weights
        x_init = np.random.dirichlet(np.ones(k) * np.random.uniform(0.5, 3.0))
        h_opt, sc = optimize_weights_for_support(support, beta=100.0, x_init=x_init)
        if sc > best_score:
            best_score = sc
            best_h = h_opt.copy()
    
    return best_h

def admm_optimize(h_init, deadline, max_iter=500, rho=10.0):
    """
    ADMM optimization for the ratio h2_sq / conv_inf.
    
    We parameterize h_i = |z_i|^2 where z is a complex vector.
    The autoconvolution peak becomes a quadratic form in z.
    
    We minimize the smooth-max surrogate of the convolution peak
    subject to L1 normalization and non-negativity (automatic via |z|^2).
    """
    h = np.maximum(np.asarray(h_init, dtype=np.float64), 0.0)
    s = np.sum(h)
    if s < 1e-15:
        return h
    
    # Initialize z such that |z|^2 = h
    z = np.sqrt(h) * (1.0 + 1j * 0.0)
    
    # Build the convolution matrix structure
    # conv[m] = sum_i h[i] * h[m-i] = sum_i |z_i|^2 * |z_{m-i}|^2
    # This is a 4-index tensor contraction, but we can use FFT for efficiency
    
    # ADMM variables:
    # z: complex vector (the main variable)
    # u: auxiliary for L1 constraint (u = sum |z|^2 = 1)
    # v: auxiliary for peak constraint (v >= conv peak)
    
    # We'll use a simplified ADMM:
    # Minimize f(z) + g(u) + h(v)
    # subject to: u = sum(|z|^2), v >= max(conv(z))
    
    # For tractability, we use a penalty method with ADMM-style updates:
    # 1. Update z to minimize: smooth_max_conv(z) + rho/2 * (sum(|z|^2) - 1)^2
    # 2. Project: enforce sum(|z|^2) = 1
    
    best_h = h / s
    best_score = compute_ratio(best_h)
    
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        
        # Compute current convolution
        h_curr = np.abs(z)**2
        s_curr = np.sum(h_curr)
        if s_curr < 1e-15:
            break
        h_curr = h_curr / s_curr
        
        conv = fftconvolve(h_curr, h_curr, mode='full')
        max_c = np.max(conv)
        
        # Gradient of smooth max with respect to h
        beta = 100.0
        shifted = conv - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        weights = exp_shifted / sum_exp
        
        # Gradient with respect to z: d/dz_i (sum_m conv[m] * w[m])
        # conv[m] = sum_i h[i]*h[m-i], so d conv[m]/d h[i] = h[m-i] + h[i] (for valid indices)
        # d h[i]/d z[i] = 2 * Re(z[i]) for real part, 2 * Im(z[i]) for imag part
        # But since h = |z|^2, we work in real space and then map back
        
        # Gradient of objective w.r.t. h (before normalization)
        grad_h = np.zeros(N)
        for i in range(N):
            # d/dh_i sum_m w[m] * conv[m]
            # conv[m] contains h[i]*h[m-i] when 0 <= m-i < N
            for j in range(N):
                m = i + j
                if 0 <= m < 2*N - 1:
                    grad_h[i] += weights[m] * h_curr[j]
        
        # Account for normalization: h_curr = h / sum(h)
        # Gradient of f(h/sum(h)) w.r.t. h:
        # = (grad_f(h) * sum(h) - f(h) * ones) / sum(h)^2
        # For simplicity, just use grad_h as direction
        
        # Update z in direction that increases |z|^2 where grad_h is small
        # (we want to minimize conv peak, so decrease h where it contributes to peak)
        # Actually, we want to minimize the smooth max, so move h in direction -grad_h
        
        # Step size
        step = 0.01 / (1.0 + it * 0.01)
        
        # Update h in real space
        h_new = h_curr - step * grad_h
        h_new = np.maximum(h_new, 0.0)
        s_new = np.sum(h_new)
        if s_new < 1e-15:
            break
        h_new = h_new / s_new
        
        # Update z: z_new = sqrt(h_new) * phase(z)
        # Preserve the phase of z
        phase = z / (np.abs(z) + 1e-15)
        z_new = np.sqrt(h_new) * phase
        z = z_new
        
        sc = compute_ratio(h_new)
        if sc > best_score:
            best_score = sc
            best_h = h_new.copy()
    
    return best_h


def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # Safety margin
    
    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    # Phase 1: Sidon set optimization (proven baseline)
    sidon_sets = generate_sidon_sets()
    sidon_sets.sort(key=lambda s: len(s), reverse=True)
    
    for s in sidon_sets[:15]:  # Limit to top 15 for time
        if time.time() > deadline - 40.0:
            break
        for beta in [50.0, 100.0]:
            if time.time() > deadline - 40.0:
                break
            h_opt, sc = optimize_weights_for_support(s, beta=beta)
            if sc > best_score:
                best_score = sc
                best_h = h_opt.copy()
    
    # Phase 2: ADMM-based optimization from current best
    if time.time() < deadline - 35.0:
        h_admm = admm_optimize(best_h, deadline, max_iter=300, rho=10.0)
        sc_admm = compute_ratio(h_admm)
        if sc_admm > best_score:
            best_score = sc_admm
            best_h = h_admm.copy()
        # Polish the ADMM result
        if time.time() < deadline - 25.0:
            support = [i for i in range(N) if best_h[i] > 1e-10]
            if len(support) >= 2:
                best_h = gradient_polish(best_h, support, max_iter=200, tol=1e-14, beta=500.0)
                best_score = compute_ratio(best_h)
    
    # Phase 3: Global Differential Evolution search
    if time.time() < deadline - 25.0:
        h_global = global_differential_search(deadline, seed=42)
        if h_global is not None:
            h_refined, sc = refine_candidate(h_global, deadline)
            if sc > best_score:
                best_score = sc
                best_h = h_refined.copy()
    
    # Phase 4: Global random support search
    if time.time() < deadline - 20.0:
        h_random = global_random_support_search(deadline, seed=99)
        if h_random is not None:
            h_refined, sc = refine_candidate(h_random, deadline)
            if sc > best_score:
                best_score = sc
                best_h = h_refined.copy()
    
    # Phase 5: Multiple random restarts from different seeds
    if time.time() < deadline - 15.0:
        for seed in [7, 13, 21, 33, 47]:
            if time.time() > deadline - 12.0:
                break
            h_global = global_differential_search(deadline, seed=seed)
            if h_global is not None:
                h_refined, sc = refine_candidate(h_global, deadline)
                if sc > best_score:
                    best_score = sc
                    best_h = h_refined.copy()
    
    # Phase 6: ADMM from multiple initializations
    if time.time() < deadline - 10.0:
        np.random.seed(12345)
        for trial in range(5):
            if time.time() > deadline - 8.0:
                break
            # Random initial condition
            h_rand = np.random.dirichlet(np.ones(N) * 0.5)
            h_admm = admm_optimize(h_rand, deadline, max_iter=200, rho=5.0)
            sc_admm = compute_ratio(h_admm)
            if sc_admm > best_score:
                best_score = sc_admm
                best_h = h_admm.copy()
    
    # Final high-precision polish on the best solution
    if time.time() < deadline - 5.0:
        support = [i for i in range(N) if best_h[i] > 1e-10]
        if len(support) >= 2:
            best_h = gradient_polish(best_h, support, max_iter=300, tol=1e-14, beta=500.0)
            best_score = compute_ratio(best_h)
            if time.time() < deadline - 3.0:
                best_h = local_search_polish(best_h, support, max_iter=300, tol=1e-13)
                best_score = compute_ratio(best_h)
    
    return list(best_h)
