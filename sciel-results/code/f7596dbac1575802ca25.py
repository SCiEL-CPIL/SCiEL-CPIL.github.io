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
    """
    Generate a variety of Sidon sets. 
    We prioritize sets with larger cardinality as they offer higher potential ratios (approx K/2).
    """
    sidon_sets = []
    
    # Known good Sidon sets for N=50
    base_sets = [
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30, 35, 40], # Size 12
        [0, 1, 4, 9, 15, 21, 27, 33, 39, 44, 48], # Size 11
        [0, 2, 5, 11, 14, 20, 26, 32, 38, 43, 47], # Size 11
        [0, 1, 3, 7, 12, 18, 25, 31, 38, 44, 49], # Size 11
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30], # Size 10
        [0, 1, 4, 9, 15, 21, 27, 33, 39], # Size 9
        [0, 1, 4, 10, 12, 17, 20, 24], # Size 8
        [0, 1, 4, 9, 15, 21, 27], # Size 7
        [0, 1, 4, 10, 12, 17], # Size 6
        [0, 1, 4, 9, 15, 21], # Size 6
        [0, 1, 4, 10, 12], # Size 5
        [0, 1, 4, 9, 15], # Size 5
        [0, 1, 4, 10], # Size 4
        [0, 1, 4, 9], # Size 4
        [0, 1, 4], # Size 3
        [0, 1, 3], # Size 3
    ]
    
    for s in base_sets:
        if all(0 <= x < N for x in s) and is_sidon_set(s):
            sidon_sets.append(list(s))
            
    # Try to find larger or different Sidon sets via random search
    np.random.seed(42)
    for _ in range(2000):
        k = np.random.randint(6, 13)
        if k > N: continue
        indices = sorted(np.random.choice(N, k, replace=False).tolist())
        if is_sidon_set(indices):
            # Check for duplicates
            found = False
            for existing in sidon_sets:
                if set(existing) == set(indices):
                    found = True
                    break
            if not found:
                sidon_sets.append(indices)
                if len(sidon_sets) > 30:
                    break
                    
    # Sort by size descending to prioritize high-potential supports
    sidon_sets.sort(key=lambda x: len(x), reverse=True)
    return sidon_sets

def optimize_weights_for_support(support, beta=100.0, x_init=None, max_iter=500):
    """
    Optimize weights on a given support to maximize the ratio:
    (sum h_i^2) / (smooth_max of convolution)
    
    We minimize the negative log-ratio or a related surrogate.
    Let R(h) = ||h||_2^2 / M_beta(h)
    where M_beta(h) = (1/beta) log sum exp(beta * conv_k)
    
    We want to maximize R. 
    Equivalent to minimizing -log(R) = log(M_beta) - log(||h||_2^2)
    
    Constraint: sum(h) = 1, h >= 0.
    We parameterize h using softmax of z to handle non-negativity and normalization automatically.
    h = softmax(z)
    """
    k = len(support)
    if k == 0:
        return np.zeros(N), 0.0
        
    if x_init is not None:
        x0 = np.asarray(x_init, dtype=np.float64)
        x0 = np.maximum(x0, 1e-15)
        x0 = x0 / np.sum(x0)
    else:
        x0 = np.ones(k) / k

    def softmax(z):
        z_shifted = z - np.max(z)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z)

    def objective_and_grad(z):
        x = softmax(z)
        h = np.zeros(N)
        h[support] = x
        
        conv = fftconvolve(h, h, mode='full')
        max_c = np.max(conv)
        if max_c < 1e-15:
            return 0.0, np.zeros(k)
            
        # Smooth max
        shifted = conv - max_c
        exp_shifted = np.exp(beta * shifted)
        sum_exp = np.sum(exp_shifted)
        log_sum_exp = max_c + np.log(sum_exp)
        M_beta = log_sum_exp / beta
        
        h2_sq = np.sum(h**2)
        if h2_sq < 1e-15:
            return 1e10, np.zeros(k)
            
        # Objective: -log(R) = log(M_beta) - log(h2_sq)
        # We want to minimize this.
        obj = np.log(M_beta) - np.log(h2_sq)
        
        # Gradient calculation
        # d M_beta / d h_i
        # M_beta = (1/beta) log sum_j exp(beta C_j)
        # d M_beta / d h_i = (1/beta) * (1/sum_j exp(beta C_j)) * sum_j exp(beta C_j) * beta * d C_j / d h_i
        # = sum_j w_j * d C_j / d h_i
        # where w_j = exp(beta C_j) / sum_k exp(beta C_k)
        
        weights = exp_shifted / sum_exp
        
        # d C_j / d h_i = 2 * h_{j-i} (if 0 <= j-i < N)
        # C_j = sum_m h_m h_{j-m}
        # d C_j / d h_i = h_{j-i} + h_{j-i} (if j-i != i) or 2 h_i (if j=2i)
        # Generally: d C_j / d h_i = h_{j-i} + h_{i-j} (with bounds)
        # Since h is real, h_{j-i} is non-zero only if j-i is in support.
        
        grad_h = np.zeros(N)
        for m in support:
            hm = h[m]
            if hm <= 1e-15:
                continue
            # Contribution to grad_h from h_m
            # d C_j / d h_m includes h_{j-m} and h_{m-j}
            # Let's compute grad_h[i] = sum_j weights[j] * (h_{j-i} + h_{i-j})
            # This is a convolution of weights with h?
            # grad_h[i] = sum_j weights[j] h_{j-i} + sum_j weights[j] h_{i-j}
            # Let u = j-i => j = u+i. Sum_u weights[u+i] h_u.
            # Let v = i-j => j = i-v. Sum_v weights[i-v] h_v.
            # These are cross-correlations.
            
            # Efficient computation:
            # grad_h = 2 * np.convolve(weights, h, mode='same')? No, indices are tricky.
            # Let's just loop over support for i, and sum over j where h_{j-i} is non-zero.
            # Since h is sparse, this is fast.
            
            # For each i in support:
            # grad_h[i] = sum_{m in support} h[m] * (weights[i+m] + weights[i-m])?
            # Let's verify.
            # d C_j / d h_i = h_{j-i} + h_{i-j} (if valid indices)
            # sum_j weights[j] (h_{j-i} + h_{i-j})
            # Term 1: sum_j weights[j] h_{j-i}. Let m = j-i => j = m+i.
            #   sum_m weights[m+i] h_m.
            # Term 2: sum_j weights[j] h_{i-j}. Let m = i-j => j = i-m.
            #   sum_m weights[i-m] h_m.
            
            # So grad_h[i] = sum_{m in support} h[m] * (weights[i+m] + weights[i-m])
            # Note: weights has length 2N-1. Indices must be valid.
            
            for m in support:
                hm = h[m]
                if hm <= 1e-15:
                    continue
                # Term 1: weights[i+m]
                idx1 = i + m
                if 0 <= idx1 < len(weights):
                    grad_h[i] += hm * weights[idx1]
                # Term 2: weights[i-m]
                idx2 = i - m
                if 0 <= idx2 < len(weights):
                    grad_h[i] += hm * weights[idx2]
        
        # Now we have grad_h for all i. We need grad_z.
        # h = softmax(z).
        # d h_i / d z_j = h_i (delta_ij - h_j)
        # d obj / d z_j = sum_i (d obj / d h_i) * (d h_i / d z_j)
        # d obj / d h_i = - (d M_beta / d h_i) / M_beta + (d h2_sq / d h_i) / h2_sq
        # d h2_sq / d h_i = 2 h_i
        # d M_beta / d h_i = grad_h[i] (calculated above)
        
        d_obj_d_h = - (grad_h / M_beta) + (2 * h / h2_sq)
        
        # Project onto simplex tangent? No, softmax handles the normalization.
        # grad_z = Hessian of log softmax * d_obj_d_h
        # grad_z_j = h_j * (d_obj_d_h_j - sum_i h_i d_obj_d_h_i)
        
        sum_hd = np.sum(h * d_obj_d_h)
        grad_z = h * (d_obj_d_h - sum_hd)
        
        return obj, grad_z

    z0 = np.log(x0)
    z0 -= np.max(z0)
    
    try:
        res = minimize(objective_and_grad, z0, jac=True, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        z_opt = res.x
        x_opt = softmax(z_opt)
        x_opt = np.maximum(x_opt, 0.0)
        x_opt = x_opt / np.sum(x_opt)
        
        h_opt = np.zeros(N)
        h_opt[support] = x_opt
        sc = compute_ratio(h_opt)
        return h_opt, sc
    except Exception:
        h_opt = np.zeros(N)
        h_opt[support] = 1.0 / k
        sc = compute_ratio(h_opt)
        return h_opt, sc

def local_search_polish(h, support, max_iter=100, tol=1e-10):
    h = h.copy()
    k = len(support)
    if k < 2:
        return h
    best_score = compute_ratio(h)
    step_scale = 0.05
    
    for it in range(max_iter):
        improved = False
        # Try pair moves
        for i in range(k):
            for j in range(i+1, k):
                for delta in [step_scale, step_scale*0.5, -step_scale, -step_scale*0.5]:
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
            step_scale *= 0.5
            if step_scale < 1e-8:
                break
    return h

def construct():
    t0 = time.time()
    deadline = t0 + 130.0
    
    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    sidon_sets = generate_sidon_sets()
    
    # Optimize on each Sidon set
    for s in sidon_sets:
        if time.time() > deadline - 20.0:
            break
        # Try different betas for smooth max
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 20.0:
                break
            h_opt, sc = optimize_weights_for_support(s, beta=beta, max_iter=300)
            if sc > best_score:
                best_score = sc
                best_h = h_opt.copy()
                
    # Final polish on the best solution
    if time.time() < deadline - 5.0:
        support = [i for i in range(N) if best_h[i] > 1e-10]
        if len(support) >= 2:
            best_h = local_search_polish(best_h, support, max_iter=200)
            best_score = compute_ratio(best_h)
            
    return list(best_h)
