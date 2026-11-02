import time
import numpy as np
from scipy.optimize import minimize

N = 600

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    return max_conv / (s * s)

def smooth_max_surrogate(conv, beta=100.0):
    """
    Smooth approximation of max(conv) using log-sum-exp.
    Returns: (1/beta) * log(sum(exp(beta * conv)))
    """
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    # Clip to avoid overflow in exp
    shifted = np.clip(shifted, -100, 100)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return (max_val + log_sum / beta)

def objective(params, beta=100.0):
    """
    Objective function for optimization.
    params: flattened array of h values (non-negative enforced via projection)
    """
    h = np.maximum(params, 0.0)
    s = np.sum(h)
    if s < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    smoothed_peak = smooth_max_surrogate(conv, beta)
    return smoothed_peak / (s * s)

def objective_grad(params, beta=100.0):
    """
    Gradient of the objective function.
    Note: This is an approximation. The gradient of the smooth max is weighted by softmax.
    """
    h = np.maximum(params, 0.0)
    s = np.sum(h)
    if s < 1e-10:
        return np.zeros_like(params)
    
    conv = np.convolve(h, h, mode='full')
    
    # Compute softmax weights for the max surrogate
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -100, 100)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    
    # d(smooth_max)/d(conv[k]) = weights[k]
    # conv = h * h (full mode)
    # d(conv)/d(h) is a 2N-1 x N matrix. 
    # Let's compute grad_h = (d conv/dh).T @ weights / (s*s) - 2*smooth_max/s^3 * sum_h_gradient
    
    # d(conv[k])/d(h[i]) = h[k-i] + h[i-k] (with appropriate bounds)
    # This is equivalent to:
    # grad_conv_h = np.convolve(h, weights, mode='valid') + np.convolve(h, weights[::-1], mode='valid')
    # Wait, let's derive properly.
    # conv[k] = sum_j h[j] h[k-j]
    # d conv[k] / d h[i] = h[k-i] + h[i-k] (if indices valid)
    # grad_h[i] = sum_k weights[k] * (h[k-i] + h[i-k])
    
    # This is a convolution of weights with h.
    # Specifically, grad_h = convolve(weights, h, mode='valid') + convolve(weights, h, mode='valid')? No.
    # Let's use FFT for efficiency if needed, but N=600 is small.
    # grad_h = np.convolve(weights, h, mode='valid') + np.convolve(weights[::-1], h, mode='valid')
    # Actually, let's just compute it via matrix multiplication or direct convolution logic.
    # For N=600, a simple loop or vectorized convolution is fine.
    
    # grad_h[i] = sum_k weights[k] * d(conv[k])/d(h[i])
    # d(conv[k])/d(h[i]) = h[k-i] if k>=i else 0  (term 1: h[j] where j=k-i, so h[k-i])
    #                    + h[i-k] if k<=i else 0  (term 2: h[k-j] where j=i, so h[k-i]... wait)
    
    # Let's check indices.
    # conv[k] = sum_{j=0}^{N-1} h[j] h[k-j]
    # Term 1: j such that h[j] is the variable h[i]. So j=i. Term is h[i] h[k-i].
    #   Contribution to d(conv[k])/d(h[i]) is h[k-i] if 0 <= k-i < N.
    # Term 2: k-j such that h[k-j] is the variable h[i]. So k-j=i => j=k-i. Term is h[k-i] h[i].
    #   Contribution to d(conv[k])/d(h[i]) is h[k-i] if 0 <= k-i < N.
    
    # So d(conv[k])/d(h[i]) = h[k-i] + h[k-i] = 2*h[k-i] if 0 <= k-i < N?
    # No, if i != k-i, they are distinct terms in the sum.
    # If i == k-i, i.e., k=2i, the term is h[i]^2, derivative is 2*h[i].
    # So generally, d(conv[k])/d(h[i]) = h[k-i] + h[i-k] is not quite right because h is indexed 0..N-1.
    # Let's stick to:
    # d(conv[k])/d(h[i]) = (h[k-i] if 0 <= k-i < N else 0) + (h[i-k] if 0 <= i-k < N else 0)
    # But h is only defined for 0..N-1.
    # Note that h[k-i] is non-zero only if 0 <= k-i < N.
    # h[i-k] is non-zero only if 0 <= i-k < N.
    
    # So grad_h[i] = sum_k weights[k] * ( (h[k-i] if 0<=k-i<N else 0) + (h[i-k] if 0<=i-k<N else 0) )
    
    # This can be computed as:
    # grad_h = convolve(weights, h, mode='valid') + convolve(weights, h, mode='valid')?
    # Let's test with small N.
    # Actually, it's simpler:
    # grad_h = np.convolve(weights, h, mode='valid') + np.convolve(weights, h, mode='valid') is wrong.
    
    # Let's use the property that grad_h is the gradient of the correlation.
    # grad_h = np.convolve(weights, h, mode='valid') + np.convolve(weights, h, mode='valid')
    # Wait, np.convolve(a, b, 'valid') gives output length max(len(a), len(b)) - min(len(a), len(b)) + 1.
    # Here len(weights) = 2N-1, len(h) = N.
    # 'valid' will give length N.
    
    # Let's try:
    # part1 = np.convolve(weights, h, mode='valid')
    # part2 = np.convolve(weights, h, mode='valid')
    # This doesn't seem right.
    
    # Let's just do it explicitly with a loop or vectorized slicing, N=600 is small enough.
    grad_h = np.zeros(N)
    for i in range(N):
        # Sum over k where k-i is in [0, N-1]
        # k ranges from i to i+N-1
        # But k is in [0, 2N-2]
        # So k_min = max(0, i), k_max = min(2N-2, i+N-1)
        k_min = i
        k_max = i + N - 1
        if k_max >= 2*N - 1:
            k_max = 2*N - 2
        if k_min <= k_max:
            # Term 1: h[k-i]
            # Indices of h: 0 to N-1
            # k-i ranges from 0 to N-1
            term1 = np.sum(weights[k_min:k_max+1] * h[0:k_max-k_min+1])
            
            # Term 2: h[i-k]
            # i-k ranges from 0 to N-1 => k ranges from i-(N-1) to i
            # k_min2 = max(0, i-(N-1)), k_max2 = min(2N-2, i)
            k_min2 = max(0, i - (N - 1))
            k_max2 = min(2*N - 2, i)
            if k_min2 <= k_max2:
                # i-k ranges from i-k_max2 to i-k_min2
                # i-k_max2 = i - min(2N-2, i). If i <= 2N-2, then i-i=0.
                # i-k_min2 = i - max(0, i-N+1).
                # Let's just sum directly for clarity and safety, N=600, 600*1200 ops is trivial.
                term2 = 0.0
                for k in range(k_min2, k_max2 + 1):
                    idx_h = i - k
                    if 0 <= idx_h < N:
                        term2 += weights[k] * h[idx_h]
            else:
                term2 = 0.0
            
            grad_h[i] = term1 + term2
        else:
            grad_h[i] = 0.0
    
    # Gradient of objective: f = smooth_max / s^2
    # df/dh = (1/s^2) * d(smooth_max)/dh - 2*smooth_max/s^3 * s
    # d(smooth_max)/dh = grad_h
    # s = sum(h)
    # d(s)/dh = 1
    # So df/dh = grad_h / s^2 - 2 * smooth_max / s^3 * 1
    # Wait, s is a scalar.
    # f = A / s^2
    # df = (dA * s^2 - A * 2s * ds) / s^4 = dA/s^2 - 2A ds / s^3
    # dA = grad_h . dh
    # ds = sum(dh)
    # So component i: grad_h[i]/s^2 - 2*smooth_max/s^3
    
    smooth_max_val = smooth_max_surrogate(conv, beta)
    grad_obj = grad_h / (s * s) - 2 * smooth_max_val / (s * s * s)
    
    # Apply non-negativity constraint gradient (if h[i] < 0, grad is 0, but we project)
    # Since we use np.maximum(params, 0), the gradient is only defined for params > 0.
    # For L-BFGS-B, we should return the gradient of the smooth function.
    # The projection makes the function non-smooth at 0.
    # We will return the gradient computed for h = max(params, 0).
    # If params[i] < 0, the true gradient of the projected function is 0 (if we consider the projection active).
    # But L-BFGS-B handles bounds. It's better to optimize without bounds if we use softplus or similar,
    # or just return the gradient and let L-BFGS-B handle the bounds if we provide them.
    # However, the objective function itself projects.
    # If we use bounds=[(0, inf)], L-BFGS-B will stay in the positive domain.
    # In that case, params are always >= 0, so max(params, 0) is identity.
    # So we can just return grad_obj.
    
    return grad_obj

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_h = np.ones(N) / N
    best_score = compute_score(best_h)

    np.random.seed(42)
    
    # Multiple initializations
    initializations = []
    
    # 1. Uniform
    initializations.append(np.ones(N) / N)
    
    # 2. Gaussian
    x = np.linspace(-1, 1, N)
    gauss = np.exp(-x**2 / (2 * 0.5**2))
    gauss /= np.sum(gauss)
    initializations.append(gauss)
    
    # 3. Linear ramp
    lin = np.linspace(0.1, 1.0, N)
    lin /= np.sum(lin)
    initializations.append(lin)
    
    # 4. Sine
    sine = np.sin(np.pi * (x + 1) / 2) ** 2
    sine /= np.sum(sine)
    initializations.append(sine)
    
    # 5. Step function (left half vs right half)
    step = np.zeros(N)
    step[:N//2] = 1.0
    step /= np.sum(step)
    initializations.append(step)
    
    # 6. Random
    rand_h = np.random.rand(N)
    rand_h /= np.sum(rand_h)
    initializations.append(rand_h)

    beta = 100.0
    
    for h0 in initializations:
        if time.time() > deadline - 20:
            break
            
        # Optimize
        try:
            res = minimize(
                objective, 
                h0, 
                method='L-BFGS-B', 
                jac=objective_grad, 
                args=(beta,),
                bounds=[(0, None)] * N,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0)
            
            score_opt = compute_score(h_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass

    # Final polish with coordinate descent if time permits
    if time.time() < deadline - 5:
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            idx = np.random.randint(0, N)
            target = np.random.randint(0, N)
            if idx == target:
                continue
            delta = best_h[idx] * 0.01
            h_pert = best_h.copy()
            h_pert[idx] -= delta
            h_pert[target] += delta
            if np.min(h_pert) < 0:
                continue
            score_pert = compute_score(h_pert)
            if score_pert < best_score:
                best_score = score_pert
                best_h = h_pert

    best_h = np.maximum(best_h, 0)
    return best_h.tolist()
# EVOLVE-BLOCK-END
