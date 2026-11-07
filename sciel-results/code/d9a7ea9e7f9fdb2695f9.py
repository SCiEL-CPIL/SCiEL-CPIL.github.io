"""
Strategy: Analytic Gradient Descent with Multi-Start and Momentum.

Rationale:
1. The current approach uses L-BFGS-B with finite differences, which is slow (50 evaluations per gradient step) and prone to numerical noise.
2. The objective function V(h) = l2 / (norm_1 * norm_inf) is differentiable almost everywhere (except where norm_inf has multiple maximizers or is zero).
3. We can derive the analytic gradient dV/dh.
   Let g = h * h (convolution).
   l2 = (w/3) * sum_{i=0}^{m} (y_i^2 + y_i y_{i+1} + y_{i+1}^2).
   Note y_i = g_{i-1} for i in 1..m, y_0=0, y_{m+1}=0.
   So l2 depends quadratically on g.
   norm_1 = sum(g) / (m+1).
   norm_inf = max(g).
   
   Let F = l2, G = norm_1 * norm_inf. V = F/G.
   dV = (G * dF - F * dG) / G^2.
   
   dF/dg: 
   l2 = (w/3) * [ y_0^2 + y_0 y_1 + y_1^2 + y_1^2 + y_1 y_2 + y_2^2 + ... ]
   Actually, let's expand the sum carefully.
   Sum_{i=0}^{m} (y_i^2 + y_i y_{i+1} + y_{i+1}^2).
   Term for y_k^2 appears in i=k (as y_i^2) and i=k-1 (as y_{i+1}^2).
   So coeff of y_k^2 is 2 (except boundaries).
   Term y_i y_{i+1} appears once.
   
   Let's compute dF/dg_j.
   g_j corresponds to y_{j+1}.
   y_{j+1} appears in:
   i=j: y_j^2 + y_j y_{j+1} + y_{j+1}^2 -> deriv wrt y_{j+1}: y_j + 2 y_{j+1}
   i=j-1: y_{j-1}^2 + y_{j-1} y_j + y_j^2 -> Wait, y_{j+1} is not here.
   i=j+1: y_{j+1}^2 + y_{j+1} y_{j+2} + y_{j+2}^2 -> deriv wrt y_{j+1}: 2 y_{j+1} + y_{j+2}
   
   So dF/dy_{j+1} = (y_j + 2 y_{j+1}) + (2 y_{j+1} + y_{j+2}) = y_j + 4 y_{j+1} + y_{j+2}.
   (With y_0=0, y_{m+1}=0).
   
   dF/dh_k = sum_j (dF/dg_j) * (dg_j/dh_k).
   g_j = sum_i h_i h_{j-i}.
   dg_j/dh_k = h_{j-k} + (if j-k != k) h_{j-k} ... 
   Actually, g_j = sum_{i=0}^{N-1} h_i h_{j-i}.
   If we differentiate wrt h_k:
   Terms involving h_k:
   i=k: h_k h_{j-k}. Deriv: h_{j-k} + h_k * 0 (if j-k != k) ... wait.
   If j-k != k, the term is h_k h_{j-k}. Deriv is h_{j-k}.
   Also term i=j-k: h_{j-k} h_k. Deriv is h_k.
   So if j-k != k, dg_j/dh_k = 2 h_{j-k}.
   If j-k == k (i.e., j=2k), the term is h_k^2. Deriv is 2 h_k.
   In both cases, if indices are valid (0 <= j-k < N), dg_j/dh_k = 2 h_{j-k}.
   If j-k is out of bounds, it's 0.
   
   So dF/dh_k = 2 * sum_j (y_j + 4 y_{j+1} + y_{j+2}) * h_{j-k}  (with appropriate index shifts for y).
   Let A_j = y_j + 4 y_{j+1} + y_{j+2}.
   dF/dh_k = 2 * sum_j A_j h_{j-k}.
   This is a convolution of A and h.
   
   dG/dh_k:
   G = norm_1 * norm_inf = (sum(g)/(m+1)) * max(g).
   Let S = sum(g), M = max(g). G = S*M/(m+1).
   dG = (dS*M + S*dM) / (m+1).
   dS/dh_k = sum_j dg_j/dh_k = 2 * sum_j h_{j-k} = 2 * (sum of h). Let H_sum = sum(h).
   So dS/dh_k = 2 * H_sum.
   
   dM/dh_k:
   M = max(g).
   If the max is unique at index j*, dM/dh_k = dg_{j*}/dh_k = 2 h_{j*-k} (if valid).
   If multiple indices achieve max, the subgradient is the average of the gradients at those indices.
   For simplicity in optimization, we can use the subgradient corresponding to the index of the first max, or average them. Averaging is more stable.
   
   So we can compute the gradient analytically in O(N^2) or using FFT O(N log N).
   N=50, so O(N^2) is trivial.
   
   We will use this analytic gradient with a custom gradient descent loop (Adam or SGD with momentum) to navigate the landscape more efficiently than L-BFGS-B with finite differences, especially allowing for more steps within the time limit. We will also use multiple random restarts.
"""

import numpy as np
import time

N = 50

def evaluate(h):
    """Compute the value for a given set of step heights."""
    h = np.array(h, dtype=np.float64)
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return -1.0
    h = np.clip(h, 0, None)
    if np.sum(h) == 0:
        return -1.0
        
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    
    # l2 calculation
    # y has length m+2. indices 0 to m+1.
    # sum i=0 to m of (y[i]^2 + y[i]y[i+1] + y[i+1]^2)
    terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(terms)
    
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    
    if norm_1 * norm_inf == 0:
        return -1.0
        
    return l2 / (norm_1 * norm_inf)

def gradient(h):
    """Compute analytic gradient of the objective function."""
    h = np.clip(h, 0, None)
    if np.sum(h) == 0:
        return np.zeros_like(h)
        
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    
    # F = l2
    terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(terms)
    
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    G = norm_1 * norm_inf
    
    if G == 0:
        return np.zeros_like(h)
        
    V = l2 / G
    
    # dF/dh
    # A_j = y_j + 4 y_{j+1} + y_{j+2} for j in 0..m-1 (corresponding to g_j)
    # Note: y indices in formula y_j, y_{j+1}, y_{j+2}
    # g_j is y_{j+1}.
    # dF/dg_j = y_j + 4 y_{j+1} + y_{j+2}
    A = np.zeros(m)
    for j in range(m):
        A[j] = y[j] + 4*y[j+1] + y[j+2]
        
    # dF/dh_k = 2 * sum_j A_j h_{j-k}
    # This is 2 * convolve(A, h) but reversed?
    # sum_j A_j h_{j-k} is the cross-correlation of A and h, or convolution with reversed h.
    # Let's just do it directly for N=50.
    dF_dh = np.zeros(N)
    for k in range(N):
        s = 0.0
        for j in range(m):
            idx = j - k
            if 0 <= idx < N:
                s += A[j] * h[idx]
        dF_dh[k] = 2.0 * s
        
    # dG/dh
    # G = S * M / (m+1)
    # S = sum(g) = (sum h)^2
    # dS/dh_k = 2 * sum(h)
    S = np.sum(g)
    M = np.max(g)
    
    # Find indices where g is close to max to handle non-differentiability
    # Use a small tolerance
    max_idx = np.where(g >= M - 1e-9)[0]
    
    dS_dh = np.zeros(N)
    sum_h = np.sum(h)
    for k in range(N):
        dS_dh[k] = 2.0 * sum_h
        
    dM_dh = np.zeros(N)
    for k in range(N):
        s = 0.0
        for j in max_idx:
            idx = j - k
            if 0 <= idx < N:
                s += 2.0 * h[idx]
        # Average over the subgradients
        dM_dh[k] = s / len(max_idx)
        
    dG_dh = (dS_dh * M + S * dM_dh) / (m + 1)
    
    # dV/dh = (G * dF_dh - F * dG_dh) / G^2
    dV_dh = (G * dF_dh - l2 * dG_dh) / (G * G)
    
    return dV_dh

def construct():
    deadline = time.time() + 115.0
    best_h = np.ones(N)
    best_val = evaluate(best_h)
    
    # Multi-start with diverse initializations
    np.random.seed(42)
    
    initializations = [
        np.ones(N),
        np.exp(-10*(np.linspace(0, 1, N)-0.5)**2),
        1 - np.abs(2*np.linspace(0, 1, N) - 1),
        np.random.uniform(0, 1, N),
        np.random.uniform(0, 1, N),
        np.random.uniform(0, 1, N),
        # Sparse
        np.zeros(N),
    ]
    # Add a few sparse random ones
    for _ in range(3):
        h = np.zeros(N)
        idx = np.random.choice(N, size=5, replace=False)
        h[idx] = 1.0
        initializations.append(h)
        
    for h0 in initializations:
        if time.time() > deadline:
            break
            
        h = np.clip(h0, 0, None).copy()
        if np.sum(h) == 0:
            h = np.ones(N)
            
        # Adam-like optimizer
        lr = 0.01
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        m_t = np.zeros_like(h)
        v_t = np.zeros_like(h)
        t = 0
        
        # Adaptive learning rate decay
        for step in range(2000):
            if time.time() > deadline:
                break
                
            val = evaluate(h)
            if val > best_val:
                best_val = val
                best_h = h.copy()
                
            grad = gradient(h)
            
            t += 1
            m_t = beta1 * m_t + (1 - beta1) * grad
            v_t = beta2 * v_t + (1 - beta2) * (grad ** 2)
            
            m_hat = m_t / (1 - beta1 ** t)
            v_hat = v_t / (1 - beta2 ** t)
            
            # Decay learning rate
            lr_t = lr / (1 + step * 0.001)
            
            h = h + lr_t * m_hat / (np.sqrt(v_hat) + eps)
            h = np.clip(h, 0, None)
            
            # Normalize to prevent overflow/underflow issues, though objective is scale invariant?
            # Let's check: h -> c*h.
            # g -> c^2 g.
            # l2 -> c^2 l2.
            # norm_1 -> c^2 norm_1.
            # norm_inf -> c^2 norm_inf.
            # value = (c^2 l2) / (c^2 norm_1 * c^2 norm_inf) = l2 / (c^2 norm_1 norm_inf).
            # It is NOT scale invariant. It scales as 1/c^2.
            # So we want large h?
            # Wait.
            # l2 ~ h^2. norm_1 ~ h^2. norm_inf ~ h^2.
            # value ~ h^2 / (h^2 * h^2) = 1/h^2.
            # So we want SMALL h?
            # But h must be non-negative. If h -> 0, value -> infinity?
            # Let's re-read the evaluator.
            # "value = l2 / (norm_1 * norm_inf)"
            # If h is scaled by c, g scales by c^2.
            # l2 scales by c^2.
            # norm_1 scales by c^2.
            # norm_inf scales by c^2.
            # Denominator scales by c^4.
            # So value scales by 1/c^2.
            # This implies we should make h as small as possible?
            # But h are "step heights". Usually there's a constraint or normalization I missed?
            # "Choose 50 nonnegative step heights h".
            # "seed is 50 equal heights".
            # If I can scale h down arbitrarily, the score goes to infinity.
            # Is there a constraint?
            # "The evaluator computes exactly this value... higher is better."
            # "The heights must be finite and nonnegative."
            # If I can just output [1e-10] * 50, the score should be huge.
            # Let's check the seed score.
            # Seed: h = ones(50).
            # If I output 1e-10 * ones(50), score should be 1e20 times higher.
            # Why is the current score 1.23?
            # Maybe the evaluator normalizes h?
            # "With g = np.convolve(h, h)..."
            # "y = [0, g[0], ..., g[m-1], 0]"
            # "l2 = ..."
            # "norm_1 = ..."
            # "norm_inf = ..."
            # "value = ..."
            # There is no normalization of h in the description.
            # However, if the score is only 1.23, it suggests that either:
            # 1. The evaluator normalizes h (e.g., sum(h)=1 or max(h)=1) before computation.
            # 2. The "value" is not just the raw ratio, but something else.
            # 3. I am misinterpreting the scaling.
            
            # Let's assume the evaluator DOES normalize h to sum=1 or max=1.
            # If sum(h)=1, then we are on a simplex.
            # If max(h)=1, we are in a hypercube.
            # Given "seed is 50 equal heights", and score 1.0, and my current best 1.23,
            # it implies the search is finding better shapes, not just scaling.
            # This strongly implies h is normalized.
            # Common normalization: sum(h) = 1.
            # Let's assume sum(h) = 1 for the optimization.
            
            # I will normalize h to sum 1 after each step to stay on the simplex.
            # This prevents the 1/h^2 divergence.
            if np.sum(h) > 0:
                h = h / np.sum(h)
            else:
                h = np.ones(N) / N
                
    return list(best_h)

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
