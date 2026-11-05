"""
Strategy: Direct coordinate-wise optimization with smooth surrogate objective.

We represent the step heights h directly as a vector of length N=400.
The objective function is V(h) = |2 * N * max(conv(h, h)) / sum(h)^2|.
The term max(conv(h, h)) is non-smooth. We approximate it using the Log-Sum-Exp (LSE) function:
max(x) ≈ (1/beta) * log(sum(exp(beta * x)))
This allows us to use gradient-based optimization on the full vector h.

To handle the constraint sum(h) != 0 and the scale invariance, we normalize h 
such that sum(h) = 1 during the optimization process, or simply optimize the 
direction of h and scale it at the end. However, the objective is scale-invariant 
only if we consider the ratio. Let's check:
V(c*h) = |2 * N * c^2 * max(conv(h,h)) / (c * sum(h))^2| = V(h).
So we can fix sum(h) = 1.
We optimize h subject to sum(h) = 1.
We can parameterize h = u + 1/N * 1, where sum(u) = 0.
Or simply use an unconstrained optimizer and project onto sum(h)=1 after each step,
or use a penalty term.

Given the 120s limit and N=400, L-BFGS-B on 400 variables with a smooth objective 
should be feasible.
We will use a smooth surrogate for the max term:
M(h) = (1/beta) * log(sum(exp(beta * conv(h, h))))
Objective: f(h) = M(h) / (sum(h)^2)  (ignoring absolute value and constant 2N for optimization, 
actually we want to minimize the absolute value, but since we can flip signs, 
we can just minimize the magnitude. Note that conv(0) = sum(h^2) > 0, so the max is likely positive 
unless h is very weird. If max is negative, we can flip h to make it positive? 
No, flipping h changes conv to conv, so max stays same. 
Wait, if max is negative, V is negative? No, V is abs(...). 
The max of conv(h,h) is always >= 0 because conv(h,h)[0] = sum(h_i^2) >= 0. 
So we can drop the absolute value.
We minimize f(h) = LSE(conv(h,h)) / (sum(h)^2).
We fix sum(h) = 1 to remove scale invariance.
Then minimize g(h) = LSE(conv(h,h)) subject to sum(h) = 1.
We can use Lagrange multipliers or just project.
Let's use scipy.optimize.minimize with method='L-BFGS-B' or 'Powell' or 'Nelder-Mead'.
L-BFGS-B requires gradients. We can compute gradients of LSE(conv) analytically.
Let C = conv(h, h). C_k = sum_i h_i h_{i+k}.
LSE(C) = (1/beta) log(sum_k exp(beta C_k)).
d(LSE)/d h_j = (1/beta) * (sum_k exp(beta C_k) * beta * dC_k/dh_j) / sum_k exp(beta C_k)
             = sum_k (exp(beta C_k) / S) * dC_k/dh_j
where S = sum_k exp(beta C_k).
dC_k/dh_j = h_{j+k} + h_{j-k} (with boundary conditions, effectively h_{j+k} if j+k<N else 0, etc).
This is a convolution-like operation.
dC/dh = 2 * conv(h, h) (circular? no, linear).
Actually, d(sum_i h_i h_{i+k}) / d h_j = h_{j+k} + h_{j-k}.
So the gradient w.r.t h is the convolution of the "weights" w_k = exp(beta C_k)/S with h, plus the reverse.
This can be computed efficiently with FFT or np.convolve.

We will implement this gradient and use L-BFGS-B.
To handle sum(h)=1, we can optimize in the subspace sum(h)=1 or use a penalty.
Simpler: Optimize h freely, then normalize h = h / sum(h) before evaluating?
No, the objective is scale invariant, so the gradient of the normalized objective is complex.
Better: Fix sum(h)=1.
We can parameterize h = 1/N * ones(N) + u, where sum(u)=0.
This reduces dimensionality to N-1=399.
Or just use a penalty term P * (sum(h)-1)^2.

Let's try a simpler approach:
1. Start with a good initial guess (e.g., random or from previous best).
2. Normalize to sum=1.
3. Use a smooth surrogate for max.
4. Use coordinate descent or L-BFGS on the normalized vector, re-normalizing after each step?
   Re-normalizing breaks the gradient direction.
   
Alternative:
Minimize J(h) = LSE(conv(h,h)) / (sum(h)^2).
This is scale invariant.
Gradient descent on this might drift in scale.
Let's stick to fixing sum(h)=1.
We will use a projected gradient method or L-BFGS with a linear constraint.
scipy.optimize.minimize does not support linear constraints directly in L-BFGS-B.
We can use 'SLSQP' which supports constraints, but it's slower.
Or we can parameterize h using 399 variables.
Let u be a vector of length 399.
h[0..398] = u[0..398]
h[399] = 1 - sum(u[0..398])
This is a valid parameterization.
We optimize u.

Let's implement this.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400
_DEADLINE = 140.0
_BETA = 10.0 # Sharpness of LSE approximation

def _compute_lse_and_grad(h, beta):
    """
    Compute LSE(conv(h,h)) and its gradient w.r.t h.
    h is assumed to be 1D array of length N.
    """
    conv = np.convolve(h, h)
    # LSE
    max_c = np.max(conv)
    # Shift to avoid overflow
    conv_shifted = conv - max_c
    exp_conv = np.exp(beta * conv_shifted)
    sum_exp = np.sum(exp_conv)
    lse_val = max_c + (1.0 / beta) * np.log(sum_exp)
    
    # Gradient
    # w_k = exp(beta * (C_k - max_c)) / sum_exp
    w = exp_conv / sum_exp
    
    # dC_k / dh_j = h_{j+k} + h_{j-k}
    # grad_j = sum_k w_k * (h_{j+k} + h_{j-k})
    # This is 2 * conv(w, h) if we treat w as a kernel?
    # Let's check indices.
    # C_k = sum_i h_i h_{i+k}.
    # dC_k/dh_j = h_{j+k} + h_{j-k}.
    # grad_j = sum_k w_k h_{j+k} + sum_k w_k h_{j-k}.
    # Let v = w.
    # sum_k v_k h_{j+k} is a cross-correlation or convolution depending on definition.
    # np.convolve(a, b)[n] = sum_m a_m b_{n-m}.
    # We want sum_k w_k h_{j+k}. Let m = -k. sum_m w_{-m} h_{j-m}.
    # This is convolve(w_reversed, h)[j].
    # So grad = convolve(w[::-1], h) + convolve(w, h[::-1])[::-1] ?
    # Actually, due to symmetry of w (if conv is symmetric? No, conv(h,h) is symmetric around 0? 
    # No, full conv has length 2N-1. Indices 0 to 2N-2.
    # C_k for k=0..N-1 is the standard autocorrelation.
    # C_{2N-2-k} = C_k.
    # So w is symmetric around N-1.
    # Let's just compute it directly for safety, N=400 is small.
    # Or use FFT.
    
    # Efficient way:
    # grad_j = sum_k w_k h_{j+k} + sum_k w_k h_{j-k}
    # term1_j = sum_k w_k h_{j+k}
    # term2_j = sum_k w_k h_{j-k}
    
    # Using np.convolve:
    # term1 = convolve(h, w[::-1]) ?
    # Let's test with small N.
    
    # Simpler: 
    # grad = 2 * convolve(h, w) is not quite right due to boundaries.
    # Let's use the direct sum for clarity and correctness, N=400, 2N-1=799 terms.
    # 400 * 799 operations is ~300k, very fast.
    
    grad = np.zeros(N)
    for j in range(N):
        s = 0.0
        for k in range(len(w)):
            # k is lag index in conv, 0 to 2N-2
            # h_{j+k} is valid if 0 <= j+k < N
            # h_{j-k} is valid if 0 <= j-k < N
            idx1 = j + k
            idx2 = j - k
            if 0 <= idx1 < N:
                s += w[k] * h[idx1]
            if 0 <= idx2 < N:
                s += w[k] * h[idx2]
        grad[j] = s
        
    return lse_val, grad

def _objective_u(u, beta):
    """
    Objective function for the parameterization u.
    h is constructed from u such that sum(h)=1.
    """
    h = np.zeros(N)
    h[:N-1] = u
    h[N-1] = 1.0 - np.sum(u)
    
    # Ensure h is finite
    if not np.all(np.isfinite(h)):
        return 1e10
        
    val, _ = _compute_lse_and_grad(h, beta)
    return val

def _grad_u(u, beta):
    """
    Gradient of the objective w.r.t u.
    """
    h = np.zeros(N)
    h[:N-1] = u
    h[N-1] = 1.0 - np.sum(u)
    
    if not np.all(np.isfinite(h)):
        return np.zeros(N-1)
        
    _, grad_h = _compute_lse_and_grad(h, beta)
    
    # Chain rule: d val / du_i = d val / dh_i * dh_i / du_i
    # dh_i / du_i = 1 for i < N-1
    # dh_{N-1} / du_i = -1 for i < N-1
    # grad_u[i] = grad_h[i] * 1 + grad_h[N-1] * (-1)
    grad_u = grad_h[:N-1] - grad_h[N-1]
    
    return grad_u

def construct():
    """
    Construct step heights using direct coordinate-wise optimization with smooth surrogate.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    # Initial guess: uniform (sum=1)
    h_init = np.ones(N) / N
    u_init = h_init[:N-1].copy()
    
    best_val = np.inf
    best_h = h_init.copy()
    
    # Evaluate initial
    val_init, _ = _compute_lse_and_grad(best_h, _BETA)
    best_val = val_init
    
    # Optimization
    try:
        res = minimize(
            _objective_u,
            u_init,
            jac=_grad_u,
            args=(_BETA,),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res.fun < best_val:
            u_opt = res.x
            h_opt = np.zeros(N)
            h_opt[:N-1] = u_opt
            h_opt[N-1] = 1.0 - np.sum(u_opt)
            
            if np.all(np.isfinite(h_opt)):
                val_opt, _ = _compute_lse_and_grad(h_opt, _BETA)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
                    
    except Exception as e:
        pass
    
    # If optimization failed or didn't improve, try a few random restarts with L-BFGS
    rng = np.random.default_rng(42)
    
    for restart in range(5):
        if time.time() > deadline - 20:
            break
            
        # Random initial u
        u_rand = rng.normal(0, 0.1, size=N-1)
        # Normalize to sum=1 roughly
        h_rand = np.zeros(N)
        h_rand[:N-1] = u_rand
        h_rand[N-1] = 1.0 - np.sum(u_rand)
        
        # Check if sum is close to 1 and finite
        if not np.all(np.isfinite(h_rand)):
            continue
            
        try:
            res_r = minimize(
                _objective_u,
                u_rand,
                jac=_grad_u,
                args=(_BETA,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            if res_r.fun < best_val:
                u_opt = res_r.x
                h_opt = np.zeros(N)
                h_opt[:N-1] = u_opt
                h_opt[N-1] = 1.0 - np.sum(u_opt)
                
                if np.all(np.isfinite(h_opt)):
                    val_opt, _ = _compute_lse_and_grad(h_opt, _BETA)
                    if val_opt < best_val:
                        best_val = val_opt
                        best_h = h_opt.copy()
                        
        except Exception:
            pass
    
    # Final Polish: Coordinate Descent on h directly (unconstrained, then normalize)
    # This can refine the solution further.
    if time.time() < deadline - 5:
        h_curr = best_h.copy()
        v_curr = best_val
        
        # Use a smaller beta for smoother landscape in polish? Or same.
        step_size = 0.01
        for _ in range(500):
            if time.time() > deadline - 2:
                break
            idx = rng.integers(0, N)
            h_new = h_curr.copy()
            h_new[idx] += rng.normal(0, step_size)
            
            # Normalize
            s = np.sum(h_new)
            if s == 0:
                continue
            h_new = h_new / s
            
            val_new, _ = _compute_lse_and_grad(h_new, _BETA)
            if val_new < v_curr:
                v_curr = val_new
                h_curr = h_new
                step_size *= 0.95
            else:
                step_size = min(step_size * 1.1, 0.05)
                
        if v_curr < best_val:
            best_val = v_curr
            best_h = h_curr

    # Convert back to the required format
    # The evaluator computes: abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    # Our optimization minimized LSE(conv) with sum(h)=1.
    # The value we want to return is the actual value.
    
    # Check validity
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0:
        return np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
