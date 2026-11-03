import numpy as np
import time
from scipy.optimize import linprog

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """Compute the raw score: L2^2 / (L1 * AutoConvMax)."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution peak (discrete convolution scaled by DELTA)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin

    # We want to maximize: L2^2 / (L1 * M) where M = max(autoconv)
    # By scaling invariance (ratio scales as 1/c if we scale h by c),
    # we can fix L1 = 1 and maximize L2^2 / M.
    # Equivalently, for a fixed L2^2 = S, we want to minimize M.
    # Or, for a fixed L1=1, we can use a bisection-like approach or directly optimize.
    
    # The directive suggests: "Formulate the problem as a convex optimization by fixing the L1 norm to 1 and minimizing the peak autoconvolution, which is a linear function of the height vector"
    # Wait, is the peak autoconvolution linear?
    # M(h) = max_k (conv(h,h))_k.
    # (conv(h,h))_k = sum_i h_i h_{k-i}. This is QUADRATIC in h.
    # So M(h) is the max of quadratic forms. It is convex, but not linear.
    # Minimizing a convex function is not generally a linear program.
    # However, we can use a different approach.
    
    # Let's reconsider the scaling.
    # Score = L2^2 / (L1 * M).
    # If we fix L1 = 1, Score = L2^2 / M.
    # We want to maximize this.
    # Let's define a variable S = L2^2.
    # For a given S, what is the minimal M we can achieve with L1=1 and L2^2=S?
    # This is still a non-convex problem because the feasible set {h: L1=1, L2^2=S, h>=0} is not convex (it's the intersection of a hyperplane and a sphere).
    
    # Alternative approach:
    # Note that for any h, let \bar{h} = h / L1(h). Then L1(\bar{h})=1.
    # Score(h) = Score(\bar{h}).
    # So we only need to search over h with L1=1.
    # Maximize L2^2 / M.
    
    # Let's try a grid search over the number of non-zero elements and their values, or use a heuristic.
    # Given N=50, we can try to use `scipy.optimize.minimize` with SLSQP or trust-constr, 
    # but the previous attempt did that. The directive asks for a "new program... from scratch" and mentions "convex optimization... linear programming".
    # Is there a way to make it linear?
    
    # Let's look at the structure.
    # M(h) >= (conv(h,h))_k for all k.
    # If we fix h, M is determined.
    # If we introduce a variable M and constraints M >= (conv(h,h))_k, we have quadratic constraints.
    # This is a Quadratically Constrained Quadratic Program (QCQP), which is non-convex in general.
    
    # However, the directive says "minimizing the peak autoconvolution, which is a linear function of the height vector".
    # This statement might be flawed, or it refers to a different formulation.
    # Perhaps it means that for a FIXED pattern of support, the convolution is linear in the weights? No, it's quadratic.
    # Maybe it means that we can relax the problem?
    
    # Let's try a different strategy:
    # Since we have 120 seconds, we can use a more robust numerical optimizer than SLSQP, like `trust-constr` or even a custom implementation using `linprog` if we can linearize.
    # We cannot linearize the quadratic constraints easily.
    
    # Let's try to maximize the score using `scipy.optimize.minimize` with `method='trust-constr'` which can handle non-linear constraints better than SLSQP.
    # Objective: minimize - (sum(h^2) / M)
    # Constraints:
    # 1. sum(h) = 1
    # 2. h >= 0
    # 3. M >= conv(h,h)_k for all k
    # 4. M > 0
    
    # Actually, we can eliminate M.
    # Maximize f(h) = sum(h^2) / max(conv(h,h))
    # This is differentiable almost everywhere.
    
    # Let's try a stochastic local search or simulated annealing combined with local optimization.
    # Or, simply use `minimize` with multiple restarts and `trust-constr`.
    
    # Let's implement the `trust-constr` approach.
    
    from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
    
    # Precompute convolution indices for efficiency in constraints?
    # conv(h,h) has 2N-1 = 99 terms.
    # term_k = sum_{i} h_i h_{k-i}
    
    # We can define the objective and constraints directly.
    
    def obj_and_grad(h):
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv == 0:
            return 0.0, np.zeros_like(h)
        val = l2_sq / max_conv
        # Gradient is tricky due to max.
        # We can use subgradient or just let the optimizer handle it if we use M variable.
        # Let's stick to the M variable approach for smoothness.
        
        return val, None # Gradient not easily available
    
    # Let's use the M variable approach with `trust-constr`.
    # Variables: x = [h_0, ..., h_49, M]
    # Objective: minimize - (sum(h^2) / M)
    # Constraints:
    # - sum(h) - 1 = 0
    # - h_i >= 0
    # - M > 0
    # - M - (conv(h,h))_k >= 0 for k=0..98
    
    def objective(x):
        h = x[:N]
        M = x[N]
        if M <= 0:
            return 1e10
        l2_sq = np.sum(h**2)
        return -(l2_sq / M)
    
    def objective_grad(x):
        h = x[:N]
        M = x[N]
        if M <= 0:
            return np.zeros_like(x)
        l2_sq = np.sum(h**2)
        # d/dh_i (-L2^2/M) = -2 h_i / M
        # d/dM (-L2^2/M) = L2^2 / M^2
        grad_h = -2 * h / M
        grad_M = l2_sq / (M * M)
        return np.concatenate([grad_h, [grad_M]])
    
    # Constraints for trust-constr
    
    # 1. Linear constraint: sum(h) = 1
    A_lin = np.zeros((1, N + 1))
    A_lin[0, :N] = 1.0
    b_lin = np.array([1.0])
    lc = LinearConstraint(A_lin, b_lin, b_lin)
    
    # 2. Nonlinear constraints: M - conv_k >= 0
    # There are 2N-1 = 99 constraints.
    
    def nonlinear_fun(x):
        h = x[:N]
        M = x[N]
        h_conv = np.convolve(h, h, mode='full')
        return M - h_conv
    
    def nonlinear_grad(x):
        h = x[:N]
        M = x[N]
        # h_conv = conv(h,h)
        # d/dh_i (M - h_conv_k) = - sum_j (d/dh_i (h_j h_{k-j}))
        # h_conv_k = sum_{i=0}^{k} h_i h_{k-i} (with h_i=0 if i<0 or i>=N)
        # d/dh_i h_conv_k = h_{k-i} + h_{k-i} = 2 h_{k-i} (if i != k-i) or h_{k-i} (if i == k-i, i.e., k=2i)
        # Wait, derivative of h_i * h_{k-i} wrt h_i is h_{k-i}.
        # Derivative of h_j * h_{k-j} wrt h_i is 0 if j != i.
        # So d/dh_i (sum_j h_j h_{k-j}) = sum_j (delta_{ij} h_{k-j} + h_j delta_{i,k-j})
        # = h_{k-i} + h_{k-i} = 2 h_{k-i}
        # Wait, if i = k-i, i.e., k=2i, then the term is h_i^2, derivative is 2h_i.
        # The formula 2 h_{k-i} holds generally.
        
        # We need to construct a 99 x (N+1) matrix.
        # This might be expensive. 99 * 50 = 4950 entries.
        # Let's try to compute it efficiently.
        
        # h_conv has length 2N-1.
        # For each k, d/dh_i (M - h_conv_k) = -2 h_{k-i} (if 0 <= k-i < N, else 0).
        # And d/dM (M - h_conv_k) = 1.
        
        J = np.zeros((2*N-1, N+1))
        J[:, N] = 1.0 # d/dM term
        
        # For h terms:
        # J[k, i] = -2 * h_{k-i} if valid index, else 0.
        # We can construct this using shifts.
        # Let's just loop for simplicity first, if it's too slow, optimize.
        # N=50, 99 constraints. Looping 99 times is fine.
        
        for k in range(2*N-1):
            for i in range(N):
                j = k - i
                if 0 <= j < N:
                    J[k, i] = -2 * h[j]
        
        return J
    
    nlcc = NonlinearConstraint(nonlinear_fun, 0.0, np.inf, jac=nonlinear_grad)
    
    # Bounds
    bounds = [(0.0, 1.0)] * N + [(1e-6, None)]
    
    # Initial guesses
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # Generate diverse initializations
    inits = []
    
    # 1. Uniform
    inits.append(best_h.copy())
    
    # 2. Gaussians
    x_idx = np.arange(N)
    for c in np.linspace(5, N-5, 5):
        for w in np.linspace(1, N/2, 5):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            h_g /= np.sum(h_g)
            inits.append(h_g)
            
    # 3. Sparse
    for _ in range(10):
        h_s = np.zeros(N)
        k = np.random.randint(3, 10)
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    # 4. Random smooth
    for _ in range(10):
        h_r = np.random.rand(N)
        h_r /= np.sum(h_r)
        inits.append(h_r)

    # Optimization loop
    for h_init in inits:
        if time.time() > deadline:
            break
            
        # Initialize M
        h_init_norm = h_init / np.sum(h_init)
        h_conv_init = np.convolve(h_init_norm, h_init_norm, mode='full')
        M_init = np.max(h_conv_init) * 1.1 # Add margin
        
        x0 = np.concatenate([h_init_norm, [M_init]])
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='trust-constr', 
                jac=objective_grad, 
                bounds=bounds, 
                constraints=[lc, nlcc], 
                options={'maxiter': 500, 'gtol': 1e-8}
            )
            
            if res.success:
                h_candidate = res.x[:N]
                # Ensure non-negative and sum 1 (just in case)
                h_candidate = np.maximum(0, h_candidate)
                if np.sum(h_candidate) > 0:
                    h_candidate /= np.sum(h_candidate)
                
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
                    
        except Exception as e:
            pass
            
    # Final polish
    if time.time() < deadline - 5:
        h_start = best_h.copy()
        h_conv_start = np.convolve(h_start, h_start, mode='full')
        M_start = np.max(h_conv_start) * 1.05
        x0 = np.concatenate([h_start, [M_start]])
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='trust-constr', 
                jac=objective_grad, 
                bounds=bounds, 
                constraints=[lc, nlcc], 
                options={'maxiter': 1000, 'gtol': 1e-10}
            )
            if res.success:
                h_candidate = res.x[:N]
                h_candidate = np.maximum(0, h_candidate)
                if np.sum(h_candidate) > 0:
                    h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    return list(best_h)

# EVOLVE-BLOCK-END
