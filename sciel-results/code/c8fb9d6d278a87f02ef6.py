import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT for speed."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return np.inf
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_and_grad(x, beta):
    """
    Compute the soft-max approximation of max(|ac|) and its gradient with respect to x.
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    
    if m < 1e-15:
        # If max is 0, gradient is 0 (or uniform, but 0 is safer for min)
        return 0.0, np.zeros_like(x)

    # Soft max: (1/beta) * log(sum(exp(beta * abs_ac)))
    # Shift for stability
    shifted = beta * (abs_ac - m)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    soft_max_val = m + np.log(sum_exp) / beta
    
    # Weights for the soft max
    weights = exp_shifted / sum_exp
    
    # Gradient of soft_max(|ac|) w.r.t x
    # d(soft_max)/dx_j = sum_k weights_k * sign(ac_k) * d(ac_k)/dx_j
    # d(ac_k)/dx_j = x_{k-j} + x_{j-k} (for valid indices)
    # Let W_k = weights_k * sign(ac_k)
    # grad_j = sum_k W_k (x_{k-j} + x_{j-k})
    
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
        
    # Construct W_full for FFT
    W_full = np.zeros(L_fft)
    W_full[:L] = weights * np.sign(ac)
    
    U = np.fft.rfft(W_full)
    X = np.fft.rfft(x, n=L_fft)
    
    # Term 1: sum_k W_k x_{k-j} = IFFT(U * conj(X))
    t1 = np.fft.irfft(U * np.conj(X), n=L_fft)[:N]
    # Term 2: sum_k W_k x_{j-k} = IFFT(U * X)
    t2 = np.fft.irfft(U * X, n=L_fft)[:N]
    
    grad_soft_max = t1 + t2
    
    return soft_max_val, grad_soft_max

def _objective_and_grad(x, beta):
    """
    Objective: max(|ac|) / (sum(x))^2
    We use a penalty method to enforce sum(x)=1, or just optimize the ratio directly.
    Optimizing ratio directly is tricky with L-BFGS-B if sum(x) can be negative or zero.
    The task allows signed heights.
    Let's stick to the penalty method for stability, or optimize x with constraint sum(x)=1?
    L-BFGS-B doesn't support linear equality constraints directly.
    
    Alternative: Optimize y where x = y / sum(y)? No, non-smooth.
    
    Let's use the penalty approach on the normalized objective?
    f(x) = max(|ac(x)|) / (sum(x))^2
    Let S = sum(x).
    grad f = (grad_max * S^2 - max * 2*S * 1) / S^4 = grad_max/S^2 - 2*max/S^3
    
    This is valid if S != 0.
    If S is close to 0, the objective blows up, which is good for minimization (we want large S).
    However, if S crosses 0, the sign flips and the objective becomes negative (if we don't take abs(S)^2? No, S^2 is positive).
    Wait, max(|ac|) is non-negative. S^2 is non-negative.
    So f(x) >= 0.
    The gradient derivation:
    f = M / S^2
    df/dx_j = (dM/dx_j * S^2 - M * 2S * dS/dx_j) / S^4
    dS/dx_j = 1
    df/dx_j = dM/dx_j / S^2 - 2M / S^3
    
    This requires S not to be 0.
    If we start with S > 0, we likely stay there.
    If we allow signed x, S can be small.
    To be safe, we can add a penalty if |S| is small.
    Or, simpler: Minimize M / S^2 subject to S=1?
    If we fix S=1, we minimize M.
    But the scale of x doesn't matter for the ratio.
    So we can just minimize M(x) subject to sum(x)=1.
    This is a standard constrained optimization.
    We can use the Lagrangian: L = M(x) + lambda * (sum(x) - 1)
    But L-BFGS-B is for box constraints.
    
    Let's go back to the penalty method on the unnormalized objective?
    No, the ratio is scale invariant.
    
    Let's use the gradient of the ratio directly, with a safeguard.
    If |S| < 1e-4, add a strong penalty to push S away from 0.
    
    Actually, the previous program used a penalty on (sum(x)-1)^2 added to the soft-max.
    That effectively minimizes soft_max + penalty*(sum-1)^2.
    This pushes sum towards 1 and minimizes the autoconv.
    This is a valid strategy.
    
    Let's stick to: minimize soft_max(|ac|) + penalty * (sum(x) - 1)^2
    This ensures sum(x) approx 1 and minimizes the autoconv peak.
    """
    s = np.sum(x)
    soft_max_val, grad_soft_max = _soft_max_and_grad(x, beta)
    
    # Penalty term
    penalty = 1e4
    penalty_val = penalty * (s - 1)**2
    penalty_grad = 2 * penalty * (s - 1) * np.ones(N)
    
    total_val = soft_max_val + penalty_val
    total_grad = grad_soft_max + penalty_grad
    
    # Safeguard for numerical issues
    if not np.all(np.isfinite(total_grad)):
        total_grad = np.zeros(N)
    
    return total_val, total_grad

def _structured_init(freq, phase=0.0):
    t = np.arange(N)
    x = np.cos(2 * np.pi * freq * t / N + phase)
    s = np.sum(x)
    if abs(s) < 1e-6:
        x += 0.1
        s = np.sum(x)
    x /= s
    return x

def _random_init():
    x = np.random.randn(N)
    s = np.sum(x)
    if abs(s) < 1e-6:
        x += 0.1
        s = np.sum(x)
    x /= s
    return x

def _banded_init(bandwidth=5):
    x = np.zeros(N)
    for i in range(N):
        for j in range(max(0, i-bandwidth), min(N, i+bandwidth+1)):
            x[j] += np.random.randn()
    s = np.sum(x)
    if abs(s) < 1e-6:
        x += 0.1
        s = np.sum(x)
    x /= s
    return x

def construct():
    START_TIME = time.time()
    TIME_LIMIT = 145  # Safety margin before 150s kill

    # Initial best
    best_x = np.ones(N)
    best_x /= np.sum(best_x)
    best_val = _objective(best_x)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Diverse Initializations with L-BFGS-B
    # Combine ideas from both programs: structured, random, banded
    initializations = [
        np.ones(N),
        _structured_init(1),
        _structured_init(2),
        _structured_init(5),
        _structured_init(10),
        _random_init(),
        _random_init(),
        _banded_init(3),
        _banded_init(8),
    ]
    
    # Normalize all inits
    for i, x0 in enumerate(initializations):
        s = np.sum(x0)
        if abs(s) < 1e-6: continue
        x0 = x0 / s
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
    
    # Optimization loop
    # We will run L-BFGS-B on the best current solution and potentially some others if time permits.
    # To maximize score, we should focus compute on the most promising candidates.
    # Strategy:
    # 1. Polish the best init found so far with moderate beta.
    # 2. Polish with high beta.
    # 3. If time, try a few random restarts.
    
    # Helper to run minimization
    def run_minimize(x_start, beta, maxiter, ftol, gtol):
        if time.time() - START_TIME > TIME_LIMIT:
            return x_start, _objective(x_start)
        
        def obj_and_grad(x):
            return _objective_and_grad(x, beta)
        
        try:
            res = minimize(
                obj_and_grad, 
                x_start, 
                method='L-BFGS-B', 
                jac=True,
                options={
                    'maxiter': maxiter, 
                    'ftol': ftol, 
                    'gtol': gtol, 
                    'disp': False
                }
            )
            
            x_opt = res.x
            s_opt = np.sum(x_opt)
            if abs(s_opt) < 1e-10:
                return x_start, _objective(x_start)
            
            # The penalty method drives sum towards 1, but might not be exactly 1.
            # The objective _objective uses the actual sum.
            # So we evaluate the true objective.
            val_opt = _objective(x_opt)
            
            return x_opt, val_opt
        except Exception:
            return x_start, _objective(x_start)

    # Step 1: Moderate beta polish
    x_curr = best_x.copy()
    x_curr, val_curr = run_minimize(x_curr, beta=50.0, maxiter=200, ftol=1e-8, gtol=1e-6)
    if val_curr < best_val:
        best_val = val_curr
        best_x = x_curr.copy()

    # Step 2: High beta polish
    if time.time() - START_TIME < TIME_LIMIT - 10:
        x_curr = best_x.copy()
        x_curr, val_curr = run_minimize(x_curr, beta=200.0, maxiter=300, ftol=1e-10, gtol=1e-8)
        if val_curr < best_val:
            best_val = val_curr
            best_x = x_curr.copy()

    # Step 3: Local refinement / Coordinate Descent (from Inspiration 1)
    # This helps escape shallow local minima of the soft-max approximation
    if time.time() - START_TIME < TIME_LIMIT - 5:
        x = best_x.copy()
        
        # Find peak lag
        ac = _autoconv_fft(x)
        abs_ac = np.abs(ac)
        
        step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
        
        for step in step_sizes:
            for _ in range(200):
                if time.time() - START_TIME > TIME_LIMIT - 2:
                    break
                
                idx = rng.integers(N)
                x_old = x[idx]
                
                improved = False
                for direction in [-1, 1]:
                    x[idx] = x_old + direction * step
                    sc = _objective(x)
                    if sc < best_val:
                        best_val = sc
                        best_x = x.copy()
                        improved = True
                        break
                if not improved:
                    x[idx] = x_old
                else:
                    # Keep the change, update x for next iteration
                    pass 

    # Step 4: One more high-beta polish if time remains
    if time.time() - START_TIME < TIME_LIMIT - 2:
        x_curr = best_x.copy()
        x_curr, val_curr = run_minimize(x_curr, beta=500.0, maxiter=100, ftol=1e-12, gtol=1e-10)
        if val_curr < best_val:
            best_val = val_curr
            best_x = x_curr.copy()

    return best_x
