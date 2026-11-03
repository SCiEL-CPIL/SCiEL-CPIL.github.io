import time
import numpy as np
from scipy.optimize import minimize
import sys

N = 50

def _compute_score(h):
    """Compute the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0):
        return 0.0
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.dot(h, h)
    # Autoconvolution: (h * h)[k] = sum_i h[i] h[k-i]
    # Using FFT for efficiency
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    autoconv = np.fft.irfft(H * H, n=n_fft)[:N]
    sup_norm = np.max(np.abs(autoconv))
    if sup_norm <= 0:
        return 0.0
    return l2_sq / (l1 * sup_norm)

def construct():
    """
    Construct 50 nonnegative heights on equal intervals of [-1/4, 1/4]
    to maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    
    Strategy: Use a structural approach based on the insight that
    concentrating mass on a sparse set of points (like a comb pattern)
    can reduce the autoconvolution peak relative to the L2 norm.
    
    We use a combination of:
    1. Initialize with a sparse pattern (every other point active)
    2. Use SLSQP with epigraph reformulation for the max constraint
    3. Block-coordinate descent for refinement
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    best_h = np.ones(N)
    best_score = _compute_score(best_h)
    
    # Helper to check time
    def time_remaining():
        return deadline - time.time()
    
    # Phase 1: Try several structured initializations
    initializations = []
    
    # 1. All ones (baseline)
    initializations.append(np.ones(N))
    
    # 2. Sparse comb: every other point
    comb = np.zeros(N)
    comb[0::2] = 1.0
    initializations.append(comb)
    
    # 3. Sparse comb: every 4th point
    comb4 = np.zeros(N)
    comb4[0::4] = 1.0
    initializations.append(comb4)
    
    # 4. Two-point concentration
    two_pt = np.zeros(N)
    two_pt[0] = 1.0
    two_pt[N//2] = 1.0
    initializations.append(two_pt)
    
    # 5. Gaussian-like
    x = np.linspace(-1, 1, N)
    gauss = np.exp(-2 * x**2)
    initializations.append(gauss)
    
    # 6. Uniform with noise
    initializations.append(np.ones(N) * 0.5 + 0.5 * np.random.rand(N))
    
    # 7. Triangular
    tri = 1.0 - 2.0 * np.abs(np.arange(N) - N/2) / (N/2)
    tri = np.clip(tri, 0, 1)
    initializations.append(tri)
    
    # 8. Alternating high-low
    alt = np.zeros(N)
    alt[0::2] = 1.0
    alt[1::2] = 0.1
    initializations.append(alt)
    
    # 9. Edge concentration
    edge = np.zeros(N)
    edge[:5] = 1.0
    edge[-5:] = 1.0
    initializations.append(edge)
    
    # 10. Center spike
    center = np.zeros(N)
    center[N//2] = 1.0
    center[N//2 - 1] = 0.5
    center[N//2 + 1] = 0.5
    initializations.append(center)
    
    # Evaluate all initializations
    for h0 in initializations:
        h0 = np.clip(h0, 0, None)
        if np.sum(h0) > 0:
            s = _compute_score(h0)
            if s > best_score:
                best_score = s
                best_h = h0.copy()
    
    # Phase 2: SLSQP optimization from best initialization
    # Epigraph reformulation:
    # Maximize f(h) = ||h||_2^2 / (||h||_1 * t)
    # subject to: (h*h)[k] <= t for all k, h >= 0, t > 0
    # Equivalently, minimize -||h||_2^2 / (||h||_1 * t)
    
    def objective_and_grad(x):
        h = x[:N]
        t = x[N]
        if t <= 1e-10:
            return 1e10, np.zeros(N+1)
        l1 = np.sum(h)
        if l1 <= 1e-10:
            return 1e10, np.zeros(N+1)
        l2_sq = np.dot(h, h)
        val = -l2_sq / (l1 * t)
        
        # Gradient
        grad_h = np.zeros(N)
        # d/dh_i [-l2_sq/(l1*t)] = -(2*h_i*l1 - l2_sq)/ (l1^2 * t)
        for i in range(N):
            grad_h[i] = -(2.0 * h[i] * l1 - l2_sq) / (l1 * l1 * t)
        # d/dt [-l2_sq/(l1*t)] = l2_sq/(l1*t^2)
        grad_t = l2_sq / (l1 * t * t)
        
        return val, np.concatenate([grad_h, [grad_t]])
    
    def constraint_autoconv(x):
        """Constraints: (h*h)[k] <= t for all k"""
        h = x[:N]
        t = x[N]
        n_fft = 2 * N
        H = np.fft.rfft(h, n=n_fft)
        autoconv = np.fft.irfft(H * H, n=n_fft)[:N]
        return t - autoconv
    
    # We need to handle the non-smooth max by using all autoconv constraints
    # But that's N constraints. Let's use a penalty-based approach instead.
    
    # Phase 2a: Block-coordinate descent
    # For each pair of indices, try to adjust values to improve score
    
    def block_coordinate_descent(h, max_iter=1000):
        """
        Block-coordinate descent: for each index i, try to optimize h[i]
        while keeping others fixed. Also try symmetric pairs.
        """
        h = h.copy()
        for iteration in range(max_iter):
            if time_remaining() < 30.0:
                break
            improved = False
            for i in range(N):
                if time_remaining() < 30.0:
                    break
                # Try a range of values for h[i]
                old_val = h[i]
                h[i] = 0.0
                base_l1 = np.sum(h)
                base_l2_sq = np.dot(h, h)
                
                # Compute autoconv with h[i] = 0
                n_fft = 2 * N
                H_base = np.fft.rfft(h, n=n_fft)
                autoconv_base = np.fft.irfft(H_base * H_base, n=n_fft)[:N]
                
                # When we add delta to h[i], autoconv changes by:
                # delta * (h * e_i)[k] + delta * (e_i * h)[k] + delta^2 * e_i*e_i
                # = delta * (h[k-i] + h[i-k]) + delta^2 * delta(k, 2i)
                
                best_delta = 0.0
                best_score_i = _compute_score(h)
                
                # Try several delta values
                for delta in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
                    h[i] = delta
                    s = _compute_score(h)
                    if s > best_score_i:
                        best_score_i = s
                        best_delta = delta
                
                h[i] = best_delta
                if abs(best_delta - old_val) > 1e-6:
                    improved = True
            
            if not improved:
                break
        
        return h
    
    # Apply BCD from best initialization
    h_bcd = block_coordinate_descent(best_h, max_iter=500)
    s_bcd = _compute_score(h_bcd)
    if s_bcd > best_score:
        best_score = s_bcd
        best_h = h_bcd.copy()
    
    # Phase 2b: SLSQP with penalty for autoconv constraints
    def penalty_objective(x, penalty_coeff=100.0):
        h = x[:N]
        t = x[N]
        if t <= 1e-10:
            return 1e10
        l1 = np.sum(h)
        if l1 <= 1e-10:
            return 1e10
        l2_sq = np.dot(h, h)
        base = -l2_sq / (l1 * t)
        
        # Penalty for violating autoconv <= t
        n_fft = 2 * N
        H = np.fft.rfft(h, n=n_fft)
        autoconv = np.fft.irfft(H * H, n=n_fft)[:N]
        violations = np.maximum(0, autoconv - t)
        penalty = penalty_coeff * np.sum(violations ** 2)
        
        return base + penalty
    
    # Normalize h to sum to 1 for numerical stability during optimization
    h_norm = best_h / np.sum(best_h) if np.sum(best_h) > 0 else np.ones(N) / N
    
    # Optimize over h (normalized) and t
    x0 = np.concatenate([h_norm, [1.0]])
    
    constraints = []
    # h >= 0 is handled by bounds
    bounds = [(0, None)] * N + [(1e-6, None)]
    
    if time_remaining() > 20.0:
        try:
            res = minimize(
                penalty_objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.success or res.fun < 1e10:
                h_opt = res.x[:N]
                t_opt = res.x[N]
                # Scale back: the score is scale-invariant in a specific way
                # Actually, let's just evaluate the score directly
                s_opt = _compute_score(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Phase 3: Local refinement with random perturbations
    if time_remaining() > 15.0:
        for _ in range(50):
            if time_remaining() < 15.0:
                break
            h_pert = best_h.copy()
            # Perturb a few random indices
            idxs = np.random.choice(N, size=3, replace=False)
            for idx in idxs:
                h_pert[idx] *= (1.0 + 0.3 * np.random.randn())
                h_pert[idx] = max(0, h_pert[idx])
            
            s_pert = _compute_score(h_pert)
            if s_pert > best_score:
                best_score = s_pert
                best_h = h_pert.copy()
                
                # Refine with BCD
                h_refined = block_coordinate_descent(h_pert, max_iter=100)
                s_refined = _compute_score(h_refined)
                if s_refined > best_score:
                    best_score = s_refined
                    best_h = h_refined.copy()
    
    # Ensure nonnegativity and return as list
    best_h = np.clip(best_h, 0, None)
    return list(best_h)
