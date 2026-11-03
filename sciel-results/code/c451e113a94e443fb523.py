import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill

def _linear_overlap_values(h):
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs

def _score(h):
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))

def _fix_sum(h):
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # Spectral parameterization: h = 0.5 + sum_{k=1}^{N/2} A_k * cos(2*pi*k*i/N)
    # where A_k >= 0 (power spectrum amplitudes)
    # This ensures the autocorrelation is well-defined and the problem is smooth
    
    K = N // 2  # Number of positive frequency components
    i = np.arange(N)
    
    # Precompute cosine basis: shape (K, N)
    basis = np.zeros((K, N))
    for k in range(1, K+1):
        basis[k-1] = np.cos(2.0 * np.pi * k * i / N)
    
    def h_from_spectrum(A):
        """Convert non-negative amplitude spectrum to time-domain signal."""
        return 0.5 + basis @ A
    
    def _compute_cs_from_spectrum(A):
        """Compute overlap values directly from spectrum."""
        h = h_from_spectrum(A)
        return _linear_overlap_values(h)
    
    def _smooth_max_and_grad(A, beta):
        """Smooth max of overlap values with gradient w.r.t. amplitudes."""
        cs = _compute_cs_from_spectrum(A)
        m = np.max(cs)
        exp_terms = np.exp(beta * (cs - m))
        sum_exp = np.sum(exp_terms)
        w = exp_terms / sum_exp
        val = m + np.log(sum_exp) / beta
        
        # Gradient: d(cs_k)/dA = -2/N * (h * shifted_h)_k
        # For each shift k, the contribution to gradient is:
        # grad_j = -2/N * w_k * sum over appropriate shifts
        # More efficiently: grad = -2/N * sum_k w_k * (basis_k @ (shifted correlation))
        # But let's compute via chain rule through h
        
        # Actually, let's compute gradient numerically or analytically
        # d(cs_k)/d(h_i) = -2/N * (delta_{i,j} for the correlation)
        # cs_k = 2/N * (N/2 - sum_i h_i * h_{i+k})
        # d(cs_k)/d(h_j) = -2/N * (h_{j+k} + h_{j-k}) for circular, but we have linear
        
        # Let's use the direct formula:
        # cs_k = 2/N * sum_i (0.5 - h_i*h_{i+k}) for i=0..N-1-k
        # = 2/N * (N/2 - sum_i h_i * h_{i+k})
        
        # d(cs_k)/d(h_j) = -2/N * (h_{j+k} if j+k < N else 0) - 2/N * (h_{j-k} if j-k >= 0 else 0)
        # Wait, that's not quite right for the linear (non-circular) correlation.
        
        # Let me just compute the gradient through h:
        # grad_A = basis.T @ grad_h
        # where grad_h = sum_k w_k * d(cs_k)/d(h)
        
        grad_h = np.zeros(N)
        for k in range(N):
            if w[k] == 0.0:
                continue
            # d(cs_k)/d(h_j) for j = 0..N-1
            # cs_k = 2/N * (N/2 - sum_{i=0}^{N-1-k} h_i * h_{i+k})
            # d/dh_j of sum_{i=0}^{N-1-k} h_i * h_{i+k}:
            # = h_j if j in [0, N-1-k] (from h_i term) 
            #   + h_j if j-k in [0, N-1-k] i.e. j in [k, N-1] (from h_{i+k} term)
            # So d(cs_k)/d(h_j) = -2/N * (I[j <= N-1-k] * h_{j+k} + I[j >= k] * h_{j-k})
            # Wait, I need to be more careful.
            
            # sum_{i=0}^{N-1-k} h_i * h_{i+k}
            # d/dh_j = sum_i [delta_{i,j} * h_{i+k} + h_i * delta_{i+k,j}]
            # = h_{j+k} (if j+k < N, i.e., j <= N-1-k) 
            #   + h_{j-k} (if j-k >= 0, i.e., j >= k)
            
            factor = -2.0 / N * w[k]
            if k == 0:
                # d/dh_j of h_j*h_j = 2*h_j
                grad_h += factor * 2.0 * h_from_spectrum(A)
            else:
                # First term: h_{j+k} for j = 0..N-1-k
                if N - 1 - k >= 0:
                    grad_h[:N-k] += factor * h_from_spectrum(A)[k:]
                # Second term: h_{j-k} for j = k..N-1
                grad_h[k:] += factor * h_from_spectrum(A)[:N-k]
        
        grad_A = basis @ grad_h
        return val, grad_A
    
    def _run_spectral_optimization(A0, beta, maxiter, ftol):
        nonlocal best_h, best_score
        
        def obj_func(A):
            # Penalize negative amplitudes and out-of-bounds h
            h = h_from_spectrum(A)
            penalty = 0.0
            neg_A = np.maximum(-A, 0)
            if np.any(neg_A > 0):
                penalty += 10.0 * np.sum(neg_A**2)
            if np.any(h < 0) or np.any(h > 1):
                penalty += 10.0 * (np.sum(np.minimum(h, 0)**2) + np.sum(np.maximum(h-1, 0)**2))
            val, _ = _smooth_max_and_grad(A, beta)
            return val + penalty
        
        def jac_func(A):
            val, grad = _smooth_max_and_grad(A, beta)
            # Add penalty gradients
            h = h_from_spectrum(A)
            # d(penalty for h bounds)/dA
            h_pen_grad = np.zeros(N)
            h_pen_grad[h < 0] = -20.0 * h[h < 0]
            h_pen_grad[h > 1] = 20.0 * (h[h > 1] - 1.0)
            # d(penalty for A >= 0)/dA
            A_pen_grad = np.zeros_like(A)
            A_pen_grad[A < 0] = -20.0 * A[A < 0]
            
            total_grad = grad + basis @ h_pen_grad + A_pen_grad
            return total_grad
        
        bounds = [(0.0, 0.5)] * K  # A_k should be non-negative, bounded
        
        try:
            res = minimize(obj_func, A0.copy(), jac=jac_func, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': maxiter, 'ftol': ftol})
            if res.fun < best_score or res.success:
                h_candidate = h_from_spectrum(res.x)
                _update_best(h_candidate)
        except Exception:
            pass
    
    # Multi-start spectral optimization
    num_restarts = 15
    for r in range(num_restarts):
        if time.time() - t0 > _DEADLINE - 30:
            break
        
        if r == 0:
            # Start with small fundamental
            A0 = np.zeros(K)
            A0[0] = 0.05
        elif r == 1:
            # Start with small second harmonic
            A0 = np.zeros(K)
            A0[1] = 0.05
        else:
            # Random small amplitudes
            A0 = np.random.uniform(0.0, 0.05, K)
        
        _run_spectral_optimization(A0, beta=50.0, maxiter=300, ftol=1e-10)
        
        if time.time() - t0 > _DEADLINE - 15:
            break
        
        # Refine with higher beta
        if best_h is not None and time.time() - t0 < _DEADLINE - 5:
            # Get current best spectrum
            H_best = np.fft.rfft(best_h, n=2*N)
            # Extract amplitudes (this is approximate, but good enough for init)
            A_refine = np.zeros(K)
            for k in range(1, K+1):
                A_refine[k-1] = np.abs(H_best[k]) / (N/2)
            A_refine = np.clip(A_refine, 0.0, 0.5)
            _run_spectral_optimization(A_refine, beta=100.0, maxiter=100, ftol=1e-11)
    
    # Final refinement using direct h-space optimization if time permits
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        def _smooth_max_grad_h(h, beta):
            n = len(h)
            cs = _linear_overlap_values(h)
            m = np.max(cs)
            exp_terms = np.exp(beta * (cs - m))
            sum_exp = np.sum(exp_terms)
            w = exp_terms / sum_exp
            val = m + np.log(sum_exp) / beta
            
            grad = np.zeros(n)
            for k in range(n):
                if w[k] == 0.0:
                    continue
                factor = -(2.0 / n) * w[k]
                if k == 0:
                    grad += factor * 2.0 * h
                else:
                    grad[:n-k] += factor * h[k:]
                    grad[k:] += factor * h[:n-k]
            return val, grad
        
        def _run_h_optimization(h0, beta, maxiter, ftol):
            nonlocal best_h, best_score
            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                     'jac': lambda x: np.ones(N)}]
            bounds = [(0.0, 1.0)] * N
            
            def obj_func(x):
                val, _ = _smooth_max_grad_h(x, beta)
                return val
            def jac_func(x):
                _, grad = _smooth_max_grad_h(x, beta)
                return grad
                
            try:
                res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                               bounds=bounds, constraints=cons,
                               options={'maxiter': maxiter, 'ftol': ftol})
                if res.success or res.fun < best_score:
                    _update_best(res.x)
            except Exception:
                pass
        
        _run_h_optimization(best_h, beta=100.0, maxiter=150, ftol=1e-10)
    
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _fix_sum(best_h)
    return best_h
