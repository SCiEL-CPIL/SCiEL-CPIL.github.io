import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 115.0 # 120s limit, 5s margin

def compute_ratio(h):
    h = np.array(h, dtype=np.float64)
    if np.any(h < -1e-9):
        return -1.0
    h = np.clip(h, 0, None)
    
    l1 = np.sum(h)
    if l1 < 1e-12:
        return -1.0
        
    l2_sq = np.sum(h**2)
    
    # Linear convolution
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(np.abs(h_conv))
    
    if max_conv < 1e-12:
        return -1.0
        
    return (l2_sq / (l1 * max_conv))

def construct():
    start_time = time.time()
    
    # Basis functions: DCT-II modes are orthogonal and well-suited for real-valued sequences.
    # We use a subset of modes to keep the optimization dimension manageable.
    # Using all 50 modes allows full representation but might be harder to optimize.
    # Let's try a mix: first, optimize with a small number of modes, then refine.
    
    # Precompute DCT basis matrix
    # DCT-II definition: X[k] = sum_{n=0}^{N-1} x[n] cos(pi/N * (n+0.5) * k)
    # We want h = W @ c, where W is the inverse DCT (or similar orthogonal basis).
    # scipy.fftpack.dct with type=2 is the forward transform.
    # The inverse is (2/N) * dct(h, type=2, norm='ortho')? 
    # Actually, let's just use the standard DCT-II matrix directly for reconstruction.
    # h[n] = sum_k c[k] * phi_k[n]
    # Let's use the normalized DCT-II basis functions.
    
    n = np.arange(N)
    k_vals = np.arange(N)
    # phi_k[n] = sqrt(2/N) * cos(pi/N * (n + 0.5) * k) for k>0, sqrt(1/N) for k=0
    # This forms an orthonormal basis.
    # h = Phi^T @ c (if Phi is N x N with rows as basis vectors? No, columns)
    # Let's construct the matrix B where B[n, k] = basis_k[n].
    # Then h = B @ c.
    
    B = np.zeros((N, N))
    for k in range(N):
        if k == 0:
            B[:, k] = np.sqrt(1.0 / N)
        else:
            B[:, k] = np.sqrt(2.0 / N) * np.cos(np.pi / N * (n + 0.5) * k)
            
    # Objective function in terms of coefficients c
    # h = B @ c
    # We need h >= 0.
    # Ratio R(h) = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    
    def neg_log_ratio_coeffs(c):
        h = B @ c
        # Enforce non-negativity softly? Or hard clip?
        # Hard clip creates non-differentiability. 
        # However, for Nelder-Mead or Powell, non-differentiability is okay.
        # For L-BFGS, we need smoothness.
        # Let's use a softplus-like transformation or just clip and rely on local search.
        # Given the complexity, let's use a parametric family where h is inherently non-negative.
        # h[n] = sum_k a_k * (1 + cos(...)) ? No, that's too restrictive.
        
        # Alternative: Optimize h directly but initialize from DCT modes.
        # The directive says: "parameterizing the heights via a non-negative linear combination of a small set of orthogonal basis functions"
        # If we use h = B @ c, h can be negative.
        # To ensure non-negativity, we can optimize over h directly but use DCT modes to generate good initial points 
        # and constrain the search to the subspace spanned by the first K modes.
        
        # Let's stick to optimizing h directly, but use DCT modes to create a diverse set of initial points 
        # that are "spectrally smooth".
        
        # Actually, the directive suggests parameterizing h via the basis.
        # If we restrict h to the subspace spanned by the first K DCT modes, h might go negative.
        # We can add a large constant offset (DC component) to ensure non-negativity.
        # h[n] = c_0 * phi_0[n] + sum_{k=1}^{K-1} c_k * phi_k[n]
        # Since phi_0 is constant positive, if c_0 is large enough, h >= 0.
        
        # Let's define the objective for a subset of modes K.
        # We'll search over c[0..K-1].
        
        h = B[:, :K] @ c
        # Check non-negativity
        if np.any(h < -1e-6):
            # Penalize negative values heavily
            return 1000.0 + np.sum(np.clip(-h, 0, None)**2)
            
        h = np.clip(h, 0, None)
        l1 = np.sum(h)
        if l1 < 1e-12:
            return 1000.0
            
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(np.abs(h_conv))
        
        if max_conv < 1e-12:
            return 1000.0
            
        ratio = l2_sq / (l1 * max_conv)
        if ratio <= 0:
            return 1000.0
        return -np.log(ratio)

    best_h = np.ones(N) / N
    best_val = compute_ratio(best_h)
    
    # Strategy:
    # 1. Try various numbers of DCT modes K.
    # 2. For each K, run a few random restarts of Nelder-Mead on the coefficients.
    # 3. Refine the best candidates using direct h-optimization (L-BFGS or Nelder-Mead on h).
    
    modes_to_try = [5, 10, 20, 30, 50]
    
    for K in modes_to_try:
        if time.time() > DEADLINE - 10:
            break
            
        # Initial guesses for coefficients
        # 1. Uniform DC
        c0 = np.zeros(K)
        c0[0] = np.sqrt(N) # h = 1/N * N = 1? 
        # If c0[0] = sqrt(N), h[n] = sqrt(1/N)*sqrt(N) = 1. Sum = N. Ratio = N / (N * N) = 1/N.
        
        # 2. Random
        for restart in range(5):
            if time.time() > DEADLINE - 10:
                break
                
            if restart == 0:
                c_start = c0.copy()
            else:
                c_start = np.random.randn(K) * 0.5
                c_start[0] = np.abs(c_start[0]) + 1.0 # Ensure DC is positive to help non-negativity
            
            try:
                res = minimize(neg_log_ratio_coeffs, c_start, args=(K,), method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
                
                c_opt = res.x
                h_opt = B[:, :K] @ c_opt
                h_opt = np.clip(h_opt, 0, None)
                
                if np.sum(h_opt) > 1e-12:
                    val_opt = compute_ratio(h_opt)
                    if val_opt > best_val:
                        best_val = val_opt
                        best_h = h_opt
            except Exception:
                pass

    # Refine the best candidate found so far using direct optimization on h
    # Use L-BFGS-B with bounds [0, None]
    if time.time() < DEADLINE - 5:
        def neg_log_ratio_h(h):
            h = np.clip(h, 0, None)
            l1 = np.sum(h)
            if l1 < 1e-12:
                return 1000.0
            l2_sq = np.sum(h**2)
            h_conv = np.convolve(h, h, mode='full')
            max_conv = np.max(np.abs(h_conv))
            if max_conv < 1e-12:
                return 1000.0
            ratio = l2_sq / (l1 * max_conv)
            if ratio <= 0:
                return 1000.0
            return -np.log(ratio)
        
        # Normalize best_h to sum 1 to start
        h_start = best_h / np.sum(best_h)
        
        try:
            res = minimize(neg_log_ratio_h, h_start, method='L-BFGS-B', 
                           bounds=[(0, None)] * N,
                           options={'maxiter': 500, 'ftol': 1e-12})
            if res.fun < 1000:
                h_opt = res.x
                val_opt = compute_ratio(h_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt
        except Exception:
            pass
            
        # One more Nelder-Mead refinement on h
        if time.time() < DEADLINE - 2:
            try:
                res = minimize(neg_log_ratio_h, best_h, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.fun < 1000:
                    h_opt = res.x
                    h_opt = np.clip(h_opt, 0, None)
                    val_opt = compute_ratio(h_opt)
                    if val_opt > best_val:
                        best_val = val_opt
                        best_h = h_opt
            except Exception:
                pass

    # Final cleanup
    best_h = np.clip(best_h, 0, None)
    if np.sum(best_h) == 0:
        best_h = np.ones(N) / N
        
    return list(best_h)
