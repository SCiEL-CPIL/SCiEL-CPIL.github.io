import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.optimize import minimize

N = 400

def _evaluate(h):
    """Compute the objective value."""
    s = np.sum(h)
    if s == 0:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    max_val = np.max(conv)
    return abs(2 * N * max_val / (s * s))

def _h_from_coeffs(coeffs):
    """
    Reconstruct h from DFT coefficients.
    coeffs: [A0, A1, B1, A2, B2, ...]
    We use a real-valued basis.
    To ensure h is real, we construct it from cos/sin.
    """
    n = np.arange(N)
    h = np.zeros(N)
    
    if len(coeffs) == 0:
        return h
        
    # DC term
    h += coeffs[0]
    
    # Frequency terms
    num_pairs = (len(coeffs) - 1) // 2
    for i in range(num_pairs):
        freq = i + 1
        a = coeffs[1 + 2*i]
        b = coeffs[2 + 2*i]
        h += a * np.cos(2 * np.pi * freq * n / N)
        h += b * np.sin(2 * np.pi * freq * n / N)
        
    return h

def _smooth_max_value(h, tau):
    """
    Smooth approximation of max(conv).
    Using log-sum-exp or similar.
    """
    conv = np.convolve(h, h, mode='full')
    # Soft max: tau * log(sum(exp(conv/tau)))
    # To avoid overflow, subtract max
    c_max = np.max(conv)
    soft_max = tau * np.log(np.sum(np.exp((conv - c_max) / tau))) + c_max
    s = np.sum(h)
    if s == 0:
        return 1e10
    return abs(2 * N * soft_max / (s * s))

def _objective(coeffs, tau=0.1):
    h = _h_from_coeffs(coeffs)
    return _smooth_max_value(h, tau)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    # Search over low frequency components
    # More components allow for more complex shapes
    K_freqs = 10 
    dim = 1 + 2 * K_freqs
    
    bounds = [(-5, 5)] * dim
    
    # Initial population
    init = np.zeros((20, dim))
    init[0, 0] = 1.0 # Seed
    
    # Phase 1: Global search with moderate tau
    # tau controls smoothness. Smaller is better for accuracy, larger for optimization ease.
    # Start with larger tau.
    tau_start = 0.5
    
    try:
        # We need a wrapper to pass tau
        def obj_phase1(x):
            return _objective(x, tau_start)
            
        res = differential_evolution(
            obj_phase1, 
            bounds, 
            maxiter=300, 
            popsize=20, 
            tol=1e-4, 
            init='latinhypercube',
            seed=42,
            polish=False # We will polish later
        )
        
        if res.fun < best_val:
            best_h = _h_from_coeffs(res.x)
            best_val = _evaluate(best_h)
            
    except Exception as e:
        pass
    
    # Phase 2: Local refinement with decreasing tau
    if time.time() < deadline - 10:
        # Use Nelder-Mead on the exact objective? 
        # Or minimize with decreasing tau?
        # Let's try Nelder-Mead on the exact objective from the best point found.
        def obj_exact(x):
            return _evaluate(_h_from_coeffs(x))
            
        x0 = res.x if 'res' in locals() and res.success else np.zeros(dim)
        x0[0] = 1.0 # Keep DC positive-ish
        
        try:
            res2 = minimize(
                obj_exact, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res2.fun < best_val:
                best_h = _h_from_coeffs(res2.x)
                best_val = _evaluate(best_h)
        except Exception:
            pass

    # Phase 3: If time remains, try a second DE run with different seed and smaller tau
    if time.time() < deadline - 5:
        tau_end = 0.1
        def obj_phase2(x):
            return _objective(x, tau_end)
            
        try:
            res3 = differential_evolution(
                obj_phase2, 
                bounds, 
                maxiter=100, 
                popsize=15, 
                tol=1e-5, 
                init='latinhypercube',
                seed=123,
                polish=False
            )
            if res3.fun < best_val:
                best_h = _h_from_coeffs(res3.x)
                best_val = _evaluate(best_h)
                
                # Final polish
                res4 = minimize(
                    obj_exact, 
                    res3.x, 
                    method='Nelder-Mead', 
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res4.fun < best_val:
                    best_h = _h_from_coeffs(res4.x)
                    best_val = _evaluate(best_h)
        except Exception:
            pass

    return best_h

if __name__ == "__main__":
    h = construct()
    print("Value:", _evaluate(h))
    print("Sum:", np.sum(h))
    print("Max Conv:", np.max(np.convolve(h, h)))
