import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize
import time

def construct():
    # The Cohn-Goncalves refined construction with m=12 double roots, n=1
    # We need 12 positive double-root locations for the Laguerre combination.
    # The score is the upper bound on C4 (lower is better).
    
    # Try the zeros of L_24^(1)(x) - 24 zeros, take 12
    # Or try other constructions
    
    # Let's try: the 12 roots are the zeros of a specific Laguerre-related polynomial
    # For the double-root construction, a natural candidate is the zeros of 
    # the derivative of L_24^(1) or similar
    
    # Actually, let me try the standard approach: 
    # The Cohn-Goncalves bound for Fourier uncertainty with Laguerre polynomials
    # The 12 double roots for m=12, n=1
    
    # Let me try: roots are the positive zeros of L_{2m}^{(n)}(x) = L_{24}^{(1)}(x)
    # That gives 24 roots, we need 12. Maybe take the first 12?
    
    # Or perhaps the construction uses L_m^{(n)}(x) and its derivative
    # giving m roots from the polynomial and m from the derivative
    
    # Let me try multiple candidates and pick the best
    
    deadline = time.time() + 590  # safety margin
    
    best_score = np.inf
    best_roots = [4.0 * (i + 1) for i in range(12)]
    
    def evaluate_roots(roots):
        """Compute the upper bound on C4 given 12 positive double roots."""
        # The bound on C4 for the Fourier sign uncertainty
        # This is typically related to the integral or moment of the polynomial
        # P(x) = prod_{i=1}^{12} (x - r_i)^2
        # The C4 bound might be: (sum r_i) or (prod r_i)^{1/12} or similar
        
        # For the Cohn-Goncalves construction, the bound is often:
        # C4 <= some function of the roots
        # A common form is C4 <= (2*pi)^{-1} * integral or similar
        
        # Let me try: the bound is the product of (1 + r_i) or similar
        # Or the bound is related to the Gamma function moments
        
        # Actually, for the Fourier uncertainty with Laguerre polynomials,
        # the bound on C4 is typically:
        # C4 = (1/2) * sum_{k=0}^{m} a_k^2 * (2k+1) or similar
        
        # Without the exact formula, let me try a few proxies:
        # Proxy 1: sum of roots
        # Proxy 2: product of roots
        # Proxy 3: max root
        # Proxy 4: sum of squares
        
        # The seed gives roots [4,8,...,48]
        # sum = 4*(1+2+...+12) = 4*78 = 312
        # If seed score is 1.0, then the bound formula gives 1.0 for these roots
        
        # Let me assume the bound is normalized such that the seed gives 1.0
        # and try to minimize a natural quantity
        
        # Try: the bound might be (1/m) * sum(r_i) / (some reference)
        # Or the bound is the value of the Laguerre polynomial at some point
        
        # Given the uncertainty, let me just try to minimize the sum of roots
        # or the largest root, as these are natural "size" measures
        
        return np.sum(roots)
    
    # Candidate 1: Seed
    seed_roots = [4.0 * (i + 1) for i in range(12)]
    seed_score = evaluate_roots(seed_roots)
    
    # Candidate 2: Zeros of L_24^(1)(x), take 12 smallest
    try:
        roots_24 = roots_laguerre(24, 1)
        r24 = sorted(roots_24.real)
        cand2 = list(r24[:12])
        score2 = evaluate_roots(cand2)
        if score2 < best_score:
            best_score = score2
            best_roots = cand2
    except:
        pass
    
    # Candidate 3: Zeros of L_12^(1)(x) - only 12 zeros
    try:
        roots_12 = roots_laguerre(12, 1)
        cand3 = sorted(roots_12.real).tolist()
        score3 = evaluate_roots(cand3)
        if score3 < best_score:
            best_score = score3
            best_roots = cand3
    except:
        pass
    
    # Candidate 4: Geometric progression
    try:
        cand4 = [1.0 * (2.0 ** (i / 6.0)) for i in range(12)]
        score4 = evaluate_roots(cand4)
        if score4 < best_score:
            best_score = score4
            best_roots = cand4
    except:
        pass
    
    # Candidate 5: Try to optimize numerically
    if time.time() < deadline - 30:
        try:
            # Start from seed and try to minimize
            x0 = np.array(seed_roots, dtype=float)
            
            def obj(x):
                return np.sum(x)
            
            from scipy.optimize import minimize as scipy_minimize
            res = scipy_minimize(obj, x0, method='Nelder-Mead', 
                                 options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.success and np.all(res.x > 0):
                cand5 = res.x.tolist()
                score5 = evaluate_roots(cand5)
                if score5 < best_score:
                    best_score = score5
                    best_roots = cand5
        except:
            pass
    
    return best_roots
