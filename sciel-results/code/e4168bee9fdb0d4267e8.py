import numpy as np
import scipy.optimize as so
import scipy.special as sp
import scipy.integrate as si
import time

# Hermite polynomials (physicists')
def H(n, x):
    return sp.eval_hermite(n, x)

def construct():
    start_time = time.time()
    deadline = 180.0  # Safety margin before 210s kill
    
    # Coefficients for H0, H4, H8
    # c12 is derived from g(0)=0
    # g(x) = (c0 H0 + c4 H4 + c8 H8 + c12 H12) * exp(-x^2/2)
    # H0(0)=1, H4(0)=-12, H8(0)=1344, H12(0)=-103968
    # c0 - 12 c4 + 1344 c8 - 103968 c12 = 0
    # c12 = (c0 - 12 c4 + 1344 c8) / 103968
    
    def get_c12(c0, c4, c8):
        return (c0 - 12 * c4 + 1344 * c8) / 103968.0

    def g(x, c0, c4, c8):
        c12 = get_c12(c0, c4, c8)
        poly = c0 * H(0, x) + c4 * H(4, x) + c8 * H(8, x) + c12 * H(12, x)
        return poly * np.exp(-0.5 * x**2)

    def g_hat(x, c0, c4, c8):
        # Fourier transform of P(x) exp(-x^2/2)
        # We can compute this numerically or analytically.
        # Analytical: 
        # FT of exp(-x^2/2) is sqrt(2*pi) exp(-x^2/2)
        # FT of H_n(x) exp(-x^2/2) is (-i)^n sqrt(2*pi) H_n(x) exp(-x^2/2) ? 
        # Actually, the Hermite functions are eigenfunctions of the Fourier transform.
        # phi_n(x) = (2^n n! sqrt(pi))^-1/2 H_n(x) exp(-x^2/2)
        # FT[phi_n] = (-i)^n phi_n
        # So FT[H_n(x) exp(-x^2/2)] = (-i)^n (2^n n! sqrt(pi))^(1/2) phi_n(x) * (-i)^n ? 
        # Let's stick to numerical integration for safety and simplicity, or use the property:
        # \int H_n(x) e^{-x^2/2} e^{-i x \xi} dx = (-i)^n \sqrt{2\pi} H_n(\xi) e^{-\xi^2/2}
        # This is a known identity.
        c12 = get_c12(c0, c4, c8)
        # Sum of c_k (-i)^k sqrt(2*pi) H_k(x) exp(-x^2/2)
        # Note: The standard Fourier transform definition in scipy is 1/sqrt(2*pi) integral...
        # If we use scipy.fft or manual integral, we need to be consistent.
        # Let's use the identity: \hat{g}(\xi) = \frac{1}{\sqrt{2\pi}} \int g(x) e^{-i x \xi} dx
        # \hat{g}(\xi) = \frac{1}{\sqrt{2\pi}} [ c0 (-i)^0 \sqrt{2\pi} H_0(\xi) e^{-\xi^2/2} + ... ]
        # \hat{g}(\xi) = c0 H_0(\xi) e^{-\xi^2/2} + c4 (-i)^4 H_4(\xi) e^{-\xi^2/2} + c8 (-i)^8 H_8(\xi) e^{-\xi^2/2} + c12 (-i)^12 H_12(\xi) e^{-\xi^2/2}
        # (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1.
        # So \hat{g}(\xi) = (c0 H_0(\xi) + c4 H_4(\xi) + c8 H_8(\xi) + c12 H_12(\xi)) e^{-\xi^2/2}
        # This is exactly g(\xi) ! 
        # Wait, is it?
        # H_n is even/odd.
        # If the identity holds, then \hat{g} = g for these even degrees?
        # Let's verify for n=0. \hat{e^{-x^2/2}} = \sqrt{2\pi} e^{-x^2/2} / \sqrt{2\pi} = e^{-x^2/2}. Correct.
        # For n=2. H_2(x) = 4x^2-2.
        # \int (4x^2-2) e^{-x^2/2} e^{-ix\xi} dx.
        # Using derivative property: \int x^2 e^{-x^2/2} e^{-ix\xi} dx = (-i \partial_\xi) \int x e^{-x^2/2} e^{-ix\xi} dx ...
        # It is a known result that Hermite functions are eigenfunctions with eigenvalue (-i)^n.
        # So \hat{g}(\xi) = \sum c_k (-i)^k \sqrt{2\pi} \phi_k(\xi).
        # Since \phi_k = \frac{1}{\sqrt{2^k k! \sqrt{\pi}}} H_k e^{-x^2/2},
        # \hat{g}(\xi) = \sum c_k (-i)^k \sqrt{2\pi} \frac{1}{\sqrt{2^k k! \sqrt{\pi}}} H_k(\xi) e^{-\xi^2/2}
        # This is NOT simply g(\xi) unless c_k are scaled.
        # So we must use the formula.
        
        term0 = c0 * H(0, x)
        term4 = c4 * ((-1j)**4) * H(4, x)
        term8 = c8 * ((-1j)**8) * H(8, x)
        term12 = c12 * ((-1j)**12) * H(12, x)
        
        # Wait, the eigenvalue relation is for the orthonormal basis.
        # Let's just use numerical integration for g_hat to be absolutely sure.
        # It's safer.
        pass

    def compute_objective(c4, c8):
        # c0 is free? No, we have 3 free coefficients c0, c4, c8.
        # But c12 is fixed by c0, c4, c8.
        # We can fix the scale. Let's fix c0 = 1? No, c0 is a variable.
        # The objective is scale invariant. We can set c0=1 and optimize c4, c8?
        # Or optimize c0, c4, c8 and divide by norm squared.
        # Let's optimize c0, c4, c8 directly.
        pass

    # Let's use a simpler parameterization.
    # Since the problem is homogeneous, we can fix one coefficient, e.g., c0=1, 
    # and optimize c4, c8.
    # However, c0 might be 0 at the optimum? Unlikely for this type of problem.
    # Let's use spherical coordinates or just optimize 3 variables with a penalty for large norms?
    # No, just minimize J(c0, c4, c8) = (||g||_1 * ||g_hat||_1) / (||g||_2^2)
    
    def objective(vars):
        c0, c4, c8 = vars
        if time.time() > deadline:
            return np.inf
            
        # Calculate c12
        c12 = get_c12(c0, c4, c8)
        
        # Define g and g_hat
        # For g:
        # g(x) = (c0 H0 + c4 H4 + c8 H8 + c12 H12) exp(-x^2/2)
        
        # Norms
        # ||g||_2^2 = integral g(x)^2 dx
        # Use Gauss-Hermite quadrature for g^2?
        # g(x)^2 = P(x)^2 exp(-x^2).
        # P(x) is degree 12. P(x)^2 is degree 24.
        # Gauss-Hermite with N=13 points integrates polynomials up to degree 25 exactly.
        # So we can use N=13 for ||g||_2^2.
        x_gh, w_gh = np.polynomial.hermite.hermgauss(13)
        # Note: hermgauss weights are for exp(-x^2).
        P_vals = c0 * H(0, x_gh) + c4 * H(4, x_gh) + c8 * H(8, x_gh) + c12 * H(12, x_gh)
        norm2_sq = np.sum(w_gh * P_vals**2)
        
        if norm2_sq < 1e-10:
            return np.inf
            
        # ||g||_1 = integral |g(x)| dx
        # This requires numerical integration.
        def abs_g(x):
            P = c0 * H(0, x) + c4 * H(4, x) + c8 * H(8, x) + c12 * H(12, x)
            return np.abs(P * np.exp(-0.5 * x**2))
        
        norm1, _ = si.quad(abs_g, -np.inf, np.inf, limit=100)
        
        # ||g_hat||_1
        # g_hat(xi) = 1/sqrt(2*pi) integral g(x) exp(-i x xi) dx
        # We can compute g_hat(xi) analytically using the Hermite eigenfunction property.
        # \hat{g}(\xi) = \sum_{k \in \{0,4,8,12\}} c_k (-i)^k \sqrt{2\pi} \phi_k(\xi)
        # where \phi_k(\xi) = \frac{1}{\sqrt{2^k k! \sqrt{\pi}}} H_k(\xi) e^{-\xi^2/2}
        
        def g_hat_func(xi):
            # Sum of terms
            val = 0.0
            for k, ck in zip([0, 4, 8, 12], [c0, c4, c8, c12]):
                # Eigenvalue (-i)^k
                # k=0: 1, k=4: 1, k=8: 1, k=12: 1
                # So all terms are real and positive coefficient for H_k?
                # (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1.
                # So \hat{g}(\xi) = \sum c_k \sqrt{2\pi} \phi_k(\xi)
                # \phi_k(\xi) = H_k(\xi) e^{-\xi^2/2} / \sqrt{2^k k! \sqrt{\pi}}
                # \hat{g}(\xi) = e^{-\xi^2/2} \sum c_k \sqrt{2\pi} \frac{H_k(\xi)}{\sqrt{2^k k! \sqrt{\pi}}}
                # \sqrt{2\pi} / \sqrt{\pi} = \sqrt{2}
                # \hat{g}(\xi) = \sqrt{2} e^{-\xi^2/2} \sum c_k \frac{H_k(\xi)}{\sqrt{2^k k!}}
                
                factor = c_k * H(k, xi) / np.sqrt(2**k * math.factorial(k))
                val += factor
            return np.sqrt(2) * val * np.exp(-0.5 * xi**2)

        def abs_g_hat(xi):
            return np.abs(g_hat_func(xi))
            
        norm1_hat, _ = si.quad(abs_g_hat, -np.inf, np.inf, limit=100)
        
        return (norm1 * norm1_hat) / norm2_sq

    # Optimization
    # Initial guess from seed
    c0_init, c4_init, c8_init = -113/100, 1/25, 1/3240
    x0 = np.array([c0_init, c4_init, c8_init])
    
    best_val = np.inf
    best_x = x0
    
    # Try a few random starts and the seed
    starts = [x0]
    # Add some random perturbations
    for _ in range(5):
        pert = np.random.normal(0, 0.1, 3)
        starts.append(x0 + pert)
        
    for s in starts:
        if time.time() > deadline - 10:
            break
        try:
            res = so.minimize(objective, s, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 500})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    # Return the best coefficients found
    return best_x

import math
