import numpy as np
import time
from scipy.optimize import minimize
import math

# Hermite Polynomials (Physicist's convention)
def H0(x):
    return 1.0

def H4(x):
    return x**4 - 6*x**2 + 3

def H8(x):
    return x**8 - 28*x**6 + 420*x**4 - 1680*x**2 + 105

def H12(x):
    return x**12 - 42*x**10 + 2100*x**8 - 60480*x**6 + 604800*x**4 - 1632960*x**2 + 10395

def compute_a12(a0, a4, a8):
    # g(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
    return -(a0 + 3*a4 + 105*a8) / 10395.0

def build_quadratic_forms():
    """
    Build the quadratic form matrices for:
    - ||g||_2^2 = v^T B v
    - ||x*g||_2^2 = v^T A v
    - ||g_hat||_2^2 = v^T C v
    
    Using orthonormal Hermite functions:
    psi_n(x) = H_n(x) * exp(-x^2/2) / sqrt(2^n * n! * sqrt(pi))
    
    The coefficients in the orthonormal basis are:
    c_n = a_n * sqrt(2^n * n! * sqrt(pi))
    
    For the uncertainty product, we need:
    B = (||x*g||_2 / ||g||_2) * (||g_hat||_2 / ||g||_2)
    
    But actually, for the standard uncertainty, we want the product of spreads.
    Let me use the ratio approach: minimize (v^T A v / v^T B v) * (v^T C v / v^T B v)
    where A, B, C are 3x3 matrices (since we have 3 free coefficients).
    """
    # Degrees of Hermite polynomials in our combination
    degrees = [0, 4, 8, 12]
    
    # We have 4 coefficients (a0, a4, a8, a12) but a12 is determined by a0, a4, a8
    # So we work in the 3D space of (a0, a4, a8)
    
    # Precompute the orthonormalization factors
    # For H_n: norm factor = sqrt(2^n * n! * sqrt(pi))
    def norm_factor(n):
        return math.sqrt(2**n * math.factorial(n) * math.sqrt(math.pi))
    
    # The g(x) = sum_{n in degrees} a_n * H_n(x) * exp(-x^2/2)
    # In terms of orthonormal Hermite functions: g(x) = sum c_n * psi_n(x)
    # where c_n = a_n * norm_factor(n)
    
    # For the L2 norm: ||g||_2^2 = sum |c_n|^2 = sum a_n^2 * norm_factor(n)^2
    # This is a quadratic form in a = (a0, a4, a8, a12)
    
    # But we need to express in terms of (a0, a4, a8) only, with a12 = f(a0,a4,a8)
    
    # Let's compute the matrices directly using the constraint.
    # a12 = -(a0 + 3*a4 + 105*a8) / 10395
    
    # So g(x) = a0*H0(x)*e^{-x^2/2} + a4*H4(x)*e^{-x^2/2} + a8*H8(x)*e^{-x^2/2} 
    #          + (-(a0 + 3*a4 + 105*a8)/10395)*H12(x)*e^{-x^2/2}
    
    # = a0*(H0(x)*e^{-x^2/2} - H12(x)*e^{-x^2/2}/10395)
    #   + a4*(H4(x)*e^{-x^2/2} - 3*H12(x)*e^{-x^2/2}/10395)
    #   + a8*(H8(x)*e^{-x^2/2} - 105*H12(x)*e^{-x^2/2}/10395)
    
    # Let phi0(x) = (H0(x) - H12(x)/10395) * e^{-x^2/2}
    #     phi4(x) = (H4(x) - 3*H12(x)/10395) * e^{-x^2/2}
    #     phi8(x) = (H8(x) - 105*H12(x)/10395) * e^{-x^2/2}
    
    # Then g(x) = a0*phi0(x) + a4*phi4(x) + a8*phi8(x)
    
    # Now compute the quadratic forms:
    # B_ij = ∫ phi_i(x) * phi_j(x) dx  (for ||g||_2^2)
    # A_ij = ∫ x^2 * phi_i(x) * phi_j(x) dx  (for ||x*g||_2^2)
    # C_ij = ∫ phi_i_hat(x) * phi_j_hat(x) dx  (for ||g_hat||_2^2)
    
    # For the Fourier transform, using the property:
    # F[H_n(x) * e^{-x^2/2}](ξ) = i^n * sqrt(2*pi) * H_n(2*pi*ξ) * e^{-2*pi^2*ξ^2}
    
    # So phi_n_hat(ξ) = F[phi_n(x)](ξ)
    
    # Let's compute these integrals numerically using Gauss-Hermite quadrature
    # or direct integration.
    
    from scipy.integrate import quad
    
    def phi0(x):
        return (H0(x) - H12(x)/10395.0) * np.exp(-0.5*x**2)
    
    def phi4(x):
        return (H4(x) - 3*H12(x)/10395.0) * np.exp(-0.5*x**2)
    
    def phi8(x):
        return (H8(x) - 105*H12(x)/10395.0) * np.exp(-0.5*x**2)
    
    phis = [phi0, phi4, phi8]
    
    # Compute B matrix: B_ij = ∫ phi_i(x) * phi_j(x) dx
    B = np.zeros((3, 3))
    A = np.zeros((3, 3))
    C = np.zeros((3, 3))
    
    for i in range(3):
        for j in range(3):
            # B_ij
            def integrand_B(x):
                return phis[i](x) * phis[j](x)
            B[i,j], _ = quad(integrand_B, -np.inf, np.inf, limit=200)
            
            # A_ij
            def integrand_A(x):
                return x**2 * phis[i](x) * phis[j](x)
            A[i,j], _ = quad(integrand_A, -np.inf, np.inf, limit=200)
            
            # C_ij: need Fourier transforms
            # phi_n_hat(ξ) = F[phi_n(x)](ξ)
            # For phi0: F[(H0 - H12/10395)*e^{-x^2/2}] = sqrt(2pi)*(H0(2pi*ξ) - H12(2pi*ξ)/10395)*e^{-2pi^2*ξ^2}
            # For phi4: F[(H4 - 3H12/10395)*e^{-x^2/2}] = i^4*sqrt(2pi)*(H4(2pi*ξ) - 3*H12(2pi*ξ)/10395)*e^{-2pi^2*ξ^2}
            #   = sqrt(2pi)*(H4(2pi*ξ) - 3*H12(2pi*ξ)/10395)*e^{-2pi^2*ξ^2}
            # For phi8: F[(H8 - 105H12/10395)*e^{-x^2/2}] = i^8*sqrt(2pi)*(H8(2pi*ξ) - 105*H12(2pi*ξ)/10395)*e^{-2pi^2*ξ^2}
            #   = sqrt(2pi)*(H8(2pi*ξ) - 105*H12(2pi*ξ)/10395)*e^{-2pi^2*ξ^2}
            
            def phi0_hat(xi):
                arg = 2*np.pi*xi
                return np.sqrt(2*np.pi) * (H0(arg) - H12(arg)/10395.0) * np.exp(-2*(np.pi**2)*xi**2)
            
            def phi4_hat(xi):
                arg = 2*np.pi*xi
                return np.sqrt(2*np.pi) * (H4(arg) - 3*H12(arg)/10395.0) * np.exp(-2*(np.pi**2)*xi**2)
            
            def phi8_hat(xi):
                arg = 2*np.pi*xi
                return np.sqrt(2*np.pi) * (H8(arg) - 105*H12(arg)/10395.0) * np.exp(-2*(np.pi**2)*xi**2)
            
            phis_hat = [phi0_hat, phi4_hat, phi8_hat]
            
            def integrand_C(xi):
                return phis_hat[i](xi) * phis_hat[j](xi)
            C[i,j], _ = quad(integrand_C, -np.inf, np.inf, limit=200)
    
    return A, B, C

def objective(coeffs):
    a0, a4, a8 = coeffs
    v = np.array([a0, a4, a8])
    
    A, B, C = objective.A, objective.B, objective.C
    
    norm_sq = v @ B @ v
    xnorm_sq = v @ A @ v
    fnorm_sq = v @ C @ v
    
    if norm_sq <= 0 or xnorm_sq < 0 or fnorm_sq < 0:
        return 1e10
    
    # The uncertainty product is (sqrt(xnorm_sq)/sqrt(norm_sq)) * (sqrt(fnorm_sq)/sqrt(norm_sq))
    # = sqrt(xnorm_sq * fnorm_sq) / norm_sq
    return math.sqrt(max(xnorm_sq, 0) * max(fnorm_sq, 0)) / norm_sq

def construct():
    start_time = time.time()
    deadline = start_time + 170.0
    
    # Build the quadratic forms
    A, B, C = build_quadratic_forms()
    objective.A = A
    objective.B = B
    objective.C = C
    
    # Initial guess from seed
    initial_guess = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    best_coeffs = initial_guess
    best_val = objective(initial_guess)
    
    # Optimization 1: Nelder-Mead from seed
    try:
        res = minimize(objective, initial_guess, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
        if time.time() < deadline and res.fun < best_val:
            best_val = res.fun
            best_coeffs = res.x
    except:
        pass
    
    # Optimization 2: L-BFGS-B from best found
    if time.time() < deadline - 60:
        try:
            res2 = minimize(objective, best_coeffs, method='L-BFGS-B', 
                       options={'maxiter': 200})
            if time.time() < deadline and res2.fun < best_val:
                best_val = res2.fun
                best_coeffs = res2.x
        except:
            pass
    
    # Optimization 3: Random restart if time permits
    if time.time() < deadline - 30:
        np.random.seed(42)
        for _ in range(5):
            if time.time() > deadline - 20:
                break
            noise = np.random.normal(0, 0.5, 3)
            guess = best_coeffs + noise
            try:
                res3 = minimize(objective, guess, method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
                if res3.fun < best_val:
                    best_val = res3.fun
                    best_coeffs = res3.x
            except:
                pass
    
    return best_coeffs
