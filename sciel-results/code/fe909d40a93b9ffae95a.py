# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv


def rudin_problem10() -> np.ndarray:
    """
    Returns the coefficients for a degree-10 spherical harmonic that minimizes
    the L4/L2 norm ratio. Based on the directive, we reduce the search to a 
    1-dimensional problem by considering a linear combination of the zonal 
    harmonic (m=0) and a sectorial harmonic (m=10), parameterized by alpha.
    """
    k = 10
    coeffs = np.zeros(2 * k + 1, dtype=np.float64)
    
    # We will search for the best alpha in f = Y_0 + alpha * Y_10
    # Y_0 corresponds to index k (m=0)
    # Y_10 corresponds to index 2*k (m=10)
    
    def compute_R(alpha):
        c = np.zeros(2 * k + 1, dtype=np.float64)
        c[k] = 1.0
        c[2 * k] = alpha
        
        # Compute norms using quadrature
        # 70-point Gauss-Legendre in cos(theta)
        x, w = np.polynomial.legendre.leggauss(70)
        # x is in [-1, 1], w are weights
        # theta is from 0 to pi, cos(theta) = x
        # dOmega = dphi dcos(theta)
        # phi is 130 points
        phi_vals = np.linspace(0, 2 * np.pi, 130, endpoint=False)
        dphi = 2 * np.pi / 130
        
        # Evaluate f
        # f = Y_0 + alpha * Y_10
        # Y_0 = N_0 * P_10(cos(theta))
        # Y_10 = sqrt(2) * N_10 * P_10^10(cos(theta)) * cos(10*phi)
        
        # Normalization constants
        # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
        # For m=0: N_0 = sqrt(21/(4*pi) * 10! / 10!) = sqrt(21/(4*pi))
        N_0 = np.sqrt(21.0 / (4.0 * np.pi))
        # For m=10: N_10 = sqrt(21/(4*pi) * 0! / 20!)
        N_10 = np.sqrt(21.0 / (4.0 * np.pi) * (1.0 / np.math.factorial(20)))
        
        # P_10(x)
        P_10_0 = lpmv(0, 10, x)
        # P_10^10(x)
        P_10_10 = lpmv(10, 10, x)
        
        # Y_0 values at grid points
        # Shape (70, 1)
        Y_0_vals = N_0 * P_10_0[:, np.newaxis]
        
        # Y_10 values at grid points
        # Shape (70, 130)
        # cos(10 * phi)
        cos_10_phi = np.cos(10 * phi_vals)[np.newaxis, :]
        Y_10_vals = np.sqrt(2.0) * N_10 * P_10_10[:, np.newaxis] * cos_10_phi
        
        f_vals = Y_0_vals + alpha * Y_10_vals
        
        # Norms
        # Integral over sphere: sum w_i * sum_j f_ij * dphi
        # Note: Gauss-Legendre weights w are for integral over [-1, 1]
        # The integral over sphere is \int_0^{2\pi} \int_{-1}^1 f(x, \phi) dx d\phi
        # Our quadrature: \sum_i w_i \sum_j f(x_i, \phi_j) dphi
        
        f2 = np.sum(f_vals**2 * w[:, np.newaxis]) * dphi
        f4 = np.sum(f_vals**4 * w[:, np.newaxis]) * dphi
        
        norm2 = np.sqrt(f2)
        norm4 = f4 ** 0.25
        
        if norm2 == 0:
            return np.inf
        return norm4 / norm2

    # Optimize alpha
    # We can use a simple grid or scipy.optimize
    # Given the symmetry, alpha=0 is the zonal harmonic.
    # Let's try a few values around 0 and 1.
    
    best_alpha = 0.0
    best_R = compute_R(0.0)
    
    # Grid search
    alphas = np.linspace(-2, 2, 41)
    for a in alphas:
        r = compute_R(a)
        if r < best_R:
            best_R = r
            best_alpha = a
            
    # Refine with scipy if needed, but grid might be enough for this simple case
    # Let's do a finer grid around the best found
    if best_alpha != 0.0:
        alphas_fine = np.linspace(best_alpha - 0.1, best_alpha + 0.1, 101)
        for a in alphas_fine:
            r = compute_R(a)
            if r < best_R:
                best_R = r
                best_alpha = a

    coeffs[k] = 1.0
    coeffs[2 * k] = best_alpha
    
    return coeffs

# EVOLVE-BLOCK-END
