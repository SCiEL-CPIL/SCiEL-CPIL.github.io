<<<<<<< SEARCH
def _compute_objective_exact(coeffs):
    """Compute objective using adaptive quadrature for exact integrals."""
=======
def _compute_objective_gh(coeffs):
    """Compute objective using Gauss-Hermite quadrature (exact for polynomials)."""
    a0, a4, a8 = coeffs
    
    a12 = (a0 - 12.0*a4 + 1680.0*a8) / 332640.0
    
    # P and P_minus at GH nodes for exp(-2*pi*x^2)
    P_2 = a0*_gh_H0_2 + a4*_gh_H4_2 + a8*_gh_H8_2 + a12*_gh_H12_2
    Pm_2 = a0*_gh_H0_2 + a4*_gh_H4_2 + a8*_gh_H8_2 - a12*_gh_H12_2
    
    # P and P_minus at GH nodes for exp(-4*pi*x^2)
    P_4 = a0*_gh_H0_4 + a4*_gh_H4_4 + a8*_gh_H8_4 + a12*_gh_H12_4
    Pm_4 = a0*_gh_H0_4 + a4*_gh_H4_4 + a8*_gh_H8_4 - a12*_gh_H12_4
    
    norm2_sq = np.dot(P_2*P_2, _gh_w_2)
    norm4_sq = np.dot(P_4*P_4*P_4*P_4, _gh_w_4)
    norm2_sq_hat = np.dot(Pm_2*Pm_2, _gh_w_2)
    norm4_sq_hat = np.dot(Pm_4*Pm_4*Pm_4*Pm_4, _gh_w_4)
    
    if norm2_sq < 1e-12 or norm2_sq_hat < 1e-12:
        return 1e10
        
    obj = (norm4_sq / (norm2_sq * norm2_sq)) * (norm4_sq_hat / (norm2_sq_hat * norm2_sq_hat))
    
    return obj

def _compute_objective_exact(coeffs):
    """Compute objective using adaptive quadrature for exact integrals."""
>>>>>>> REPLACE

