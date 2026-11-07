import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Hybrid basis: Polynomial + Radial Basis Functions (Gaussians)
    # This allows the model to capture both global trends (polynomials)
    # and local features/bumps (RBFs), which pure polynomials might miss
    # if the target function has non-polynomial characteristics.
    
    # Ensure x is 1D array for consistent shape handling
    x = np.asarray(x).flatten()
    
    # Polynomial components (degrees 0-4)
    p0 = np.ones_like(x)
    p1 = x
    p2 = x**2
    p3 = x**3
    p4 = x**4
    
    # RBF components (Gaussian bumps)
    # Fixed centers evenly spaced in [-3, 3]
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    sigma = 1.0
    
    rbf_terms = np.zeros((len(x), 5))
    for i, c in enumerate(centers):
        # exp(-(x-c)^2 / (2*sigma^2))
        rbf_terms[:, i] = np.exp(-((x - c)**2) / (2.0 * sigma**2))
        
    # Combine basis:
    # Indices 0-4: Polynomial coefficients
    # Indices 5-9: RBF weights
    
    # Construct basis matrix
    basis = np.column_stack((
        p0, p1, p2, p3, p4,
        rbf_terms
    ))
    
    return basis @ params
