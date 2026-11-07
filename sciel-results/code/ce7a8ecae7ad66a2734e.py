"""
Strategy: Linear-in-parameters Radial Basis Function (RBF) Network.
Instead of a non-linear physics model where parameters appear in exponents or products
(non-convex optimization), we use a fixed set of basis functions.
The model is: sigma = sum_{i=1}^{10} w_i * phi_i(strain, temp)
where phi_i are smooth functions (Gaussian bumps in strain, modulated by temperature).
The parameters w are linear coefficients. This makes the MSE landscape convex in w,
allowing BFGS to find the global minimum reliably from any starting point (including all ones).
The "intelligence" is in the choice of basis functions, which are fixed to capture
the expected shape of a stress-strain curve (elastic slope, yield plateau, hardening).
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a linear combination of 10 pre-defined basis functions.
    Inputs:
        strain: array of strain values [-0.0019, 0.96]
        temp: array of temperature values [0.066, 1.0]
        params: array of 10 weights
    Output:
        array of predicted stress values
    """
    eps = np.asarray(strain, dtype=np.float64)
    T = np.asarray(temp, dtype=np.float64)
    
    # Ensure 1D arrays for consistent broadcasting
    eps = eps.flatten()
    T = T.flatten()
    
    # Define 10 basis functions.
    # We want to capture:
    # 1. Linear elastic region (small eps)
    # 2. Yield point / Plateau (medium eps)
    # 3. Strain hardening (large eps)
    # 4. Temperature dependence (softening with T)
    
    # Basis 0: Linear term (Elastic modulus approximation)
    # We use a smooth linear term that is active everywhere but can be weighted.
    # To allow for a "kink" at yield, we rely on the combination of bases.
    phi0 = eps
    
    # Basis 1: Temperature-modulated linear term
    # Allows the slope to change with temperature
    phi1 = eps * (1.0 - 0.5 * T)
    
    # Basis 2-7: Gaussian bumps centered at various strain points
    # These capture the non-linear hardening and yield plateau shape.
    # Centers chosen to span the strain range [0, 1]
    # Widths chosen to be smooth but distinct
    
    # Center 0.05 (near yield)
    c1 = 0.05
    w1 = 0.05
    phi2 = np.exp(-((eps - c1)**2) / (2 * w1**2))
    
    # Center 0.15
    c2 = 0.15
    w2 = 0.1
    phi3 = np.exp(-((eps - c2)**2) / (2 * w2**2))
    
    # Center 0.3
    c3 = 0.3
    w3 = 0.15
    phi4 = np.exp(-((eps - c3)**2) / (2 * w3**2))
    
    # Center 0.5
    c4 = 0.5
    w4 = 0.2
    phi5 = np.exp(-((eps - c4)**2) / (2 * w4**2))
    
    # Center 0.7
    c5 = 0.7
    w5 = 0.2
    phi6 = np.exp(-((eps - c5)**2) / (2 * w5**2))
    
    # Center 0.9
    c6 = 0.9
    w6 = 0.2
    phi7 = np.exp(-((eps - c6)**2) / (2 * w6**2))
    
    # Basis 8: Temperature interaction with hardening
    # A term that decreases with T for large strains
    # Use a sigmoid or smooth step to isolate high strain region
    # Let's use a smooth power of strain multiplied by T
    # phi8 = (eps / (1 + eps)) * T  # Bounded [0, 1]
    phi8 = (eps**2) * T
    
    # Basis 9: Constant offset (Bias)
    # Useful if the zero point isn't perfectly at 0, or for global shift
    phi9 = np.ones_like(eps)
    
    # Construct design matrix
    # Shape: (N, 10)
    # We can do this via matrix multiplication for speed and clarity
    
    # Let's assemble the basis matrix
    # Note: We must ensure no NaNs or Infs. Exps of negative numbers are fine.
    # The inputs are bounded, so this is safe.
    
    # Instead of concatenating, let's just compute the dot product directly
    # This is faster and cleaner.
    
    # w0..w9 are params[0]..params[9]
    
    sigma = (params[0] * phi0 + 
             params[1] * phi1 + 
             params[2] * phi2 + 
             params[3] * phi3 + 
             params[4] * phi4 + 
             params[5] * phi5 + 
             params[6] * phi6 + 
             params[7] * phi7 + 
             params[8] * phi8 + 
             params[9] * phi9)
             
    # Safety check: ensure finite
    if not np.all(np.isfinite(sigma)):
        sigma = np.where(np.isfinite(sigma), sigma, 0.0)
        
    return sigma
# EVOLVE-BLOCK-END
