"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a Generalized Additive Model (GAM)
    with fixed Gaussian Radial Basis Functions (RBFs).
    
    Inputs:
    - b: [1, 5]
    - s: [1, 5]
    - temp: [1, 40]
    - pH: [2, 12]
    
    Parameters (10 total):
    - params[0]: Bias
    - params[1:3]: Coefficients for b RBFs (2 bases)
    - params[3:5]: Coefficients for s RBFs (2 bases)
    - params[5:8]: Coefficients for temp RBFs (3 bases)
    - params[8:10]: Coefficients for pH RBFs (2 bases)
    
    This approach uses fixed centers and widths for the RBFs, making the
    model linear in the parameters and thus convex for BFGS optimization.
    """
    
    # Scaling factors to normalize inputs to roughly [-1, 1]
    # b: center 3, scale 2
    b_scaled = (b - 3.0) / 2.0
    # s: center 3, scale 2
    s_scaled = (s - 3.0) / 2.0
    # temp: center 20.5, scale 10
    temp_scaled = (temp - 20.5) / 10.0
    # pH: center 7, scale 5
    ph_scaled = (pH - 7.0) / 5.0
    
    # Define RBF centers and widths in scaled space
    # Width is chosen to provide sufficient overlap for smooth interpolation
    width = 1.0
    
    # b RBFs: centers at -1.0 and 1.0 (corresponding to b=1 and b=5)
    b_center_1 = -1.0
    b_center_2 = 1.0
    
    # s RBFs: centers at -1.0 and 1.0 (corresponding to s=1 and s=5)
    s_center_1 = -1.0
    s_center_2 = 1.0
    
    # temp RBFs: centers at -1.0, 0.0, 1.0 (corresponding to temp=10.5, 20.5, 30.5)
    t_center_1 = -1.0
    t_center_2 = 0.0
    t_center_3 = 1.0
    
    # pH RBFs: centers at -0.6, 0.6 (corresponding to pH=4, 10)
    # Using 2 bases for pH to save parameters, focusing on the optimal range
    ph_center_1 = -0.6
    ph_center_2 = 0.6
    
    # Calculate Basis Functions
    b_basis_1 = np.exp(-((b_scaled - b_center_1) / width) ** 2)
    b_basis_2 = np.exp(-((b_scaled - b_center_2) / width) ** 2)
    
    s_basis_1 = np.exp(-((s_scaled - s_center_1) / width) ** 2)
    s_basis_2 = np.exp(-((s_scaled - s_center_2) / width) ** 2)
    
    temp_basis_1 = np.exp(-((temp_scaled - t_center_1) / width) ** 2)
    temp_basis_2 = np.exp(-((temp_scaled - t_center_2) / width) ** 2)
    temp_basis_3 = np.exp(-((temp_scaled - t_center_3) / width) ** 2)
    
    ph_basis_1 = np.exp(-((ph_scaled - ph_center_1) / width) ** 2)
    ph_basis_2 = np.exp(-((ph_scaled - ph_center_2) / width) ** 2)
    
    # Combine linearly
    # params[0]: Bias
    # params[1], params[2]: b coefficients
    # params[3], params[4]: s coefficients
    # params[5], params[6], params[7]: temp coefficients
    # params[8], params[9]: pH coefficients
    
    pred = (params[0] +
            params[1] * b_basis_1 + params[2] * b_basis_2 +
            params[3] * s_basis_1 + params[4] * s_basis_2 +
            params[5] * temp_basis_1 + params[6] * temp_basis_2 + params[7] * temp_basis_3 +
            params[8] * ph_basis_1 + params[9] * ph_basis_2)
            
    return pred
# EVOLVE-BLOCK-END
