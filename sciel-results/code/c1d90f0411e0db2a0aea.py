"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time

def _build_matrices():
    """Build x^2 and p^2 matrices for indices 0, 4, 8, 12."""
    indices = [0, 4, 8, 12]
    n = len(indices)
    M_x2 = np.zeros((n, n))
    M_p2 = np.zeros((n, n))
    
    for i_idx, i in enumerate(indices):
        for j_idx, j in enumerate(indices):
            # x^2 matrix element <psi_i, x^2 psi_j>
            # x^2 psi_j = 0.5 * sqrt(j*(j-1)) psi_{j-2} + (j+0.5) psi_j + 0.5 * sqrt((j+1)*(j+2)) psi_{j+2}
            # Non-zero if i == j, i == j-2, i == j+2
            val_x2 = 0.0
            if i == j:
                val_x2 += (j + 0.5)
            if i == j - 2:
                val_x2 += 0.5 * np.sqrt(j * (j - 1))
            if i == j + 2:
                val_x2 += 0.5 * np.sqrt((j + 1) * (j + 2))
            M_x2[i_idx, j_idx] = val_x2
            
            # p^2 matrix element <psi_i, p^2 psi_j>
            # p^2 psi_j = -0.5 * sqrt(j*(j-1)) psi_{j-2} + (j+0.5) psi_j - 0.5 * sqrt((j+1)*(j+2)) psi_{j+2}
            val_p2 = 0.0
            if i == j:
                val_p2 += (j + 0.5)
            if i == j - 2:
                val_p2 += -0.5 * np.sqrt(j * (j - 1))
            if i == j + 2:
                val_p2 += -0.5 * np.sqrt((j + 1) * (j + 2))
            M_p2[i_idx, j_idx] = val_p2
            
    return M_x2, M_p2

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 150  # 150s out of 180s/210s limit
    
    M_x2, M_p2 = _build_matrices()
    
    # Coefficients for H0, H4, H8, H12
    # Indices: 0->0, 1->4, 2->8, 3->12
    # g(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    # H0(0)=1, H4(0)=-6, H8(0)=210, H12(0)=-55440
    # c0 - 6c4 + 210c8 - 55440c12 = 0
    # c12 = (c0 - 6c4 + 210c8) / 55440
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
            
        theta, phi = params
        c0 = np.sin(theta) * np.cos(phi)
        c4 = np.sin(theta) * np.sin(phi)
        c8 = np.cos(theta)
        
        c12 = (c0 - 6*c4 + 210*c8) / 55440.0
        
        c = np.array([c0, c4, c8, c12])
        
        # Calculate variances
        # <x^2> = c^T M_x2 c
        # <p^2> = c^T M_p2 c
        var_x = np.dot(c, M_x2 @ c)
        var_p = np.dot(c, M_p2 @ c)
        
        # Avoid division by zero or negative values due to numerical error
        if var_x <= 0 or var_p <= 0:
            return 1e10
            
        return np.sqrt(var_x * var_p)

    # Differential Evolution bounds
    bounds = [(0, np.pi), (-np.pi, np.pi)]
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=500, 
            popsize=10, 
            tol=1e-8, 
            seed=42
        )
        
        best_params = result.x
        best_val = result.fun
        
        # Calculate final coefficients
        theta, phi = best_params
        c0 = np.sin(theta) * np.cos(phi)
        c4 = np.sin(theta) * np.sin(phi)
        c8 = np.cos(theta)
        
        # Return the three free coefficients
        return np.array([c0, c4, c8])
        
    except Exception:
        # Fallback to seed values if optimization fails
        return np.array([-113 / 100, 1 / 25, 1 / 3240])
# EVOLVE-BLOCK-END
