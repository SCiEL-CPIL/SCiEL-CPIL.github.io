import numpy as np
from scipy.optimize import linear_sum_assignment

def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    
    # Solve the linear assignment problem
    try:
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
    except Exception:
        # Fallback to inverse distance if LAP fails
        return 1.0 / (distance_matrix + 1e-9)
    
    u = np.zeros(n)
    v = np.zeros(n)
    
    for i, j in zip(row_ind, col_ind):
        cost = distance_matrix[i, j]
        u[i] = 0.5 * cost
        v[j] = 0.5 * cost
        
    # Compute reduced costs
    # R_ij = D_ij - u_i - v_j
    R = distance_matrix - u[:, np.newaxis] - v[np.newaxis, :]
    
    # Clip negative reduced costs to 0
    # This ensures matched edges have R=0 (if the dual is tight)
    # And prevents H > 1 if R < 0
    R = np.maximum(R, 0)
    
    # Scale the reduced costs to create a reasonable gradient
    # We want exp(-R/scale) to vary meaningfully
    # If R is small, use a small scale. If R is large, use a larger scale.
    # Let's use the mean of positive R as a characteristic scale.
    positive_R = R[R > 0]
    if len(positive_R) > 0:
        scale = np.mean(positive_R)
        # Ensure scale is not too small to avoid huge gradients
        if scale < 1e-6:
            scale = 1.0
    else:
        scale = 1.0
        
    H = np.exp(-R / scale)
    
    # Ensure positive values
    H = np.where(H > 0, H, 1e-9)
    
    return H
