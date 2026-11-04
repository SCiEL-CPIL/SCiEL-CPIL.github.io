import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 55
    rank = 55
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen's algorithm coefficients for 2x2 matrix multiplication
    # Each entry is (coeff_U, coeff_V, coeff_W) for the 7 multiplications
    # We map the 2x2 subproblem indices (0,1) to the global indices.
    
    # For each pair of columns (k0, k1) in B and C
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        base_col = t * 11
        
        # --- Strassen Part (7 ranks) ---
        # Handles A[:, 0:2] * B[0:2, :] -> C[:, :]
        
        # M1: (A00 + A11) * (B00 + B11)
        # U: A[0,0]=1, A[1,1]=1
        # V: B[0,k0]=1, B[1,k1]=1
        # W: C[0,k0]=1, C[1,k1]=1
        # Note: The W coefficients for Strassen are distributed.
        # We need to be careful with the W matrix.
        # The tensor decomposition is sum_r U[:,r] V[:,r] W[:,r]^T
        # For each rank r, we add to W the coefficients that multiply the result.
        
        # Let's define the 7 multiplications explicitly.
        
        # 1. (A00 + A11)(B00 + B11)
        # U: A[0,0], A[1,1]
        # V: B[0,k0], B[1,k1]
        # W: C[0,k0] += 1, C[0,k1] += 0, C[1,k0] += 0, C[1,k1] += 1 ? 
        # Wait, C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for M1 (rank base_col + 0):
        # U[0*3+0] += 1
        # U[1*3+1] += 1
        # V[0*10+k0] += 1
        # V[1*10+k1] += 1
        # W[k0*2+0] += 1  # C00
        # W[k1*2+1] += 1  # C11
        
        # 2. (A10 + A11) B00
        # U: A[1,0], A[1,1]
        # V: B[0,k0]
        # W: C10 += 1, C11 -= 1
        U[1*3+0][base_col + 1] += 1
        U[1*3+1][base_col + 1] += 1
        V[0*10+k0][base_col + 1] += 1
        W[k0*2+1][base_col + 1] += 1  # C10
        W[k1*2+1][base_col + 1] -= 1  # C11
        
        # 3. A00 (B01 - B11)
        # U: A[0,0]
        # V: B[0,k1], B[1,k1]
        # W: C01 += 1, C11 += 1
        U[0*3+0][base_col + 2] += 1
        V[0*10+k1][base_col + 2] += 1
        V[1*10+k1][base_col + 2] -= 1
        W[k1*2+0][base_col + 2] += 1  # C01
        W[k1*2+1][base_col + 2] += 1  # C11
        
        # 4. A11 (B10 - B00)
        # U: A[1,1]
        # V: B[1,k0], B[0,k0]
        # W: C00 += 1, C10 += 1
        U[1*3+1][base_col + 3] += 1
        V[1*10+k0][base_col + 3] += 1
        V[0*10+k0][base_col + 3] -= 1
        W[k0*2+0][base_col + 3] += 1  # C00
        W[k0*2+1][base_col + 3] += 1  # C10
        
        # 5. (A00 + A01) B11
        # U: A[0,0], A[0,1]
        # V: B[1,k1]
        # W: C00 -= 1, C01 += 1
        U[0*3+0][base_col + 4] += 1
        U[0*3+1][base_col + 4] += 1
        V[1*10+k1][base_col + 4] += 1
        W[k0*2+0][base_col + 4] -= 1  # C00
        W[k1*2+0][base_col + 4] += 1  # C01
        
        # 6. (A10 - A00) (B00 + B01)
        # U: A[1,0], A[0,0]
        # V: B[0,k0], B[0,k1]
        # W: C11 += 1
        U[1*3+0][base_col + 5] += 1
        U[0*3+0][base_col + 5] -= 1
        V[0*10+k0][base_col + 5] += 1
        V[0*10+k1][base_col + 5] += 1
        W[k1*2+1][base_col + 5] += 1  # C11
        
        # 7. (A01 - A11) (B10 + B11)
        # U: A[0,1], A[1,1]
        # V: B[1,k0], B[1,k1]
        # W: C00 += 1
        U[0*3+1][base_col + 6] += 1
        U[1*3+1][base_col + 6] -= 1
        V[1*10+k0][base_col + 6] += 1
        V[1*10+k1][base_col + 6] += 1
        W[k0*2+0][base_col + 6] += 1  # C00
        
        # --- Direct Part (4 ranks) ---
        # Handles A[:, 2] * B[2, :] -> C[:, :]
        
        # 8. A02 * B20
        U[0*3+2][base_col + 7] += 1
        V[2*10+k0][base_col + 7] += 1
        W[k0*2+0][base_col + 7] += 1
        
        # 9. A02 * B21
        U[0*3+2][base_col + 8] += 1
        V[2*10+k1][base_col + 8] += 1
        W[k1*2+0][base_col + 8] += 1
        
        # 10. A12 * B20
        U[1*3+2][base_col + 9] += 1
        V[2*10+k0][base_col + 9] += 1
        W[k0*2+1][base_col + 9] += 1
        
        # 11. A12 * B21
        U[1*3+2][base_col + 10] += 1
        V[2*10+k1][base_col + 10] += 1
        W[k1*2+1][base_col + 10] += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
