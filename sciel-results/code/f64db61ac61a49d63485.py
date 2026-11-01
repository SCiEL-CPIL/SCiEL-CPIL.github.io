"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m  # 4
    dim_B = m * p  # 8
    dim_C = p * n  # 8
    
    # Build the target tensor T of shape (dim_A, dim_B, dim_C)
    T = np.zeros((dim_A, dim_B, dim_C))
    for i_n in range(n):
        for i_m in range(m):
            for k_p in range(p):
                i = i_n * m + i_m
                j = i_m * p + k_p
                k = i_n * p + k_p
                T[i, j, k] = 1.0
                
    def verify_decomposition(U, V, W, rank):
        # U: (dim_A, rank), V: (dim_B, rank), W: (dim_C, rank)
        # T_recon[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T_recon - T)
        return err
        
    def try_peeling(rank, time_limit=10):
        """Try to find a rank-`rank` decomposition using ALS."""
        start_time = time.time()
        rng = np.random.RandomState(42)
        
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        # Number of restarts depends on available time
        # We will run restarts until time_limit is reached
        restarts = 0
        while time.time() - start_time < time_limit:
            restarts += 1
            
            # Random initialization
            U = rng.randn(dim_A, rank)
            V = rng.randn(dim_B, rank)
            W = rng.randn(dim_C, rank)
            
            # Normalize columns to prevent scaling issues
            U /= np.linalg.norm(U, axis=0, keepdims=True)
            V /= np.linalg.norm(V, axis=0, keepdims=True)
            W /= np.linalg.norm(W, axis=0, keepdims=True)
            
            prev_err = verify_decomposition(U, V, W, rank)
            
            # Alternating Least Squares
            for step in range(200):
                if time.time() - start_time > time_limit:
                    break
                    
                # Update U: T_flat (dim_A, dim_B*dim_C) = U * (V^T \otimes W^T)^T
                # M_U[r, (j,k)] = V[j,r] * W[k,r]
                M_U = np.einsum('jr,kr->rjk', V, W).reshape(rank, dim_B * dim_C)
                T_flat = T.reshape(dim_A, dim_B * dim_C)
                # Solve U * M_U^T = T_flat  => U = T_flat * (M_U^T)^+
                try:
                    U_new = np.linalg.lstsq(M_U.T, T_flat.T, rcond=None)[0].T
                    U = U_new
                except:
                    pass
                    
                # Update V
                M_V = np.einsum('ir,kr->rik', U, W).reshape(rank, dim_A * dim_C)
                T_flat_V = T.transpose(1, 0, 2).reshape(dim_B, dim_A * dim_C)
                try:
                    V_new = np.linalg.lstsq(M_V.T, T_flat_V.T, rcond=None)[0].T
                    V = V_new
                except:
                    pass
                    
                # Update W
                M_W = np.einsum('ir,jr->rij', U, V).reshape(rank, dim_A * dim_B)
                T_flat_W = T.transpose(2, 0, 1).reshape(dim_C, dim_A * dim_B)
                try:
                    W_new = np.linalg.lstsq(M_W.T, T_flat_W.T, rcond=None)[0].T
                    W = W_new
                except:
                    pass
                    
                err = verify_decomposition(U, V, W, rank)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
                if err < 1e-8:
                    return U, V, W
                if err > prev_err * 1.01:
                    break
                prev_err = err
                
        # Return the best found so far if it's very good
        if best_err < 1e-6:
            return best_U, best_V, best_W
        return None

    # 1. Try to find a rank 13 decomposition (theoretical minimum might be 14, but let's verify)
    # We allocate 30 seconds for this search.
    res_13 = try_peeling(13, time_limit=30)
    if res_13 is not None:
        U, V, W = res_13
        # Check for integer/half-integer property as per task description
        # The task says "integer/half-integer entries".
        # If the solution is not close to these, it might be invalid for the specific checker?
        # The prompt says "exactly decompose... with integer/half-integer entries".
        # If my random search finds a non-integer solution, it might be rejected?
        # "Interface: ... with integer/half-integer entries that exactly decompose..."
        # This suggests the output MUST have these properties.
        # Random ALS usually produces floats.
        # So, if I find a rank 13 solution, I must snap it to half-integers and verify.
        
        def snap_and_verify(U, V, W, rank):
            U_list = [[U[i, r] for r in range(rank)] for i in range(dim_A)]
            V_list = [[V[i, r] for r in range(rank)] for i in range(dim_B)]
            W_list = [[W[i, r] for r in range(rank)] for i in range(dim_C)]
            
            is_valid = True
            for mat in [U_list, V_list, W_list]:
                for row in mat:
                    for val in row:
                        k = round(val * 2)
                        if abs(val - k * 0.5) > 1e-4:
                            is_valid = False
                            break
                    if not is_valid:
                        break
                if not is_valid:
                    break
            
            if is_valid:
                U_list = [[round(val * 2) * 0.5 for val in row] for row in U_list]
                V_list = [[round(val * 2) * 0.5 for val in row] for row in V_list]
                W_list = [[round(val * 2) * 0.5 for val in row] for row in W_list]
                U_np = np.array(U_list)
                V_np = np.array(V_list)
                W_np = np.array(W_list)
                if verify_decomposition(U_np, V_np, W_np, rank) < 1e-6:
                    return U_list, V_list, W_list
            return None

        snapped = snap_and_verify(U, V, W, 13)
        if snapped:
            return {"U": snapped[0], "V": snapped[1], "W": snapped[2]}

    # 2. Fallback to known rank-14 decomposition
    # This is the standard two-Strassen-block decomposition.
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen for B1 (cols 0,1 of B, which correspond to k=0,1)
    # Indices for U (A): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for V (B): 0:(0,0), 1:(0,1), 2:(0,2), 3:(0,3), 4:(1,0), 5:(1,1), 6:(1,2), 7:(1,3)
    # Indices for W (C): 0:(0,0), 1:(0,1), 2:(0,2), 3:(0,3), 4:(1,0), 5:(1,1), 6:(1,2), 7:(1,3)
    
    # Block 1: A * B[:, 0:2]
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1 # Wait, W index for C[0,0] and C[0,3]?
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's map this to our indices.
    # A00=U[0], A01=U[1], A10=U[2], A11=U[3]
    # B00=V[0], B01=V[1], B10=V[4], B11=V[5] (for first 2 cols)
    # C00=W[0], C01=W[1], C10=W[4], C11=W[5] (for first 2 cols)
    
    # M1: (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    # Contribution to C00: +1, C11: +1
    W[0][0] = 1; W[5][0] = 1 
    
    # M2: (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    # Contribution to C10: +1, C11: -1
    W[4][1] = 1; W[5][1] = -1
    
    # M3: A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    # Contribution to C01: +1, C11: +1
    W[1][2] = 1; W[5][2] = 1
    
    # M4: A11(B10-B00)
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    # Contribution to C00: +1, C10: +1
    W[0][3] = 1; W[4][3] = 1
    
    # M5: (A00+A10)B11
    U[0][4] = 1; U[2][4] = 1
    V[5][4] = 1
    # Contribution to C00: -1, C01: +1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # Contribution to C11: +1
    W[5][5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    # Contribution to C00: -1
    W[0][6] = -1

    # Block 2: A * B[:, 2:4]
    # B00=V[2], B01=V[3], B10=V[6], B11=V[7]
    # C00=W[2], C01=W[3], C10=W[6], C11=W[7]
    
    # M8: (A00+A11)(B00+B11)
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[2][7] = 1; W[7][7] = 1 
    
    # M9: (A10+A11)B00
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[6][8] = 1; W[7][8] = -1
    
    # M10: A00(B01-B11)
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[3][9] = 1; W[7][9] = 1
    
    # M11: A11(B10-B00)
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[2][10] = 1; W[6][10] = 1
    
    # M12: (A00+A10)B11
    U[0][11] = 1; U[2][11] = 1
    V[7][11] = 1
    W[2][11] = -1; W[3][11] = 1
    
    # M13: (A10-A00)(B00+B01)
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    # M14: (A01-A11)(B10+B11)
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[2][13] = -1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
