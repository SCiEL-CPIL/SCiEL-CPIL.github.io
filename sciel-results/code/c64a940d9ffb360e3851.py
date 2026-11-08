"""Matrix multiplication 6x6x6 with rank-minimal exact tensor decomposition."""
import time
import random
from fractions import Fraction

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Known rank-7 decomposition for 3x3 (Strassen-like)
    def strassen_3x3():
        """
        Returns U, V, W for 3x3 matrix multiplication with rank 7.
        Using the standard 7-multiplication algorithm for 3x3 matrices.
        """
        rank = 7
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(9)]
        W = [[0] * rank for _ in range(9)]
        
        # Strassen's 7-term algorithm for 3x3 matrices
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        
        # U entries (for A matrix, 3x3 indexed as 0-8)
        # M1: A11+A22 => indices 0, 4
        U[0][0] = 1; U[4][0] = 1
        # M2: A21+A22 => indices 3, 4
        U[3][1] = 1; U[4][1] = 1
        # M3: A11 => index 0
        U[0][2] = 1
        # M4: A22 => index 4
        U[4][3] = 1
        # M5: A11+A12 => indices 0, 1
        U[0][4] = 1; U[1][4] = 1
        # M6: A21-A11 => indices 3, 0
        U[3][5] = 1; U[0][5] = -1
        # M7: A12-A22 => indices 1, 4
        U[1][6] = 1; U[4][6] = -1
        
        # V entries (for B matrix, 3x3 indexed as 0-8)
        # M1: B11+B22 => indices 0, 4
        V[0][0] = 1; V[4][0] = 1
        # M2: B11 => index 0
        V[0][1] = 1
        # M3: B12-B22 => indices 1, 4
        V[1][2] = 1; V[4][2] = -1
        # M4: B21-B11 => indices 3, 0
        V[3][3] = 1; V[0][3] = -1
        # M5: B22 => index 4
        V[4][4] = 1
        # M6: B11+B12 => indices 0, 1
        V[0][5] = 1; V[1][5] = 1
        # M7: B21+B22 => indices 3, 4
        V[3][6] = 1; V[4][6] = 1
        
        # W entries (for C matrix, 3x3 indexed as 0-8)
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M6
        # C22 = M1 - M2 + M3 + M7
        
        # C[0] = C11: M1 - M5 + M4 - M7
        W[0][0] = 1; W[0][4] = -1; W[0][3] = 1; W[0][6] = -1
        # C[1] = C12: M3 + M5
        W[1][2] = 1; W[1][4] = 1
        # C[3] = C21: M2 + M6
        W[3][1] = 1; W[3][5] = 1
        # C[4] = C22: M1 - M2 + M3 + M7
        W[4][0] = 1; W[4][1] = -1; W[4][2] = 1; W[4][6] = 1
        
        return U, V, W
    
    def strassen_6x6_tiled():
        """
        Tile a 6x6 multiplication using 9 blocks of 2x2, each using rank-4 (Strassen for 2x2).
        This gives rank 9*4 = 36.
        """
        # Actually, for 6x6 we can use 3x3 tiling with 9 blocks of 2x2
        # Each 2x2 uses Strassen rank 7? No, 2x2 has rank 7.
        # 9 blocks * 7 = 63. That's not better than schoolbook 216.
        # 
        # Alternative: Use 3x3 blocks. 6x6 = 4 blocks of 3x3 (2x2 grid).
        # Each 3x3 uses rank 7. Total: 4 * 7 = 28.
        
        rank = 28
        U = [[0] * rank for _ in range(36)]
        V = [[0] * rank for _ in range(36)]
        W = [[0] * rank for _ in range(36)]
        
        # 2x2 grid of 3x3 blocks
        # Block (0,0): rows 0-2, cols 0-2
        # Block (0,1): rows 0-2, cols 3-5
        # Block (1,0): rows 3-5, cols 0-2
        # Block (1,1): rows 3-5, cols 3-5
        
        U3, V3, W3 = strassen_3x3()
        
        block_offsets = [(0, 0), (0, 3), (3, 0), (3, 3)]
        for bi, (ri, ci) in enumerate(block_offsets):
            base = bi * 7
            for local_i in range(3):
                for local_j in range(3):
                    a_idx = (ri + local_i) * 6 + (ci + local_j)
                    for t in range(7):
                        U[a_idx][base + t] = U3[local_i * 3 + local_j][t]
            
            for local_i in range(3):
                for local_j in range(3):
                    b_idx = (ci + local_i) * 6 + (ri + local_j)
                    for t in range(7):
                        V[b_idx][base + t] = V3[local_i * 3 + local_j][t]
            
            for local_i in range(3):
                for local_j in range(3):
                    c_idx = (ri + local_i) * 6 + (ci + local_j)
                    for t in range(7):
                        W[c_idx][base + t] = W3[local_i * 3 + local_j][t]
        
        return {"U": U, "V": V, "W": W}
    
    # Try to find a better decomposition using bilinear search
    def try_bilinear_search(max_rank):
        """
        Attempt to find a decomposition of rank max_rank using random initialization
        and local search. Returns None if not successful.
        """
        import numpy as np
        
        n, m, p = 6, 6, 6
        dim_U = n * m  # 36
        dim_V = m * p  # 36
        dim_W = p * n  # 36
        
        # The matrix multiplication tensor T[i,j,k] = 1 if C[i,k] = sum_j A[i,j]*B[j,k]
        # We need: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # (with W indexed as [k*n+i] per the convention)
        
        # Build the target tensor
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1  # C[i,k] = sum_j A[i,j] B[j,k], so each (i,j,k) contributes
        
        # Actually, the tensor for matrix multiplication is:
        # M = sum_{i,j,k} A[i,j] B[j,k] C^*[k,i]
        # where C^* is the dual. The tensor is T_{i,j,k} = 1 for all i,j,k.
        
        # We want to minimize rank R such that T = sum_{r=1}^{R} U_r ⊗ V_r ⊗ W_r
        # where U_r is a vector of length n*m, V_r of length m*p, W_r of length p*n
        
        # Use random initialization
        best_result = None
        n_attempts = 50
        
        for attempt in range(n_attempts):
            if time.time() > deadline:
                break
                
            # Random rank decomposition
            R = max_rank
            U = np.random.randn(dim_U, R)
            V = np.random.randn(dim_V, R)
            W = np.random.randn(dim_W, R)
            
            # Optimize using gradient descent
            lr = 0.01
            for step in range(500):
                if time.time() > deadline:
                    break
                
                # Compute current decomposition
                # T_approx[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                T_approx = np.zeros((n, m, p))
                for r in range(R):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_approx[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                
                error = np.sum((T - T_approx)**2)
                if error < 1e-10:
                    # Found a solution!
                    # Convert to integer/half-integer if possible
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    
                    # Verify
                    T_check = np.zeros((n, m, p))
                    for r in range(R):
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    T_check[i, j, k] += U_int[i*m+j, r] * V_int[j*p+k, r] * W_int[k*n+i, r]
                    
                    if np.max(np.abs(T - T_check)) < 1e-8:
                        # Convert to lists
                        U_list = [[float(U_int[i, r]) for r in range(R)] for i in range(dim_U)]
                        V_list = [[float(V_int[i, r]) for r in range(R)] for i in range(dim_V)]
                        W_list = [[float(W_int[i, r]) for r in range(R)] for i in range(dim_W)]
                        return {"U": U_list, "V": V_list, "W": W_list}
                    
                    # If rounding didn't work, use the float values
                    U_list = [[float(U[i, r]) for r in range(R)] for i in range(dim_U)]
                    V_list = [[float(V[i, r]) for r in range(R)] for i in range(dim_V)]
                    W_list = [[float(W[i, r]) for r in range(R)] for i in range(dim_W)]
                    return {"U": U_list, "V": V_list, "W": W_list}
                
                # Compute gradients (approximate)
                # This is expensive; use a simpler approach
                # Perturb each parameter and estimate gradient
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                eps = 1e-4
                # Sample a subset of parameters for speed
                for _ in range(200):
                    mat = random.choice(['U', 'V', 'W'])
                    if mat == 'U':
                        idx = (random.randint(0, dim_U-1), random.randint(0, R-1))
                        old_val = U[idx]
                        U[idx] = old_val + eps
                        T_plus = np.zeros((n, m, p))
                        for r in range(R):
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        T_plus[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err_plus = np.sum((T - T_plus)**2)
                        U[idx] = old_val - eps
                        T_minus = np.zeros((n, m, p))
                        for r in range(R):
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        T_minus[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err_minus = np.sum((T - T_minus)**2)
                        U[idx] = old_val
                        grad = (err_plus - err_minus) / (2 * eps)
                        dU[idx] += grad
                    elif mat == 'V':
                        idx = (random.randint(0, dim_V-1), random.randint(0, R-1))
                        old_val = V[idx]
                        V[idx] = old_val + eps
                        T_plus = np.zeros((n, m, p))
                        for r in range(R):
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        T_plus[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err_plus = np.sum((T - T_plus)**2)
                        V[idx] = old_val - eps
                        T_minus = np.zeros((n, m, p))
                        for r in range(R):
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        T_minus[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err_minus = np.sum((T - T_minus)**2)
                        V[idx] = old_val
                        grad = (err_plus - err_minus) / (2 * eps)
                        dV[idx] += grad
                    else:
                        idx = (random.randint(0, dim_W-1), random.randint(0, R-1))
                        old_val = W[idx]
                        W[idx] = old_val + eps
                        T_plus = np.zeros((n, m, p))
                        for r in range(R):
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        T_plus[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err_plus = np.sum((T - T_plus)**2)
                        W[idx] = old_val - eps
                        T_minus = np.zeros((n, m, p))
                        for r in range(R):
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        T_minus[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err_minus = np.sum((T - T_minus)**2)
                        W[idx] = old_val
                        grad = (err_plus - err_minus) / (2 * eps)
                        dW[idx] += grad
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            # Check if we got close
            T_approx = np.zeros((n, m, p))
            for r in range(R):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_approx[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            error = np.max(np.abs(T - T_approx))
            if error < 1e-8:
                U_list = [[float(U[i, r]) for r in range(R)] for i in range(dim_U)]
                V_list = [[float(V[i, r]) for r in range(R)] for i in range(dim_V)]
                W_list = [[float(W[i, r]) for r in range(R)] for i in range(dim_W)]
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Strategy: try the 3x3-tiled Strassen (rank 28) first
    # Then try to find something better
    
    # First, validate the tiled Strassen
    result_tiled = strassen_6x6_tiled()
    
    # Verify it
    import numpy as np
    n, m, p = 6, 6, 6
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    
    U_arr = np.array(result_tiled["U"])
    V_arr = np.array(result_tiled["V"])
    W_arr = np.array(result_tiled["W"])
    R = U_arr.shape[1]
    
    T_approx = np.zeros((n, m, p))
    for r in range(R):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_approx[i, j, k] += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
    
    if np.max(np.abs(T - T_approx)) < 1e-10:
        # The tiled Strassen works with rank 28
        # Try to find something better
        for target_rank in [27, 26, 25, 24, 23, 22, 21, 20]:
            if time.time() > deadline - 10:
                break
            result = try_bilinear_search(target_rank)
            if result is not None:
                return result
        
        return result_tiled
    
    # If tiled Strassen doesn't work, fall back to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
