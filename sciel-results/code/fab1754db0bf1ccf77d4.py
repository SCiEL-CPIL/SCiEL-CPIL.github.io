import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    n, m, p = DIMENSIONS
    
    # Precompute the 2x2x2 Strassen decomposition (rank 7)
    # Returns (U, V, W) where U is 4x7, V is 4x7, W is 4x7
    # U[i,j] means input A[i] at term j
    # V[i,j] means input B[i] at term j
    # W[i,j] means output C[i] at term j
    
    def strassen_222():
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # Term 0: (a+d)(e+h) -> c00 += 1, c11 += 1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # Term 1: (c+d)e -> c10 += 1, c11 -= 1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # Term 2: a(f-h) -> c01 += 1, c11 += 1
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # Term 3: d(g-e) -> c00 += 1, c10 += 1
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # Term 4: (a+b)h -> c00 -= 1, c01 += 1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # Term 5: (c-a)(e+f) -> c11 += 1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # Term 6: (b-d)(g+h) -> c00 += 1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    # Precompute 2x2x1 schoolbook (rank 4)
    def schoolbook_221():
        U = np.zeros((4, 4), dtype=int)
        V = np.zeros((4, 4), dtype=int)
        W = np.zeros((4, 4), dtype=int)
        
        # ae -> c00
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # be -> c10
        U[1, 1] = 1; V[0, 1] = 1; W[2, 1] = 1
        # ce -> c10 (wait, c is row 1 of A)
        # A = [[a,b],[c,d]], B = [[e],[f]] (2x1)
        # C = [[ae, af],[ce, cf]] -- no, B is 2x1 so C is 2x1
        # C = [[a*e], [c*e]] -- no
        # A is 2x2, B is 2x1: C[i,j] = sum_k A[i,k] B[k,j]
        # C[0,0] = a*e + b*f... but B is 2x1, so B[0,0]=e, B[1,0]=f
        # C[0,0] = a*e + b*f? No, B has shape (2,1), so B[k,0]
        # C[i,0] = sum_k A[i,k] B[k,0]
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = a*e + b*f
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] = c*e + d*f
        
        U = np.zeros((4, 4), dtype=int)
        V = np.zeros((4, 4), dtype=int)
        W = np.zeros((4, 4), dtype=int)
        
        # ae -> c00
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # bf -> c00
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        # ce -> c10
        U[2, 2] = 1; V[0, 2] = 1; W[2, 2] = 1
        # df -> c10
        U[3, 3] = 1; V[1, 3] = 1; W[2, 3] = 1
        
        return U, V, W
    
    # Now build the full 2x4x7 decomposition
    # Split the 4 middle dimension into two groups of 2
    # Group 0: cols 0,1 of B, rows 0,1 of A
    # Group 1: cols 2,3 of B, rows 2,3 of A
    
    # For each group, we have a 2x2 block of A and 2x7 block of B
    # Split the 7 output columns into: 0,1,2,3,4,5,6
    # For each group of 2 input columns, we can process 7 output columns
    
    # Strategy: For each of the 2 input groups (j=0,1), process all 7 output cols
    # Each 2x2x7 can be decomposed as:
    #   - 3 blocks of 2x2x2 (for cols 0-1, 2-3, 4-5) using Strassen: 3*7 = 21
    #   - 1 block of 2x2x1 (for col 6) using schoolbook: 4
    # Total per group: 25, total: 50
    
    # But we can try to do better by using 2x2x3 and 2x2x4 decompositions
    
    # Let's first implement the rank-50 baseline
    S_U, S_V, S_W = strassen_222()
    T_U, T_V, T_W = schoolbook_221()
    
    # Build full matrices
    rank_50 = 50
    U50 = np.zeros((n*m, rank_50), dtype=int)  # 8 x 50
    V50 = np.zeros((m*p, rank_50), dtype=int)  # 28 x 50
    W50 = np.zeros((p*n, rank_50), dtype=int)  # 14 x 50
    
    col = 0
    for j0 in [0, 2]:  # input column groups
        # 3 blocks of 2x2x2
        for k0 in [0, 2, 4]:
            for t in range(7):
                # U: rows 0-3 (A[0,0], A[0,1], A[1,0], A[1,1]) -> map to actual indices
                # A group j0: rows 0,1 of A, cols j0, j0+1
                # U[0] = A[0,j0], U[1] = A[0,j0+1], U[2] = A[1,j0], U[3] = A[1,j0+1]
                # Actual index: row*m + col
                for r in range(4):
                    if S_U[r, t] != 0:
                        actual_r = r // 2 * m + (r % 2) + j0
                        U50[actual_r, col] += S_U[r, t]
                
                # V: cols 0-3 of B group -> B[j0+k0, :], B[j0+k0+1, :]
                # V[0] = B[j0, k0], V[1] = B[j0, k0+1], V[2] = B[j0+1, k0], V[3] = B[j0+1, k0+1]
                # Actual index: col*p + row (since B is m x p, index = i*p + j)
                for r in range(4):
                    if S_V[r, t] != 0:
                        b_row = r // 2 + j0  # which row of B (0-3)
                        b_col = r % 2 + k0    # which col of this 2x2 block
                        actual_c = b_col * p + b_row
                        V50[actual_c, col] += S_V[r, t]
                
                # W: output 2x2 -> C[k0: k0+2, 0:2]
                # W[0] = C[k0,0], W[1] = C[k0+1,0], W[2] = C[k0,1], W[3] = C[k0+1,1]
                # Trace convention: W[k*n + i] where k is col, i is row of C
                # So C[k, i] is at index k*n + i
                for r in range(4):
                    if S_W[r, t] != 0:
                        c_col = r % 2 + k0   # output column
                        c_row = r // 2       # output row
                        actual_w = c_col * n + c_row
                        W50[actual_w, col] += S_W[r, t]
                col += 1
        
        # 1 block of 2x2x1 for col 6
        for t in range(4):
            for r in range(4):
                if T_U[r, t] != 0:
                    actual_r = r // 2 * m + (r % 2) + j0
                    U50[actual_r, col] += T_U[r, t]
            for r in range(4):
                if T_V[r, t] != 0:
                    b_row = r // 2 + j0
                    b_col = r % 2  # 0 or 1, but we only have 1 col, so col 6
                    # For 2x2x1, B has 2 rows, 1 col
                    # V[0] = B[j0, 6], V[1] = B[j0+1, 6]
                    # T_V indices: 0,1 for the two rows
                    actual_c = 6 * p + b_row
                    V50[actual_c, col] += T_V[r, t]
            for r in range(4):
                if T_W[r, t] != 0:
                    c_col = 6
                    c_row = r // 2
                    actual_w = c_col * n + c_row
                    W50[actual_w, col] += T_W[r, t]
            col += 1
    
    # Verify rank 50
    def verify(U, V, W, tol=1e-9):
        # Check that for random inputs, the decomposition gives correct result
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute via decomposition
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        
        # Flatten A and B
        A_flat = A.flatten()  # n*m
        B_flat = B.flatten()  # m*p
        
        # Compute sum_k U[:,k] * V[:,k] * W[:,k] outer products
        # C[i,j] = sum_k U[i,k] * V[j,k] * W[k_out,k]
        # Actually: C = sum_k (U[:,k] outer V[:,k]) weighted by W[:,k]
        # C[i,j] = sum_k U[i,k] * V[j,k] * W[k_out,k] where k_out maps to (i,j)
        
        C_computed = np.zeros((n, p))
        for k in range(U.shape[1]):
            u_col = U_mat[:, k]
            v_col = V_mat[:, k]
            w_col = W_mat[:, k]
            # u_col is n*m, v_col is m*p, w_col is p*n
            # C[i,j] += u_col[i*m + something]... 
            # Actually the tensor is: C[i,j] = sum_k A_flat[idx_A] * B_flat[idx_B] * W_coeff
            # Where idx_A = i*m + j_A, idx_B = j_B*p + j
            # u_col has nonzeros at certain A_flat indices
            # v_col has nonzeros at certain B_flat indices
            # w_col has nonzeros at certain C indices
            
            for a_idx in np.nonzero(u_col)[0]:
                for b_idx in np.nonzero(v_col)[0]:
                    for c_idx in np.nonzero(w_col)[0]:
                        i = a_idx // m
                        j = b_idx % p
                        c_i = c_idx % n
                        c_j = c_idx // n
                        if i == c_i and j == c_j:
                            C_computed[i, j] += u_col[a_idx] * v_col[b_idx] * w_col[c_idx]
        
        return np.max(np.abs(C_computed - C_expected)) < tol
    
    # Verify rank 50
    is_valid_50 = verify(U50.tolist(), V50.tolist(), W50.tolist())
    
    if is_valid_50:
        return {"U": U50.tolist(), "V": V50.tolist(), "W": W50.tolist()}
    else:
        # Fallback: return zeros (shouldn't happen)
        rank = 0
        return {"U": [], "V": [], "W": []}
