import numpy as np
from itertools import combinations
import time

def construct():
    n, m, p = 2, 4, 4
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Build the target tensor
    # T[i, k, j] = 1 if this is a valid (i,k,j) triple for matrix mult
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # So T = sum_{i,k,j} e_i ⊗ e_k ⊗ e_j
    
    # Represent tensor as a 3D array
    T = np.zeros((n, m, p))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T[i, k, j] = 1
    
    # Strassen decomposition (rank 28)
    def get_strassen_decomposition():
        rank = 28
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Strassen 2x2: X (2x2) * Y (2x2) = Z (2x2)
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y01-Y11)
        # M4 = X11(Y10-Y00)
        # M5 = (X00+X01)Y11
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X01-X11)(Y10+Y11)
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        
        def add_strassen(X_rows, X_cols, Y_rows, Y_cols, Z_rows, Z_cols, offset):
            # M1
            U[X_rows[0]*m + X_cols[0]][offset+0] += 1
            U[X_rows[1]*m + X_cols[1]][offset+0] += 1
            V[Y_rows[0]*p + Y_cols[0]][offset+0] += 1
            V[Y_rows[1]*p + Y_cols[1]][offset+0] += 1
            
            # M2
            U[X_rows[1]*m + X_cols[0]][offset+1] += 1
            U[X_rows[1]*m + X_cols[1]][offset+1] += 1
            V[Y_rows[0]*p + Y_cols[0]][offset+1] += 1
            
            # M3
            U[X_rows[0]*m + X_cols[0]][offset+2] += 1
            V[Y_rows[0]*p + Y_cols[1]][offset+2] += 1
            V[Y_rows[1]*p + Y_cols[1]][offset+2] -= 1
            
            # M4
            U[X_rows[1]*m + X_cols[1]][offset+3] += 1
            V[Y_rows[1]*p + Y_cols[0]][offset+3] += 1
            V[Y_rows[0]*p + Y_cols[0]][offset+3] -= 1
            
            # M5
            U[X_rows[0]*m + X_cols[0]][offset+4] += 1
            U[X_rows[0]*m + X_cols[1]][offset+4] += 1
            V[Y_rows[1]*p + Y_cols[1]][offset+4] += 1
            
            # M6
            U[X_rows[1]*m + X_cols[0]][offset+5] += 1
            U[X_rows[0]*m + X_cols[0]][offset+5] -= 1
            V[Y_rows[0]*p + Y_cols[0]][offset+5] += 1
            V[Y_rows[0]*p + Y_cols[1]][offset+5] += 1
            
            # M7
            U[X_rows[0]*m + X_cols[1]][offset+6] += 1
            U[X_rows[1]*m + X_cols[1]][offset+6] -= 1
            V[Y_rows[1]*p + Y_cols[0]][offset+6] += 1
            V[Y_rows[1]*p + Y_cols[1]][offset+6] += 1
            
            # W updates
            # Z00 = M1 + M4 - M5 + M7
            w_idx = Z_cols[0]*n + Z_rows[0]
            W[w_idx][offset+0] += 1
            W[w_idx][offset+3] += 1
            W[w_idx][offset+4] -= 1
            W[w_idx][offset+6] += 1
            
            # Z01 = M3 + M5
            w_idx = Z_cols[1]*n + Z_rows[0]
            W[w_idx][offset+2] += 1
            W[w_idx][offset+4] += 1
            
            # Z10 = M2 + M4
            w_idx = Z_cols[0]*n + Z_rows[1]
            W[w_idx][offset+1] += 1
            W[w_idx][offset+3] += 1
            
            # Z11 = M1 - M2 + M3 + M6
            w_idx = Z_cols[1]*n + Z_rows[1]
            W[w_idx][offset+0] += 1
            W[w_idx][offset+1] -= 1
            W[w_idx][offset+2] += 1
            W[w_idx][offset+5] += 1
        
        # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        add_strassen([0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 0)
        # Block 1: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        add_strassen([0,1], [2,3], [2,3], [0,1], [0,1], [0,1], 7)
        # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        add_strassen([0,1], [0,1], [0,1], [2,3], [0,1], [2,3], 14)
        # Block 3: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
        add_strassen([0,1], [2,3], [2,3], [2,3], [0,1], [2,3], 21)
        
        return U, V, W
    
    U, V, W = get_strassen_decomposition()
    
    # Verify the decomposition
    def verify(U, V, W):
        # Reconstruct tensor: sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # Need to check if sum_r outer(U[:,r], V[:,r]) gives correct W contribution
        # Actually, the tensor is: T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
        # Wait, let me re-read the convention: W[k*n+i, r]
        # So W index is j*n + i where j is col of C, i is row of C
        # U index is i*m + k where i is row of A, k is col of A
        # V index is k*p + j where k is row of B, j is col of B
        
        # T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
        # This should equal 1 for all valid (i,k,j)
        
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    r = i*m*p + k*p + j
                    U[i*m+k, r] = 1
                    V[k*p+j, r] = 1
                    W[j*n+i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by finding linear dependencies
    # The rank-1 tensors are u_r ⊗ v_r ⊗ w_r
    # If we can find a linear combination that sums to zero, we can reduce rank
    
    # Convert to list format for the return
    def to_list(mat):
        return mat.tolist()
    
    # Check time
    if time.time() > deadline:
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # Try to find a rank reduction
    # Strategy: look for pairs of rank-1 terms that can be merged
    # or find a linear dependency among the terms
    
    # For now, return the rank-28 solution
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
