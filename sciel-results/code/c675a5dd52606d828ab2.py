import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2_matrices():
    """
    Returns U, V, W matrices (2x2 x 7) for Strassen's 2x2x2 decomposition.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C00, C11
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C10
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1 # C01
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1 # C00, C10
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[0, 4] = -1 # C01, C00
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C00
    
    return U, V, W

def _construct():
    n, m, p = DIMENSIONS
    # Rank 26 for 3x3x3, total rank 52
    rank = 52
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Helper to add a block
    def add_block(b_offset, c_offset, start_rank):
        # 1. Strassen for A11 B11 -> C11
        U_s, V_s, W_s = _strassen_2x2_matrices()
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    if U_s[i*2+j, r] != 0:
                        U[i*m+j, start_rank+r] = U_s[i*2+j, r]
            for j in range(2):
                for k in range(2):
                    if V_s[j*2+k, r] != 0:
                        V[j*p+(k+b_offset), start_rank+r] = V_s[j*2+k, r]
            for k in range(2):
                for i in range(2):
                    if W_s[k*2+i, r] != 0:
                        W[(k+c_offset)*n+i, start_rank+r] = W_s[k*2+i, r]
                        
        # 2. A12 B21 -> C11 (Outer product)
        # A12 is A[0:2, 2], B21 is B[2, 0:2]
        # C11 is C[0:2, 0:2]
        # Term: sum_{i,j} A[i,2] B[2,j] C[i,j]
        # This is 4 rank-1 terms.
        curr = start_rank + 7
        for i in range(2):
            for j in range(2):
                U[i*m+2, curr] = 1
                V[2*p+(j+b_offset), curr] = 1
                W[(j+c_offset)*n+i, curr] = 1
                curr += 1
                
        # 3. A11 B12 -> C12
        # A11 is A[0:2, 0:2], B12 is B[0:2, 2]
        # C12 is C[0:2, 2]
        # Term: sum_{i,k} A[i,k] B[k,2] C[i,2]
        # This is 4 rank-1 terms.
        for i in range(2):
            for k in range(2):
                U[i*m+k, curr] = 1
                V[k*p+(2+b_offset), curr] = 1
                W[(2+c_offset)*n+i, curr] = 1
                curr += 1
                
        # 4. A12 B22 -> C12
        # A12 is A[0:2, 2], B22 is B[2, 2]
        # C12 is C[0:2, 2]
        # Term: sum_{i} A[i,2] B[2,2] C[i,2]
        # This is 2 rank-1 terms.
        for i in range(2):
            U[i*m+2, curr] = 1
            V[2*p+(2+b_offset), curr] = 1
            W[(2+c_offset)*n+i, curr] = 1
            curr += 1
            
        # 5. A21 B11 -> C21
        # A21 is A[2, 0:2], B11 is B[0:2, 0:2]
        # C21 is C[2, 0:2]
        # Term: sum_{k,j} A[2,k] B[k,j] C[2,j]
        # This is 4 rank-1 terms.
        for k in range(2):
            for j in range(2):
                U[2*m+k, curr] = 1
                V[k*p+(j+b_offset), curr] = 1
                W[(j+c_offset)*n+2, curr] = 1
                curr += 1
                
        # 6. A22 B12 -> C21
        # A22 is A[2, 2], B12 is B[0:2, 2]
        # C21 is C[2, 0:2]
        # Term: sum_{j} A[2,2] B[j,2] C[2,j]
        # This is 2 rank-1 terms.
        for j in range(2):
            U[2*m+2, curr] = 1
            V[j*p+(2+b_offset), curr] = 1
            W[(j+c_offset)*n+2, curr] = 1
            curr += 1
            
        # 7. A21 B21 -> C22
        # A21 is A[2, 0:2], B21 is B[2, 0:2]
        # C22 is C[2, 2]
        # Term: sum_{k} A[2,k] B[2,k] C[2,2]
        # This is 2 rank-1 terms.
        for k in range(2):
            U[2*m+k, curr] = 1
            V[2*p+(k+b_offset), curr] = 1
            W[(2+c_offset)*n+2, curr] = 1
            curr += 1
            
        # 8. A22 B22 -> C22
        # A22 is A[2, 2], B22 is B[2, 2]
        # C22 is C[2, 2]
        # Term: A[2,2] B[2,2] C[2,2]
        # This is 1 rank-1 term.
        U[2*m+2, curr] = 1
        V[2*p+(2+b_offset), curr] = 1
        W[(2+c_offset)*n+2, curr] = 1
        curr += 1
        
        return curr

    # Block 0: cols 0-2
    curr = add_block(0, 0, 0)
    # Block 1: cols 3-5
    curr = add_block(3, 3, curr)
    
    # curr should be 52
    if curr != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {curr}")
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    return _construct()
