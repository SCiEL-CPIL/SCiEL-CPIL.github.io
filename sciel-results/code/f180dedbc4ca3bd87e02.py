import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 4, 4, 5
    
    def time_remaining():
        return deadline - time.time()
    
    def verify(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        if U_arr.ndim != 2 or V_arr.ndim != 2 or W_arr.ndim != 2:
            return False
        
        rank = U_arr.shape[1]
        if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
            return False
        
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        if not np.allclose(T, 1.0, atol=1e-8):
            return False
        return True
    
    def get_schoolbook():
        schoolbook_rank = n * m * p
        U = np.zeros((n * m, schoolbook_rank), dtype=float)
        V = np.zeros((m * p, schoolbook_rank), dtype=float)
        W = np.zeros((p * n, schoolbook_rank), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Build rank-65 decomposition using recursive Strassen
    U = np.zeros((n * m, 65), dtype=float)
    V = np.zeros((m * p, 65), dtype=float)
    W = np.zeros((p * n, 65), dtype=float)
    
    # 2x2 Strassen factors (U2, V2, W2) of shape (4, 7)
    # U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 0: (A00+A11)(B00+B11) -> C00 + C11
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # Term 1: (A10+A11)B00 -> C10 + (-1)*C11
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # Term 2: A00(B01-B11) -> C01 + C11
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # Term 3: A11(B10-B00) -> C00 + C10
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # Term 4: (A00+A01)B11 -> (-1)*C00 + C01
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # Term 5: (A10-A00)(B00+B01) -> C11
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # Term 6: (A01-A11)(B10+B11) -> C00
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Verify 2x2 Strassen
    U2_3d = U2.reshape(2, 2, 7)
    V2_3d = V2.reshape(2, 2, 7)
    W2_3d = W2.reshape(2, 2, 7)
    T2 = np.einsum('ijr,jkr,kir->ik', U2_3d, V2_3d, W2_3d)
    # T2[i,k] should be 1 if i==k, 0 otherwise (for 2x2 identity)
    # Actually, the tensor T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B gives (i,k) entry of C
    # For 2x2: T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # The decomposition should give T[i,j,k] = delta_{i,j} * delta_{j,k} ... no
    # T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiplication term
    # For matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (each term A[i,j]*B[j,k] contributes to C[i,k])
    # Wait, that's not right either. Let me think again.
    
    # The tensor for matrix multiplication C = AB is:
    # T[i,j,k] = 1 if we're computing C[i,k] and the term is A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k where 0<=i<2, 0<=j<2, 0<=k<2
    # That's 8 terms, but we have rank 7.
    
    # Actually, the tensor is T[i,j,k] = 1 for all (i,j,k) in {0,1}^3
    # No wait. The multiplication tensor has T[i,j,k] = 1 if the (i,j) entry of first matrix 
    # and (j,k) entry of second matrix contribute to (i,k) entry of product.
    # So T[i,j,k] = 1 for all i,j,k (since every A[i,j]*B[j,k] term exists)
    
    # Let me just check: sum_r U2[i,j,r]*V2[j,k,r]*W2[k,i,r] should equal 1 for all i,j,k
    T2_check = np.zeros((2,2,2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for r in range(7):
                    s += U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r]
                T2_check[i,j,k] = s
    
    # T2_check should be all 1s
    if not np.allclose(T2_check, 1.0, atol=1e-10):
        # Fallback to schoolbook
        U_sb, V_sb, W_sb = get_schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Now build 4x4x4 decomposition using 2x2 Strassen on blocks
    # 4x4 matrix A has blocks A00,A01,A10,A11 (each 2x2)
    # 4x4 matrix B has blocks B00,B01,B10,B11 (each 2x2)
    # 4x4 matrix C has blocks C00,C01,C10,C11 (each 2x2)
    
    # Outer Strassen: 7 terms
    # Each outer term s has L_s (4x4) and R_s (4x4)
    # L_s and R_s are linear combinations of the 2x2 blocks of A and B
    
    # For each outer term s, we decompose L_s * R_s using inner 2x2 Strassen
    # This gives 7 inner terms t for each outer term s
    # Total: 7*7 = 49 terms
    
    col = 0
    for s in range(7):
        # Define L_s and R_s as 4x4 matrices of coefficients
        L_s = np.zeros((4, 4))
        R_s = np.zeros((4, 4))
        
        if s == 0:
            # L = A00+A11, R = B00+B11
            L_s[0:2, 0:2] = np.eye(2)  # A00
            L_s[2:4, 2:4] = np.eye(2)  # A11
            R_s[0:2, 0:2] = np.eye(2)  # B00
            R_s[2:4, 2:4] = np.eye(2)  # B11
        elif s == 1:
            # L = A10+A11, R = B00
            L_s[2:4, 0:2] = np.eye(2)  # A10
            L_s[2:4, 2:4] = np.eye(2)  # A11
            R_s[0:2, 0:2] = np.eye(2)  # B00
        elif s == 2:
            # L = A00, R = B01-B11
            L_s[0:2, 0:2] = np.eye(2)  # A00
            R_s[0:2, 2:4] = np.eye(2)  # B01
            R_s[2:4, 2:4] = -np.eye(2)  # -B11
        elif s == 3:
            # L = A11, R = B10-B00
            L_s[2:4, 2:4] = np.eye(2)  # A11
            R_s[2:4, 0:2] = np.eye(2)  # B10
            R_s[0:2, 0:2] = -np.eye(2)  # -B00
        elif s == 4:
            # L = A00+A01, R = B11
            L_s[0:2, 0:2] = np.eye(2)  # A00
            L_s[0:2, 2:4] = np.eye(2)  # A01
            R_s[2:4, 2:4] = np.eye(2)  # B11
        elif s == 5:
            # L = A10-A00, R = B00+B01
            L_s[2:4, 0:2] = np.eye(2)  # A10
            L_s[0:2, 0:2] = -np.eye(2)  # -A00
            R_s[0:2, 0:2] = np.eye(2)  # B00
            R_s[0:2, 2:4] = np.eye(2)  # B01
        elif s == 6:
            # L = A01-A11, R = B10+B11
            L_s[0:2, 2:4] = np.eye(2)  # A01
            L_s[2:4, 2:4] = -np.eye(2)  # -A11
            R_s[2:4, 0:2] = np.eye(2)  # B10
            R_s[2:4, 2:4] = np.eye(2)  # B11
        
        # Decompose L_s * R_s using 2x2 Strassen on blocks
        # L_s has blocks L00,L01,L10,L11 (each 2x2)
        # R_s has blocks R00,R01,R10,R11 (each 2x2)
        
        # Apply 2x2 Strassen: 7 inner terms
        for t in range(7):
            # Inner term t uses U2, V2, W2
            # But we need to map the 2x2 block indices to 4x4 indices
            
            # For inner term t, the U factor uses:
            # U2[i*2+j, t] for block positions (i,j) in the 2x2 block structure
            # Block (i,j) in L_s corresponds to rows 2*i:2*i+2, cols 2*j:2*j+2 in the 4x4 matrix
            
            # U factor for this term:
            for i in range(2):
                for j in range(2):
                    coeff = U2[i*2+j, t]
                    if coeff != 0:
                        # Block (i,j) of L_s is at rows 2*i:2*i+2, cols 2*j:2*j+2
                        for di in range(2):
                            for dj in range(2):
                                a_row = 2*i + di
                                a_col = 2*j + dj
                                U[a_row * m + a_col, col] += coeff * L_s[a_row, a_col]
            
            # V factor for this term:
            for j in range(2):
                for k in range(2):
                    coeff = V2[j*2+k, t]
                    if coeff != 0:
                        # Block (j,k) of R_s is at rows 2*j:2*j+2, cols 2*k:2*k+2
                        for dj in range(2):
                            for dk in range(2):
                                b_row = 2*j + dj
                                b_col = 2*k + dk
                                V[b_row * p + b_col, col] += coeff * R_s[b_row, b_col]
            
            # W factor for this term:
            # W2[k*2+i, t] where (k,i) is the output block position
            # Block (k,i) of C is at rows 2*k:2*k+2, cols 2*i:2*i+2
            # But W index is k*n+i where n=4 for the full 4x4 case
            # Wait, for the 4x4 case, W[k*n+i, r] where n=4
            # Block (k,i) means output block row k, col i
            # So the actual (row, col) in C is (2*k+di, 2*i+dj)
            # W index = col * n + row = (2*i+dj) * 4 + (2*k+di)
            
            for k in range(2):
                for i in range(2):
                    coeff = W2[k*2+i, t]
                    if coeff != 0:
                        # Block (k,i) of C
                        for di in range(2):
                            for dj in range(2):
                                c_row = 2*k + di
                                c_col = 2*i + dj
                                w_idx = c_col * n + c_row
                                W[w_idx, col] += coeff
            
            col += 1
    
    # Add 16 terms for the 5th column (k=4)
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0
            W[4 * n + i, col] = 1.0
            col += 1
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U_sb, V_sb, W_sb = get_schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
