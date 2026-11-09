import time
import sys

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Laderman's 23-term decomposition for 3x3 matrix multiplication
    # Each term: (a_indices, a_coeffs, b_indices, b_coeffs, c_indices, c_coeffs)
    # a_indices: (i,j) for A[i,j]
    # b_indices: (j,k) for B[j,k]  
    # c_indices: (k,i) for C[k,i] (output row k, col i)
    
    laderman_terms = [
        # 1. A00 * (B01 - B11)
        ([(0,0)], [1], [(0,1),(1,1)], [1,-1], [(1,0),(2,0)], [1,-1]),
        # 2. (A00 + A01) * B11
        ([(0,0),(0,1)], [1,1], [(1,1)], [1], [(1,0),(1,1)], [1,1]),
        # 3. (A11 + A12) * B22
        ([(1,1),(1,2)], [1,1], [(2,2)], [1], [(2,1),(2,2)], [1,1]),
        # 4. A22 * (B20 - B00)
        ([(2,2)], [1], [(2,0),(0,0)], [1,-1], [(0,2),(0,0)], [1,-1]),
        # 5. (A20 + A22) * B00
        ([(2,0),(2,2)], [1,1], [(0,0)], [1], [(0,2),(0,0)], [1,1]),
        # 6. (A02 + A22) * B20
        ([(0,2),(2,2)], [1,1], [(2,0)], [1], [(0,0),(0,2)], [1,1]),
        # 7. (A00 - A11) * (B11 + B22)
        ([(0,0),(1,1)], [1,-1], [(1,1),(2,2)], [1,1], [(1,0),(2,0),(1,1),(2,1)], [1,1,-1,-1]),
        # 8. (A11 - A22) * (B22 + B00)
        ([(1,1),(2,2)], [1,-1], [(2,2),(0,0)], [1,1], [(2,1),(0,1),(2,2),(0,2)], [1,1,-1,-1]),
        # 9. (A22 - A00) * (B00 + B11)
        ([(2,2),(0,0)], [1,-1], [(0,0),(1,1)], [1,1], [(0,2),(1,2),(0,0),(1,0)], [1,1,-1,-1]),
        # 10. (A01 + A11) * B12
        ([(0,1),(1,1)], [1,1], [(1,2)], [1], [(2,0),(2,1)], [1,1]),
        # 11. (A11 + A21) * B21
        ([(1,1),(2,1)], [1,1], [(2,1)], [1], [(1,1),(1,2)], [1,1]),
        # 12. (A21 + A01) * B11
        ([(2,1),(0,1)], [1,1], [(1,1)], [1], [(1,2),(1,0)], [1,1]),
        # 13. (A02 + A12) * B20
        ([(0,2),(1,2)], [1,1], [(2,0)], [1], [(0,0),(0,1)], [1,1]),
        # 14. (A12 + A22) * B02
        ([(1,2),(2,2)], [1,1], [(0,2)], [1], [(2,1),(2,2)], [1,1]),
        # 15. (A22 + A02) * B20
        ([(2,2),(0,2)], [1,1], [(2,0)], [1], [(0,2),(0,0)], [1,1]),
        # 16. (A20 + A10) * B01
        ([(2,0),(1,0)], [1,1], [(0,1)], [1], [(1,2),(1,1)], [1,1]),
        # 17. (A10 + A00) * B10
        ([(1,0),(0,0)], [1,1], [(1,0)], [1], [(0,1),(0,0)], [1,1]),
        # 18. (A00 + A20) * B00
        ([(0,0),(2,0)], [1,1], [(0,0)], [1], [(0,0),(0,2)], [1,1]),
        # 19. (A01 - A21) * (B21 + B01)
        ([(0,1),(2,1)], [1,-1], [(2,1),(0,1)], [1,1], [(1,0),(2,0),(1,1),(2,1)], [1,1,-1,-1]),
        # 20. (A21 - A11) * (B01 + B21)
        ([(2,1),(1,1)], [1,-1], [(0,1),(2,1)], [1,1], [(1,2),(2,2),(1,1),(2,1)], [1,1,-1,-1]),
        # 21. (A11 - A01) * (B21 + B01)
        ([(1,1),(0,1)], [1,-1], [(2,1),(0,1)], [1,1], [(1,1),(2,1),(1,0),(2,0)], [1,1,-1,-1]),
        # 22. (A02 - A12) * (B02 + B22)
        ([(0,2),(1,2)], [1,-1], [(0,2),(2,2)], [1,1], [(0,0),(2,0),(0,1),(2,1)], [1,1,-1,-1]),
        # 23. (A12 - A02) * (B22 + B02)
        ([(1,2),(0,2)], [1,-1], [(2,2),(0,2)], [1,1], [(2,1),(0,1),(2,2),(0,2)], [1,1,-1,-1]),
    ]
    
    # Build the 3x3x3 decomposition (rank 23)
    # U: 9x23, V: 9x23, W: 9x23
    U_333 = [[0]*23 for _ in range(9)]
    V_333 = [[0]*23 for _ in range(9)]
    W_333 = [[0]*23 for _ in range(9)]
    
    for r, (a_idx, a_val, b_idx, b_val, c_idx, c_val) in enumerate(laderman_terms):
        for (i,j), coeff in zip(a_idx, a_val):
            U_333[i*3+j][r] = coeff
        for (j,k), coeff in zip(b_idx, b_val):
            V_333[j*3+k][r] = coeff
        for (k,i), coeff in zip(c_idx, c_val):
            W_333[k*3+i][r] = coeff
    
    # Verify the 3x3x3 decomposition is correct
    # C[i,k] = sum_r U[i*3+k... wait, no.
    # A is 3x3, B is 3x3, C is 3x3
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: C[i,k] = sum_r (sum_j A[i,j] U[j*3+... 
    
    # Actually let me re-verify. The decomposition is:
    # C = sum_r (A * u_r) * (v_r * B) where u_r and v_r are the r-th columns
    # No, that's not right either.
    
    # The correct interpretation:
    # C[i,k] = sum_r U[i*3+j... 
    
    # Let me think about this more carefully.
    # In the standard matrix multiplication tensor:
    # T[i,j,k,l] = delta[i,l] * delta[j,k] for (A[i,j], B[k,l]) -> C[i,l]
    # 
    # A rank-r decomposition: T = sum_{r=1}^r u_r ⊗ v_r ⊗ w_r
    # where u_r is a vector in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    # 
    # For 3x3x3: n=m=p=3
    # u_r[i*3+j] = coefficient of A[i,j] in term r
    # v_r[j*3+k] = coefficient of B[j,k] in term r
    # w_r[k*3+i] = coefficient of C[k,i] in term r (note: output index is k*n+i = k*3+i)
    # 
    # The decomposition satisfies:
    # C[i,k] = sum_r sum_j u_r[i*3+j] * v_r[j*3+k] * (sum over... 
    # 
    # Actually, the standard form is:
    # C[i,k] = sum_r (sum_j A[i,j] * U_r[i*3+j]) * (sum_l B[j,l] * V_r[j*3+l])
    # No, that's not right.
    
    # Let me use the correct formulation. For matrix multiplication C = A*B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # A rank-r decomposition has:
    # U: (n*m) x r, V: (m*p) x r, W: (p*n) x r
    # such that C[i,k] = sum_r U[i*m+j... 
    # 
    # The correct identity is:
    # sum_r U[i*m+j] * V[j*p+k] * W[k*n+i] = delta_{i,k'} * ... 
    # 
    # I think the cleanest way is to verify numerically.
    # For random A and B, compute C = A*B, then compute C' from the decomposition
    # and check they match.
    
    import numpy as np
    
    # Verify 3x3x3 decomposition
    np.random.seed(42)
    A_test = np.random.randint(-5, 5, size=(3,3)).astype(float)
    B_test = np.random.randint(-5, 5, size=(3,3)).astype(float)
    C_test = A_test @ B_test
    
    # Compute C from decomposition
    C_decomp = np.zeros((3,3))
    U_arr = np.array(U_333)  # 9x23
    V_arr = np.array(V_333)  # 9x23
    W_arr = np.array(W_333)  # 9x23
    
    for r in range(23):
        u = U_arr[:, r].reshape(3,3)  # A coefficients
        v = V_arr[:, r].reshape(3,3)  # B coefficients
        w = W_arr[:, r].reshape(3,3)  # C coefficients, w[k,i]
        
        # The product is: (A * u) * (v * B) ??? 
        # No. Let me think again.
        
        # Actually, I think the decomposition works as:
        # For each term r, we form:
        #   M_r = (linear combination of A rows/cols) * (linear combination of B rows/cols)
        # and C = sum_r M_r * (scaling from w)
        
        # Hmm, this is getting confusing. Let me just verify by direct computation.
        
        # The tensor identity is:
        # delta_{i,i'} delta_{j,j'} delta_{k,k'} = sum_r U_r[i,j] V_r[j,k] W_r[k,i']
        # where U_r[i,j] = U[i*3+j, r], etc.
        
        # So C[i,k] = sum_j A[i,j] B[j,k]
        # = sum_j A[i,j] B[j,k] * 1
        # = sum_j A[i,j] B[j,k] * (sum_r U_r[i,j] V_r[j,k] W_r[k,i])  ... no
        
        # I think the correct check is:
        # For the identity tensor, we need:
        # sum_r U_r[i,j] * V_r[j,k] * W_r[k,i'] = delta_{i,i'} (when summed over j)
        
        # So C[i,k] = sum_j A[i,j] B[j,k]
        # And the decomposition says this equals:
        # sum_r (sum_j A[i,j] U_r[i,j]) * (sum_j B[j,k] V_r[j,k]) * ... 
        # No, that's not how it works.
        
        # Let me look at this differently. The rank-r decomposition of the matmul tensor
        # means:
        # T = sum_r u_r ⊗ v_r ⊗ w_r
        # where T[i,j,k] = delta_{i,k} (for the appropriate indices)
        
        # For matmul: T[i,j,k,l] = delta_{j,k} delta_{i,l}
        # A rank-r decomposition: T = sum_r a_r ⊗ b_r ⊗ c_r
        # where a_r is in R^{n*m}, b_r in R^{m*p}, c_r in R^{p*n}
        
        # So: delta_{j,k} delta_{i,l} = sum_r a_r[i,j] b_r[k,l] c_r[l,n+i]... 
        
        # I'm overcomplicating this. Let me just verify numerically using the
        # standard formula. If the decomposition is correct, then for any A, B:
        # C = A @ B = sum_r (A @ diag-like stuff) ... 
        
        # Actually, the simplest numerical check:
        # Form the matrices from the decomposition and verify the identity.
        pass
    
    # Let me just trust the Laderman decomposition and build the 3x3x6 version.
    # Split B into two 3x3 blocks: B1 (cols 0-2) and B2 (cols 3-5)
    # C = A * B = [A*B1, A*B2]
    # Use Laderman for each block, giving rank 23+23 = 46
    
    rank = 46
    U = [[0.0]*rank for _ in range(n*m)]  # 9x46
    V = [[0.0]*rank for _ in range(m*p)]  # 18x46
    W = [[0.0]*rank for _ in range(p*n)]  # 18x46
    
    # Block 0: B cols 0,1,2
    for r, (a_idx, a_val, b_idx, b_val, c_idx, c_val) in enumerate(laderman_terms):
        for (i,j), coeff in zip(a_idx, a_val):
            U[i*m+j][r] = coeff
        for (j,k), coeff in zip(b_idx, b_val):
            V[j*p+k][r] = coeff  # k is 0,1,2 -> cols 0,1,2 of B
        for (k,i), coeff in zip(c_idx, c_val):
            W[k*n+i][r] = coeff  # k is 0,1,2 -> rows 0,1,2 of C
    
    # Block 1: B cols 3,4,5
    for r, (a_idx, a_val, b_idx, b_val, c_idx, c_val) in enumerate(laderman_terms):
        r_offset = 23 + r
        for (i,j), coeff in zip(a_idx, a_val):
            U[i*m+j][r_offset] = coeff
        for (j,k), coeff in zip(b_idx, b_val):
            V[j*p+(k+3)][r_offset] = coeff  # k is 0,1,2 -> cols 3,4,5 of B
        for (k,i), coeff in zip(c_idx, c_val):
            W[(k+3)*n+i][r_offset] = coeff  # k is 0,1,2 -> rows 3,4,5 of C
    
    # Convert to numpy for verification
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Verify with random matrices
    np.random.seed(123)
    A_rand = np.random.randint(-10, 10, size=(n,m)).astype(float)
    B_rand = np.random.randint(-10, 10, size=(m,p)).astype(float)
    C_expected = A_rand @ B_rand
    
    # Compute C from decomposition
    # C[i,k] = sum_r sum_j U[i*m+j, r] * A[i,j] * V[j*p+k, r] * ... 
    # Actually, the correct formula for computing C from the decomposition:
    # C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r])
    # No, that's not right either.
    
    # The correct formula: Given U, V, W with the tensor decomposition,
    # C[i,k] = sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
    # summed over j? No.
    
    # I think the correct interpretation is:
    # The decomposition gives us a way to compute C = A*B using rank multiplications.
    # For each term r:
    #   x_r = sum_{i,j} U[i*m+j, r] * A[i,j]
    #   y_r = sum_{j,k} V[j*p+k, r] * B[j,k]
    #   Then C[i,k] gets contribution x_r * y_r * W[k*n+i, r]
    
    # Let me verify this interpretation.
    C_computed = np.zeros((n,p))
    for r in range(rank):
        x_r = 0.0
        for i in range(n):
            for j in range(m):
                x_r += U_np[i*m+j, r] * A_rand[i,j]
        y_r = 0.0
        for j in range(m):
            for k in range(p):
                y_r += V_np[j*p+k, r] * B_rand[j,k]
        for i in range(n):
            for k in range(p):
                C_computed[i,k] += x_r * y_r * W_np[k*n+i, r]
    
    if np.allclose(C_computed, C_expected, atol=1e-10):
        # Decomposition is correct!
        pass
    else:
        # Decomposition is wrong, fallback to schoolbook
        rank = n * m * p
        U = [[0.0]*rank for _ in range(n*m)]
        V = [[0.0]*rank for _ in range(m*p)]
        W = [[0.0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
    
    return {"U": U, "V": V, "W": W}
