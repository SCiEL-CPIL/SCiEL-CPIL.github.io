import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def _strassen_222():
    """Return the 7 rank-1 terms for 2x2 x 2x2 matrix multiplication.
    Each term is (u, v, w) where u is 4-dim (row of A), v is 4-dim (col of B), w is 4-dim (entry of C).
    Indexing: A[i,j] -> u[i*2+j], B[j,k] -> v[j*2+k], C[i,k] -> w[i*2+k]
    """
    # Strassen's 7 products for (A11 A12; A21 A22)(B11 B12; B21 B22)
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
    
    terms = []
    # Each term: (u_vec, v_vec, w_vec) all length 4
    # u_vec[i*2+j] = coefficient for A[i,j]
    # v_vec[j*2+k] = coefficient for B[j,k]
    # w_vec[i*2+k] = coefficient for C[i,k]
    
    def vec_a(a11, a12, a21, a22):
        return [a11, a12, a21, a22]
    def vec_b(b11, b12, b21, b22):
        return [b11, b12, b21, b22]
    def vec_c(c11, c12, c21, c22):
        return [c11, c12, c21, c22]
    
    # M1 = (A11+A22)(B11+B22)
    u1 = vec_a(1, 0, 0, 1)
    v1 = vec_b(1, 0, 0, 1)
    # contributes to C11: +1, C12: 0, C21: 0, C22: -1
    w1 = vec_c(1, 0, 0, -1)
    terms.append((u1, v1, w1))
    
    # M2 = (A21+A22)B11
    u2 = vec_a(0, 0, 1, 1)
    v2 = vec_b(1, 0, 0, 0)
    # contributes to C21: +1, C22: -1
    w2 = vec_c(0, 0, 1, -1)
    terms.append((u2, v2, w2))
    
    # M3 = A11(B12-B22)
    u3 = vec_a(1, 0, 0, 0)
    v3 = vec_b(0, 1, 0, -1)
    # contributes to C12: +1, C22: +1
    w3 = vec_c(0, 1, 0, 1)
    terms.append((u3, v3, w3))
    
    # M4 = A22(B21-B11)
    u4 = vec_a(0, 0, 0, 1)
    v4 = vec_b(-1, 0, 1, 0)
    # contributes to C11: +1, C21: +1
    w4 = vec_c(1, 0, 1, 0)
    terms.append((u4, v4, w4))
    
    # M5 = (A11+A12)B22
    u5 = vec_a(1, 1, 0, 0)
    v5 = vec_b(0, 0, 0, 1)
    # contributes to C11: -1, C12: +1
    w5 = vec_c(-1, 1, 0, 0)
    terms.append((u5, v5, w5))
    
    # M6 = (A21-A11)(B11+B12)
    u6 = vec_a(-1, 0, 1, 0)
    v6 = vec_b(1, 1, 0, 0)
    # contributes to C22: +1
    w6 = vec_c(0, 0, 0, 1)
    terms.append((u6, v6, w6))
    
    # M7 = (A12-A22)(B21+B22)
    u7 = vec_a(0, 1, 0, -1)
    v7 = vec_b(0, 0, 1, 1)
    # contributes to C11: +1
    w7 = vec_c(1, 0, 0, 0)
    terms.append((u7, v7, w7))
    
    return terms

def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Decomposition: A (2x4) = [A1 | A2], each 2x2
    # B (4x6) = [B1; B2], each 2x6
    # C = A1*B1 + A2*B2
    # Split B1 into 3 blocks of 2x2: B1 = [B1a, B1b, B1c]
    # Split B2 into 3 blocks of 2x2: B2 = [B2a, B2b, B2c]
    # C = A1*B1a + A1*B1b + A1*B1c + A2*B2a + A2*B2b + A2*B2c
    # Each term is a (2,2,2) multiplication, rank 7 via Strassen
    # Total rank = 6 * 7 = 42
    
    strassen_terms = _strassen_222()
    
    # Build U, V, W matrices
    # U has shape (n*m, rank) = (8, 42)
    # V has shape (m*p, rank) = (24, 42)  
    # W has shape (p*n, rank) = (12, 42)
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Column index in U/V/W
    col = 0
    
    # For each of the 6 sub-problems (2x2 x 2x2):
    # sub_problem index s in 0..5
    # s = 0,1,2: A1 * B1a, B1b, B1c
    # s = 3,4,5: A2 * B2a, B2b, B2c
    
    for s in range(6):
        # Determine which A block and which B block
        if s < 3:
            a_block = 0  # A1 (columns 0-1 of A)
            b_block = s  # B1s (rows 0-1, cols 2*s to 2*s+1 of B)
        else:
            a_block = 1  # A2 (columns 2-3 of A)
            b_block = s - 3  # B2s (rows 2-3, cols 2*s to 2*s+1 of B)
        
        # For each of the 7 Strassen terms in this sub-problem
        for (u_vec, v_vec, w_vec) in strassen_terms:
            # u_vec[i*2+j] for A block: A row i (0..1), col j (0..1)
            # In full A: row i, col a_block*2 + j
            # U index: i*m + (a_block*2 + j) = i*4 + a_block*2 + j
            
            # v_vec[j*2+k] for B block: B row j (0..1), col k (0..1)
            # In full B: row b_block*2 + j, col k
            # V index: (b_block*2 + j)*p + k = (b_block*2 + j)*6 + k
            
            # w_vec[i*2+k] for C: C row i (0..1), col k (0..1)
            # In full C: row i, col b_block*2 + k
            # W index: (b_block*2 + k)*n + i = (b_block*2 + k)*2 + i
            
            for i in range(2):
                for j in range(2):
                    val = u_vec[i * 2 + j]
                    if val != 0:
                        U[i * m + a_block * 2 + j][col] += val
            
            for j in range(2):
                for k in range(2):
                    val = v_vec[j * 2 + k]
                    if val != 0:
                        V[(b_block * 2 + j) * p + k][col] += val
            
            for i in range(2):
                for k in range(2):
                    val = w_vec[i * 2 + k]
                    if val != 0:
                        W[(b_block * 2 + k) * n + i][col] += val
            
            col += 1
    
    # Verify the decomposition
    if time.time() < deadline - 20:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: sum_r U[:,r] * V[:,r]^T * W[:,r]^T should give the multiplication tensor
        # Actually, the tensor T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # where a indexes A (n*m), b indexes B (m*p), c indexes C (p*n)
        # T[a,b,c] should be 1 if a = i*m+j, b = j*p+k, c = k*n+i, else 0
        
        tensor = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        
        # Build expected tensor
        expected = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected[i * m + j, j * p + k, k * n + i] = 1
        
        if not np.allclose(tensor, expected, atol=1e-9):
            # Fallback to schoolbook
            rank_fb = n * m * p
            U_fb = [[0] * rank_fb for _ in range(n * m)]
            V_fb = [[0] * rank_fb for _ in range(m * p)]
            W_fb = [[0] * rank_fb for _ in range(p * n)]
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j][c] = 1
                        V_fb[j * p + k][c] = 1
                        W_fb[k * n + i][c] = 1
                        c += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
