import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    N = 6  # Embedding dimension
    
    # We'll build a 6x6x6 decomposition using recursive Strassen,
    # then project to 5x5x5 by zeroing out index 5.
    
    # For 6x6x6: partition into 3x3 blocks of size 2x2
    # Block-level: 3x3x3 schoolbook (27 terms)
    # Each block op: 2x2x2 Strassen (7 terms)
    # Total 6x6x6 rank: 27 * 7 = 189
    # After projection to 5x5x5, many terms vanish.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term_6(u_coeffs, v_coeffs, w_coeffs):
        """Add a term to the 6x6x6 decomposition."""
        U_col = [0.0] * (N * N)
        V_col = [0.0] * (N * N)
        W_col = [0.0] * (N * N)
        
        for idx, coeff in u_coeffs:
            if 0 <= idx < N * N:
                U_col[idx] = coeff
        for idx, coeff in v_coeffs:
            if 0 <= idx < N * N:
                V_col[idx] = coeff
        for idx, coeff in w_coeffs:
            if 0 <= idx < N * N:
                W_col[idx] = coeff
        return U_col, V_col, W_col
    
    def project_to_5(U_col6, V_col6, W_col6):
        """Project a 6x6x6 term to 5x5x5. Returns None if term is zero."""
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        
        has_content = False
        
        for i6 in range(N):
            for k6 in range(N):
                if i6 < n and k6 < m:
                    U_col[i6 * m + k6] = U_col6[i6 * N + k6]
                    if U_col[i6 * m + k6] != 0:
                        has_content = True
        
        for k6 in range(N):
            for j6 in range(N):
                if k6 < m and j6 < p:
                    V_col[k6 * p + j6] = V_col6[k6 * N + j6]
                    if V_col[k6 * p + j6] != 0:
                        has_content = True
        
        for j6 in range(N):
            for i6 in range(N):
                if j6 < p and i6 < n:
                    W_col[j6 * n + i6] = W_col6[j6 * N + i6]
                    if W_col[j6 * n + i6] != 0:
                        has_content = True
        
        if not has_content:
            return None
        return U_col, V_col, W_col
    
    def map_u_6(block_i, block_k, p_local, q_local):
        """Map to 6x6 U index. Blocks are size 2."""
        row = block_i * 2 + p_local
        col = block_k * 2 + q_local
        return row * N + col
    
    def map_v_6(block_k, block_j, q_local, t_local):
        """Map to 6x6 V index."""
        row = block_k * 2 + q_local
        col = block_j * 2 + t_local
        return row * N + col
    
    def map_w_6(block_i, block_j, p_local, t_local):
        """Map to 6x6 W index. W[k*n+i, r] convention: W[(j)*n + i, ...] wait."""
        # W index: the output C[i,j] = sum_k A[i,k]*B[k,j]
        # W tensor: T[i,k,j] where T[i,k,j] = 1 if... 
        # In the factorization: T[i,k,j] = sum_r U[i,k,r] * V[k,j,r] * W[j,i,r]
        # So W is indexed by (j, i) -> W[j*n + i, r]
        # For 6x6: W index = block_j * 2 + t_local for row, block_i * 2 + p_local for col
        row = block_j * 2 + t_local
        col = block_i * 2 + p_local
        return row * N + col
    
    def strassen_2x2_6(block_i, block_j, block_k):
        """Generate 7 Strassen terms for a 2x2x2 block in the 6x6 space."""
        terms = []
        
        # Term 1: (a00+a11)(b00+b11) -> c00 + c11
        u = [(map_u_6(block_i, block_k, 0, 0), 1), (map_u_6(block_i, block_k, 1, 1), 1)]
        v = [(map_v_6(block_k, block_j, 0, 0), 1), (map_v_6(block_k, block_j, 1, 1), 1)]
        w = [(map_w_6(block_i, block_j, 0, 0), 1), (map_w_6(block_i, block_j, 1, 1), 1)]
        terms.append((u, v, w))
        
        # Term 2: (a10+a11)(b00) -> c10 - c11  ... wait let me use standard Strassen
        # Standard Strassen for C = A*B where A,B,C are 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Term 2: (a10+a11)*b00
        u = [(map_u_6(block_i, block_k, 1, 0), 1), (map_u_6(block_i, block_k, 1, 1), 1)]
        v = [(map_v_6(block_k, block_j, 0, 0), 1)]
        w = [(map_w_6(block_i, block_j, 1, 0), 1), (map_w_6(block_i, block_j, 1, 1), -1)]
        terms.append((u, v, w))
        
        # Term 3: a00*(b01-b11)
        u = [(map_u_6(block_i, block_k, 0, 0), 1)]
        v = [(map_v_6(block_k, block_j, 0, 1), 1), (map_v_6(block_k, block_j, 1, 1), -1)]
        w = [(map_w_6(block_i, block_j, 0, 1), 1), (map_w_6(block_i, block_j, 1, 1), 1)]
        terms.append((u, v, w))
        
        # Term 4: a11*(b10-b00)
        u = [(map_u_6(block_i, block_k, 1, 1), 1)]
        v = [(map_v_6(block_k, block_j, 1, 0), 1), (map_v_6(block_k, block_j, 0, 0), -1)]
        w = [(map_w_6(block_i, block_j, 0, 0), 1), (map_w_6(block_i, block_j, 1, 0), 1)]
        terms.append((u, v, w))
        
        # Term 5: (a00+a01)*b11
        u = [(map_u_6(block_i, block_k, 0, 0), 1), (map_u_6(block_i, block_k, 0, 1), 1)]
        v = [(map_v_6(block_k, block_j, 1, 1), 1)]
        w = [(map_w_6(block_i, block_j, 0, 0), -1), (map_w_6(block_i, block_j, 0, 1), 1)]
        terms.append((u, v, w))
        
        # Term 6: (a10-a00)*(b00+b01)
        u = [(map_u_6(block_i, block_k, 1, 0), 1), (map_u_6(block_i, block_k, 0, 0), -1)]
        v = [(map_v_6(block_k, block_j, 0, 0), 1), (map_v_6(block_k, block_j, 0, 1), 1)]
        w = [(map_w_6(block_i, block_j, 1, 1), 1)]
        terms.append((u, v, w))
        
        # Term 7: (a01-a11)*(b10+b11)
        u = [(map_u_6(block_i, block_k, 0, 1), 1), (map_u_6(block_i, block_k, 1, 1), -1)]
        v = [(map_v_6(block_k, block_j, 1, 0), 1), (map_v_6(block_k, block_j, 1, 1), 1)]
        w = [(map_w_6(block_i, block_j, 0, 0), 1)]
        terms.append((u, v, w))
        
        return terms
    
    # Build 6x6x6 decomposition: 3x3 blocks of 2x2
    # For each block position (bi, bk, bj), apply Strassen
    # This gives 27 * 7 = 189 terms for 6x6x6
    all_terms_6 = []
    for bi in range(3):
        for bk in range(3):
            for bj in range(3):
                terms = strassen_2x2_6(bi, bj, bk)
                all_terms_6.extend(terms)
    
    # Project to 5x5x5
    for u, v, w in all_terms_6:
        U6, V6, W6 = add_term_6(u, v, w)
        projected = project_to_5(U6, V6, W6)
        if projected is not None:
            U_cols.append(projected[0])
            V_cols.append(projected[1])
            W_cols.append(projected[2])
    
    rank = len(U_cols)
    
    # Convert to standard format
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n*m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m*p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
