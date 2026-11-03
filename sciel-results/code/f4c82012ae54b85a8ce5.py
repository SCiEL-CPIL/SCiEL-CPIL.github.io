import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n
    target_dim = n * m * p
    
    T = np.zeros(target_dim, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * (m * p) + j * p + k
                T[idx] = 1.0

    def verify(U, V, W):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        S = np.zeros(target_dim, dtype=float)
        for c in range(rank):
            u = U[:, c]
            v = V[:, c]
            w = W[:, c]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * (m * p) + j * p + k
                        S[idx] += u[i*m+j] * v[j*p+k] * w[k*n+i]
        if not np.allclose(S, T, atol=1e-9):
            return False
        for mat in [U, V, W]:
            for val in mat.flatten():
                val_float = float(val)
                if abs(val_float * 2 - round(val_float * 2)) > 1e-9:
                    return False
        return True

    def build_strassen_decomposition():
        """
        Construct rank-54 decomposition using Strassen on 2x2x2 blocks.
        Partition j: {0,1}, {2,3}, {4}
        Partition k: {0,1}, {2,3}, {4,5}
        6 Strassen blocks (2x2x2) + 3 small blocks (2x1x2) = 6*7 + 3*4 = 54
        """
        # Strassen coefficients for 2x2 matrix multiplication
        # Each product r has: A_coeffs[2][2], B_coeffs[2][2], C_coeffs[2][2]
        # A_coeffs[i'][j'] = coeff of A[i',j'] in the A-part of product r
        # B_coeffs[j'][k'] = coeff of B[j',k'] in the B-part of product r
        # C_coeffs[i'][k'] = coeff of C[i',k'] in the output contribution of product r
        
        strassen_products = [
            # (A_coeffs, B_coeffs, C_coeffs)
            # Product 1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
            ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
            # Product 2: (A10+A11)*B00 -> C10+=1
            ([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,0]]),
            # Product 3: A00*(B01-B11) -> C01+=1
            ([[1,0],[0,0]], [[0,1],[0,-1]], [[0,1],[0,0]]),
            # Product 4: A11*(B10-B00) -> C00+=1, C10+=1
            ([[0,0],[0,1]], [[-1,0],[1,0]], [[1,0],[1,0]]),
            # Product 5: (A00+A01)*B11 -> C00-=1, C01+=1
            ([[1,1],[0,0]], [[0,0],[0,1]], [[-1,1],[0,0]]),
            # Product 6: (A10-A00)*(B00+B01) -> C11+=1
            ([[-1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),
            # Product 7: (A01-A11)*(B10+B11) -> C00+=1
            ([[0,1],[0,-1]], [[0,0],[1,1]], [[1,0],[0,0]]),
        ]
        
        # j-blocks: (j1, j2)
        j_blocks = [(0, 1), (2, 3), (4,)]  # last one is size 1
        # k-blocks: (k1, k2)
        k_blocks = [(0, 1), (2, 3), (4, 5)]
        
        rank = 0
        U_cols = []
        V_cols = []
        W_cols = []
        
        for j_block in j_blocks:
            for k_block in k_blocks:
                if len(j_block) == 2 and len(k_block) == 2:
                    # Strassen block: 2x2x2
                    j1, j2 = j_block
                    k1, k2 = k_block
                    for A_c, B_c, C_c in strassen_products:
                        U_col = np.zeros(N)
                        V_col = np.zeros(M)
                        W_col = np.zeros(P)
                        
                        # Fill U: U[i*m + j]
                        for i_prime in range(2):
                            for j_prime in range(2):
                                j = j1 if j_prime == 0 else j2
                                U_col[i * m + j] = A_c[i_prime][j_prime]
                        
                        # Fill V: V[j*p + k]
                        for j_prime in range(2):
                            for k_prime in range(2):
                                j = j1 if j_prime == 0 else j2
                                k = k1 if k_prime == 0 else k2
                                V_col[j * p + k] = B_c[j_prime][k_prime]
                        
                        # Fill W: W[k*n + i]
                        for i_prime in range(2):
                            for k_prime in range(2):
                                k = k1 if k_prime == 0 else k2
                                W_col[k * n + i_prime] = C_c[i_prime][k_prime]
                        
                        U_cols.append(U_col)
                        V_cols.append(V_col)
                        W_cols.append(W_col)
                        rank += 1
                elif len(j_block) == 1 and len(k_block) == 2:
                    # Small block: 2x1x2, use schoolbook (4 rank-1 tensors)
                    j1 = j_block[0]
                    k1, k2 = k_block
                    for i_prime in range(2):
                        for k_prime in range(2):
                            k = k1 if k_prime == 0 else k2
                            U_col = np.zeros(N)
                            V_col = np.zeros(M)
                            W_col = np.zeros(P)
                            
                            U_col[i_prime * m + j1] = 1
                            V_col[j1 * p + k] = 1
                            W_col[k * n + i_prime] = 1
                            
                            U_cols.append(U_col)
                            V_cols.append(V_col)
                            W_cols.append(W_col)
                            rank += 1
                else:
                    raise ValueError(f"Unexpected block sizes: j={j_block}, k={k_block}")
        
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        
        return U, V, W, rank

    # Try the Strassen-based construction first
    U, V, W, rank = build_strassen_decomposition()
    
    # Verify the construction
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank_fb = n * m * p
    U_fb = np.zeros((N, rank_fb), dtype=int)
    V_fb = np.zeros((M, rank_fb), dtype=int)
    W_fb = np.zeros((P, rank_fb), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
