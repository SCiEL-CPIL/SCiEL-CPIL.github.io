"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Total standard terms
    total_terms = n * m * p
    # Apply Strassen to two disjoint 2x2x2 blocks: each saves 1 rank
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {0,1}, j in {2,3}, k in {4,5}
    rank = total_terms - 2  # 60 - 2 = 58
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the two disjoint Strassen blocks
    strassen_block1 = {(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)}
    strassen_block2 = {(0,2,4), (0,2,5), (0,3,4), (0,3,5), (1,2,4), (1,2,5), (1,3,4), (1,3,5)}
    strassen_blocks = strassen_block1 | strassen_block2
    
    col = 0
    
    # Add all standard terms except those in the Strassen blocks
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in strassen_blocks:
                    continue
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
                # Time check
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Helper to add a Strassen term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for u_idx, coeff in u_coeffs:
            U[u_idx][col] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx][col] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx][col] += coeff
        col += 1
    
    # Strassen formulas for C = A*B where A is 2x2, B is 2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # --- Block 1: i in {0,1}, j in {0,1}, k in {0,1} ---
    u00, u01 = 0*5+0, 0*5+1
    u10, u11 = 1*5+0, 1*5+1
    v00, v01 = 0*6+0, 0*6+1
    v10, v11 = 1*6+0, 1*6+1
    w00, w01 = 0*2+0, 1*2+0
    w10, w11 = 0*2+1, 1*2+1
    
    # M1
    add_term([(u00, 1), (u11, 1)], [(v00, 1), (v11, 1)], [(w00, 1), (w11, 1)])
    # M2
    add_term([(u10, 1), (u11, 1)], [(v00, 1)], [(w10, 1), (w11, -1)])
    # M3
    add_term([(u00, 1)], [(v01, 1), (v11, -1)], [(w01, 1), (w11, 1)])
    # M4
    add_term([(u11, 1)], [(v10, 1), (v00, -1)], [(w00, 1), (w10, 1)])
    # M5
    add_term([(u00, 1), (u01, 1)], [(v11, 1)], [(w00, -1), (w01, 1)])
    # M6
    add_term([(u10, 1), (u00, -1)], [(v00, 1), (v01, 1)], [(w11, 1)])
    # M7
    add_term([(u01, 1), (u11, -1)], [(v10, 1), (v11, 1)], [(w00, 1)])
    
    # --- Block 2: i in {0,1}, j in {2,3}, k in {4,5} ---
    u20, u21 = 0*5+2, 0*5+3
    u30, u31 = 1*5+2, 1*5+3
    v20, v21 = 2*6+4, 2*6+5
    v30, v31 = 3*6+4, 3*6+5
    w40, w41 = 4*2+0, 5*2+0
    w50, w51 = 4*2+1, 5*2+1
    
    # M1
    add_term([(u20, 1), (u31, 1)], [(v20, 1), (v31, 1)], [(w40, 1), (w51, 1)])
    # M2
    add_term([(u30, 1), (u31, 1)], [(v20, 1)], [(w50, 1), (w51, -1)])
    # M3
    add_term([(u20, 1)], [(v21, 1), (v31, -1)], [(w41, 1), (w51, 1)])
    # M4
    add_term([(u31, 1)], [(v30, 1), (v20, -1)], [(w40, 1), (w50, 1)])
    # M5
    add_term([(u20, 1), (u21, 1)], [(v31, 1)], [(w40, -1), (w41, 1)])
    # M6
    add_term([(u30, 1), (u20, -1)], [(v20, 1), (v21, 1)], [(w51, 1)])
    # M7
    add_term([(u21, 1), (u31, -1)], [(v30, 1), (v31, 1)], [(w40, 1)])
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, n, m, p, rank):
        U_np = np.array(U_mat, dtype=np.float64)
        V_np = np.array(V_mat, dtype=np.float64)
        W_np = np.array(W_mat, dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i*m+j, :]
                    v_row = V_np[j*p+k, :]
                    w_row = W_np[k*n+i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > 1e-10:
                        return False
        return True
    
    if verify(U, V, W, n, m, p, rank):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback: return rank 59 (single Strassen block)
        rank_fallback = total_terms - 1
        U_f = [[0] * rank_fallback for _ in range(n * m)]
        V_f = [[0] * rank_fallback for _ in range(m * p)]
        W_f = [[0] * rank_fallback for _ in range(p * n)]
        col_f = 0
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in strassen_block1:
                        continue
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U_f[u_idx][col_f] = 1
                    V_f[v_idx][col_f] = 1
                    W_f[w_idx][col_f] = 1
                    col_f += 1
        
        def add_term_f(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col_f
            for u_idx, coeff in u_coeffs:
                U_f[u_idx][col_f] += coeff
            for v_idx, coeff in v_coeffs:
                V_f[v_idx][col_f] += coeff
            for w_idx, coeff in w_coeffs:
                W_f[w_idx][col_f] += coeff
            col_f += 1
        
        add_term_f([(u00, 1), (u11, 1)], [(v00, 1), (v11, 1)], [(w00, 1), (w11, 1)])
        add_term_f([(u10, 1), (u11, 1)], [(v00, 1)], [(w10, 1), (w11, -1)])
        add_term_f([(u00, 1)], [(v01, 1), (v11, -1)], [(w01, 1), (w11, 1)])
        add_term_f([(u11, 1)], [(v10, 1), (v00, -1)], [(w00, 1), (w10, 1)])
        add_term_f([(u00, 1), (u01, 1)], [(v11, 1)], [(w00, -1), (w01, 1)])
        add_term_f([(u10, 1), (u00, -1)], [(v00, 1), (v01, 1)], [(w11, 1)])
        add_term_f([(u01, 1), (u11, -1)], [(v10, 1), (v11, 1)], [(w00, 1)])
        
        return {"U": U_f, "V": V_f, "W": W_f}

# EVOLVE-BLOCK-END
