"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Target rank 56 using 4 Strassen blocks
    # Blocks cover j in {0,1,2,3} and k in {0,1,2,3}
    # This leaves j=4 and k in {4,5} to be handled by standard terms
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Define the Strassen blocks
    # Each block is defined by (j1, j2, k1, k2)
    # j1 < j2, k1 < k2
    blocks = [
        (0, 1, 0, 1),
        (2, 3, 2, 3),
        (0, 1, 2, 3),
        (2, 3, 0, 1)
    ]
    
    # Set of indices covered by Strassen blocks
    covered = set()
    for j1, j2, k1, k2 in blocks:
        for i in range(n):
            for j in [j1, j2]:
                for k in [k1, k2]:
                    covered.add((i, j, k))
    
    # Add standard terms for uncovered indices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
                
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Add Strassen terms for each block
    for j1, j2, k1, k2 in blocks:
        # Indices in U (A matrix)
        # A[i, j] -> u_idx = i*m + j
        # i=0: u_0j1, u_0j2
        # i=1: u_1j1, u_1j2
        u_00 = 0 * m + j1
        u_01 = 0 * m + j2
        u_10 = 1 * m + j1
        u_11 = 1 * m + j2
        
        # Indices in V (B matrix)
        # B[j, k] -> v_idx = j*p + k
        # j=j1: v_j1k1, v_j1k2
        # j=j2: v_j2k1, v_j2k2
        v_00 = j1 * p + k1
        v_01 = j1 * p + k2
        v_10 = j2 * p + k1
        v_11 = j2 * p + k2
        
        # Indices in W (C matrix)
        # C[k, i] -> w_idx = k*n + i
        # k=k1: w_k1i0, w_k1i1
        # k=k2: w_k2i0, w_k2i1
        w_00 = k1 * n + 0
        w_01 = k1 * n + 1
        w_10 = k2 * n + 0
        w_11 = k2 * n + 1
        
        # Strassen M1: (A00 + A11)(B00 + B11)
        # U: u00, u11; V: v00, v11; W: w00, w01, w10, w11
        # C00 += M1, C01 += 0, C10 += 0, C11 += M1
        # Wait, the output mapping is:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to add contributions to W for each M term.
        
        # M1
        U[u_00, col] = 1
        U[u_11, col] = 1
        V[v_00, col] = 1
        V[v_11, col] = 1
        W[w_00, col] = 1  # C00
        W[w_11, col] = 1  # C11
        col += 1
        
        # M2: (A10 + A11)B00
        U[u_10, col] = 1
        U[u_11, col] = 1
        V[v_00, col] = 1
        W[w_10, col] = 1  # C10
        W[w_11, col] = -1 # C11
        col += 1
        
        # M3: A00(B01 - B11)
        U[u_00, col] = 1
        V[v_01, col] = 1
        V[v_11, col] = -1
        W[w_01, col] = 1  # C01
        W[w_11, col] = 1  # C11
        col += 1
        
        # M4: A11(B10 - B00)
        U[u_11, col] = 1
        V[v_10, col] = 1
        V[v_00, col] = -1
        W[w_00, col] = 1  # C00
        W[w_10, col] = 1  # C10
        col += 1
        
        # M5: (A00 + A01)B11
        U[u_00, col] = 1
        U[u_01, col] = 1
        V[v_11, col] = 1
        W[w_00, col] = -1 # C00
        W[w_01, col] = 1  # C01
        col += 1
        
        # M6: (A10 - A00)(B00 + B01)
        U[u_10, col] = 1
        U[u_00, col] = -1
        V[v_00, col] = 1
        V[v_01, col] = 1
        W[w_11, col] = 1  # C11
        col += 1
        
        # M7: (A01 - A10)(B10 + B11)
        U[u_01, col] = 1
        U[u_10, col] = -1
        V[v_10, col] = 1
        V[v_11, col] = 1
        W[w_00, col] = 1  # C00
        col += 1

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Verify the decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i*m+j, :]
                v_row = V[j*p+k, :]
                w_row = W[k*n+i, :]
                result = np.sum(u_row * v_row * w_row)
                if abs(result - 1) > 1e-10:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to rank 60 if verification fails
        U_fallback = np.zeros((n * m, 60), dtype=float)
        V_fallback = np.zeros((m * p, 60), dtype=float)
        W_fallback = np.zeros((p * n, 60), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fallback[i*m+j, c] = 1
                    V_fallback[j*p+k, c] = 1
                    W_fallback[k*n+i, c] = 1
                    c += 1
        return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}

# EVOLVE-BLOCK-END
