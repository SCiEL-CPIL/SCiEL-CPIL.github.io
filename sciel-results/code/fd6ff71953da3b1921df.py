"""Matrix multiplication 2x2x14 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

# Global deadline guard
START_TIME = time.time()
DEADLINE = 140.0

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    # Deadline check
    if time.time() > DEADLINE:
        # Return a valid fallback (e.g., schoolbook) if we are running out of time
        # Though this block is O(1), we satisfy the directive.
        n, m, p = DIMENSIONS
        rank_school = n * m * p
        U_s = np.zeros((n * m, rank_school), dtype=np.int64)
        V_s = np.zeros((m * p, rank_school), dtype=np.int64)
        W_s = np.zeros((p * n, rank_school), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, col] = 1
                    V_s[j * p + k, col] = 1
                    W_s[k * n + i, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    n, m, p = DIMENSIONS
    # n=2, m=2, p=14
    # The problem decomposes into 7 independent 2x2 * 2x2 multiplications.
    # Each 2x2 * 2x2 can be done in rank 7 (Strassen).
    # Total rank = 7 * 7 = 49.
    
    size_a = n * m  # 4
    size_b = m * p  # 28
    size_c = p * n  # 28
    rank = 49
    
    # Initialize matrices
    U = np.zeros((size_a, rank), dtype=np.int64)
    V = np.zeros((size_b, rank), dtype=np.int64)
    W = np.zeros((size_c, rank), dtype=np.int64)
    
    # Indices for A (2x2): u = i*m + j
    # u00=0, u01=1, u10=2, u11=3
    
    # Indices for B (2x14): v = j*p + k
    # For block q (0..6), columns k=2q, 2q+1
    # v00 = 14*0 + 2q, v01 = 14*0 + (2q+1)
    # v10 = 14*1 + 2q, v11 = 14*1 + (2q+1)
    
    # Indices for C (2x14): w = k*n + i
    # For block q, columns k=2q, 2q+1
    # w00 = (2q)*2 + 0, w01 = (2q+1)*2 + 0
    # w10 = (2q)*2 + 1, w11 = (2q+1)*2 + 1
    
    for q in range(7):
        k0 = 2 * q
        k1 = 2 * q + 1
        
        # B indices
        v_B00 = 14 * 0 + k0
        v_B01 = 14 * 0 + k1
        v_B10 = 14 * 1 + k0
        v_B11 = 14 * 1 + k1
        
        # C indices
        w_C00 = 2 * k0 + 0
        w_C01 = 2 * k1 + 0
        w_C10 = 2 * k0 + 1
        w_C11 = 2 * k1 + 1
        
        # A indices
        u_A00 = 0
        u_A01 = 1
        u_A10 = 2
        u_A11 = 3
        
        col_base = q * 7
        
        # Strassen products for this block:
        # M1 = (A00 + A11)(B00 + B11)
        c = col_base + 0
        U[u_A00, c] += 1
        U[u_A11, c] += 1
        V[v_B00, c] += 1
        V[v_B11, c] += 1
        W[w_C00, c] += 1
        W[w_C11, c] += 1
        
        # M2 = (A10 + A11)B00
        c = col_base + 1
        U[u_A10, c] += 1
        U[u_A11, c] += 1
        V[v_B00, c] += 1
        W[w_C10, c] += 1
        W[w_C11, c] -= 1
        
        # M3 = A00(B01 - B11)
        c = col_base + 2
        U[u_A00, c] += 1
        V[v_B01, c] += 1
        V[v_B11, c] -= 1
        W[w_C01, c] += 1
        W[w_C11, c] += 1
        
        # M4 = A11(B10 - B00)
        c = col_base + 3
        U[u_A11, c] += 1
        V[v_B10, c] += 1
        V[v_B00, c] -= 1
        W[w_C00, c] += 1
        W[w_C10, c] += 1
        
        # M5 = (A00 + A01)B11
        c = col_base + 4
        U[u_A00, c] += 1
        U[u_A01, c] += 1
        V[v_B11, c] += 1
        W[w_C00, c] -= 1
        W[w_C01, c] += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        c = col_base + 5
        U[u_A10, c] += 1
        U[u_A00, c] -= 1
        V[v_B00, c] += 1
        V[v_B01, c] += 1
        W[w_C11, c] += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        c = col_base + 6
        U[u_A01, c] += 1
        U[u_A11, c] -= 1
        V[v_B10, c] += 1
        V[v_B11, c] += 1
        W[w_C00, c] += 1

    # Verification (optional but good for safety, should be fast)
    # T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    # T_true = np.zeros((4, 28, 28))
    # for i in range(2):
    #     for j in range(2):
    #         for k in range(14):
    #             T_true[i*2+j, j*14+k, k*2+i] = 1.0
    # if not np.allclose(T_recon, T_true, atol=1e-9):
    #     # Fallback to schoolbook if verification fails (should not happen)
    #     rank_school = 56
    #     U_s = np.zeros((4, 56), dtype=np.int64)
    #     V_s = np.zeros((28, 56), dtype=np.int64)
    #     W_s = np.zeros((28, 56), dtype=np.int64)
    #     col = 0
    #     for i in range(2):
    #         for j in range(2):
    #             for k in range(14):
    #                 U_s[i*2+j, col] = 1
    #                 V_s[j*14+k, col] = 1
    #                 W_s[k*2+i, col] = 1
    #                 col += 1
    #     return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Final deadline check before return
    if time.time() > DEADLINE:
        # This should never happen for this O(1) construction
        pass
        
    # Final check
    if time.time() > deadline:
        pass # Or fallback, but construction is done
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
