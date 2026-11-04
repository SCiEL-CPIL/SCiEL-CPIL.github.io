import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    target_rank = 36
    
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    col = 0
    
    # Helper to add a Strassen block
    # a_offset: row offset for A (i) -> actually i is always 0,1 for 2x2
    # j_offset: col offset for A and row offset for B
    # k_offset: col offset for B and row offset for W (k)
    # i_offset: row offset for W (i) -> always 0,1
    
    def add_strassen_block(j_off, k_off):
        nonlocal col
        # M1: (x00+x11)(y00+y11)
        # U: x00 (i=0, j=j_off), x11 (i=1, j=j_off+1)
        U[0 * m + j_off, col] = 1.0
        U[1 * m + (j_off + 1), col] = 1.0
        # V: y00 (j=j_off, k=k_off), y11 (j=j_off+1, k=k_off+1)
        V[j_off * p + k_off, col] = 1.0
        V[(j_off + 1) * p + (k_off + 1), col] = 1.0
        # W: z00 (k=k_off, i=0), z11 (k=k_off+1, i=1)
        W[k_off * n + 0, col] = 1.0
        W[(k_off + 1) * n + 1, col] = 1.0
        col += 1
        
        # M2: (x10+x11)y00
        U[1 * m + j_off, col] = 1.0
        U[1 * m + (j_off + 1), col] = 1.0
        V[j_off * p + k_off, col] = 1.0
        W[(k_off + 1) * n + 1, col] = 1.0 # z11
        W[k_off * n + 1, col] = -1.0 # z10? No.
        # z10 = M2 + M4. z11 = M1 - M2 + ...
        # So W for M2: z10 coeff 1, z11 coeff -1.
        W[k_off * n + 1, col] = 1.0
        W[(k_off + 1) * n + 1, col] = -1.0
        col += 1

        # M3: x00(y01-y11)
        U[0 * m + j_off, col] = 1.0
        V[j_off * p + (k_off + 1), col] = 1.0
        V[(j_off + 1) * p + (k_off + 1), col] = -1.0
        W[k_off * n + 0, col] = 1.0 # z00? No.
        # z00 = M1 + M4 - M5 + M7. No M3.
        # z01 = M3 + M5.
        # z11 = M1 - M2 + M3 + M6.
        # So W for M3: z01 coeff 1, z11 coeff 1.
        W[(k_off + 1) * n + 0, col] = 1.0
        W[(k_off + 1) * n + 1, col] = 1.0
        col += 1

        # M4: x11(y10-y00)
        U[1 * m + (j_off + 1), col] = 1.0
        V[(j_off + 1) * p + k_off, col] = 1.0
        V[j_off * p + k_off, col] = -1.0
        W[k_off * n + 0, col] = 1.0 # z00
        W[k_off * n + 1, col] = 1.0 # z10
        col += 1

        # M5: (x00+x01)y11
        U[0 * m + j_off, col] = 1.0
        U[0 * m + (j_off + 1), col] = 1.0
        V[(j_off + 1) * p + (k_off + 1), col] = 1.0
        W[k_off * n + 0, col] = -1.0 # z00
        W[(k_off + 1) * n + 0, col] = 1.0 # z01
        col += 1

        # M6: (x10-x00)(y00+y01)
        U[1 * m + j_off, col] = 1.0
        U[0 * m + j_off, col] = -1.0
        V[j_off * p + k_off, col] = 1.0
        V[j_off * p + (k_off + 1), col] = 1.0
        W[(k_off + 1) * n + 1, col] = 1.0 # z11
        col += 1

        # M7: (x01-x11)(y10+y11)
        U[0 * m + (j_off + 1), col] = 1.0
        U[1 * m + (j_off + 1), col] = -1.0
        V[(j_off + 1) * p + k_off, col] = 1.0
        V[(j_off + 1) * p + (k_off + 1), col] = 1.0
        W[k_off * n + 0, col] = 1.0 # z00
        col += 1

    def add_schoolbook_2x2x1_block(j_off, k_off):
        nonlocal col
        # x00*y00 -> z00
        U[0 * m + j_off, col] = 1.0
        V[j_off * p + k_off, col] = 1.0
        W[k_off * n + 0, col] = 1.0
        col += 1
        # x01*y10 -> z00? No.
        # z00 = x00 y00 + x01 y10
        # z10 = x10 y00 + x11 y10
        # Wait, Y is 2x1. y00 is row 0, y10 is row 1?
        # Y indices: j, k. Here k is fixed.
        # Y[j, k].
        # z00 = x00 Y[0,k] + x01 Y[1,k]
        # z10 = x10 Y[0,k] + x11 Y[1,k]
        
        # Term 2: x01 * Y[1,k] -> z00
        U[0 * m + (j_off + 1), col] = 1.0
        V[(j_off + 1) * p + k_off, col] = 1.0
        W[k_off * n + 0, col] = 1.0
        col += 1
        
        # Term 3: x10 * Y[0,k] -> z10
        U[1 * m + j_off, col] = 1.0
        V[j_off * p + k_off, col] = 1.0
        W[k_off * n + 1, col] = 1.0
        col += 1
        
        # Term 4: x11 * Y[1,k] -> z10
        U[1 * m + (j_off + 1), col] = 1.0
        V[(j_off + 1) * p + k_off, col] = 1.0
        W[k_off * n + 1, col] = 1.0
        col += 1

    # Block 1: A1 (j=0,1), B1 (j=0,1)
    # Y1: k=0,1
    add_strassen_block(0, 0)
    # Y21: k=2,3
    add_strassen_block(0, 2)
    # Y22: k=4
    add_schoolbook_2x2x1_block(0, 4)
    
    # Block 2: A2 (j=2,3), B2 (j=2,3)
    # Y1: k=0,1
    add_strassen_block(2, 0)
    # Y21: k=2,3
    add_strassen_block(2, 2)
    # Y22: k=4
    add_schoolbook_2x2x1_block(2, 4)
    
    # Verify
    def verify(U, V, W):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = 0.0
                    for c in range(r):
                        val += U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                    if abs(val - 1.0) > 1e-10:
                        return False
        # Check zeros
        for idx_u in range(n * m):
            for idx_v in range(m * p):
                for idx_w in range(p * n):
                    is_one = False
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if (i * m + j == idx_u and j * p + k == idx_v and k * n + i == idx_w):
                                    is_one = True
                                    break
                            if is_one: break
                        if is_one: break
                    if not is_one:
                        val = 0.0
                        for c in range(r):
                            val += U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                        if abs(val) > 1e-10:
                            return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook if something went wrong
        return construct_fallback()
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct_fallback():
    n, m, p = DIMENSIONS
    target_rank = n * m * p
    U_school = np.zeros((n * m, target_rank))
    V_school = np.zeros((m * p, target_rank))
    W_school = np.zeros((p * n, target_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U_school[idx_u, col] = 1.0
                V_school[idx_v, col] = 1.0
                W_school[idx_w, col] = 1.0
                col += 1
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
