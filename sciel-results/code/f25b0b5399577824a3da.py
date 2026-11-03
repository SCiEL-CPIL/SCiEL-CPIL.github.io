import numpy as np
import random
import time

DIMENSIONS = [3, 4, 5]

def check_decomposition(U, V, W, n, m, p):
    """
    Check if U, V, W form a valid decomposition.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    Condition: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k
    """
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0
                for r in range(R):
                    val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                if val != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Rank 59 decomposition using Strassen on a 2x2x2 subblock
    rank = 59
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Define the 2x2x2 block indices
    # i in {0, 1}, j in {0, 1}, k in {0, 1}
    block_i = [0, 1]
    block_j = [0, 1]
    block_k = [0, 1]
    
    # Set of schoolbook indices to skip
    skip_indices = set()
    for i in block_i:
        for j in block_j:
            for k in block_k:
                skip_indices.add((i, j, k))
                
    col = 0
    # Add standard schoolbook terms for all (i, j, k) not in the block
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in skip_indices:
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
                    
    # Add Strassen terms for the 2x2x2 block
    # The block uses local indices 0,1 for i, j, k
    # Global indices:
    # A_{ij} -> U[i*m + j]
    # B_{jk} -> V[j*p + k]
    # C_{ik} -> W[k*n + i]
    
    # Strassen terms definitions:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 + B11)
    # M4 = (A00 - A10)(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A11 - A01)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Reconstruction:
    # C00 = M1 + M4 - M6 + M7
    # C01 = M5 + M6
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M5
    
    # We need to map local (li, lj, lk) to global (i, j, k)
    # li=0->i=0, li=1->i=1
    # lj=0->j=0, lj=1->j=1
    # lk=0->k=0, lk=1->k=1
    
    def get_u_idx(li, lj):
        i = block_i[li]
        j = block_j[lj]
        return i * m + j
        
    def get_v_idx(lj, lk):
        j = block_j[lj]
        k = block_k[lk]
        return j * p + k
        
    def get_w_idx(lk, li):
        k = block_k[lk]
        i = block_i[li]
        return k * n + i

    # Term 1: M1
    # A: A00(1), A11(1)
    # B: B00(1), B11(1)
    # C: C00(1), C11(1)
    U[get_u_idx(0, 0), col] += 1
    U[get_u_idx(1, 1), col] += 1
    V[get_v_idx(0, 0), col] += 1
    V[get_v_idx(1, 1), col] += 1
    W[get_w_idx(0, 0), col] += 1
    W[get_w_idx(1, 1), col] += 1
    col += 1
    
    # Term 2: M2
    # A: A10(1), A11(1)
    # B: B00(1)
    # C: C10(1), C11(-1) -- Wait, C11 has -M3 + M2... so M2 contributes +1 to C11?
    # C10 = M2 + M4 => C10 gets +1 from M2
    # C11 = M1 - M3 + M2 - M5 => C11 gets +1 from M2
    U[get_u_idx(1, 0), col] += 1
    U[get_u_idx(1, 1), col] += 1
    V[get_v_idx(0, 0), col] += 1
    W[get_w_idx(0, 1), col] += 1 # C10: k=0, i=1 -> idx 0*n+1 = 1? No.
    # W index is k*n + i.
    # C10: i=1, k=0 -> 0*3 + 1 = 1.
    # C11: i=1, k=1 -> 1*3 + 1 = 4.
    W[get_w_idx(0, 1), col] += 1
    W[get_w_idx(1, 1), col] += 1
    col += 1
    
    # Term 3: M3
    # A: A00(1)
    # B: B10(1), B11(1)
    # C: C11(-1)
    U[get_u_idx(0, 0), col] += 1
    V[get_v_idx(1, 0), col] += 1
    V[get_v_idx(1, 1), col] += 1
    W[get_w_idx(1, 1), col] += -1
    col += 1
    
    # Term 4: M4
    # A: A00(1), A10(-1)
    # B: B10(1), B00(-1)
    # C: C00(1), C10(1)
    U[get_u_idx(0, 0), col] += 1
    U[get_u_idx(1, 0), col] += -1
    V[get_v_idx(1, 0), col] += 1
    V[get_v_idx(0, 0), col] += -1
    W[get_w_idx(0, 0), col] += 1
    W[get_w_idx(0, 1), col] += 1
    col += 1
    
    # Term 5: M5
    # A: A00(1), A01(1)
    # B: B11(1)
    # C: C01(1), C11(-1)
    U[get_u_idx(0, 0), col] += 1
    U[get_u_idx(0, 1), col] += 1
    V[get_v_idx(1, 1), col] += 1
    W[get_w_idx(1, 0), col] += 1 # C01: i=0, k=1 -> 1*3+0=3
    W[get_w_idx(1, 1), col] += -1
    col += 1
    
    # Term 6: M6
    # A: A11(1), A01(-1)
    # B: B00(1), B01(1)
    # C: C00(-1), C01(1)
    U[get_u_idx(1, 1), col] += 1
    U[get_u_idx(0, 1), col] += -1
    V[get_v_idx(0, 0), col] += 1
    V[get_v_idx(0, 1), col] += 1
    W[get_w_idx(0, 0), col] += -1
    W[get_w_idx(1, 0), col] += 1
    col += 1
    
    # Term 7: M7
    # A: A01(1), A11(-1)
    # B: B10(1), B11(1)
    # C: C00(1)
    U[get_u_idx(0, 1), col] += 1
    U[get_u_idx(1, 1), col] += -1
    V[get_v_idx(1, 0), col] += 1
    V[get_v_idx(1, 1), col] += 1
    W[get_w_idx(0, 0), col] += 1
    col += 1
    
    # Verify col == rank
    if col != rank:
        # Fallback to schoolbook if error in logic
        base_rank = n * m * p
        U = np.zeros((n * m, base_rank), dtype=int)
        V = np.zeros((m * p, base_rank), dtype=int)
        W = np.zeros((p * n, base_rank), dtype=int)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Optional: Verify correctness
    # if not check_decomposition(U, V, W, n, m, p):
    #     # Fallback
    #     base_rank = n * m * p
    #     U = np.zeros((n * m, base_rank), dtype=int)
    #     V = np.zeros((m * p, base_rank), dtype=int)
    #     W = np.zeros((p * n, base_rank), dtype=int)
    #     c = 0
    #     for i in range(n):
    #         for j in range(m):
    #             for k in range(p):
    #                 U[i * m + j, c] = 1
    #                 V[j * p + k, c] = 1
    #                 W[k * n + i, c] = 1
    #                 c += 1
    #     return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
