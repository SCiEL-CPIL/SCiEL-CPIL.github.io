import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify if U, V, W form a valid decomposition for M_{n,m,p}.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = U.shape[1]
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We check if this sum equals 1 for all i, j, k.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _construct_strassen_222():
    """
    Returns U, V, W for M_{2,2,2} using Strassen's algorithm (rank 7).
    """
    n, m, p = 2, 2, 2
    rank = 7
    
    # Strassen formulas:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B11 - B12)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # U indices: i*m + j. i in {0,1}, j in {0,1}
    # U[0, :] -> A00
    # U[1, :] -> A01
    # U[2, :] -> A10
    # U[3, :] -> A11
    
    # V indices: j*p + k. j in {0,1}, k in {0,1}
    # V[0, :] -> B00
    # V[1, :] -> B01
    # V[2, :] -> B10
    # V[3, :] -> B11
    
    # W indices: k*n + i. k in {0,1}, i in {0,1}
    # W[0, :] -> C00
    # W[1, :] -> C01
    # W[2, :] -> C10
    # W[3, :] -> C11
    
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = -1; W[2, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[0, 1] = 1; W[2, 1] = 1
    
    # M3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[0, 3] = 1; V[1, 3] = -1
    W[0, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1
    
    return U, V, W

def _construct_221():
    """
    Returns U, V, W for M_{2,2,1} (rank 4, schoolbook).
    """
    n, m, p = 2, 2, 1
    rank = 4
    
    U = np.zeros((4, rank))
    V = np.zeros((2, rank))
    W = np.zeros((2, rank))
    
    # C0 = A00*B00 + A01*B10
    # C1 = A10*B00 + A11*B10
    
    # r=0: A00 B00 -> C0
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # r=1: A01 B10 -> C0
    U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
    # r=2: A10 B00 -> C1
    U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
    # r=3: A11 B10 -> C1
    U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
    
    return U, V, W

def construct():
    _start = time.time()
    _deadline = _start + 135.0
    
    n, m, p = DIMENSIONS
    
    # We construct M_{2,4,5} by splitting the inner dimension (j) into two blocks of size 2.
    # Block 1: j in {0,1}
    # Block 2: j in {2,3}
    
    # Each block is a multiplication of (2x2) A_block by (2x5) B_block.
    # We further split the output dimension (k) into:
    # k in {0,1} -> M_{2,2,2} (Strassen, rank 7)
    # k in {2,3} -> M_{2,2,2} (Strassen, rank 7)
    # k in {4}   -> M_{2,2,1} (Schoolbook, rank 4)
    
    # Total rank per block = 7 + 7 + 4 = 18
    # Total rank = 18 * 2 = 36
    
    U_str, V_str, W_str = _construct_strassen_222()
    U_221, V_221, W_221 = _construct_221()
    
    rank_block = 18
    total_rank = 2 * rank_block
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Helper to place a sub-decomposition
    def place_block(block_id, j_start, k_start, sub_U, sub_V, sub_W, sub_rank):
        """
        Places a sub-decomposition for A_block (2x2) * B_subblock (2xk_len) -> C_subblock (2xk_len).
        block_id: 0 or 1 (which half of j)
        j_start: starting index of j in the full 4-dim space
        k_start: starting index of k in the full 5-dim space
        """
        r_start = block_id * rank_block
        r_offset = 0
        
        # sub_U shape: (4, sub_rank) -> rows i*2+j_local
        # sub_V shape: (2*k_len, sub_rank) -> rows j_local*k_len+k_local
        # sub_W shape: (k_len*2, sub_rank) -> rows k_local*2+i
        
        k_len = sub_V.shape[0] // 2
        
        for i in range(2):
            for j_local in range(2):
                j_global = j_start + j_local
                for r in range(sub_rank):
                    U[i*4 + j_global, r_start + r_offset + r] = sub_U[i*2 + j_local, r]
        
        for j_local in range(2):
            j_global = j_start + j_local
            for k_local in range(k_len):
                k_global = k_start + k_local
                for r in range(sub_rank):
                    V[j_global*5 + k_global, r_start + r_offset + r] = sub_V[j_local*k_len + k_local, r]
        
        for k_local in range(k_len):
            k_global = k_start + k_local
            for i in range(2):
                for r in range(sub_rank):
                    W[k_global*2 + i, r_start + r_offset + r] = sub_W[k_local*2 + i, r]
        
        return r_offset + sub_rank

    # Block 1 (j=0,1)
    r_off = 0
    # k=0,1
    r_off = place_block(0, 0, 0, U_str, V_str, W_str, 7) + r_off
    # k=2,3
    r_off = place_block(0, 0, 2, U_str, V_str, W_str, 7) + r_off
    # k=4
    r_off = place_block(0, 0, 4, U_221, V_221, W_221, 4) + r_off
    
    # Block 2 (j=2,3)
    r_off = 0
    # k=0,1
    r_off = place_block(1, 2, 0, U_str, V_str, W_str, 7) + r_off
    # k=2,3
    r_off = place_block(1, 2, 2, U_str, V_str, W_str, 7) + r_off
    # k=4
    r_off = place_block(1, 2, 4, U_221, V_221, W_221, 4) + r_off
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails (should not happen)
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", res["U"].shape[1])
    print(res)
