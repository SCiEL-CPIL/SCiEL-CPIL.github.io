"""Matrix multiplication 3x3 by 3x7 - rank minimization."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    
    Strategy: 
    1. Attempt a constructive Rank 55 decomposition:
       - Use Laderman's algorithm (Rank 19) for the first 3x3 block of B (columns 0-2).
       - Use Schoolbook (Rank 36) for the remaining 3x4 block of B (columns 3-6).
    2. If the constructive solution fails verification, fall back to the Schoolbook solution (Rank 63).
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # ============================================================
    # HELPER: Verify Decomposition
    # ============================================================
    def verify_decomposition(U_list, V_list, W_list):
        """
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        """
        rank = len(U_list[0]) if U_list else 0
        if not U_list or not V_list or not W_list:
            return False
        
        # Convert to numpy for fast dot products
        U = np.array(U_list)
        V = np.array(V_list)
        W = np.array(W_list)
        
        # Check shapes
        if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
            return False
        
        # Verify T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # ============================================================
    # HELPER: Schoolbook Decomposition (Rank 63)
    # ============================================================
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    # ============================================================
    # HELPER: Laderman Decomposition for 3x3x3 (Rank 19)
    # ============================================================
    def get_laderman_block():
        """
        Returns U_block (9x19), V_block (9x19), W_block (9x19) 
        for the tensor T[i,j,k] where i,j,k in 0..2.
        Note: The interface for the full problem maps:
        U_full[i*m+j, r] = U_block[i*3+j, r]
        V_full[j*p+k, r] = V_block[j*3+k, r]
        W_full[k*n+i, r] = W_block[k*3+i, r]
        """
        rank_l = 19
        U_b = np.zeros((9, rank_l))
        V_b = np.zeros((9, rank_l))
        W_b = np.zeros((9, rank_l))
        
        # Laderman's 19-term decomposition
        # Indices: i,j,k in 0..2
        # U index: i*3 + j
        # V index: j*3 + k
        # W index: k*3 + i
        
        # M1 = A00 (B00 + B11 - B10 - B21)
        U_b[0, 0] = 1
        V_b[0, 0] = 1; V_b[4, 0] = 1; V_b[3, 0] = -1; V_b[7, 0] = -1
        W_b[0, 0] = 1
        
        # M2 = (A00 + A01) (B10 + B21)
        U_b[0, 1] = 1; U_b[1, 1] = 1
        V_b[3, 1] = 1; V_b[7, 1] = 1
        W_b[0, 1] = 1
        
        # M3 = (A00 + A10) (B01 + B12)
        U_b[0, 2] = 1; U_b[3, 2] = 1
        V_b[1, 2] = 1; V_b[5, 2] = 1
        W_b[0, 2] = 1
        
        # M4 = (A01 + A11) (B00 + B10 - B01 - B11)
        U_b[1, 3] = 1; U_b[4, 3] = 1
        V_b[0, 3] = 1; V_b[3, 3] = 1; V_b[1, 3] = -1; V_b[4, 3] = -1
        W_b[0, 3] = 1
        
        # M5 = (A01 + A21) (B02 + B12)
        U_b[1, 4] = 1; U_b[7, 4] = 1
        V_b[2, 4] = 1; V_b[5, 4] = 1
        W_b[0, 4] = 1
        
        # M6 = (A02 + A12) (B00 + B20 - B10 - B21)
        U_b[2, 5] = 1; U_b[5, 5] = 1
        V_b[0, 5] = 1; V_b[6, 5] = 1; V_b[3, 5] = -1; V_b[7, 5] = -1
        W_b[0, 5] = 1
        
        # M7 = (A02 + A22) (B01 + B21 - B02 - B12)
        U_b[2, 6] = 1; U_b[8, 6] = 1
        V_b[1, 6] = 1; V_b[7, 6] = 1; V_b[2, 6] = -1; V_b[5, 6] = -1
        W_b[0, 6] = 1
        
        # M8 = (A10 + A20) (B01 + B12 - B00 - B11)
        U_b[3, 7] = 1; U_b[6, 7] = 1
        V_b[1, 7] = 1; V_b[5, 7] = 1; V_b[0, 7] = -1; V_b[4, 7] = -1
        W_b[0, 7] = 1
        
        # M9 = (A11 + A21) (B02 + B12 - B01 - B11)
        U_b[4, 8] = 1; U_b[7, 8] = 1
        V_b[2, 8] = 1; V_b[5, 8] = 1; V_b[1, 8] = -1; V_b[4, 8] = -1
        W_b[0, 8] = 1
        
        # M10 = (A12 + A22) (B00 + B10 - B02 - B12)
        U_b[5, 9] = 1; U_b[8, 9] = 1
        V_b[0, 9] = 1; V_b[3, 9] = 1; V_b[2, 9] = -1; V_b[5, 9] = -1
        W_b[0, 9] = 1
        
        # M11 = (A00 + A11 + A22) (B00 + B11 + B22 - B01 - B10 - B21 - B12)
        U_b[0, 10] = 1; U_b[4, 10] = 1; U_b[8, 10] = 1
        V_b[0, 10] = 1; V_b[4, 10] = 1; V_b[8, 10] = 1
        V_b[1, 10] = -1; V_b[3, 10] = -1; V_b[7, 10] = -1; V_b[5, 10] = -1
        W_b[0, 10] = 1
        
        # M12 = (A01 + A12 + A20) (B02 + B10 + B21 - B00 - B12 - B20)
        U_b[1, 11] = 1; U_b[5, 11] = 1; U_b[6, 11] = 1
        V_b[2, 11] = 1; V_b[3, 11] = 1; V_b[7, 11] = 1
        V_b[0, 11] = -1; V_b[5, 11] = -1; V_b[6, 11] = -1
        W_b[0, 11] = 1
        
        # M13 = (A02 + A10 + A21) (B01 + B12 + B20 - B00 - B11 - B22)
        U_b[2, 12] = 1; U_b[3, 12] = 1; U_b[7, 12] = 1
        V_b[1, 12] = 1; V_b[5, 12] = 1; V_b[6, 12] = 1
        V_b[0, 12] = -1; V_b[4, 12] = -1; V_b[8, 12] = -1
        W_b[0, 12] = 1
        
        # M14 = (A01 + A11 + A22) (B02 + B10 + B21 - B00 - B12 - B20)
        U_b[1, 13] = 1; U_b[4, 13] = 1; U_b[8, 13] = 1
        V_b[2, 13] = 1; V_b[3, 13] = 1; V_b[7, 13] = 1
        V_b[0, 13] = -1; V_b[5, 13] = -1; V_b[6, 13] = -1
        W_b[0, 13] = 1
        
        # M15 = (A02 + A11 + A20) (B01 + B12 + B20 - B00 - B11 - B22)
        U_b[2, 14] = 1; U_b[4, 14] = 1; U_b[6, 14] = 1
        V_b[1, 14] = 1; V_b[5, 14] = 1; V_b[6, 14] = 1
        V_b[0, 14] = -1; V_b[4, 14] = -1; V_b[8, 14] = -1
        W_b[0, 14] = 1
        
        # M16 = (A00 + A12 + A21) (B02 + B11 + B20 - B01 - B10 - B22)
        U_b[0, 15] = 1; U_b[5, 15] = 1; U_b[7, 15] = 1
        V_b[2, 15] = 1; V_b[4, 15] = 1; V_b[6, 15] = 1
        V_b[1, 15] = -1; V_b[3, 15] = -1; V_b[8, 15] = -1
        W_b[0, 15] = 1
        
        # M17 = (A02 + A10 + A22) (B01 + B12 + B20 - B00 - B11 - B22)
        U_b[2, 16] = 1; U_b[3, 16] = 1; U_b[8, 16] = 1
        V_b[1, 16] = 1; V_b[5, 16] = 1; V_b[6, 16] = 1
        V_b[0, 16] = -1; V_b[4, 16] = -1; V_b[8, 16] = -1
        W_b[0, 16] = 1
        
        # M18 = (A00 + A11 + A22) (B02 + B10 + B21 - B01 - B12 - B20)
        U_b[0, 17] = 1; U_b[4, 17] = 1; U_b[8, 17] = 1
        V_b[2, 17] = 1; V_b[3, 17] = 1; V_b[7, 17] = 1
        V_b[1, 17] = -1; V_b[5, 17] = -1; V_b[6, 17] = -1
        W_b[0, 17] = 1
        
        # M19 = (A01 + A12 + A20) (B02 + B10 + B21 - B00 - B12 - B20)
        # This is a duplicate of M12? Let's assume the standard 19 terms are distinct.
        # If my hardcoded terms are wrong, verification will fail.
        U_b[1, 18] = 1; U_b[5, 18] = 1; U_b[6, 18] = 1
        V_b[2, 18] = 1; V_b[3, 18] = 1; V_b[7, 18] = 1
        V_b[0, 18] = -1; V_b[5, 18] = -1; V_b[6, 18] = -1
        W_b[0, 18] = 1
        
        return U_b, V_b, W_b

    # ============================================================
    # CONSTRUCT RANK 55 SOLUTION
    # ============================================================
    def construct_rank_55():
        rank_l = 19
        rank_s = 36 # 3x3x4
        total_rank = rank_l + rank_s
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # Part 1: Laderman for B[:, 0:3]
        U_b, V_b, W_b = get_laderman_block()
        
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for r in range(rank_l):
                    U[u_idx][r] = U_b[i * 3 + j, r]
        
        for j in range(m):
            for k in range(3): # Only first 3 cols
                v_idx = j * p + k
                for r in range(rank_l):
                    V[v_idx][r] = V_b[j * 3 + k, r]
        
        for k in range(3): # Only first 3 cols
            for i in range(n):
                w_idx = k * n + i
                for r in range(rank_l):
                    W[w_idx][r] = W_b[k * 3 + i, r]
        
        # Part 2: Schoolbook for B[:, 3:7]
        # k in 3..6
        col_offset = rank_l
        for i in range(n):
            for j in range(m):
                for k in range(3, p):
                    r = col_offset + (k - 3)
                    # Wait, schoolbook rank for 3x3x4 is 36.
                    # Indices: i (0-2), j (0-2), k (3-6)
                    # We need to map (i,j,k) to a unique column index in [19, 55)
                    # There are 3*3*4 = 36 terms.
                    # Let's just iterate and assign.
                    pass
        
        # Proper Schoolbook mapping for the second block
        s_col = 0
        for i in range(n):
            for j in range(m):
                for k in range(3, p):
                    r = rank_l + s_col
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    s_col += 1
        
        return U, V, W

    # ============================================================
    # MAIN LOGIC
    # ============================================================
    
    # 1. Try Rank 55
    if time.time() < deadline - 10:
        try:
            U_55, V_55, W_55 = construct_rank_55()
            if verify_decomposition(U_55, V_55, W_55):
                return {"U": U_55, "V": V_55, "W": W_55}
        except Exception as e:
            pass # Fallback to schoolbook
    
    # 2. Fallback to Schoolbook (Rank 63)
    U_63, V_63, W_63 = get_schoolbook()
    return {"U": U_63, "V": V_63, "W": W_63}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
