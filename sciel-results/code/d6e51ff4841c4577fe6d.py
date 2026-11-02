import time
import numpy as np

DIMENSIONS = [3, 4, 8]
DEADLINE = time.time() + 145.0  # 5s safety margin

def construct():
    n, m, p = DIMENSIONS
    naive_rank = n * m * p
    
    # Build naive rank-96 decomposition as fallback
    U_naive = [[0] * naive_rank for _ in range(n * m)]
    V_naive = [[0] * naive_rank for _ in range(m * p)]
    W_naive = [[0] * naive_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i * m + j][col] = 1
                V_naive[j * p + k][col] = 1
                W_naive[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W, rank):
        """Check if the decomposition is valid"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] = np.dot(U[i*m+j, :rank], V[j*p+k, :rank] * W[k*n+i, :rank])
        
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True
    
    # Verify naive solution
    if not verify(U_naive, V_naive, W_naive, naive_rank):
        return {"U": U_naive, "V": V_naive, "W": W_naive}
    
    # Try block decomposition with Strassen for 2x2x2 blocks
    # Strategy: 
    # - Split n=3 into 2+1
    # - For the 2-row part: use 4 Strassen 2x2x2 blocks (for j=0..1, k=0..1 within each 4x4 sub-block)
    #   Actually, m=4, p=8. Let's split m into 2+2 and p into 4+4.
    #   For A[0:2, :] x B[:, 0:4] -> C[0:2, 0:4]: use 2x4x4
    #   For A[0:2, :] x B[:, 4:8] -> C[0:2, 4:8]: use 2x4x4
    #   For A[2:3, :] x B[:, 0:8] -> C[2:3, 0:8]: use 1x4x8 (naive)
    
    # 2x4x4 can be decomposed as 2x(2x2x4) = 2 * (2x2x4)
    # 2x2x4 can be decomposed as 2x(2x2x2) = 2 * 7 = 14
    # So 2x4x4 = 2 * 14 = 28
    # Two 2x4x4 blocks: 2 * 28 = 56
    # 1x4x8 naive: 4*8 = 32
    # Total: 56 + 32 = 88
    
    # Better: split m=4 into 2+2, p=8 into 4+4
    # A[0:2, 0:2] x B[0:2, 0:4] -> C[0:2, 0:4]: 2x2x4 = 14
    # A[0:2, 0:2] x B[0:2, 4:8] -> C[0:2, 4:8]: 2x2x4 = 14
    # A[0:2, 2:4] x B[2:4, 0:4] -> C[0:2, 0:4]: 2x2x4 = 14
    # A[0:2, 2:4] x B[2:4, 4:8] -> C[0:2, 4:8]: 2x2x4 = 14
    # A[2:3, 0:2] x B[0:2, 0:8] -> C[2:3, 0:8]: 1x2x8 = 16
    # A[2:3, 0:2] x B[0:2, 4:8] already covered
    # Wait, I need to be more careful.
    
    # Let me use a cleaner approach:
    # Total rank target: use Strassen where possible
    # 
    # Block structure:
    # i in {0,1,2}, j in {0,1,2,3}, k in {0,...,7}
    # 
    # For i=0,1 (2 rows):
    #   For each j_pair in {(0,1), (2,3)} and k_block in {0:4, 4:8}:
    #     Use 2x2x4 Strassen-like: rank 14 per block
    #   That's 4 blocks * 14 = 56
    # 
    # For i=2 (1 row):
    #   Naive: 4*8 = 32
    # 
    # Total: 56 + 32 = 88
    
    # Let's implement 2x2x4 using two 2x2x2 Strassen blocks
    # 2x2x4: A is 2x2, B is 2x4, C is 2x4
    # Split B into B[:,0:2] and B[:,2:4], each 2x2
    # So 2x2x4 = 2x2x2 (for B[:,0:2]) + 2x2x2 (for B[:,2:4])
    # Each 2x2x2 uses Strassen rank 7
    # Total: 14
    
    # Strassen 2x2x2 decomposition:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A10)(B10+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # In tensor form T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # For 2x2x2: i,j in {0,1}, k in {0,1}
    # U has 4 entries (2x2), V has 4 entries (2x2), W has 4 entries (2x2)
    # W[k*n+i] with n=2: W[0]=C[0,0], W[1]=C[1,0], W[2]=C[0,1], W[3]=C[1,1]
    
    # Term 0 (M1): (A00+A11)(B00+B11)
    # U: A00=1, A11=1 -> [1,0,0,1]
    # V: B00=1, B11=1 -> [1,0,0,1]
    # W: C00=+1, C11=+1 -> [1,0,0,1]
    
    # Term 1 (M2): (A10+A11)B00
    # U: A10=1, A11=1 -> [0,0,1,1]
    # V: B00=1 -> [1,0,0,0]
    # W: C10=+1, C11=+1 -> [0,1,0,1]
    
    # Term 2 (M3): A00(B10-B11)
    # U: A00=1 -> [1,0,0,0]
    # V: B10=1, B11=-1 -> [0,0,1,-1]
    # W: C01=+1 -> [0,0,1,0]
    
    # Term 3 (M4): A11(B01-B00)
    # U: A11=1 -> [0,0,0,1]
    # V: B01=1, B00=-1 -> [-1,1,0,0]
    # W: C00=+1, C10=+1 -> [1,1,0,0]
    
    # Term 4 (M5): (A00+A01)B11
    # U: A00=1, A01=1 -> [1,1,0,0]
    # V: B11=1 -> [0,0,0,1]
    # W: C00=-1, C01=+1 -> [-1,0,1,0]
    
    # Term 5 (M6): (A10-A00)(B00+B01)
    # U: A10=1, A00=-1 -> [-1,0,1,0]
    # V: B00=1, B01=1 -> [1,1,0,0]
    # W: C10=+1 -> [0,1,0,0]
    
    # Term 6 (M7): (A01-A10)(B10+B11)
    # U: A01=1, A10=-1 -> [0,1,-1,0]
    # V: B10=1, B11=1 -> [0,0,1,1]
    # W: C00=-1, C11=+1 -> [-1,0,0,1]
    
    strassen_terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,1,0,1]),
        ([1,0,0,0], [0,0,1,-1], [0,0,1,0]),
        ([0,0,0,1], [-1,1,0,0], [1,1,0,0]),
        ([1,1,0,0], [0,0,0,1], [-1,0,1,0]),
        ([-1,0,1,0], [1,1,0,0], [0,1,0,0]),
        ([0,1,-1,0], [0,0,1,1], [-1,0,0,1]),
    ]
    
    # Now build the full decomposition
    # Total rank: 56 (for i=0,1) + 32 (for i=2) = 88
    total_rank = 88
    U_full = [[0] * total_rank for _ in range(n * m)]
    V_full = [[0] * total_rank for _ in range(m * p)]
    W_full = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Add a 2x2x2 Strassen block at offsets (i_off, j_off, k_off)
    # i_off: row offset in A and C (0 or 1 for the 2-row block)
    # j_off: col offset in A and row offset in B
    # k_off: col offset in B and col offset in C
    def add_strassen_2x2x2(i_off, j_off, k_off, start_col):
        nonlocal col
        for term_idx in range(7):
            u_term, v_term, w_term = strassen_terms[term_idx]
            r = start_col + term_idx
            
            # U: A[i_off+di, j_off+dj]
            for di in range(2):
                for dj in range(2):
                    U_full[(i_off+di)*m + (j_off+dj)][r] = u_term[di*2+dj]
            
            # V: B[j_off+dj, k_off+dk]
            for dj in range(2):
                for dk in range(2):
                    V_full[(j_off+dj)*p + (k_off+dk)][r] = v_term[dj*2+dk]
            
            # W: C[i_off+di, k_off+dk] -> W[(k_off+dk)*n + (i_off+di)]
            for dk in range(2):
                for di in range(2):
                    W_full[(k_off+dk)*n + (i_off+di)][r] = w_term[dk*2+di]
    
    # Add a 1x2x4 naive block at offsets (i_off, j_off, k_off)
    # i_off: row offset in A and C
    # j_off: col offset in A and row offset in B
    # k_off: col offset in B and col offset in C
    def add_naive_1x2x4(i_off, j_off, k_off, start_col):
        nonlocal col
        for dj in range(2):
            for dk in range(4):
                r = start_col + dj*4 + dk
                U_full[(i_off)*m + (j_off+dj)][r] = 1
                V_full[(j_off+dj)*p + (k_off+dk)][r] = 1
                W_full[(k_off+dk)*n + (i_off)][r] = 1
    
    # For i=0,1 (2 rows):
    # 4 blocks of 2x2x4, each consisting of two 2x2x2 Strassen blocks
    # Block (i_off=0, j_off=0, k_off=0): covers A[0:2,0:2] x B[0:2,0:4] -> C[0:2,0:4]
    #   Sub-block 1: 2x2x2 at (0,0,0)
    #   Sub-block 2: 2x2x2 at (0,0,2)
    add_strassen_2x2x2(0, 0, 0, col)
    col += 7
    add_strassen_2x2x2(0, 0, 2, col)
    col += 7
    
    # Block (i_off=0, j_off=0, k_off=4): covers A[0:2,0:2] x B[0:2,4:8] -> C[0:2,4:8]
    add_strassen_2x2x2(0, 0, 4, col)
    col += 7
    add_strassen_2x2x2(0, 0, 6, col)
    col += 7
    
    # Block (i_off=0, j_off=2, k_off=0): covers A[0:2,2:4] x B[2:4,0:4] -> C[0:2,0:4]
    add_strassen_2x2x2(0, 2, 0, col)
    col += 7
    add_strassen_2x2x2(0, 2, 2, col)
    col += 7
    
    # Block (i_off=0, j_off=2, k_off=4): covers A[0:2,2:4] x B[2:4,4:8] -> C[0:2,4:8]
    add_strassen_2x2x2(0, 2, 4, col)
    col += 7
    add_strassen_2x2x2(0, 2, 6, col)
    col += 7
    
    # For i=2 (1 row):
    # 2 blocks of 1x2x4
    # Block (i_off=2, j_off=0, k_off=0): covers A[2:3,0:2] x B[0:2,0:4] -> C[2:3,0:4]
    add_naive_1x2x4(2, 0, 0, col)
    col += 8
    
    # Block (i_off=2, j_off=0, k_off=4): covers A[2:3,0:2] x B[0:2,4:8] -> C[2:3,4:8]
    add_naive_1x2x4(2, 0, 4, col)
    col += 8
    
    # Block (i_off=2, j_off=2, k_off=0): covers A[2:3,2:4] x B[2:4,0:4] -> C[2:3,0:4]
    add_naive_1x2x4(2, 2, 0, col)
    col += 8
    
    # Block (i_off=2, j_off=2, k_off=4): covers A[2:3,2:4] x B[2:4,4:8] -> C[2:3,4:8]
    add_naive_1x2x4(2, 2, 4, col)
    col += 8
    
    # Verify col == total_rank
    if col != total_rank:
        # Fall back to naive
        return {"U": U_naive, "V": V_naive, "W": W_naive}
    
    result = {"U": U_full, "V": V_full, "W": W_full}
    
    # Verify
    if verify(U_full, V_full, W_full, total_rank):
        return result
    
    # If verification fails, return naive
    return {"U": U_naive, "V": V_naive, "W": W_naive}
