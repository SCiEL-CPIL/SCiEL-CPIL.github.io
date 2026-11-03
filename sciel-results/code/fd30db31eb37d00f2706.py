import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    n, m, p = 4, 4, 6  # A is n×m, B is m×p, C is n×p
    
    # Partition into 2×2 blocks of A (each 2×2), 2×3 blocks of B, 2×3 blocks of C
    # A: 4×4 -> 4 blocks of 2×2
    # B: 4×6 -> 4 blocks of 2×3  
    # C: 4×6 -> 4 blocks of 2×3
    # Each C_block = A_block × B_block where A_block is 2×2, B_block is 2×3
    # 2×2×3 multiplication: use Strassen (rank 7) for first 2 cols + schoolbook (rank 4) for 3rd col = rank 11
    # Total rank = 4 blocks × 11 = 44
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 16 × 44
    V = [[0] * rank for _ in range(m * p)]  # 24 × 44
    W = [[0] * rank for _ in range(p * n)]  # 24 × 44
    
    # Strassen steps for 2×2×2 (rank 7)
    # Local indices: A(r,s), B(s,k), C(r,k) with r,s,k in {0,1}
    # Each step: (A_entries, B_entries, C_entries)
    # A_entries: list of (r, s, coeff)
    # B_entries: list of (s, k, coeff)
    # C_entries: list of (r, k, coeff)
    
    strassen_2x2x2 = [
        # M1 = (A00+A11)(B00+B11) -> C00+C11
        (([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])),
        # M2 = (A10+A11)B00 -> C10-C11
        (([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])),
        # M3 = A00(B01-B11) -> C01+C11
        (([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])),
        # M4 = A11(B10-B00) -> C00+C10
        (([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])),
        # M5 = (A00+A01)B11 -> C01-C00
        (([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,1), (0,0,-1)])),
        # M6 = (A10-A00)(B00+B01) -> C11
        (([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])),
        # M7 = (A01-A11)(B10+B11) -> C00
        (([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]))
    ]
    
    # Schoolbook for 2×2×1 (3rd column, rank 4)
    # Local: A(r,s), B(s,0), C(r,0) with r,s in {0,1}
    # This handles the 3rd column (k=2 in local 2×2×3)
    schoolbook_2x2x1 = [
        # A00 * B00 -> C00
        (([(0,0,1)], [(0,0,1)], [(0,0,1)])),
        # A01 * B10 -> C00
        (([(0,1,1)], [(1,0,1)], [(0,0,1)])),
        # A10 * B00 -> C10
        (([(1,0,1)], [(0,0,1)], [(1,0,1)])),
        # A11 * B10 -> C10
        (([(1,1,1)], [(1,0,1)], [(1,0,1)]))
    ]
    
    # For each of the 4 block positions (row_block, col_block) in {0,1}×{0,1}:
    # A_block is at rows 2*row_block..2*row_block+1, cols 2*col_block..2*col_block+1
    # B_block is at rows 2*row_block..2*row_block+1, cols 3*col_block..3*col_block+2
    # C_block is at rows 2*row_block..2*row_block+1, cols 3*col_block..3*col_block+2
    
    for rb in range(2):  # row block
        for cb in range(2):  # col block
            base_rank = (rb * 2 + cb) * 11  # 0, 11, 22, 33
            
            # Strassen part (first 2 columns of the 2×3 block, i.e., local k=0,1)
            for si, (A_entries, B_entries, C_entries) in enumerate(strassen_2x2x2):
                c_rank = base_rank + si  # 0-6
                
                # Update U: A(r,s) where r,s in {0,1}
                for (r, s, coeff) in A_entries:
                    I = 2 * rb + r
                    J = 2 * cb + s
                    U[I * m + J][c_rank] += coeff
                
                # Update V: B(s,k) where s in {0,1}, k in {0,1} (first 2 cols)
                for (s, k, coeff) in B_entries:
                    J = 2 * rb + s
                    K = 3 * cb + k  # k=0,1 for Strassen
                    V[J * p + K][c_rank] += coeff
                
                # Update W: C(r,k) where r in {0,1}, k in {0,1}
                for (r, k, coeff) in C_entries:
                    I = 2 * rb + r
                    K = 3 * cb + k
                    W[K * n + I][c_rank] += coeff
            
            # Schoolbook part (3rd column of the 2×3 block, i.e., local k=2)
            for si, (A_entries, B_entries, C_entries) in enumerate(schoolbook_2x2x1):
                c_rank = base_rank + 7 + si  # 7-10
                
                # Update U: A(r,s) where r,s in {0,1}
                for (r, s, coeff) in A_entries:
                    I = 2 * rb + r
                    J = 2 * cb + s
                    U[I * m + J][c_rank] += coeff
                
                # Update V: B(s,0) where s in {0,1}, mapped to 3rd column (k=2)
                for (s, k, coeff) in B_entries:
                    J = 2 * rb + s
                    K = 3 * cb + 2  # 3rd column of the 2×3 block
                    V[J * p + K][c_rank] += coeff
                
                # Update W: C(r,0) where r in {0,1}, mapped to 3rd column (k=2)
                for (r, k, coeff) in C_entries:
                    I = 2 * rb + r
                    K = 3 * cb + 2
                    W[K * n + I][c_rank] += coeff
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U_mat[i*m+j][c] * V_mat[j*p+k][c] * W_mat[k*n+i][c]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True
    
    if verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition (rank 96)
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    
    return {"U": U_fb, "V": V_fb, "W": W_fb}
