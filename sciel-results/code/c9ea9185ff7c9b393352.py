"""Program for matrix multiplication tensor decomposition 3x3x6."""
import time

DIMENSIONS = [3, 3, 6]

def construct():
    # Use two independent 3x3x3 blocks (k mod 3 = 0 and k mod 3 = 1)
    # Each 3x3x3 multiplication uses Strassen's rank-7 algorithm.
    # Total rank = 14.
    
    # Strassen's 7 terms for 2x2 matrix multiplication:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    n, m, p = 3, 3, 6
    rank = 14  # 2 slices * 7 terms per slice
    
    U = [[0] * rank for _ in range(n * m)]  # shape (9, 14)
    V = [[0] * rank for _ in range(m * p)]  # shape (18, 14)
    W = [[0] * rank for _ in range(p * n)]  # shape (18, 14)
    
    # For each slice (k0 in {0, 1}), we handle columns k = k0, k0+3
    # The 3x3 multiplication for each slice uses Strassen on 2x2 submatrices
    # plus schoolbook for the remaining terms involving the 3rd row/column.
    
    # Actually, let me reconsider. Strassen works for 2x2. For 3x3, we can
    # split into a 2x2 block and handle the rest with schoolbook terms.
    # 3x3 = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, etc.
    # This gets complex. Let me use a known rank for 3x3 matrix multiplication.
    # The tensor rank of 3x3 matrix multiplication is known to be at most 19
    # (from de Groote et al.) but finding an explicit rank-19 decomposition is complex.
    
    # Let me use a simpler approach: schoolbook for 3x3x6 gives rank 54.
    # Strassen for two 3x3x3 blocks: if I can get rank 7 per 3x3x3, total is 14.
    # But Strassen is for 2x2. For 3x3, I need to extend it.
    
    # Alternative: use the fact that 3x3x6 can be seen as two 3x3x3 multiplications.
    # For each 3x3x3, use a rank-19 decomposition (best known upper bound is 19).
    # Total rank would be 38.
    
    # Let me implement a known rank-19 decomposition for 3x3 matrix multiplication.
    # This is complex, so let me use a safer approach.
    
    # Safe approach: schoolbook rank 54.
    # Or use two 3x3x3 with rank ~19 each = 38.
    
    # Let me just do schoolbook to be safe, then try to optimize.
    rank = n * m * p  # 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
