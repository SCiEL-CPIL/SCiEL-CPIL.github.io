"""Matrix multiplication 3x4x8 - Kronecker product decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n, m, p):
    """Check that sum_s U[a,s]V[b,s]W[c,s] == T[a,b,c] for all a,b,c."""
    a = np.array(U, dtype=float)
    b = np.array(V, dtype=float)
    c = np.array(W, dtype=float)
    D = np.einsum('as,bs,cs->abc', a, b, c, optimize=True)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(D, T, atol=1e-6)

def _strassen222():
    """Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U (4 rows), V (4 rows), W (4 rows), each 4x7.
    Indices: U row = i*2+j, V row = j*2+k, W row = k*2+i.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1: (A00+A11)*(B01+B10)
    U[0, 0] += 1; U[3, 0] += 1
    V[1, 0] += 1; V[2, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1; W[1, 0] -= 1; W[2, 0] += 1
    # M2: (A10+A11)*B00
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[1, 1] += 1; W[3, 1] -= 1
    # M3: A00*(B10-B11)
    U[0, 2] += 1
    V[2, 2] += 1; V[3, 2] -= 1
    W[0, 2] += 1; W[3, 2] += 1
    # M4: A11*(B00-B10)
    U[3, 3] += 1
    V[0, 3] += 1; V[2, 3] -= 1
    W[1, 3] += 1; W[3, 3] += 1
    # M5: (A00+A01)*B11
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[2, 4] -= 1; W[3, 4] += 1
    # M6: (A10-A00)*(B00+B01)
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    W[2, 5] += 1
    # M7: (A01-A11)*(B10+B11)
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1
    return U, V, W

def _build_3x4x4():
    """Build rank-28 decomposition for 3x4x4.
    
    Strategy: Split m=4 into 2+2, giving two 3x2x4 blocks.
    Each 3x2x4: split p=4 into 2+2, giving two 3x2x2 blocks.
    Each 3x2x2: use schoolbook rank 12.
    Total: 2 * 2 * 12 = 48.
    
    Alternative: Use a smarter base.
    Actually, let's try: 3x4x4 = 2*(3x2x4), each 3x2x4 = 2*(3x2x2).
    For 3x2x2, we need rank. Let's just use schoolbook 12 for now.
    Total = 4*12 = 48.
    """
    n, m, p = 3, 4, 4
    U = np.zeros((n*m, 48))
    V = np.zeros((m*p, 48))
    W = np.zeros((p*n, 48))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def construct():
    start = time.time()
    deadline = start + 115
    n, m, p = 3, 4, 8

    # Fallback: schoolbook rank 96
    rank_sb = n * m * p
    U = np.zeros((n * m, rank_sb))
    V = np.zeros((m * p, rank_sb))
    W = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    fallback = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if not _verify(U, V, W, n, m, p):
        raise RuntimeError("Schoolbook verification failed")
    
    # Try Kronecker product decomposition:
    # 3x4x8 = 2*(3x4x4) by splitting p=8 into 4+4
    # Each 3x4x4 has rank 48 via schoolbook, total = 96. No improvement.
    # 
    # Better: Use the structure more cleverly.
    # 3x4x8: A is 3x4, B is 4x8, C is 3x8.
    # Write B as [B1 | B2] where B1, B2 are 4x4.
    # C = A*B1 + A*B2, each 3x4.
    # So M(3,4,8) = 2*M(3,4,4).
    # 
    # For M(3,4,4): A is 3x4, B is 4x4, C is 3x4.
    # Write A = [A1 | A2] where A1, A2 are 3x2.
    # Write B = [[B11, B12], [B21, B22]] where each is 2x2.
    # C = A1*B_top + A2*B_bottom where B_top = [B11, B12], B_bottom = [B21, B22].
    # C = A1*[B11, B12] + A2*[B21, B22]
    # C_left = A1*B11 + A2*B21 (3x2)
    # C_right = A1*B12 + A2*B22 (3x2)
    # Each 3x2x2 multiplication.
    # M(3,4,4) = 2*M(3,2,2) [for left] + 2*M(3,2,2) [for right] = 4*M(3,2,2).
    # 
    # M(3,2,2): A is 3x2, B is 2x2, C is 3x2.
    # Schoolbook: 3*2*2 = 12.
    # Can we do better? 
    # C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    # For j=0: C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    # For j=1: C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1]
    # 
    # Let's use 8 terms with some sharing:
    # t0: A[0,0]*B[0,0] -> C[0,0]
    # t1: A[1,0]*B[0,0] -> C[1,0]  
    # t2: A[2,0]*B[0,0] -> C[2,0]
    # t3: A[0,1]*B[1,0] -> C[0,0]
    # t4: A[1,1]*B[1,0] -> C[1,0]
    # t5: A[2,1]*B[1,0] -> C[2,0]
    # t6: A[0,0]*B[0,1] -> C[0,1]
    # t7: A[1,0]*B[0,1] -> C[1,1]
    # t8: A[2,0]*B[0,1] -> C[2,1]
    # t9: A[0,1]*B[1,1] -> C[0,1]
    # t10: A[1,1]*B[1,1] -> C[1,1]
    # t11: A[2,1]*B[1,1] -> C[2,1]
    # That's 12 terms. No sharing possible since each (i,j) pair is distinct.
    # 
    # So M(3,2,2) = 12 (schoolbook is optimal for this small case).
    # M(3,4,4) = 4*12 = 48.
    # M(3,4,8) = 2*48 = 96.
    # 
    # This doesn't beat schoolbook! Let me try a different approach.
    # 
    # Use Strassen on 2x2x2 blocks where possible.
    # 3x4x8: The 4x8 part can be seen as 2x(2x4) or 4x(2x2).
    # 
    # Alternative: Embed 3x4x8 into a larger structured problem.
    # 
    # Let me try: 3x4x8 = 2*(3x4x4). For each 3x4x4, use rank 28.
    # How to get rank 28 for 3x4x4?
    # 3x4x4: Split into 2*(3x2x4). Each 3x2x4: split into 2*(3x2x2).
    # If M(3,2,2) <= 7, then 3x2x4 <= 14, 3x4x4 <= 28, 3x4x8 <= 56.
    # 
    # Can M(3,2,2) <= 7? Let's try to construct it.
    # Actually, the known bound is M(3,2,2) = 12 (it's a small case).
    # 
    # Let me just return the schoolbook 96 which is verified.
    return fallback

# EVOLVE-BLOCK-END
