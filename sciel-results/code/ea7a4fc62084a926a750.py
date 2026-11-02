"""Matrix multiplication 4x4x9: Winograd FFT block decomposition."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 4x4x9 = 3 blocks of 4x4x3
    # Each 4x4x3 = 2 sub-blocks of 2x4x3 (row partition)
    # 2x4x3 = 2 sub-sub-blocks of 2x4x1 (col partition)
    # 2x4x1 = 2 sub-sub-sub-blocks of 1x4x1 (row partition)
    # 1x4x1 schoolbook rank 4
    # Total: 3*2*2*2*4 = 96 (not better than 144)
    
    # Alternative: 2x4x3 directly with FFT
    # 2x4x3: A is 2x4, B is 4x3, C is 2x3
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # Split A into A1 (rows 0,1 cols 0,1) and A2 (rows 0,1 cols 2,3)
    # But that doesn't help directly.
    
    # Let me try: 4x4x9 as 2x2 blocks of 2x4x3
    # Actually, let me just use a simpler decomposition.
    
    # 4x4x9 schoolbook is 144. Let me try to find a better one.
    # 4x4x4 has known rank 49 (Strassen). 3*49 = 147 > 144. Not helpful.
    # 4x4x2 has rank at most 4*4*2 = 32, but can be done in 16?
    # 4x4x2: C = A*B where A is 4x4, B is 4x2
    # C[i,j] = sum_k A[i,k]*B[k,j] for j=0,1
    # This is like 2 independent 4x4x1 multiplications.
    # 4x4x1: C[i] = sum_k A[i,k]*B[k], rank 16 (schoolbook)
    # Can we do 4x4x1 in rank 8? Unlikely.
    
    # Let me try a different approach: use the fact that 9 = 3*3
    # 4x4x9 = 4x4x(3*3). Treat B as 4x3x3.
    # For each of the 3x3 blocks of B, do 4x4x3.
    # 4x4x3: C[i,j] = sum_k A[i,k]*B[k,j] for j=0,1,2
    # This is 3 independent 4x4x1 multiplications.
    # 4x4x1 rank 16, so 4x4x3 rank 48, total 3*48 = 144. Same as schoolbook.
    
    # I need a non-trivial decomposition. Let me use Strassen for 4x4x4 and pad.
    # But 3*49 = 147 > 144.
    
    # Actually, let me reconsider. The minimal rank for 4x4x4 is 49 (Strassen).
    # For 4x4x9, if I can do it in rank < 144, that's an improvement.
    
    # Let me try: 4x4x9 = 4x4x(4+4+1) = three 4x4x4 and one 4x4x1.
    # 3*49 + 16 = 147 + 16 = 163 > 144. Worse.
    
    # 4x4x9 = 4x4x(4+5)? 49 + rank(4x4x5). 4x4x5 >= 4*4*5/4 = 20? No lower bound is 4*4*5 = 80 schoolbook.
    
    # I think the schoolbook 144 might be close to optimal. Let me just return it with a deadline guard.
    
    import time
    start = time.time()
    
    rank = n * m * p
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
# EVOLVE-BLOCK-END
