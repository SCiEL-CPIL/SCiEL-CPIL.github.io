import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=3, m=5, p=7
    
    # 6 Strassen 2x2x2 blocks
    strassen_blocks = []
    for i0 in [0]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                strassen_blocks.append((i0, j0, k0))
    
    # Determine which (i,j,k) triples are covered
    covered = set()
    for i0, j0, k0 in strassen_blocks:
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    covered.add((i0+a, j0+b, k0+c))
    
    rank_strassen = 7 * len(strassen_blocks)
    rank_schoolbook = n * m * p - len(covered)
    rank = rank_strassen + rank_schoolbook
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i0, j0, k0 in strassen_blocks:
        # P1 = (A00+A11)(B00+B11) -> C00=+1, C11=+1
        U[i0*m + j0][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        V[j0*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        
        # P2 = (A10+A11)B00 -> C10=+1, C11=-1
        U[(i0+1)*m + j0][col] = 1
        U[(i0+1)*m + (j0+1)][col] = 1
        V[j0*p + k0][col] = 1
        W[k0*n + (i0+1)][col] = 1
        W[(k0+1)*n + (i0+1)][col] = -1
        col += 1
        
        # P3 = A00(B10-B11) -> C01=+1, C11=+1
        U[i0*m + j0][col] = 1
        V[(j0+1)*p + k0][col] = 1
        V[(j0+1)*p + (k0+1)][col] = -1
        W[(k0+1)*n + i0][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        
        # P4 = A11(B01-B10) -> C00=+1, C10=+1
        U[(i0+1)*m + (j0+1)][col] = 1
        V[j0*p + (k0+1)][col] = 1
        V[(j0+1)*p + k0][col] = -1
        W[k0*n + i0][col] = 1
        W[k0*n + (i0+1)][col] = 1
        col += 1
        
        # P5 = (A00+A01)B11 -> C01=+1, C00=-1
        U[i0*m + j0][col] = 1
        U[i0*m + (j0+1)][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[(k0+1)*n + i0][col] = 1
        W[k0*n + i0][col] = -1
        col += 1
        
        # P6 = (A10-A00)(B00+B10) -> C11=+1
        U[(i0+1)*m + j0][col] = 1
        U[i0*m + j0][col] = -1
        V[j0*p + k0][col] = 1
        V[(j0+1)*p + k0][col] = 1
        W[(k0+1)*n + (i0+1)][col] = 1
        col += 1
        
        # P7 = (A01-A11)(B01+B11) -> C00=+1
        U[i0*m + (j0+1)][col] = 1
        U[(i0+1)*m + (j0+1)][col] = -1
        V[j0*p + (k0+1)][col] = 1
        V[(j0+1)*p + (k0+1)][col] = 1
        W[k0*n + i0][col] = 1
        col += 1
    
    # Schoolbook for remaining
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
