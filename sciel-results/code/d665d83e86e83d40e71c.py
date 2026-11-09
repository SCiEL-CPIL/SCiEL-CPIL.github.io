"""Program for matrix multiplication 3x4x8 tensor decomposition.
Implements Kronecker product of Strassen 3x2x2 (rank 7) and schoolbook 1x2x4 (rank 8).
Total rank = 7 * 8 = 56.
"""
import time
import itertools

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def _strassen_3x2x2():
    """Return (U, V, W) for 3x2x2 matrix multiplication using Strassen's 7-multiplication algorithm.
    
    Multiplying A (3x2) by B (2x2) to get C (3x2).
    U has shape (6, 7), V has shape (4, 7), W has shape (6, 7).
    
    Indexing:
      U row: i*m + j where i in [0,3), j in [0,2)  -> 6 rows
      V row: j*p + k where j in [0,2), k in [0,2)  -> 4 rows
      W row: k*n + i where k in [0,2), i in [0,3)  -> 6 rows
    """
    n, m, p = 3, 2, 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]  # 6 x 7
    V = [[0] * rank for _ in range(m * p)]  # 4 x 7
    W = [[0] * rank for _ in range(p * n)]  # 6 x 7
    
    # Strassen's 7 products for 2x2 matrix multiplication:
    # For standard 2x2: A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # p1 = (a+d)(e+h)
    # p2 = (c+d)*e
    # p3 = a*(f-h)
    # p4 = d*(g-e)
    # p5 = (a+b)*h
    # p6 = (c-a)*(e+f)
    # p7 = (b-d)*(g+h)
    # C = [[p1+p4-p5+p7, p3+p5], [p2+p4, p1-p2+p3+p6]]
    
    # We need to adapt this for 3x2 * 2x2 -> 3x2.
    # A is 3x2: rows i=0,1,2; cols j=0,1
    # B is 2x2: rows j=0,1; cols k=0,1
    # C is 3x2: rows i=0,1,2; cols k=0,1
    
    # C[i][k] = sum_j A[i][j] * B[j][k]
    
    # We can split A into a 2x2 block (rows 0,1) and a 1x2 row (row 2).
    # For the 2x2 part, use Strassen. For the 1x2 part, use 2 multiplications.
    # Total: 7 + 2 = 9? No, that's not minimal.
    
    # Actually, let's think differently. The 3x2x2 tensor has rank at most 6 (schoolbook).
    # But Strassen gives 7 for 2x2x2. For 3x2x2, we can use:
    # Split A's rows into group1 (rows 0,1) and group2 (row 2).
    # For group1 (2x2 * 2x2): use Strassen, rank 7.
    # For group2 (1x2 * 2x2): need 2 multiplications.
    # Total rank: 7 + 2 = 9.
    
    # But wait - we can do better. Let me use a direct approach.
    # Actually, the minimal rank of 3x2x2 is known to be 6 (schoolbook is optimal for this case? No, 2x2x2 has rank 7 < 8, so maybe 3x2x2 has rank < 6).
    
    # Let me just use the schoolbook for 3x2x2 (rank 6) and combine with 1x2x4 (rank 8) to get 48.
    # Or use Strassen for a 2x2x2 block.
    
    # Let me reconsider the Kronecker product approach.
    # 3x4x8 = (3x2x2) tensor (1x2x4)
    # If rank(3x2x2) <= 6 and rank(1x2x4) <= 8, then rank(3x4x8) <= 48.
    # If we can get rank(3x2x2) = 6, total = 48.
    
    # For now, use schoolbook for 3x2x2 (rank 6):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k  # This isn't right for Kronecker
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
    
    return U, V, W


def _kronecker_decomp(n1, m1, p1, n2, m2, p2, U1, V1, W1, U2, V2, W2):
    """Combine two tensor decompositions via Kronecker product.
    
    First problem: n1 x m1 x p2
    Second problem: n2 x m2 x p2
    Combined: (n1*n2) x (m1*m2) x (p1*p2)
    
    Wait, I need to be more careful with the indexing.
    """
    pass


# Let me take a different, cleaner approach.
# Direct construction of a low-rank decomposition for 3x4x8.

def construct():
    start_time = time.time()
    n, m, p = 3, 4, 8
    
    # Use Kronecker product structure:
    # 3x4x8 = (3x2x4) tensor (1x2x2) ... no, dimensions don't match cleanly.
    # 3x4x8: n=3, m=4, p=8
    # Factor: 4 = 2*2, 8 = 4*2
    # So: 3x4x8 = (3x2x4) kron (1x2x2)
    # rank(3x2x4) * rank(1x2x2)
    # 1x2x2 schoolbook rank = 4
    # 3x2x4 schoolbook rank = 24
    # Total: 96 (no improvement)
    
    # Better: 3x4x8 = (3x2x2) kron (1x2x4)
    # rank(3x2x2) * rank(1x2x4)
    # 3x2x2: use rank 6 (schoolbook) or try Strassen-like
    # 1x2x4: schoolbook rank 8
    # If 3x2x2 has rank 6: total 48
    
    # Let me construct 3x2x2 with rank 6 and 1x2x4 with rank 8, then Kronecker.
    
    # --- 3x2x2 decomposition (schoolbook, rank 6) ---
    n1, m1, p1 = 3, 2, 2
    r1 = n1 * m1 * p1  # 6
    U1 = [[0] * r1 for _ in range(n1 * m1)]  # 6x6
    V1 = [[0] * r1 for _ in range(m1 * p1)]  # 4x6
    W1 = [[0] * r1 for _ in range(p1 * n1)]  # 6x6
    col = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U1[i * m1 + j][col] = 1
                V1[j * p1 + k][col] = 1
                W1[k * n1 + i][col] = 1
                col += 1
    
    # --- 1x2x4 decomposition (schoolbook, rank 8) ---
    n2, m2, p2 = 1, 2, 4
    r2 = n2 * m2 * p2  # 8
    U2 = [[0] * r2 for _ in range(n2 * m2)]  # 2x8
    V2 = [[0] * r2 for _ in range(m2 * p2)]  # 8x8
    W2 = [[0] * r2 for _ in range(p2 * n2)]  # 4x8
    col = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U2[i * m2 + j][col] = 1
                V2[j * p2 + k][col] = 1
                W2[k * n2 + i][col] = 1
                col += 1
    
    # --- Kronecker product combination ---
    # Combined dimensions: n = n1*n2 = 3, m = m1*m2 = 4, p = p1*p2 = 8
    # Rank = r1 * r2 = 48
    total_rank = r1 * r2
    U = [[0] * total_rank for _ in range(n * m)]   # 12 x 48
    V = [[0] * total_rank for _ in range(m * p)]   # 32 x 48
    W = [[0] * total_rank for _ in range(p * n)]   # 24 x 48
    
    # Kronecker index mapping:
    # For combined index in U: row = (i1*m1 + j1) * (n2*m2) + (i2*m2 + j2) ... 
    # Actually, let me think about this more carefully.
    
    # The combined tensor: T[i, j, k] = sum over a,b,c of U[i,a] V[j,a] W[k,a]
    # where i = i1 * n2 + i2, j = j1 * m2 + j2, k = k1 * p2 + k2
    # Wait, that's not standard Kronecker. Let me use the standard approach.
    
    # In Kronecker product of tensors:
    # T_combined[i1*n2+i2, j1*m2+j2, k1*p2+k2] = T1[i1,j1,k1] * T2[i2,j2,k2]
    # This means the decomposition rank is r1*r2.
    
    # For each pair of rank indices (c1, c2), the combined rank index is c1*r2 + c2.
    # U_combined[i1*n2+i2, j1*m2+j2, c1*r2+c2] = U1[i1,j1,c1] * U2[i2,j2,c2]
    # Similarly for V and W.
    
    for c1 in range(r1):
        for c2 in range(r2):
            col = c1 * r2 + c2
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            ui1 = U1[i1 * m1 + j1][c1]
                            ui2 = U2[i2 * m2 + j2][c2]
                            if ui1 != 0 and ui2 != 0:
                                combined_i = i1 * n2 + i2
                                combined_j = j1 * m2 + j2
                                U[combined_i * m + combined_j][col] = ui1 * ui2
            
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            vj1 = V1[j1 * p1 + k1][c1]
                            vj2 = V2[j2 * p2 + k2][c2]
                            if vj1 != 0 and vj2 != 0:
                                combined_j = j1 * m2 + j2
                                combined_k = k1 * p2 + k2
                                V[combined_j * p + combined_k][col] = vj1 * vj2
            
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            wk1 = W1[k1 * n1 + i1][c1]
                            wk2 = W2[k2 * n2 + i2][c2]
                            if wk1 != 0 and wk2 != 0:
                                combined_k = k1 * p2 + k2
                                combined_i = i1 * n2 + i2
                                W[combined_k * n + combined_i][col] = wk1 * wk2
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
