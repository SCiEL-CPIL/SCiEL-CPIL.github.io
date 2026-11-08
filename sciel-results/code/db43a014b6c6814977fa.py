"""Tensor decomposition for 2x3 matrix multiplication (A@B, B is 3x10)."""
import time

def construct():
    _t0 = time.time()
    n, m, p = 2, 3, 10
    # We'll build a decomposition using block decomposition.
    # A is 2x3 (indices: row 0: a00,a01,a02; row 1: a10,a11,a12)
    # B is 3x10 (indices: row j, col k: B[j*10+k])
    # C is 2x10 (indices: C[i*10+k] = sum_j A[i,j]*B[j,k])
    # W index convention: W[k*n + i, r] = W[2*k + i, r]
    
    # Strategy: Use 2x3x2 blocks for each pair of columns in B.
    # For each pair of columns (k0, k1), we decompose:
    #   C[i,k0] = sum_j A[i,j]*B[j,k0]
    #   C[i,k1] = sum_j A[i,j]*B[j,k1]
    # This is a 2x3x2 matrix multiplication tensor.
    
    # For 2x3x2, we can use a rank-7 decomposition for the 2x2x2 part 
    # (using A's first 2 columns) plus 2 direct terms for A's 3rd column.
    # Actually, let's use a direct approach:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # For 2x3x2, rank is at most 6 (direct) but can be lower.
    # Let's use the known rank-6 decomposition for 2x3x2:
    # Actually, the trivial decomposition has rank 6 (one term per product).
    # But we can do better. Let me think...
    
    # For 2x2x2, rank is 7 (Laderman). For 2x3x2, it should be at least 7.
    # Let's just use the direct rank-6 per pair: 5 pairs * 6 = 30 total.
    # Wait, that's already better than 55!
    
    # Direct decomposition: C[i,k] = sum_j A[i,j]*B[j,k]
    # This requires n*m*p = 2*3*10 = 60 terms if fully direct.
    # But we can factor: for fixed (i,j), A[i,j]*B[j,k] contributes to C[i,k].
    # The tensor is sum over i,j,k of A[i,j]*B[j,k]*C[i,k]^*
    # Each (i,j) pair gives a factor A[i,j] * B[j,*] which is a rank-p factor.
    # So we need at most n*m = 6 factors if we can share B columns... but we can't.
    
    # Actually, the standard matrix multiplication tensor M_{2,3,10} has:
    # T = sum_{i=0}^{1} sum_{j=0}^{2} sum_{k=0}^{9} A[i,j] * B[j,k] * C[i,k]
    # The rank is at most 60 (each term separate), but we can group.
    # For fixed i,j: sum_k A[i,j]*B[j,k]*C[i,k] = A[i,j] * (sum_k B[j,k]*C[i,k])
    # This doesn't help directly.
    
    # Let me just use the block approach with 2x3x2 blocks.
    # For each 2-column block, use rank-7 (Laderman for 2x2x2) + 2 extra terms.
    # Actually, let me just do the straightforward thing:
    # Total rank = 60 (fully expanded), but we can do 30 with some sharing.
    
    # Let's try: for each k (10 columns), we need C[0,k] and C[1,k].
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # Using Laderman on the 2x2 part (A[0:2,0:2] * B[0:2,k]):
    # For each k, we need rank-7 for 2x2x1... but that's not standard.
    
    # Let me just use the trivial decomposition with rank 60, then see if I can reduce.
    # Actually, let me use rank 30: pair up columns and use rank-6 per pair.
    
    # For columns (k0, k1), define:
    # C[i,k0] and C[i,k1] for i=0,1
    # Using A's columns 0,1,2 and B's rows 0,1,2
    
    # Direct: 6 terms per pair (A[i,j]*B[j,k] for i in 0,1; j in 0,1,2; but k is shared)
    # Wait, for fixed k, it's 3 terms (j=0,1,2) per row i.
    # For 2 columns, it's 2*3*2 = 12 terms if fully expanded per column.
    # But A[i,j] is shared across k, so we can factor.
    
    # Let me just build it directly with rank 60 and verify.
    rank = 60
    U = [[0] * rank for _ in range(n * m)]  # 6 x rank
    V = [[0] * rank for _ in range(m * p)]  # 30 x rank
    W = [[0] * rank for _ in range(p * n)]  # 20 x rank
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    if time.time() - _t0 > 140:
        return {"U": U, "V": V, "W": W}
    
    # Now try to reduce rank using algebraic structure.
    # For 2x3x10, we can use the fact that 10 = 5*2 and use 2x3x2 blocks.
    # Let me try a cleaner block decomposition.
    
    # Actually, let me try to use rank 30 with a smarter approach.
    # For each pair of B columns (k0, k1), use a rank-6 decomposition:
    # Terms:
    # 1. A[0,0] * B[0,k0] -> C[0,k0]
    # 2. A[0,0] * B[0,k1] -> C[0,k1]
    # 3. A[1,0] * B[0,k0] -> C[1,k0]
    # 4. A[1,0] * B[0,k1] -> C[1,k1]
    # 5. A[0,1] * B[1,k0] -> C[0,k0]
    # 6. A[0,1] * B[1,k1] -> C[0,k1]
    # 7. A[1,1] * B[1,k0] -> C[1,k0]
    # 8. A[1,1] * B[1,k1] -> C[1,k1]
    # 9. A[0,2] * B[2,k0] -> C[0,k0]
    # 10. A[0,2] * B[2,k1] -> C[0,k1]
    # 11. A[1,2] * B[2,k0] -> C[1,k0]
    # 12. A[1,2] * B[2,k1] -> C[1,k1]
    # That's 12 per pair, 60 total. Not better.
    
    # The current rank-60 is valid. Let me try to verify and return it.
    # But the task says "rank-minimal". Let me at least try rank 55 from the original.
    
    # Rebuild with the original block approach (5 blocks of 11) = rank 55.
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        # M1: (A00+A11)*(B0k0+B1k1) -> C0k0, C1k1
        U[0][col] += 1; U[4][col] += 1
        V[0 * 10 + k0][col] += 1; V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1; W[k1 * 2 + 1][col] += 1
        col += 1
        # M2: (A10+A11)*B0k0 -> C1k0, -C1k1
        U[3][col] += 1; U[4][col] += 1
        V[0 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1; W[k1 * 2 + 1][col] -= 1
        col += 1
        # M3: A00*(B0k1-B1k1) -> C0k1, C1k1
        U[0][col] += 1
        V[0 * 10 + k1][col] += 1; V[1 * 10 + k1][col] -= 1
        W[k1 * 2 + 0][col] += 1; W[k1 * 2 + 1][col] += 1
        col += 1
        # M4: A11*(B1k0-B0k0) -> C0k0, C1k0
        U[4][col] += 1
        V[1 * 10 + k0][col] += 1; V[0 * 10 + k0][col] -= 1
        W[k0 * 2 + 0][col] += 1; W[k0 * 2 + 1][col] += 1
        col += 1
        # M5: (A00+A01)*B1k1 -> -C0k0, C0k1
        U[0][col] += 1; U[1][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] -= 1; W[k1 * 2 + 0][col] += 1
        col += 1
        # M6: (A10-A00)*(B0k0+B0k1) -> C1k1
        U[3][col] += 1; U[0][col] -= 1
        V[0 * 10 + k0][col] += 1; V[0 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        # M7: (A01-A11)*(B1k0+B1k1) -> C0k0
        U[1][col] += 1; U[4][col] -= 1
        V[1 * 10 + k0][col] += 1; V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        # Row 2 of B terms
        U[2][col] += 1; V[2 * 10 + k0][col] += 1; W[k0 * 2 + 0][col] += 1
        col += 1
        U[2][col] += 1; V[2 * 10 + k1][col] += 1; W[k1 * 2 + 0][col] += 1
        col += 1
        U[5][col] += 1; V[2 * 10 + k0][col] += 1; W[k0 * 2 + 1][col] += 1
        col += 1
        U[5][col] += 1; V[2 * 10 + k1][col] += 1; W[k1 * 2 + 1][col] += 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
