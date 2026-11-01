import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = 110.0

    n, m, p = DIMENSIONS  # 2, 3, 7
    # n=2 rows, m=3 inner, p=7 cols
    
    # Construct rank-39 decomposition
    # 
    # For each k (0..6), we compute:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    #
    # Use Strassen on the 2x2 part (j=0,1) for each k:
    # This gives 7 multiplications per k, but Strassen computes a full 2x2 product
    # (4 outputs), while we only need 2 outputs (one per row, for this k).
    #
    # Actually, Strassen for 2x2x2 computes C00, C01, C10, C11 from A(2x2) and B(2x2).
    # In our case, for fixed k, A is 2x2 (rows 0,1; cols 0,1) and B is 2x1 (rows 0,1; col k).
    # So it's a 2x2x1 problem, not 2x2x2.
    #
    # For 2x2x1: C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    #             C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # Rank of 2x2x1 is 4 (schoolbook) or 3? 
    # Actually, 2x2x1 has rank 4 (it's just two independent 2-term dot products).
    #
    # So Strassen doesn't help for 2x2x1.
    #
    # The saving must come from a different structure.
    
    # Let me try: group the 7 columns into groups and use Strassen on 2x2x2 blocks.
    # For example, use columns (0,1) together:
    # C[:,0:2] = A[:,0:2] * B[0:2, 0:2] + A[:,2] * B[2, 0:2]
    # The first part is 2x2x2, which can use Strassen (rank 7 vs 8).
    # Save 1 rank.
    #
    # Do this for column pairs (0,1), (2,3), (4,5), and handle col 6 separately.
    # 3 pairs * save 1 = save 3 ranks. Total: 42-3 = 39. YES!
    
    # Construction:
    # For column pairs (0,1), (2,3), (4,5): use Strassen on 2x2x2 (j=0,1; k pair)
    #   Wait, but j has 3 values, not 2. Let me reconsider.
    #
    # For a column pair (k1, k2):
    # C[:,k1:k2+1] = A * B[:,k1:k2+1]
    # = A[:,0:2] * B[0:2,k1:k2+1] + A[:,2] * B[2,k1:k2+1]
    #
    # A[:,0:2] is 2x2, B[0:2,k1:k2+1] is 2x2. Their product is 2x2x2, rank 7 with Strassen.
    # A[:,2] is 2x1, B[2,k1:k2+1] is 1x2. Their product is 2x1x2, rank 4.
    # Total per pair: 7 + 4 = 11 vs schoolbook 8+4=12. Save 1.
    #
    # For 3 pairs: save 3. For the remaining column (col 6):
    # C[:,6] = A[:,0:2]*B[0:2,6] + A[:,2]*B[2,6]
    # = 2x2x1 (rank 4) + 2x1x1 (rank 2) = 6 vs schoolbook 6. No saving.
    #
    # Total: 3*11 + 6 = 33+6 = 39. 
    
    # Now let me build the explicit U, V, W matrices.
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]  # 6 x 39
    V = [[0] * rank for _ in range(m * p)]  # 21 x 39
    W = [[0] * rank for _ in range(p * n)]  # 14 x 39
    
    col = 0
    
    # Process column pairs (0,1), (2,3), (4,5) with Strassen
    for pair_start in [0, 2, 4]:
        k1 = pair_start
        k2 = pair_start + 1
        
        # Strassen for A[:,0:2] (2x2) * B[0:2,k1:k2+1] (2x2)
        # A indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        # B indices: (0,k1)->k1, (0,k2)->k2, (1,k1)->p+k1, (1,k2)->p+k2
        # W indices: (k1,0)->k1*2+0, (k1,1)->k1*2+1, (k2,0)->k2*2+0, (k2,1)->k2*2+1
        
        # M1 = (A00+A11) * (B00+B11)
        U[0][col] = 1; U[3][col] = 1
        V[k1][col] = 1; V[p + k2][col] = 1
        W[k1 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
        W[k2 * n + 0][col] = 1; W[k2 * n + 1][col] = 1
        col += 1
        
        # M2 = (A10+A11) * B00
        U[2][col] = 1; U[3][col] = 1
        V[k1][col] = 1
        W[k2 * n + 0][col] = 1; W[k2 * n + 1][col] = -1
        col += 1
        
        # M3 = A00 * (B10-B11)
        U[0][col] = 1
        V[p + k1][col] = 1; V[p + k2][col] = -1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M4 = A11 * (B01-B00)
        U[3][col] = 1
        V[k2][col] = 1; V[k1][col] = -1
        W[k1 * n + 0][col] = 1; W[k2 * n + 0][col] = 1
        col += 1
        
        # M5 = (A00+A01) * B11
        U[0][col] = 1; U[1][col] = 1
        V[p + k2][col] = 1
        W[k1 * n + 0][col] = -1; W[k2 * n + 1][col] = 1
        col += 1
        
        # M6 = (A10-A00) * (B00+B10)
        U[2][col] = 1; U[0][col] = -1
        V[k1][col] = 1; V[p + k1][col] = 1
        W[k2 * n + 0][col] = 1
        col += 1
        
        # M7 = (A01-A11) * (B01+B11)
        U[1][col] = 1; U[3][col] = -1
        V[k2][col] = 1; V[p + k2][col] = 1
        W[k1 * n + 1][col] = 1; W[k2 * n + 1][col] = 1
        col += 1
        
        # Schoolbook for A[:,2] (2x1) * B[2,k1:k2+1] (1x2)
        # C[0,k1] += A[0,2]*B[2,k1]
        U[5][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # C[0,k2] += A[0,2]*B[2,k2]
        U[5][col] = 1
        V[2 * p + k2][col] = 1
        W[k2 * n + 0][col] = 1
        col += 1
        
        # C[1,k1] += A[1,2]*B[2,k1]
        U[5][col] = 1  # Wait, A[1,2] is index 1*m+2 = 5
        V[2 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # C[1,k2] += A[1,2]*B[2,k2]
        U[5][col] = 1
        V[2 * p + k2][col] = 1
        W[k2 * n + 1][col] = 1
        col += 1
    
    # Process remaining column 6 with schoolbook
    k = 6
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # A[0,0]*B[0,k]
    U[0][col] = 1; V[k][col] = 1; W[k * n + 0][col] = 1
    col += 1
    # A[0,1]*B[1,k]
    U[1][col] = 1; V[p + k][col] = 1; W[k * n + 0][col] = 1
    col += 1
    # A[0,2]*B[2,k]
    U[5][col] = 1; V[2 * p + k][col] = 1; W[k * n + 0][col] = 1
    col += 1
    # A[1,0]*B[0,k]
    U[2][col] = 1; V[k][col] = 1; W[k * n + 1][col] = 1
    col += 1
    # A[1,1]*B[1,k]
    U[3][col] = 1; V[p + k][col] = 1; W[k * n + 1][col] = 1
    col += 1
    # A[1,2]*B[2,k]
    U[5][col] = 1; V[2 * p + k][col] = 1; W[k * n + 1][col] = 1
    col += 1
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
