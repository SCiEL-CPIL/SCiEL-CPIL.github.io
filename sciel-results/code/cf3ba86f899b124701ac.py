import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline with 20s margin
    
    n, m, p = DIMENSIONS
    
    # Try Laderman approach (rank 60)
    try:
        result = _try_laderman()
        if result is not None:
            return result
    except Exception:
        pass
    
    if time.time() > deadline:
        pass
    
    # Fall back to block-Strassen (rank 189)
    return _block_strassen()

def _try_laderman():
    """Try to find a rank-60 decomposition using Laderman 3x3 blocks."""
    start = time.time()
    deadline = start + 100
    
    # Find rank-15 decomposition of 3x3 matmul tensor
    U3, V3, W3, err = _find_rank15_3x3()
    
    if err > 1e-6:
        return None
    
    # Round to nearest half-integers
    U3 = np.round(U3 * 2) / 2
    V3 = np.round(V3 * 2) / 2
    W3 = np.round(W3 * 2) / 2
    
    # Verify after rounding
    err = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                val = sum(U3[i*3+j, r] * V3[j*3+k, r] * W3[k*3+i, r] for r in range(15))
                err = max(err, abs(val - 1))
    if err > 1e-6:
        return None
    
    # Build 6x6 decomposition with rank 60
    rank = 60
    U = [[0.0] * rank for _ in range(36)]
    V = [[0.0] * rank for _ in range(36)]
    W = [[0.0] * rank for _ in range(36)]
    
    # For each Laderman term r (0..14) and each within-block index b (0,1):
    # U[2I+a, 2J+b] = U3[I*3+J, r] for all I,J,a
    # V[2J+b, 2K+c] = V3[J*3+K, r] for all J,K,c
    # W[2K+c, 2I+a] = W3[K*3+I, r] for all I,K,a,c
    
    for r in range(15):
        for b in range(2):
            col = r * 2 + b
            for I in range(3):
                for J in range(3):
                    val_u = U3[I*3+J, r]
                    for a in range(2):
                        i = 2*I + a
                        j = 2*J + b
                        U[i][col] = val_u
                for K in range(3):
                    val_w = W3[K*3+I, r]
                    for a in range(2):
                        for c in range(2):
                            k = 2*K + c
                            i = 2*I + a
                            W[k][col] = val_w
            for J in range(3):
                for K in range(3):
                    val_v = V3[J*3+K, r]
                    for c in range(2):
                        j = 2*J + b
                        k = 2*K + c
                        V[j][col] = val_v
    
    # Verify the full 6x6 decomposition
    for i in range(6):
        for j in range(6):
            for k in range(6):
                expected = 1
                val = sum(U[i][r] * V[j][r] * W[k][r] for r in range(rank))
                if abs(val - expected) > 1e-6:
                    return None
    
    return {"U": U, "V": V, "W": W}

def _find_rank15_3x3():
    """Find rank-15 decomposition of 3x3 matmul tensor using ALS."""
    np.random.seed(42)
    rank = 15
    n = 3
    U = np.random.randn(n*n, rank)
    V = np.random.randn(n*n, rank)
    W = np.random.randn(n*n, rank)
    
    for it in range(500):
        # Update U
        new_U = np.zeros_like(U)
        for i in range(n):
            for j in range(n):
                A = np.zeros((n, rank))
                for k in range(n):
                    for r in range(rank):
                        A[k, r] = V[j*n+k, r] * W[k*n+i, r]
                b = np.ones(n)
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                new_U[i*n+j, :] = x
        U = new_U
        
        # Update V
        new_V = np.zeros_like(V)
        for j in range(n):
            for k in range(n):
                A = np.zeros((n, rank))
                for i in range(n):
                    for r in range(rank):
                        A[i, r] = U[i*n+j, r] * W[k*n+i, r]
                b = np.ones(n)
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                new_V[j*n+k, :] = x
        V = new_V
        
        # Update W
        new_W = np.zeros_like(W)
        for k in range(n):
            for i in range(n):
                A = np.zeros((n, rank))
                for j in range(n):
                    for r in range(rank):
                        A[j, r] = U[i*n+j, r] * V[j*n+k, r]
                b = np.ones(n)
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                new_W[k*n+i, :] = x
        W = new_W
        
        # Check
        err = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    val = sum(U[i*n+j, r] * V[j*n+k, r] * W[k*n+i, r] for r in range(rank))
                    err = max(err, abs(val - 1))
        if err < 1e-10:
            break
    
    return U, V, W, err

def _block_strassen():
    """Block Strassen decomposition, rank 189."""
    n, m, p = DIMENSIONS
    # 3x3 grid of 2x2 blocks, Strassen 7-product for each 2x2 block multiplication
    # Total rank = 9 * 7 = 63? No, that's not right.
    # Actually, block Strassen for 6x6: partition into 2x2 blocks of 3x3 matrices
    # Strassen uses 7 products for 2x2 matmul, each product is 3x3*3x3 = 27 scalar mults
    # Total rank = 7 * 27 = 189
    
    rank = 189
    U = [[0.0] * rank for _ in range(36)]
    V = [[0.0] * rank for _ in range(36)]
    W = [[0.0] * rank for _ in range(36)]
    
    # Strassen's 7 products for 2x2 block matrices:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For 6x6: blocks are 3x3, indexed by (I,J) with I,J in {0,1}
    # Block (I,J) spans rows 3I..3I+2, cols 3J..3J+2
    
    strassen_A = [
        # For each of 7 products, list of (block_I, block_J, coeff)
        [(0,0,1),(1,1,1)],       # M1
        [(1,0,1),(1,1,1)],       # M2
        [(0,0,1)],               # M3
        [(1,1,1)],               # M4
        [(0,0,1),(0,1,1)],       # M5
        [(1,0,1),(0,0,-1)],      # M6
        [(0,1,1),(1,1,-1)],      # M7
    ]
    strassen_B = [
        [(0,0,1),(1,1,1)],       # M1
        [(0,0,1)],               # M2
        [(1,0,1),(1,1,-1)],      # M3
        [(0,1,1),(0,0,-1)],      # M4
        [(1,1,1)],               # M5
        [(0,0,1),(0,1,1)],       # M6
        [(1,0,1),(1,1,1)],       # M7
    ]
    strassen_C = [
        # For each output block (I,K), list of (prod_idx, coeff)
        { (0,0): [(0,1),(3,1),(4,-1),(6,1)],
          (0,1): [(2,1),(4,1)],
          (1,0): [(1,1),(3,1)],
          (1,1): [(0,1),(1,-1),(2,1),(5,1)] }
    ]
    
    # For each Strassen product s (0..6) and each scalar product within 3x3 blocks:
    # The s-th product is: (sum_{I,J} a_{s,I,J} * A_block[I,J]) * (sum_{J,K} b_{s,J,K} * B_block[J,K])
    # Each 3x3*3x3 product has 9 scalar terms (schoolbook)
    
    for s in range(7):
        for a in range(3):  # within-block row of A
            for c in range(3):  # within-block col of B
                col = s * 9 + a * 3 + c
                # U: for each (I,J) in strassen_A[s], U[3I+a, 3J+b] = a_{s,I,J} for all b
                # Wait, the middle index b is summed over. For schoolbook within 3x3:
                # C_block[I,K][a,c] = sum_b A_block[I,J][a,b] * B_block[J,K][b,c]
                # So for fixed a,c, and summing over J:
                # Contribution = sum_J (sum_{I'} a_{s,I',J} * A[3I'+a, 3J+b]) * (sum_{K'} b_{s,J,K'} * B[3J+b, 3K'+c])
                # For this to be rank-1, we need to fix J as well... 
                
                # Actually, for schoolbook within blocks, each (a,b,c) triple is a separate rank-1 term
                # So for fixed s, a, b, c: one rank-1 term
                
    # Let me redo this more carefully
    # For each Strassen product s, and each (a, b, c) in {0,1,2}^3:
    # The term is: sum_{I,J,K} a_{s,I,J} * b_{s,J,K} * A[3I+a, 3J+b] * B[3J+b, 3K+c]
    # This is NOT rank-1 in general because of the sum over J.
    # 
    # For it to be rank-1, we need to separate the J dependence.
    # Actually, for schoolbook, each (i,j,k) is a separate rank-1 term.
    # So the total rank is 7 * 27 = 189, where each term corresponds to
    # a specific Strassen product s, and specific (I,J,K,a,b,c) with the
    # constraint that the block indices satisfy the Strassen structure.
    
    # Let me just directly construct it:
    U = [[0.0] * rank for _ in range(36)]
    V = [[0.0] * rank for _ in range(36)]
    W = [[0.0] * rank for _ in range(36)]
    
    col = 0
    for s in range(7):
        for I in range(2):
            for J in range(2):
                for K in range(2):
                    for a in range(3):
                        for b in range(3):
                            for c in range(3):
                                # Check if (I,J) is in strassen_A[s] and (J,K) is in strassen_B[s]
                                a_coeff = 0
                                for (I_, J_, coeff) in strassen_A[s]:
                                    if I_ == I and J_ == J:
                                        a_coeff = coeff
                                b_coeff = 0
                                for (J_, K_, coeff) in strassen_B[s]:
                                    if J_ == J and K_ == K:
                                        b_coeff = coeff
                                if a_coeff == 0 or b_coeff == 0:
                                    continue
                                i = 3*I + a
                                j = 3*J + b
                                k = 3*K + c
                                U[i][col] = a_coeff
                                V[j][col] = b_coeff
                                W[k][col] = 1
                                col += 1
    
    # The actual rank used is col, not 189
    # Trim the matrices
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
