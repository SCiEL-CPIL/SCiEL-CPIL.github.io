import time
import random

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    deadline = time.time() + 140
    
    # M_{2,4,2} rank 8 decomposition:
    # A is 2x4 (rows i=0,1; cols j=0,1,2,3)
    # B is 4x2 (rows j=0,1,2,3; cols k=0,1)
    # C is 2x2 (rows i=0,1; cols k=0,1)
    #
    # Linear combos of A:
    #   s1 = A[0,0] + A[0,2]
    #   s2 = A[0,1] + A[0,3]
    #   s3 = A[1,0] + A[1,2]
    #   s4 = A[1,1] + A[1,3]
    #
    # Multiplications:
    #   t1 = s1 * B[0,0]  -> C[0,0]
    #   t2 = s1 * B[1,0]  -> C[0,0]
    #   t3 = s2 * B[0,1]  -> C[0,1]
    #   t4 = s2 * B[1,1]  -> C[0,1]
    #   t5 = s3 * B[0,0]  -> C[1,0]
    #   t6 = s3 * B[1,0]  -> C[1,0]
    #   t7 = s4 * B[0,1]  -> C[1,1]
    #   t8 = s4 * B[1,1]  -> C[1,1]
    #
    # Check: C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0] + A[0,3]B[3,0]
    #   s1*B[0,0] + s2*B[1,0] = (A[0,0]+A[0,2])B[0,0] + (A[0,1]+A[0,3])B[1,0]
    #   = A[0,0]B[0,0] + A[0,2]B[0,0] + A[0,1]B[1,0] + A[0,3]B[1,0]
    #   But we need A[0,2]B[2,0] + A[0,3]B[3,0], not A[0,2]B[0,0] + A[0,3]B[1,0]
    #
    # This doesn't work. Let me reconsider.
    #
    # The issue is that the linear combinations mix columns that correspond to
    # different B rows. We need:
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # For M_{2,4,2}:
    #   C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0] + A[0,3]B[3,0]
    #   C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1] + A[0,3]B[3,1]
    #   C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0] + A[1,2]B[2,0] + A[1,3]B[3,0]
    #   C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1] + A[1,2]B[2,1] + A[1,3]B[3,1]
    #
    # Let me try a different approach. Use Strassen-like on pairs of columns.
    #
    # Actually, let me try to use the fact that we can share computations.
    #
    # Alternative: use 2x2 Strassen on each half.
    # Split A into [A_left, A_right] where A_left is 2x2, A_right is 2x2.
    # Split B into [B_top, B_bottom] where B_top is 2x2, B_bottom is 2x2.
    # C = A_left * B_top + A_right * B_bottom
    #
    # Use Strassen for A_left * B_top (7 mults) and A_right * B_bottom (7 mults).
    # Total: 14 mults for M_{2,4,2}. That's worse than schoolbook (8).
    #
    # So schoolbook M_{2,4,2} has rank 8. Total for M_{2,4,8} would be 4*8 = 32.
    # But can we do better by sharing across the 4 blocks?
    #
    # The 4 blocks are:
    #   Block 0: A (2x4) * B[0:4, 0:2] (4x2) -> C[:, 0:2] (2x2)
    #   Block 1: A (2x4) * B[0:4, 2:4] (4x2) -> C[:, 2:4] (2x2)
    #   Block 2: A (2x4) * B[0:4, 4:6] (4x2) -> C[:, 4:6] (2x2)
    #   Block 3: A (2x4) * B[0:4, 6:8] (4x2) -> C[:, 6:8] (2x2)
    #
    # Since A is shared, we can compute linear combinations of A once and reuse.
    # But each block uses different columns of B, so the multiplications are distinct.
    #
    # If we use schoolbook for each block: 8 mults per block, 32 total.
    # The linear combinations of A are the same across blocks (just the raw A entries),
    # so no additional sharing benefit beyond what's already counted.
    #
    # Let me implement the schoolbook rank-32 solution.
    
    rank = 32
    nm = n * m  # 8
    mp = m * p  # 32
    pn = p * n  # 16
    
    U_mat = [[0] * rank for _ in range(nm)]
    V_mat = [[0] * rank for _ in range(mp)]
    W_mat = [[0] * rank for _ in range(pn)]
    
    col = 0
    for block in range(4):
        for i in range(n):
            for j in range(m):
                for k in range(2):
                    # U[i*m+j, col] = 1
                    # V[j*p + (block*2+k), col] = 1
                    # W[(block*2+k)*n + i, col] = 1
                    U_mat[i*m+j][col] = 1
                    V_mat[j*p + (block*2+k)][col] = 1
                    W_mat[(block*2+k)*n + i][col] = 1
                    col += 1
    
    # Verify with a test
    random.seed(42)
    A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
    
    C_true = [[sum(A[i][jj] * B[jj][k] for jj in range(m)) for k in range(p)] for i in range(n)]
    
    C_fact = [[0 for _ in range(p)] for _ in range(n)]
    for r in range(rank):
        for i in range(n):
            ua = sum(U_mat[i*m+jj][r] * A[i][jj] for jj in range(m))
            for k in range(p):
                vb = sum(V_mat[jj*p+k][r] * B[jj][k] for jj in range(m))
                C_fact[i][k] += W_mat[k*n+i][r] * ua * vb
    
    for i in range(n):
        for k in range(p):
            if C_fact[i][k] != C_true[i][k]:
                # Fall back to schoolbook full
                rank_sb = n * m * p
                U_sb = [[0] * rank_sb for _ in range(nm)]
                V_sb = [[0] * rank_sb for _ in range(mp)]
                W_sb = [[0] * rank_sb for _ in range(pn)]
                c2 = 0
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            U_sb[i2*m+j2][c2] = 1
                            V_sb[j2*p+k2][c2] = 1
                            W_sb[k2*n+i2][c2] = 1
                            c2 += 1
                return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    if time.time() > deadline:
        return {"U": U_mat, "V": V_mat, "W": W_mat}
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
