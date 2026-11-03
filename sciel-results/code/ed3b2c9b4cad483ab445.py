"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time

def _validate(U, V, W, n, m, p):
    """Check that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k."""
    R = len(U[0]) if U else 0
    if len(V[0]) != R or len(W[0]) != R:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def construct():
    t0 = time.time()
    DEADLINE = 145.0

    n, m, p = DIMENSIONS  # 3, 3, 4

    # Schoolbook fallback: rank 36
    U_fb = [[0]*36 for _ in range(n*m)]
    V_fb = [[0]*36 for _ in range(m*p)]
    W_fb = [[0]*36 for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1

    # Try to get rank 35 by using Strassen (rank 7) for the 2x2x2 sub-block
    # (i in {0,1}, j in {0,1}, k in {0,1}) instead of schoolbook (rank 8).
    # Total rank = 36 - 8 + 7 = 35

    total_rank = 35
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]

    col = 0

    # Strassen 2x2x2 block: i in {0,1}, j in {0,1}, k in {0,1}
    # The 2x2 matmul tensor: for each (i,j,k) in {0,1}^3, we need
    #   sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = 1
    # Strassen's 7 multiplications for 2x2 matmul:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11) B00
    # M3 = A00 (B10-B11)
    # M4 = A11 (B01-B00)
    # M5 = (A00+A01) B11
    # M6 = (A10-A00) (B00+B10)
    # M7 = (A01-A11) (B01+B11)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M6
    # C11 = M1 + M3 - M2 + M6
    #
    # Each Mr contributes a rank-1 term to the tensor.
    # For each r, we have U[i,j,r], V[j,k,r], W[i,k,r] such that
    # sum_r U[i,j,r]*V[j,k,r]*W[i,k,r] = 1 for all (i,j,k).
    #
    # M1: A factors: A00=1, A11=1; B factors: B00=1, B11=1
    #   U[0,0]=1, U[1,1]=1; V[0,0]=1, V[1,1]=1
    #   C: C00=1, C01=1, C10=1, C11=1 → W[0,0]=1, W[0,1]=1, W[1,0]=1, W[1,1]=1
    #   But wait, that's not rank 1. Each M_r is one multiplication, so it's one rank-1 term.
    #   The rank-1 term for M1 has:
    #     U part: coefficient for A[i,j] → U[i,j,1]
    #     V part: coefficient for B[j,k] → V[j,k,1]
    #     W part: coefficient for C[i,k] → W[i,k,1]
    #   M1 = (A00+A11)(B00+B11) contributes to all four C entries.
    #   So: U[0,0]=1, U[1,1]=1; V[0,0]=1, V[1,1]=1; W[0,0]=1, W[0,1]=1, W[1,0]=1, W[1,1]=1
    #
    # M2 = (A10+A11) B00
    #   U[1,0]=1, U[1,1]=1; V[0,0]=1; W[1,0]=1
    #
    # M3 = A00 (B10-B11)
    #   U[0,0]=1; V[1,0]=1, V[1,1]=-1; W[1,0]=1
    #
    # M4 = A11 (B01-B00)
    #   U[1,1]=1; V[0,1]=1, V[0,0]=-1; W[0,1]=1
    #
    # M5 = (A00+A01) B11
    #   U[0,0]=1, U[0,1]=1; V[1,1]=1; W[0,1]=1
    #
    # M6 = (A10-A00) (B00+B10)
    #   U[1,0]=1, U[0,0]=-1; V[0,0]=1, V[1,0]=1; W[1,0]=1
    #
    # M7 = (A01-A11) (B01+B11)
    #   U[0,1]=1, U[1,1]=-1; V[0,1]=1, V[1,1]=1; W[0,1]=1
    #
    # Let me verify C00 = M1 - M4 + M5 + M7:
    #   From M1: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1 (i=0,j=0,k=0)
    #   From M1: U[1,1]*V[1,1]*W[0,0] = 1*1*1 = 1 (i=1,j=1,k=0)
    #   Wait, I need to sum over ALL (i,j,k) for a fixed (i,k) to get C[i,k].
    #   Actually the tensor condition is: for each (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[i,k,r] = 1.
    #   This is NOT the same as C[i,k] = sum_j A[i,j]B[j,k].
    #
    # Let me re-derive. The matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k
    # (meaning the only nonzero entry in the "product" for each (i,j,k) triple is 1).
    # The decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # For schoolbook: U[i,j,r]=1 if r=(i,j,k), else 0. Similarly for V and W.
    # So T[i,j,k] = 1 (only r=(i,j,k) contributes).
    #
    # For Strassen on 2x2x2: we need 7 terms such that for each (i,j,k) in {0,1}^3,
    # sum_{r=1}^{7} U_r[i,j] * V_r[j,k] * W_r[i,k] = 1.
    #
    # Let me define the 7 terms explicitly. Each term r has:
    #   U_r: a 2x2 matrix (entries U_r[i,j] for i,j in {0,1})
    #   V_r: a 2x2 matrix (entries V_r[j,k] for j,k in {0,1})
    #   W_r: a 2x2 matrix (entries W_r[i,k] for i,k in {0,1})
    #
    # Strassen's 7 multiplications correspond to 7 rank-1 tensors:
    # r=1: U1 = [[1,0],[0,1]], V1 = [[1,0],[0,1]], W1 = [[1,1],[1,1]]
    #   Check: for (i,j,k): U1[i,j]*V1[j,k]*W1[i,k]
    #   (0,0,0): 1*1*1=1; (0,0,1): 1*0*1=0; (0,1,0): 0*1*1=0; (0,1,1): 0*0*1=0
    #   (1,0,0): 0*1*1=0; (1,0,1): 0*0*1=0; (1,1,0): 1*1*1=1; (1,1,1): 1*0*1=0
    #   So term 1 contributes 1 to (0,0,0) and (1,1,0).
    #
    # This is getting complex. Let me just directly construct and verify numerically.
    # I'll use a known correct Strassen decomposition.

    # Known Strassen rank-7 decomposition for 2x2x2 matmul tensor:
    # The tensor T[i,j,k] = 1 for all i,j,k in {0,1}.
    # 7 rank-1 terms (U_r, V_r, W_r) where each is a 2x2 matrix:
    #
    # r=1: U=[[1,0],[0,1]], V=[[1,0],[0,1]], W=[[1,1],[1,1]]
    # r=2: U=[[0,0],[1,1]], V=[[1,0],[0,0]], W=[[0,0],[1,0]]
    # r=3: U=[[1,0],[0,0]], V=[[0,0],[1,-1]], W=[[0,0],[1,0]]
    # r=4: U=[[0,0],[0,1]], V=[[0,1],[-1,0]], W=[[0,1],[0,0]]
    # r=5: U=[[1,1],[0,0]], V=[[0,0],[0,1]], W=[[0,1],[0,0]]
    # r=6: U=[[-1,0],[1,0]], V=[[1,0],[1,0]], W=[[0,0],[1,0]]
    # r=7: U=[[0,1],[-1,0]], V=[[0,1],[0,1]], W=[[0,1],[0,0]]

    # Let me verify this numerically.
    strassen_U = [
        [[1,0],[0,1]],  # r=1
        [[0,0],[1,1]],  # r=2
        [[1,0],[0,0]],  # r=3
        [[0,0],[0,1]],  # r=4
        [[1,1],[0,0]],  # r=5
        [[-1,0],[1,0]], # r=6
        [[0,1],[-1,0]], # r=7
    ]
    strassen_V = [
        [[1,0],[0,1]],  # r=1
        [[1,0],[0,0]],  # r=2
        [[0,0],[1,-1]], # r=3
        [[0,1],[-1,0]], # r=4
        [[0,0],[0,1]],  # r=5
        [[1,0],[1,0]],  # r=6
        [[0,1],[0,1]],  # r=7
    ]
    strassen_W = [
        [[1,1],[1,1]],  # r=1
        [[0,0],[1,0]],  # r=2
        [[0,0],[1,0]],  # r=3
        [[0,1],[0,0]],  # r=4
        [[0,1],[0,0]],  # r=5
        [[0,0],[1,0]],  # r=6
        [[0,1],[0,0]],  # r=7
    ]

    # Verify: for each (i,j,k) in {0,1}^3, sum_r U_r[i,j]*V_r[j,k]*W_r[i,k] should = 1
    valid = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0.0
                for r in range(7):
                    s += strassen_U[r][i][j] * strassen_V[r][j][k] * strassen_W[r][i][k]
                if abs(s - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if valid:
        # Use Strassen for the 2x2x2 block, schoolbook for the rest
        # Strassen block: i in {0,1}, j in {0,1}, k in {0,1}
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*m + j][col] = strassen_U[r][i][j]
                for k in range(2):
                    W[k*n + i][col] = strassen_W[r][i][k]
            for j in range(2):
                for k in range(2):
                    V[j*p + k][col] = strassen_V[r][j][k]
            col += 1

        # Schoolbook for remaining (i,j,k) not in the 2x2x2 block
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i < 2 and j < 2 and k < 2:
                        continue  # already handled by Strassen
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1

        # Verify the full decomposition
        if _validate(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
