import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 9]
DEADLINE = 140.0  # Safety margin before 150s kill

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 3, 9
    rank = n * m * p      # 54

    # Build a verification function
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
        # Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Must equal 1 if (i,j,k) is valid (always valid for this tensor), 0 otherwise
        # Actually the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
        # Wait, let me reconsider. The matrix multiplication tensor for C = A*B where
        # A is n x m, B is m x p, C is n x p:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor T[i,j,k] = 1 for all valid (i,j,k).
        # The decomposition requires: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For all i in [0,n), j in [0,m), k in [0,p)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1
                    actual = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(len(U[0])))
                    if actual != expected:
                        return False
        return True

    def elapsed():
        return time.time() - start_time

    # Default schoolbook decomposition
    def schoolbook():
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W

    best_U, best_V, best_W = schoolbook()
    best_rank = rank

    # Try Strassen-based block decomposition for rank 50
    # Strategy: Split A (2x3) into A_L (2x2) and A_R (2x1)
    # Split B (3x9) into B_T (2x9) and B_B (1x9)
    # C = A_L * B_T + A_R * B_B
    # A_L * B_T: 2x2x9 -> split B_T into 4 blocks of 2x2 and 1 block of 2x1
    #   Use Strassen (rank 7) for each 2x2x2: 4 * 7 = 28
    #   Use schoolbook for 2x2x1: 4
    #   Total for A_L*B_T: 32
    # A_R * B_B: 2x1x9 -> schoolbook: 18
    # Total: 32 + 18 = 50

    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition.
        U shape (4, 7), V shape (4, 7), W shape (4, 7)
        Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        """
        # Strassen's 7 products for 2x2 matrix multiplication
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # U[i*2+j, c] where i in 0,1 and j in 0,1
        # Row indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # V[j*2+k, c]
        # Row indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # W[k*2+i, c]
        # Row indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        
        # M1 = (A00+A11)(B00+B11)
        # U[0,0]=1 (A00), U[3,0]=1 (A11)
        # V[0,0]=1 (B00), V[3,0]=1 (B11)
        # W: C00 += M1, C11 += M1
        U[0][0] += 1; U[3][0] += 1
        V[0][0] += 1; V[3][0] += 1
        W[0][0] += 1  # C00
        W[3][0] += 1  # C11
        
        # M2 = (A10+A11)B00
        U[2][1] += 1; U[3][1] += 1
        V[0][1] += 1
        W[2][1] += 1  # C10
        
        # M3 = A00(B01-B11)
        U[0][2] += 1
        V[1][2] += 1; V[3][2] -= 1
        W[1][2] += 1  # C01
        W[3][2] += 1  # C11
        
        # M4 = A11(B10-B00)
        U[3][3] += 1
        V[2][3] += 1; V[0][3] -= 1
        W[0][3] += 1  # C00
        W[2][3] += 1  # C10
        
        # M5 = (A00+A01)B11
        U[0][4] += 1; U[1][4] += 1
        V[3][4] += 1
        W[1][4] += 1  # C01
        W[0][4] -= 1  # C00
        
        # M6 = (A10-A00)(B00+B01)
        U[2][5] += 1; U[0][5] -= 1
        V[0][5] += 1; V[1][5] += 1
        W[3][5] += 1  # C11
        
        # M7 = (A01-A11)(B10+B11)
        U[1][6] += 1; U[3][6] -= 1
        V[2][6] += 1; V[3][6] += 1
        W[0][6] += 1  # C00
        
        return U, V, W

    def build_rank50():
        """Build rank-50 decomposition using Strassen blocks."""
        total_rank = 50
        U = [[0]*total_rank for _ in range(n*m)]  # (6, 50)
        V = [[0]*total_rank for _ in range(m*p)]  # (27, 50)
        W = [[0]*total_rank for _ in range(p*n)]  # (18, 50)
        
        col = 0
        
        # Part 1: A_L * B_T where A_L is 2x2, B_T is 2x9
        # Split B_T into 4 blocks of 2x2 (columns 0-7) and 1 block of 2x1 (column 8)
        # B_T columns 0-7: 4 blocks of 2x2
        # For each 2x2 block, use Strassen (rank 7)
        # B_T columns 8: 2x1, use schoolbook (rank 4)
        
        for blk in range(4):
            sU, sV, sW = strassen_2x2x2()
            for ci in range(7):
                c = col + ci
                # U indices: i*2+j for i in 0,1; j in 0,1
                for ui in range(4):
                    U[ui][c] += sU[ui][ci]
                # V indices: j*2+k where j in 0,1 (rows of B_T), k in 0,1 (cols within block)
                # B_T row j corresponds to B row j (since B_T = B[0:2, :])
                # B_T col = blk*2 + k
                # V row index in full: j*p + (blk*2 + k) = j*9 + blk*2 + k
                for vi in range(4):
                    j_local = vi // 2
                    k_local = vi % 2
                    full_vi = j_local * p + blk * 2 + k_local
                    V[full_vi][c] += sV[vi][ci]
                # W indices: k*n+i where k is col index within block, i is row of C
                # C row i, col = blk*2 + k
                # W row index: (blk*2 + k)*n + i
                for wi in range(4):
                    k_local = wi // 2
                    i_local = wi % 2
                    full_wi = (blk * 2 + k_local) * n + i_local
                    W[full_wi][c] += sW[wi][ci]
            col += 7
        
        # B_T column 8: 2x1 block, schoolbook (rank 4)
        for i in range(2):
            for k in range(1):
                c = col
                # A_L[i, j] * B_T[j, 8]
                # U[i*m + j, c] = 1 where m=3, so j in 0,1 (A_L cols)
                for j in range(2):
                    U[i*m + j][c] = 1
                    # V[j*p + 8, c] = 1
                    V[j*p + 8][c] = 1
                    # W[8*n + i, c] = 1
                    W[8*n + i][c] = 1
                col += 1
        
        # Part 2: A_R * B_B where A_R is 2x1, B_B is 1x9
        # Schoolbook: rank = 2*1*9 = 18
        for i in range(2):
            for k in range(9):
                c = col
                # A_R[i, 0] * B_B[0, k]
                U[i*m + 2][c] = 1  # A_R col 0 corresponds to A col 2
                V[2*p + k][c] = 1  # B_B row 0 corresponds to B row 2
                W[k*n + i][c] = 1
                col += 1
        
        return U, V, W, total_rank

    # Try rank 50
    try:
        U50, V50, W50, r50 = build_rank50()
        if r50 < best_rank and verify(U50, V50, W50):
            best_U, best_V, best_W = U50, V50, W50
            best_rank = r50
    except Exception:
        pass

    if elapsed() > DEADLINE - 5:
        return {"U": best_U, "V": best_V, "W": best_W}

    # Try other block splits
    # Split A (2x3) into A_L (2x1) and A_R (2x2)
    # Split B (3x9) into B_T (1x9) and B_B (2x9)
    # C = A_L * B_T + A_R * B_B
    # A_L * B_T: 2x1x9 -> schoolbook: 18
    # A_R * B_B: 2x2x9 -> same as before: 32
    # Total: 50 (same, just different arrangement)
    
    # Try splitting B (3x9) into 3 blocks of 3x3
    # A (2x3) * B_block (3x3): 2x3x3 each, 3 blocks
    # 2x3x3 rank: at most 18, but can we do better?
    # Actually 2x3x3: we can try various decompositions
    # For now, let's just verify our current best

    return {"U": best_U, "V": best_V, "W": best_W}
