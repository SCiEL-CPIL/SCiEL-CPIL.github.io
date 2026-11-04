"""Tensor decomposition for 2x3x10 matrix multiplication using Strassen on 2x2x2 blocks."""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _strassen_factors():
    # Strassen 2x2x2 multiplication: C = A*B where A,B are 2x2, C is 2x2.
    # Index convention: A is 2x2 (rows i in 0..1, cols j in 0..1), B is 2x2 (j rows, k cols).
    # Product C[i,k] = sum_j A[i,j]*B[j,k].
    # Strassen factors M1..M7 with integer coefficients.
    # We return U (2*2 x 7), V (2*2 x 7), W (2*2 x 7) in the trace convention:
    #   U[i*2 + j, c], V[j*2 + k, c], W[k*2 + i, c]
    # such that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = delta (i.e. the product entry).
    #
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M4
    # C21 = M2 + M5
    # C22 = M1 - M2 + M3 + M6
    #
    # We express each M as sum over (j) of A[i,j]*B[j,k]*coeff.
    # For a given (i,k), the contribution to C[i,k] is sum_j A[i,j]*B[j,k].
    # We need: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 if this is the (i,k) product term.
    #
    # Let's build it carefully. For each of the 7 products, the "A side" and "B side"
    # are linear combinations. We encode:
    #   U[i,j,c] = coefficient of A[i,j] in the A-factor of M_c
    #   V[j,k,c] = coefficient of B[j,k] in the B-factor of M_c
    #   W[k,i,c] = coefficient of C[i,k] from M_c
    #
    # A-factors (as function of i,j):
    # M1: (A11+A22) -> A[0,0]=1, A[1,1]=1
    # M2: (A21+A22) -> A[1,0]=1, A[1,1]=1
    # M3: A11 -> A[0,0]=1
    # M4: A22 -> A[1,1]=1
    # M5: (A11+A12) -> A[0,0]=1, A[0,1]=1
    # M6: (A21-A11) -> A[1,0]=1, A[0,0]=-1
    # M7: (A12-A22) -> A[0,1]=1, A[1,1]=-1
    #
    # B-factors (as function of j,k):
    # M1: (B11+B22) -> B[0,0]=1, B[1,1]=1
    # M2: B11 -> B[0,0]=1
    # M3: (B21-B22) -> B[1,0]=1, B[1,1]=-1
    # M4: (B12-B11) -> B[0,1]=1, B[0,0]=-1
    # M5: B22 -> B[1,1]=1
    # M6: (B11+B12) -> B[0,0]=1, B[0,1]=1
    # M7: (B21+B22) -> B[1,0]=1, B[1,1]=1
    #
    # C-coefficients W[k,i,c] (contribution to C[i,k]):
    # M1: C11=+1, C22=+1
    # M2: C21=+1, C22=-1
    # M3: C12=+1, C22=+1
    # M4: C11=-1, C12=+1
    # M5: C11=+1, C21=+1
    # M6: C22=+1
    # M7: C11=+1

    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # A-side: U[i*2+j, c]
    U[0*2+0][0] = 1; U[1*2+1][0] = 1   # M1: A11+A22
    U[1*2+0][1] = 1; U[1*2+1][1] = 1   # M2: A21+A22
    U[0*2+0][2] = 1                     # M3: A11
    U[1*2+1][3] = 1                     # M4: A22
    U[0*2+0][4] = 1; U[0*2+1][4] = 1   # M5: A11+A12
    U[1*2+0][5] = 1; U[0*2+0][5] = -1  # M6: A21-A11
    U[0*2+1][6] = 1; U[1*2+1][6] = -1  # M7: A12-A22

    # B-side: V[j*2+k, c]
    V[0*2+0][0] = 1; V[1*2+1][0] = 1   # M1: B11+B22
    V[0*2+0][1] = 1                     # M2: B11
    V[1*2+0][2] = 1; V[1*2+1][2] = -1  # M3: B21-B22
    V[0*2+1][3] = 1; V[0*2+0][3] = -1  # M4: B12-B11
    V[1*2+1][4] = 1                     # M5: B22
    V[0*2+0][5] = 1; V[0*2+1][5] = 1   # M6: B11+B12
    V[1*2+0][6] = 1; V[1*2+1][6] = 1   # M7: B21+B22

    # C-side: W[k*2+i, c]  (note: W index is k*n+i with n=2)
    # M1: C[0,0]+=1, C[1,1]+=1
    W[0*2+0][0] = 1; W[1*2+1][0] = 1
    # M2: C[1,0]+=1, C[1,1]-=1
    W[0*2+1][1] = 1; W[1*2+1][1] = -1
    # M3: C[0,1]+=1, C[1,1]+=1
    W[1*2+0][2] = 1; W[1*2+1][2] = 1
    # M4: C[0,0]-=1, C[0,1]+=1
    W[0*2+0][3] = -1; W[1*2+0][3] = 1
    # M5: C[0,0]+=1, C[1,0]+=1
    W[0*2+0][4] = 1; W[0*2+1][4] = 1
    # M6: C[1,1]+=1
    W[1*2+1][5] = 1
    # M7: C[0,0]+=1
    W[0*2+0][6] = 1

    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    # Check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == delta for all i,j,k
    # Actually the tensor decomposition condition is:
    # For all (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if this is a valid product term...
    # Wait, the condition is that the tensor T_{i,j,k} = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # equals the "multiplication tensor" which is 1 for all (i,j,k) since C[i,k] = sum_j A[i,j]*B[j,k].
    # So T_{i,j,k} should be 1 for all i,j,k.
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > tol:
                    return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit

    n, m, p = DIMENSIONS  # 2, 3, 10

    # Strategy: split j into j=0,1 (2 rows) and j=2 (1 row).
    # Part 1: 2x2x10 (i in 0..1, j in 0..1, k in 0..9)
    #   Split k into 5 blocks of 2: k in {0,1}, {2,3}, ..., {8,9}
    #   Each 2x2x2 block uses Strassen (rank 7) -> 5*7 = 35
    # Part 2: 2x1x10 (i in 0..1, j=2, k in 0..9)
    #   Schoolbook: rank 2*1*10 = 20
    # Total: 55

    U_parts = []
    V_parts = []
    W_parts = []

    # Part 1: 2x2x10 via 5 Strassen blocks
    sU, sV, sW = _strassen_factors()  # each is 4x7 (4 = 2*2)
    for block in range(5):
        k0 = block * 2
        # For this block, the local indices are i (0..1), j (0..1), k_local (0..1)
        # Global: i, j, k = k0 + k_local
        # U local: [i*2 + j] -> global U row: i*m + j = i*3 + j
        # V local: [j*2 + k_local] -> global V row: j*p + (k0 + k_local)
        # W local: [k_local*2 + i] -> global W row: (k0 + k_local)*n + i
        for c in range(7):
            col = []
            for i in range(2):
                for j in range(2):
                    col.append(sU[i*2 + j][c])
            U_parts.append(col)
            col = []
            for j in range(2):
                for kk in range(2):
                    col.append(sV[j*2 + kk][c])
            V_parts.append(col)
            col = []
            for kk in range(2):
                for i in range(2):
                    col.append(sW[kk*2 + i][c])
            W_parts.append(col)

    # Part 2: 2x1x10 schoolbook
    for i in range(2):
        for kk in range(10):
            U_parts.append([1] + [0]*54)  # placeholder, will fix sizes
            V_parts.append([0]*55)
            W_parts.append([0]*55)

    # Actually let me build this more carefully with explicit column construction
    rank = 55
    U = [[0]*rank for _ in range(n*m)]  # 6 x 55
    V = [[0]*rank for _ in range(m*p)]  # 30 x 55
    W = [[0]*rank for _ in range(p*n)]  # 20 x 55

    col = 0
    # Part 1: 5 Strassen blocks
    for block in range(5):
        k0 = block * 2
        for c in range(7):
            # U: rows i*3+j for i in 0..1, j in 0..1
            for i in range(2):
                for j in range(2):
                    U[i*3 + j][col] = sU[i*2 + j][c]
            # V: rows j*10 + (k0+kk) for j in 0..1, kk in 0..1
            for j in range(2):
                for kk in range(2):
                    V[j*10 + (k0 + kk)][col] = sV[j*2 + kk][c]
            # W: rows (k0+kk)*2 + i for kk in 0..1, i in 0..1
            for kk in range(2):
                for i in range(2):
                    W[(k0 + kk)*2 + i][col] = sW[kk*2 + i][c]
            col += 1

    # Part 2: 2x1x10 schoolbook (i in 0..1, j=2, kk in 0..9)
    for i in range(2):
        for kk in range(10):
            U[i*3 + 2][col] = 1  # row i*m + j = i*3 + 2
            V[2*10 + kk][col] = 1  # row j*p + kk = 2*10 + kk
            W[kk*2 + i][col] = 1  # row kk*n + i
            col += 1

    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank 60
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for kk in range(p):
                U[i*m + j][column] = 1
                V[j*p + kk][column] = 1
                W[kk*n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
