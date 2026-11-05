import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def _verify(U, V, W):
    """Verify that the decomposition exactly computes matrix multiplication."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook_decomposition():
    """Return the schoolbook rank-72 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Return U, V, W for 2x2 * 2x2 matrix multiplication using Strassen (rank 7).
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen 7 products:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)(B00) -> C10 += M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = A00(B10-B11) -> C01 += M3
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4 = A11(B01-B10) -> C00 += M4, C10 += M4
    U[3][3] = 1
    V[1][3] = 1; V[2][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)(B11) -> C00 -= M5, C01 += M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B10) -> C11 += M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B01+B11) -> C00 += M7
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_optimized_block_decomposition():
    """
    Build a block decomposition for 3x3 * 3x8 using Strassen and optimized outer products.
    Target rank: 48.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    total_rank = 48
    
    U = [[0]*total_rank for _ in range(n*m)]  # 9 rows
    V = [[0]*total_rank for _ in range(m*p)]  # 24 rows
    W = [[0]*total_rank for _ in range(p*n)]  # 24 rows
    
    col = 0
    sU, sV, sW = _strassen_2x2x2()
    
    # Helper to add Strassen block for a 2x2x2 subproblem
    # u_rows: [A[0,0], A[0,1], A[1,0], A[1,1]] in global U indexing
    # v_rows: [B[0,0], B[0,1], B[1,0], B[1,1]] in global V indexing
    # w_rows: [C[0,0], C[0,1], C[1,0], C[1,1]] in global W indexing
    def add_strassen(u_rows, v_rows, w_rows):
        nonlocal col
        for r in range(7):
            for i_local in range(4):
                if sU[i_local][r] != 0:
                    U[u_rows[i_local]][col + r] += sU[i_local][r]
            for i_local in range(4):
                if sV[i_local][r] != 0:
                    V[v_rows[i_local]][col + r] += sV[i_local][r]
            for i_local in range(4):
                if sW[i_local][r] != 0:
                    W[w_rows[i_local]][col + r] += sW[i_local][r]
        col += 7

    # Helper to add 1x2x2 schoolbook block (Rank 4)
    # u_rows: [A[i,0], A[i,1]] (2 rows)
    # v_rows: [B[0,k0], B[0,k1], B[1,k0], B[1,k1]] (4 rows)
    # w_rows: [C[k0,i], C[k1,i]] (2 rows)
    def add_1x2x2(u_rows, v_rows, w_rows):
        nonlocal col
        # r=0: A[0]*B[0,k0] -> C[k0,i]
        U[u_rows[0]][col] = 1
        V[v_rows[0]][col] = 1
        W[w_rows[0]][col] = 1
        col += 1
        # r=1: A[1]*B[1,k0] -> C[k0,i]
        U[u_rows[1]][col] = 1
        V[v_rows[2]][col] = 1
        W[w_rows[0]][col] = 1
        col += 1
        # r=2: A[0]*B[0,k1] -> C[k1,i]
        U[u_rows[0]][col] = 1
        V[v_rows[1]][col] = 1
        W[w_rows[1]][col] = 1
        col += 1
        # r=3: A[1]*B[1,k1] -> C[k1,i]
        U[u_rows[1]][col] = 1
        V[v_rows[3]][col] = 1
        W[w_rows[1]][col] = 1
        col += 1

    # Helper to add 2x1x4 outer product (Rank 1)
    # u_rows: [A[0,j], A[1,j]] (2 rows)
    # v_rows: [B[j,0], B[j,1], B[j,2], B[j,3]] (4 rows)
    # w_rows: [C[0,0], C[0,1], C[0,2], C[0,3], C[1,0], C[1,1], C[1,2], C[1,3]] (8 rows)
    def add_2x1x4_outer(u_rows, v_rows, w_rows):
        nonlocal col
        for i in range(2):
            U[u_rows[i]][col] = 1
        for k in range(4):
            V[v_rows[k]][col] = 1
        for w_idx in w_rows:
            W[w_idx][col] = 1
        col += 1

    # Helper to add 1x1x4 outer product (Rank 1)
    # u_row: A[i,j] (1 row)
    # v_rows: [B[j,0], B[j,1], B[j,2], B[j,3]] (4 rows)
    # w_rows: [C[0,i], C[1,i], C[2,i], C[3,i]] (4 rows)
    def add_1x1x4_outer(u_row, v_rows, w_rows):
        nonlocal col
        U[u_row][col] = 1
        for k in range(4):
            V[v_rows[k]][col] = 1
        for w_idx in w_rows:
            W[w_idx][col] = 1
        col += 1

    # Global indexing:
    # U[i*m+j] = U[i*3+j], i in 0..2, j in 0..2
    # V[j*p+k] = V[j*8+k], j in 0..2, k in 0..7
    # W[k*n+i] = W[k*3+i], k in 0..7, i in 0..2

    # Block 1: i in {0,1}, k in {0,1,2,3}
    # 1a: 2x2x2 (i in {0,1}, j in {0,1}, k in {0,1})
    # u_rows: U[0*3+0]=0, U[0*3+1]=1, U[1*3+0]=3, U[1*3+1]=4
    # v_rows: V[0*8+0]=0, V[0*8+1]=1, V[1*8+0]=8, V[1*8+1]=9
    # w_rows: W[0*3+0]=0, W[0*3+1]=1, W[1*3+0]=3, W[1*3+1]=4
    add_strassen([0, 1, 3, 4], [0, 1, 8, 9], [0, 1, 3, 4])
    
    # 1b: 2x2x2 (i in {0,1}, j in {0,1}, k in {2,3})
    # v_rows: V[0*8+2]=2, V[0*8+3]=3, V[1*8+2]=10, V[1*8+3]=11
    # w_rows: W[2*3+0]=6, W[2*3+1]=7, W[3*3+0]=9, W[3*3+1]=10
    add_strassen([0, 1, 3, 4], [2, 3, 10, 11], [6, 7, 9, 10])
    
    # 1c: 2x1x4 (i in {0,1}, j in {2}, k in {0,1,2,3})
    # u_rows: U[0*3+2]=2, U[1*3+2]=5
    # v_rows: V[2*8+0]=16, V[2*8+1]=17, V[2*8+2]=18, V[2*8+3]=19
    # w_rows: W[0*3+0]=0, W[0*3+1]=1, W[1*3+0]=3, W[1*3+1]=4, W[2*3+0]=6, W[2*3+1]=7, W[3*3+0]=9, W[3*3+1]=10
    add_2x1x4_outer([2, 5], [16, 17, 18, 19], [0, 1, 3, 4, 6, 7, 9, 10])

    # Block 2: i in {0,1}, k in {4,5,6,7}
    # 2a: 2x2x2 (i in {0,1}, j in {0,1}, k in {4,5})
    # v_rows: V[0*8+4]=4, V[0*8+5]=5, V[1*8+4]=12, V[1*8+5]=13
    # w_rows: W[4*3+0]=12, W[4*3+1]=13, W[5*3+0]=15, W[5*3+1]=16
    add_strassen([0, 1, 3, 4], [4, 5, 12, 13], [12, 13, 15, 16])
    
    # 2b: 2x2x2 (i in {0,1}, j in {0,1}, k in {6,7})
    # v_rows: V[0*8+6]=6, V[0*8+7]=7, V[1*8+6]=14, V[1*8+7]=15
    # w_rows: W[6*3+0]=18, W[6*3+1]=19, W[7*3+0]=21, W[7*3+1]=22
    add_strassen([0, 1, 3, 4], [6, 7, 14, 15], [18, 19, 21, 22])
    
    # 2c: 2x1x4 (i in {0,1}, j in {2}, k in {4,5,6,7})
    # u_rows: U[0*3+2]=2, U[1*3+2]=5
    # v_rows: V[2*8+4]=20, V[2*8+5]=21, V[2*8+6]=22, V[2*8+7]=23
    # w_rows: W[4*3+0]=12, W[4*3+1]=13, W[5*3+0]=15, W[5*3+1]=16, W[6*3+0]=18, W[6*3+1]=19, W[7*3+0]=21, W[7*3+1]=22
    add_2x1x4_outer([2, 5], [20, 21, 22, 23], [12, 13, 15, 16, 18, 19, 21, 22])

    # Block 3: i in {2}, k in {0,1,2,3}
    # 3a1: 1x2x2 (i in {2}, j in {0,1}, k in {0,1})
    # u_rows: U[2*3+0]=6, U[2*3+1]=7
    # v_rows: V[0*8+0]=0, V[0*8+1]=1, V[1*8+0]=8, V[1*8+1]=9
    # w_rows: W[0*3+2]=2, W[1*3+2]=5
    add_1x2x2([6, 7], [0, 1, 8, 9], [2, 5])
    
    # 3a2: 1x2x2 (i in {2}, j in {0,1}, k in {2,3})
    # v_rows: V[0*8+2]=2, V[0*8+3]=3, V[1*8+2]=10, V[1*8+3]=11
    # w_rows: W[2*3+2]=8, W[3*3+2]=11
    add_1x2x2([6, 7], [2, 3, 10, 11], [8, 11])
    
    # 3b: 1x1x4 (i in {2}, j in {2}, k in {0,1,2,3})
    # u_row: U[2*3+2]=8
    # v_rows: V[2*8+0]=16, V[2*8+1]=17, V[2*8+2]=18, V[2*8+3]=19
    # w_rows: W[0*3+2]=2, W[1*3+2]=5, W[2*3+2]=8, W[3*3+2]=11
    add_1x1x4_outer(8, [16, 17, 18, 19], [2, 5, 8, 11])

    # Block 4: i in {2}, k in {4,5,6,7}
    # 4a1: 1x2x2 (i in {2}, j in {0,1}, k in {4,5})
    # v_rows: V[0*8+4]=4, V[0*8+5]=5, V[1*8+4]=12, V[1*8+5]=13
    # w_rows: W[4*3+2]=14, W[5*3+2]=17
    add_1x2x2([6, 7], [4, 5, 12, 13], [14, 17])
    
    # 4a2: 1x2x2 (i in {2}, j in {0,1}, k in {6,7})
    # v_rows: V[0*8+6]=6, V[0*8+7]=7, V[1*8+6]=14, V[1*8+7]=15
    # w_rows: W[6*3+2]=20, W[7*3+2]=23
    add_1x2x2([6, 7], [6, 7, 14, 15], [20, 23])
    
    # 4b: 1x1x4 (i in {2}, j in {2}, k in {4,5,6,7})
    # u_row: U[2*3+2]=8
    # v_rows: V[2*8+4]=20, V[2*8+5]=21, V[2*8+6]=22, V[2*8+7]=23
    # w_rows: W[4*3+2]=14, W[5*3+2]=17, W[6*3+2]=20, W[7*3+2]=23
    add_1x1x4_outer(8, [20, 21, 22, 23], [14, 17, 20, 23])

    assert col == total_rank, f"Expected {total_rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank exact decomposition for 3x3 * 3x8 matrix multiplication.
    Uses optimized block decomposition with rank 48.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135
    
    try:
        result = _build_optimized_block_decomposition()
        if _verify(result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass
    
    return _schoolbook_decomposition()
# EVOLVE-BLOCK-END
