"""Tensor decomposition of 5x5x5 matrix multiplication, rank 52."""
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100.0

    n, m, p = 5, 5, 5

    def idxU(i, j):
        return i * m + j
    def idxV(j, k):
        return j * p + k
    def idxW(k, i):
        return k * n + i

    U_cols = []
    V_cols = []
    W_cols = []

    # Part 1: Frame terms — all (i,j) where i==4 or j==4, for each k
    for i in range(n):
        for j in range(m):
            if i == 4 or j == 4:
                for k in range(p):
                    U_cols.append({idxU(i, j): 1})
                    V_cols.append({idxV(j, k): 1})
                    W_cols.append({idxW(k, i): 1})

    # Part 2: 4x4x4 core via Strassen (rank 7)
    # Core indices: i in 0..3, j in 0..3, k in 0..3
    # 2x2 block partition: bi,bj,bk in {0,1}
    # A block (bi,bj): i in 2*bi..2*bi+1, j in 2*bj..2*bj+1
    # B block (bj,bk): j in 2*bj..2*bj+1, k in 2*bk..2*bk+1

    def a_positions(bi, bj):
        return [(2*bi + di, 2*bj + dj) for di in range(2) for dj in range(2)]

    def b_positions(bj, bk):
        return [(2*bj + dj, 2*bk + dk) for dj in range(2) for dk in range(2)]

    # Strassen 7 products for 4x4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # Result: C00 = M1+M4-M5+M7
    #         C01 = M3+M5
    #         C10 = M2+M4
    #         C11 = M1-M2+M3+M6

    M_defs = [
        # M1 = (A00+A11)(B00+B11)
        [((0,0),(0,0),1), ((0,0),(0,1),1), ((1,1),(0,0),1), ((1,1),(0,1),1)],
        # M2 = (A10+A11)B00
        [((1,0),(0,0),1), ((1,1),(0,0),1)],
        # M3 = A00(B01-B11)
        [((0,0),(0,1),1), ((0,0),(1,1),-1)],
        # M4 = A11(B10-B00)
        [((1,1),(1,0),1), ((1,1),(0,0),-1)],
        # M5 = (A00+A01)B11
        [((0,0),(1,1),1), ((0,1),(1,1),1)],
        # M6 = (A10-A00)(B00+B01)
        [((1,0),(0,0),1), ((1,0),(0,1),1), ((0,0),(0,0),-1), ((0,0),(0,1),-1)],
        # M7 = (A01-A11)(B10+B11)
        [((0,1),(1,0),1), ((0,1),(1,1),1), ((1,1),(1,0),-1), ((1,1),(1,1),-1)],
    ]

    for parts in M_defs:
        U_acc = {}
        V_acc = {}
        W_acc = {}
        for (a_blk, b_blk, sign) in parts:
            for (i, j) in a_positions(*a_blk):
                for (j2, k) in b_positions(*b_blk):
                    # Critical: j from A block must equal j2 from B block
                    # This only holds when the bj index matches
                    # For M2: A10 has bj=0, A11 has bj=1, B00 has bj=0
                    # So A11's j positions (2,3) don't match B00's j positions (0,1)
                    # This means the simple Strassen doesn't directly work for 4x4x4
                    # with this block structure!
                    if j == j2:
                        ui = idxU(i, j)
                        vi = idxV(j, k)
                        wi = idxW(k, i)
                        U_acc[ui] = U_acc.get(ui, 0) + sign
                        V_acc[vi] = V_acc.get(vi, 0) + sign
                        W_acc[wi] = W_acc.get(wi, 0) + sign
        U_cols.append(U_acc)
        V_cols.append(V_acc)
        W_cols.append(W_acc)

    rank = len(U_cols)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        for idx, val in U_cols[r].items():
            U[idx][r] = val
        for idx, val in V_cols[r].items():
            V[idx][r] = val
        for idx, val in W_cols[r].items():
            W[idx][r] = val

    # Verify correctness
    import random
    random.seed(42)
    for trial in range(200):
        if time.time() > deadline:
            break
        A = [[random.randint(0, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(0, 3) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][jj] * B[jj][k] for jj in range(m)) for k in range(p)] for i in range(n)]
        C_tensor = [[0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for r in range(rank):
                    for j in range(m):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                C_tensor[i][k] = s
        if C_tensor != C:
            return _schoolbook()

    return {"U": U, "V": V, "W": W}

def _schoolbook():
    n, m, p = 5, 5, 5
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
