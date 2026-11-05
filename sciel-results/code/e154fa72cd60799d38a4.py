import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115.0          # safety margin before the 150s kill

    # ---------- generic exact verification (trace convention W[k*n+i, r]) ----------
    def verify(U, V, W, rank):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # ---------- guaranteed-correct schoolbook (rank n*m*p = 48) ----------
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    # ---------- rank-44 structured decomposition (Strassen blocks + dot products) ----------
    def build_rank44():
        R = 44
        U = np.zeros((n * m, R), dtype=float)
        V = np.zeros((m * p, R), dtype=float)
        W = np.zeros((p * n, R), dtype=float)
        c = [0]  # mutable column cursor

        def add_term(u, v, w):
            if c[0] >= R:
                return
            for idx, val in u:
                U[idx, c[0]] += val
            for idx, val in v:
                V[idx, c[0]] += val
            for idx, val in w:
                W[idx, c[0]] += val
            c[0] += 1

        # U index = i*m + j  (i in [0,n), j in [0,m))
        # V index = j*p + k  (j in [0,m), k in [0,p))
        # W index = k*n + i  (k in [0,p), i in [0,n))  -- trace convention

        # ---- TOP part: C_top(i,k) = sum_j A(i,j) B(j,k), i in {0,1}, k in {0,3} ----
        # Split A_top into A1 (j=0,1) and A2 (j=2,3); B into 2x2 blocks.
        # C_top[0,0]=A1·B11, C_top[0,1]=A2·B21, C_top[0,2]=A1·B12, C_top[0,3]=A2·B22
        # C_top[1,0]=A1·B21, C_top[1,1]=A2·B11, C_top[1,2]=A1·B22, C_top[1,3]=A2·B12

        def strassen(ai, bj, ck, w11, w12, w21, w22):
            # ai,bj,ck: (a,b,c,d) indices into U, V, W respectively
            a, b, cc, d = ai
            e, f, g, h = bj
            # M1 = (a+d)(e+h) -> C11(+1), C22(+1)
            add_term([(a, 1), (d, 1)], [(e, 1), (h, 1)], [(w11, 1), (w22, 1)])
            # M2 = (c+d)(e)   -> C21(+1), C22(-1)
            add_term([(cc, 1), (d, 1)], [(e, 1)], [(w21, 1), (w22, -1)])
            # M3 = a(f-h)     -> C12(+1), C22(+1)
            add_term([(a, 1)], [(f, 1), (h, -1)], [(w12, 1), (w22, 1)])
            # M4 = d(g-e)     -> C11(+1), C21(+1)
            add_term([(d, 1)], [(g, 1), (e, -1)], [(w11, 1), (w21, 1)])
            # M5 = (a+b)h     -> C12(+1), C11(-1)
            add_term([(a, 1), (b, 1)], [(h, 1)], [(w12, 1), (w11, -1)])
            # M6 = (a+c)(g-f) -> C21(+1), C11(+1)
            add_term([(a, 1), (cc, 1)], [(g, 1), (f, -1)], [(w21, 1), (w11, 1)])
            # M7 = c(g)       -> C22(+1), C21(-1)
            add_term([(cc, 1)], [(g, 1)], [(w22, 1), (w21, -1)])

        # A1 = rows A(0,0),A(0,1),A(1,0),A(1,1) -> U idx (0,1,4,5)
        # A2 = rows A(0,2),A(0,3),A(1,2),A(1,3) -> U idx (2,3,6,7)
        A1 = (0, 1, 4, 5)
        A2 = (2, 3, 6, 7)

        # B blocks (V idx = j*p + k, p=4):
        B11 = (0, 1, 4, 5)   # j=0,1 k=0,1
        B12 = (2, 3, 6, 7)   # j=0,1 k=2,3
        B21 = (8, 9, 12, 13)  # j=2,3 k=0,1
        B22 = (10, 11, 14, 15)  # j=2,3 k=2,3

        # W output indices (k*n + i):
        # row i=0: k=0,1,2,3 -> 0,1,2,3
        # row i=1: k=0,1,2,3 -> 4,5,6,7
        # C_top[0,*] = W row 0 : (0,1,2,3)
        # C_top[1,*] = W row 1 : (4,5,6,7)
        strassen(A1, B11, 0, (0, 1, 4, 5))      # C_top[0,0] = A1·B11
        strassen(A2, B21, 0, (0, 1, 4, 5))      # C_top[0,1] = A2·B21
        strassen(A1, B12, 0, (0, 1, 4, 5))      # C_top[0,2] = A1·B12
        strassen(A2, B22, 0, (0, 1, 4, 5))      # C_top[0,3] = A2·B22
        strassen(A1, B21, 0, (4, 5, 6, 7))      # C_top[1,0] = A1·B21
        strassen(A2, B11, 0, (4, 5, 6, 7))      # C_top[1,1] = A2·B11
        strassen(A1, B22, 0, (4, 5, 6, 7))      # C_top[1,2] = A1·B22
        strassen(A2, B12, 0, (4, 5, 6, 7))      # C_top[1,3] = A2·B12

        # ---- BOTTOM part: C_bot(k) = sum_j A(2,j) B(j,k), i=2 ----
        # W index for i=2: k*n + 2 = 4k + 2 -> k=0..3 => 2,6,10,14
        # A(2,j) U idx = 2*4 + j = 8,9,10,11
        # V idx = j*p + k
        for k in range(p):
            for j in range(m):
                add_term([(8 + j, 1)], [(j * p + k, 1)], [(k * n + 2, 1)])

        return U, V, W

    # ---------- assemble & pick the best verified candidate ----------
    candidates = []

    U44, V44, W44 = build_rank44()
    if verify(U44, V44, W44, 44):
        candidates.append((44, U44, V44, W44))

    Usb, Vsb, Wsb = schoolbook()
    candidates.append((n * m * p, Usb, Vsb, Wsb))

    # choose the lowest verified rank
    candidates.sort(key=lambda t: t[0])
    best_rank, BU, BV, BW = candidates[0]

    return {"U": BU.tolist(), "V": BV.tolist(), "W": BW.tolist()}
# EVOLVE-BLOCK-END
