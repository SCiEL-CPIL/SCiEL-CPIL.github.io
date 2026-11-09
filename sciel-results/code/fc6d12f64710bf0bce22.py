"""Matrix multiplication 2x3 x 3x3: rank-minimal exact tensor decomposition."""
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    n, m, p = 2, 3, 3
    DEADLINE = 150 - 15  # safety margin

    def build_candidate_17():
        # 17 rank-1 terms: 7 Strassen terms for the 2x2x2 core
        # (i in {0,1}, j in {0,1}, k in {0,1}) + 10 schoolbook terms for the rest.
        # U row = i*m + j, V row = j*p + k, W row = k*n + i.
        U = [[Fraction(0)]*17 for _ in range(n*m)]
        V = [[Fraction(0)]*17 for _ in range(m*p)]
        W = [[Fraction(0)]*17 for _ in range(p*n)]

        def add_term(col, u_rows, v_rows, w_rows):
            for (ri, val) in u_rows:
                U[ri][col] += val
            for (rj, val) in v_rows:
                V[rj][col] += val
            for (rk, val) in w_rows:
                W[rk][col] += val

        # Core 2x2x2 Strassen. a=A[0][0], b=A[0][1], c=A[1][0], d=A[1][1]
        # e=B[0][0], f=B[0][1], g=B[1][0], h=B[1][1]
        # m1=(a+d)(e+h); m2=(c+d)(e); m3=a(f-h); m4=d(g-e)
        # m5=(a+b)h; m6=(c-a)(e+f); m7=(b-d)(g+h)
        # C[0][0]=m1+m4-m5+m7; C[0][1]=m3+m5; C[1][0]=m2+m4; C[1][1]=m1-m2+m3+m6
        add_term(0, [(0,1),(4,1)], [(0,1),(4,1)], [(0,1),(2,1),(1,1),(3,1)])  # m1
        add_term(1, [(3,1),(4,1)], [(0,1)], [(1,1),(3,-1)])                    # m2
        add_term(2, [(0,1)], [(1,1),(4,-1)], [(2,1),(3,1)])                    # m3
        add_term(3, [(4,1)], [(3,1),(0,-1)], [(0,1),(1,1)])                    # m4
        add_term(4, [(0,1),(1,1)], [(4,1)], [(0,-1),(2,1)])                    # m5
        add_term(5, [(3,1),(0,-1)], [(0,1),(1,1)], [(3,1)])                    # m6
        add_term(6, [(1,1),(4,-1)], [(3,1),(4,1)], [(0,1)])                    # m7
        # Schoolbook for j=2 (6 terms), cols 7..12
        col = 7
        for i in range(n):
            for k in range(p):
                U[i*m+2][col] = 1
                V[2*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
        # Schoolbook for k=2, j in {0,1} (4 terms), cols 13..16
        for i in range(n):
            for j in range(2):
                U[i*m+j][col] = 1
                V[j*p+2][col] = 1
                W[2*n+i][col] = 1
                col += 1
        return U, V, W

    def verify(U, V, W):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True

    U17, V17, W17 = build_candidate_17()
    if verify(U17, V17, W17) and (time.time() - start) < DEADLINE:
        def to_plain(mat):
            return [[int(x) if x.denominator == 1 else float(x) for x in row] for row in mat]
        return {"U": to_plain(U17), "V": to_plain(V17), "W": to_plain(W17)}

    # Fallback: schoolbook rank 18.
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1
                V[j*p+k][column] = 1
                W[k*n+i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
