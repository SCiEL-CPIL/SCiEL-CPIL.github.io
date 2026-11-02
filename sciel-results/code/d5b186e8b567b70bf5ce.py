"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = 120.0  # seconds; evaluator kills at 150s, keep margin
    n, m, p = DIMENSIONS
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8

    def _build_rank14():
        rank = 14
        U = [[0] * rank for _ in range(u_dim)]
        V = [[0] * rank for _ in range(v_dim)]
        W = [[0] * rank for _ in range(w_dim)]

        # A indices: U[0]=A00, U[1]=A01, U[2]=A10, U[3]=A11
        # B indices: V[0]=B00,V[1]=B01,V[2]=B02,V[3]=B03,V[4]=B10,V[5]=B11,V[6]=B12,V[7]=B13
        # C indices (W): W[0]=C00,W[1]=C10,W[2]=C01,W[3]=C11,W[4]=C02,W[5]=C12,W[6]=C03,W[7]=C13

        # Left block (B cols 0,1) - Strassen
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        # M3 = A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1
        # M4 = A11(B10-B00)
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        # M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1
        # M6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        # M7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1

        # Right block (B cols 2,3) - Strassen
        # N1 = (A00+A11)(B02+B13)
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        # N2 = (A10+A11)B02
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1
        # N3 = A00(B03-B13)
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1
        # N4 = A11(B12-B02)
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1
        # N5 = (A00+A01)B13
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1
        # N6 = (A10-A00)(B02+B03)
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1
        # N7 = (A01-A11)(B12+B13)
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1

        return U, V, W

    def _verify(U, V, W):
        rank = len(U[0])
        # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        # Must equal 1 iff i=a*m+b, j=b*p+c, k=c*n+a
        for i in range(u_dim):
            a = i // m
            b = i % m
            for j in range(v_dim):
                b2 = j // p
                c = j % p
                for k in range(w_dim):
                    c2 = k // n
                    a2 = k % n
                    expected = 1.0 if (a == a2 and b == b2 and c == c2) else 0.0
                    s = 0.0
                    for r in range(rank):
                        s += U[i][r] * V[j][r] * W[k][r]
                    if abs(s - expected) > 1e-9:
                        return False
        return True

    def _build_schoolbook():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(u_dim)]
        V = [[0] * rank for _ in range(v_dim)]
        W = [[0] * rank for _ in range(w_dim)]
        col = 0
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    i = a * m + b
                    j = b * p + c
                    k = c * n + a
                    U[i][col] = 1
                    V[j][col] = 1
                    W[k][col] = 1
                    col += 1
        return U, V, W

    try:
        U, V, W = _build_rank14()
        if time.time() - t_start < deadline - 5.0:
            if _verify(U, V, W):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    # Fallback: schoolbook (rank 16)
    U, V, W = _build_schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
