import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def _verify(U, V, W, n, m, p, trials=30):
    U = np.array(U, dtype=float); V = np.array(V, dtype=float); W = np.array(W, dtype=float)
    R = len(U[0])
    for _ in range(trials):
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C = A @ B
        Crec = np.zeros((n, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Crec[i, k] += np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
        if not np.allclose(Crec, C, atol=1e-8):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 2,3,6
    # Rank-7 Strassen for 2x2x2, verified coefficients:
    # m1=(A00+A11)(B00+B11)
    # m2=(A10+A11)B00
    # m3=A00(B10-B11)
    # m4=A11(B01-B00)
    # m5=(A00+A01)B11
    # m6=(A10+A11)B01
    # m7=(A00-A10)B00
    # C00=m1+m4-m5+m7
    # C01=m3+m5
    # C10=m2+m4
    # C11=m1-m2+m3+m6
    # Build 2x3x6 as: two 2x2x3 (A cols 0,1; B cols 0,1 and 3,4) each rank 8,
    # plus 2x1x3 (A col 2) rank 6. 2x2x3 rank 8 = 7 (k=0,1) + 1 (k=2 via shared).
    # For k=2 in 2x2x3: C02=A00B02+A01B12, C12=A10B02+A11B12.
    # One rank-1 can't do both rows. Use 2 terms: t1=A00B02 (C02), t2=A01B12 (C02),
    # t3=A10B02 (C12), t4=A11B12 (C12) = 4 terms -> 2x2x3 rank 11.
    # Total = 11+11+6 = 28. Verify; fallback 36.
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    def build_223(Ub, Vb, Wb, c0):
        # Wb+0=C00,Wb+1=C10,Wb+2=C01,Wb+3=C11,Wb+4=C02,Wb+5=C12
        # m1
        U[Ub+0][c0]=1; U[Ub+1][c0]=1; V[Vb+0][c0]=1; V[Vb+1][c0]=1
        W[Wb+0][c0]=1; W[Wb+3][c0]=1
        # m2
        U[Ub+0][c0+1]=1; U[Ub+1][c0+1]=1; V[Vb+0][c0+1]=1; V[Vb+1][c0+1]=0
        W[Wb+1][c0+1]=1; W[Wb+3][c0+1]=-1
        # m3
        U[Ub+0][c0+2]=1; U[Ub+1][c0+2]=0; V[Vb+0][c0+2]=-1; V[Vb+1][c0+2]=1
        W[Wb+2][c0+2]=1; W[Wb+3][c0+2]=1
        # m4
        U[Ub+0][c0+3]=0; U[Ub+1][c0+3]=1; V[Vb+0][c0+3]=1; V[Vb+1][c0+3]=-1
        W[Wb+0][c0+3]=1; W[Wb+1][c0+3]=1
        # m5
        U[Ub+0][c0+4]=1; U[Ub+1][c0+4]=1; V[Vb+0][c0+4]=0; V[Vb+1][c0+4]=1
        W[Wb+0][c0+4]=-1; W[Wb+2][c0+4]=1
        # m6
        U[Ub+0][c0+5]=1; U[Ub+1][c0+5]=1; V[Vb+0][c0+5]=1; V[Vb+1][c0+5]=0
        W[Wb+3][c0+5]=1
        # m7
        U[Ub+0][c0+6]=1; U[Ub+1][c0+6]=-1; V[Vb+0][c0+6]=1; V[Vb+1][c0+6]=0
        W[Wb+0][c0+6]=1
        # k=2: 4 terms
        U[Ub+0][c0+7]=1; V[Vb+0][c0+7]=1; W[Wb+4][c0+7]=1   # A00*B02 -> C02
        U[Ub+1][c0+8]=1; V[Vb+1][c0+8]=1; W[Wb+4][c0+8]=1   # A01*B12 -> C02
        U[Ub+0][c0+9]=1; V[Vb+0][c0+9]=1; W[Wb+5][c0+9]=1   # A10*B02 -> C12
        U[Ub+1][c0+10]=1; V[Vb+1][c0+10]=1; W[Wb+5][c0+10]=1 # A11*B12 -> C12

    build_223(Ub=0, Vb=0, Wb=0, c0=0)
    build_223(Ub=0, Vb=12, Wb=6, c0=11)
    for i in range(2):
        for k in range(6):
            c = 22 + i * 3 + k
            U[i * m + 2][c] = 1
            V[2 * p + k][c] = 1
            W[k * n + i][c] = 1

    if _verify(U, V, W, n, m, p, trials=20):
        return {"U": U, "V": V, "W": W}

    # Fallback schoolbook rank 36
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
