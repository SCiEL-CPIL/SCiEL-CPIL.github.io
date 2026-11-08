import time

DIMENSIONS = [2, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    rank = 5
    nm = n * m
    mp = m * p
    pn = p * n

    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    # Term 0: (i={0,1}) x (j=0) x (k={0,1})
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[1][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1

    # Term 1: (i={0,1}) x (j=1) x (k={0,1})
    U[1][1] = 1; U[4][1] = 1
    V[3][1] = 1; V[4][1] = 1
    W[0][1] = 1; W[1][1] = 1; W[2][1] = 1; W[3][1] = 1

    # Term 2: (i={0,1}) x (j=2) x (k={0,1})
    U[2][2] = 1; U[5][2] = 1
    V[6][2] = 1; V[7][2] = 1
    W[0][2] = 1; W[1][2] = 1; W[2][2] = 1; W[3][2] = 1

    # Term 3: (i={0,1}) x (j={0,1}) x (k=2)
    U[0][3] = 1; U[1][3] = 1; U[3][3] = 1; U[4][3] = 1
    V[2][3] = 1; V[5][3] = 1
    W[4][3] = 1; W[5][3] = 1

    # Term 4: (i={0,1}) x (j=2) x (k=2)
    U[2][4] = 1; U[5][4] = 1
    V[8][4] = 1
    W[4][4] = 1; W[5][4] = 1

    return {"U": U, "V": V, "W": W}
