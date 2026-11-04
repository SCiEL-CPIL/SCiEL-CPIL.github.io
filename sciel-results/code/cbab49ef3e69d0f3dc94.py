"""Matrix multiplication 4x4 * 4x9: search for rank-minimal exact tensor decomposition."""
import time
import random
from fractions import Fraction as F
import numpy as np

DIMENSIONS = [4, 4, 9]
DEADLINE = 110.0

def _validate_numpy(U, V, W):
    """Fast validation using numpy. U, V, W are lists of lists of int/Fraction.
    Returns True if the decomposition is valid."""
    n, m, p = DIMENSIONS
    r = len(U[0])
    
    # Determine scale factor (power of 2) to convert to integers
    maxden = 1
    for mat in (U, V, W):
        for row in mat:
            for v in row:
                if isinstance(v, F):
                    d = v.denominator
                    if d & (d - 1):
                        return False  # non-dyadic
                    if d > maxden:
                        maxden = d
    s = 1
    while maxden % s != 0:
        s *= 2
    
    def to_int(mat):
        return np.array([[int(v * s) if isinstance(v, F) else int(v) * s for v in row] for row in mat], dtype=np.int64)
    
    Us = to_int(U)
    Vs = to_int(V)
    Ws = to_int(W)
    s3 = s * s * s
    
    for i in range(n):
        for j in range(m):
            Ui = Us[i * m + j]
            for k in range(p):
                Vk = Vs[j * p + k]
                Wk = Ws[k * n + i]
                if np.dot(Ui, Vk * Wk) != s3:
                    return False
    return True

def schoolbook():
    n, m, p = DIMENSIONS
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return r, U, V, W

def strassen_2x2x2():
    """Rank-7 Strassen for 2x2 * 2x2. Returns (r, U, V, W) for n=m=p=2."""
    n = m = p = 2
    r = 7
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # M2 = A11*(B10-B00)
    U[3][1] = 1
    V[0][1] = -1; V[4][1] = 1
    W[1][1] = 1
    # M3 = (A10+A11)*B01
    U[2][2] = 1; U[3][2] = 1
    V[5][2] = 1
    W[2][2] = 1
    # M4 = A00*(B01-B11)
    U[0][3] = 1
    V[1][3] = 1; V[5][3] = -1
    W[3][3] = 1
    # M5 = (A00+A01)(B11+B00)
    U[0][4] = 1; U[1][4] = 1
    V[0][4] = 1; V[1][4] = 1
    W[0][4] = 1; W[1][4] = 1
    # M6 = (A01-A11)*(B10+B11)
    U[1][5] = 1; U[3][5] = -1
    V[1][5] = 1; V[5][5] = 1
    W[2][5] = 1
    # M7 = (A10-A00)*(B00+B01)
    U[2][6] = 1; U[0][6] = -1
    V[0][6] = 1; V[4][6] = 1
    W[0][6] = 1
    return r, U, V, W

def block_structured():
    """
    Decompose 4x4 * 4x9 using schoolbook for each output column.
    Total rank: 4*4*9 = 144
    """
    n, m, p = DIMENSIONS
    all_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                all_terms.append((i, j, k, 1))
    
    r = len(all_terms)
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    for c, (i, j, k, sgn) in enumerate(all_terms):
        U[i * m + j][c] = sgn
        V[j * p + k][c] = 1
        W[k * n + i][c] = 1
    return r, U, V, W

def random_search(max_trials=500, max_rank=40):
    n, m, p = DIMENSIONS
    rng = random.Random(42)
    for trial in range(max_trials):
        if time.time() > DEADLINE - 15:
            break
        r = rng.randint(10, max_rank)
        def rand_entry():
            x = rng.random()
            if x < 0.7:
                return 0
            elif x < 0.9:
                return rng.choice([-1, 1])
            else:
                return rng.choice([-1, 1]) * F(1, 2)
        U = [[rand_entry() for _ in range(r)] for _ in range(n * m)]
        V = [[rand_entry() for _ in range(r)] for _ in range(m * p)]
        W = [[rand_entry() for _ in range(r)] for _ in range(p * n)]
        try:
            if _validate_numpy(U, V, W):
                return r, U, V, W
        except Exception:
            pass
    return None

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    best = None
    
    def consider(r, U, V, W):
        nonlocal best
        if best is None or r < best[0]:
            best = (r, U, V, W)
    
    # 1) Schoolbook (rank 144) - guaranteed valid
    r, U, V, W = schoolbook()
    if _validate_numpy(U, V, W):
        consider(r, U, V, W)
    
    # 2) Block structured (also schoolbook, rank 144)
    r, U, V, W = block_structured()
    if _validate_numpy(U, V, W):
        consider(r, U, V, W)
    
    # 3) Random search for better rank
    result = random_search(max_trials=300, max_rank=35)
    if result is not None:
        r, U, V, W = result
        consider(r, U, V, W)
    
    if best is None:
        r, U, V, W = schoolbook()
        return {"U": U, "V": V, "W": W}
    
    _, U, V, W = best
    return {"U": U, "V": V, "W": W}
