import time
from functools import lru_cache

DIMENSIONS = [4, 5, 5]
_start_time = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill

@lru_cache(maxsize=None)
def _decompose(m, n, p):
    """
    Return (U, V, W) for m×n * n×p matrix multiplication.
    U: shape (m*n, rank)
    V: shape (n*p, rank)
    W: shape (p*m, rank), W[k*m+i, r] convention
    """
    if time.time() - _start_time > TIME_LIMIT:
        raise TimeoutError()
    
    # Base case: schoolbook
    if m == 0 or n == 0 or p == 0:
        return ([], [], [])
    if m == 1 and n == 1 and p == 1:
        return ([[1]], [[1]], [[1]])
    
    # Special case: Strassen for 2×2 * 2×2
    if m == 2 and n == 2 and p == 2:
        # Rank 7 Strassen decomposition
        U = []
        V = []
        W = []
        # M1 = (a+d)(e+h)
        U.append([1, 0, 0, 1])  # a=1, d=1 (indices 0,1,2,3 for i*2+j)
        V.append([1, 0, 0, 1])  # e=1, h=1
        W.append([1, 0, 0, 1])  # C[0,0]=1, C[1,1]=1 (indices k*2+i)
        # M2 = (c+d)e
        U.append([0, 0, 1, 1])  # c=1, d=1
        V.append([1, 0, 0, 0])  # e=1
        W.append([0, -1, 0, -1])  # C[1,0]=-1, C[1,1]=-1
        # M3 = a(f-h)
        U.append([1, 0, 0, 0])  # a=1
        V.append([0, 1, 0, -1])  # f=1, h=-1
        W.append([0, 1, 0, 1])  # C[0,1]=1, C[1,1]=1
        # M4 = d(g-e)
        U.append([0, 0, 0, 1])  # d=1
        V.append([-1, 0, 1, 0])  # g=1, e=-1
        W.append([1, 0, 1, 0])  # C[0,0]=1, C[1,0]=1
        # M5 = (a+b)h
        U.append([1, 1, 0, 0])  # a=1, b=1
        V.append([0, 0, 0, 1])  # h=1
        W.append([-1, 0, 0, 1])  # C[0,0]=-1, C[0,1]=1
        # M6 = (c-a)(e+f)
        U.append([-1, 0, 1, 0])  # c=1, a=-1
        V.append([1, 1, 0, 0])  # e=1, f=1
        W.append([0, 0, 0, 1])  # C[1,1]=1
        # M7 = (b-d)(g+h)
        U.append([0, 1, 0, -1])  # b=1, d=-1
        V.append([0, 0, 1, 1])  # g=1, h=1
        W.append([1, 0, 0, 0])  # C[0,0]=1
        return (U, V, W)
    
    # Try all possible splits and pick the best
    best = None
    best_rank = m * n * p  # schoolbook
    
    for m1 in range(1, m):
        m2 = m - m1
        for n1 in range(1, n):
            n2 = n - n1
            for p1 in range(1, p):
                p2 = p - p1
                try:
                    # 8 sub-decompositions
                    U1, V1, W1 = _decompose(m1, n1, p1)
                    U2, V2, W2 = _decompose(m1, n1, p2)
                    U3, V3, W3 = _decompose(m1, n2, p1)
                    U4, V4, W4 = _decompose(m1, n2, p2)
                    U5, V5, W5 = _decompose(m2, n1, p1)
                    U6, V6, W6 = _decompose(m2, n1, p2)
                    U7, V7, W7 = _decompose(m2, n2, p1)
                    U8, V8, W8 = _decompose(m2, n2, p2)
                    
                    total_rank = (len(U1) + len(U2) + len(U3) + len(U4) +
                                 len(U5) + len(U6) + len(U7) + len(U8))
                    if total_rank < best_rank:
                        best_rank = total_rank
                        # Build combined U, V, W
                        U_new = []
                        V_new = []
                        W_new = []
                        
                        def add_decomp(U_s, V_s, W_s, m_i, n_i, p_i, offset_m, offset_n, offset_p):
                            for r in range(len(U_s)):
                                u_row = [0] * (m * n)
                                v_row = [0] * (n * p)
                                w_row = [0] * (p * m)
                                for ii in range(m_i):
                                    for jj in range(n_i):
                                        u_row[(offset_m + ii) * n + (offset_n + jj)] = U_s[r][ii * n_i + jj]
                                for jj in range(n_i):
                                    for kk in range(p_i):
                                        v_row[(offset_n + jj) * p + (offset_p + kk)] = V_s[r][jj * p_i + kk]
                                for kk in range(p_i):
                                    for ii in range(m_i):
                                        w_row[(offset_p + kk) * m + (offset_m + ii)] = W_s[r][kk * m_i + ii]
                                U_new.append(u_row)
                                V_new.append(v_row)
                                W_new.append(w_row)
                        
                        # A11*B11: m1×n1 * n1×p1
                        add_decomp(U1, V1, W1, m1, n1, p1, 0, 0, 0)
                        # A12*B21: m1×n2 * n2×p1
                        add_decomp(U3, V3, W3, m1, n2, p1, 0, n1, 0)
                        # A11*B12: m1×n1 * n1×p2
                        add_decomp(U2, V2, W2, m1, n1, p2, 0, 0, p1)
                        # A12*B22: m1×n2 * n2×p2
                        add_decomp(U4, V4, W4, m1, n2, p2, 0, n1, p1)
                        # A21*B11: m2×n1 * n1×p1
                        add_decomp(U5, V5, W5, m2, n1, p1, m1, 0, 0)
                        # A21*B12: m2×n1 * n1×p2
                        add_decomp(U6, V6, W6, m2, n1, p2, m1, 0, p1)
                        # A22*B21: m2×n2 * n2×p1
                        add_decomp(U7, V7, W7, m2, n2, p1, m1, n1, 0)
                        # A22*B22: m2×n2 * n2×p2
                        add_decomp(U8, V8, W8, m2, n2, p2, m1, n1, p1)
                        
                        best = (U_new, V_new, W_new)
                except TimeoutError:
                    raise
                except Exception:
                    continue
    
    if best is not None:
        return best
    else:
        # Schoolbook
        rank = m * n * p
        U = [[0] * rank for _ in range(m * n)]
        V = [[0] * rank for _ in range(n * p)]
        W = [[0] * rank for _ in range(p * m)]
        col = 0
        for i in range(m):
            for j in range(n):
                for k in range(p):
                    U[i * n + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * m + i][col] = 1
                    col += 1
        return (U, V, W)

def construct():
    global _start_time
    _start_time = time.time()
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    U, V, W = _decompose(n, m, p)
    return {"U": U, "V": V, "W": W}
