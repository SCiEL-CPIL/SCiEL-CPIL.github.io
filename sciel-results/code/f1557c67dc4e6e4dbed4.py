import time
from functools import lru_cache
import numpy as np

DIMENSIONS = [4, 5, 5]
_start_time = time.time()
TIME_LIMIT = 145  # Safety margin

def _combine_decomps(sub_decomps, m, n, p):
    """
    Combine 8 sub-decompositions into a full decomposition.
    sub_decomps: list of 8 (U_s, V_s, W_s, m_i, n_i, p_i, off_m, off_n, off_p) tuples
    Order: (m1,n1,p1), (m1,n1,p2), (m1,n2,p1), (m1,n2,p2),
            (m2,n1,p1), (m2,n1,p2), (m2,n2,p1), (m2,n2,p2)
    """
    U_new = []
    V_new = []
    W_new = []
    for U_s, V_s, W_s, m_i, n_i, p_i, off_m, off_n, off_p in sub_decomps:
        for r in range(len(U_s)):
            u_row = [0] * (m * n)
            v_row = [0] * (n * p)
            w_row = [0] * (p * m)
            for ii in range(m_i):
                for jj in range(n_i):
                    u_row[(off_m + ii) * n + (off_n + jj)] = U_s[r][ii * n_i + jj]
            for jj in range(n_i):
                for kk in range(p_i):
                    v_row[(off_n + jj) * p + (off_p + kk)] = V_s[r][jj * p_i + kk]
            for kk in range(p_i):
                for ii in range(m_i):
                    w_row[(off_p + kk) * m + (off_m + ii)] = W_s[r][kk * m_i + ii]
            U_new.append(u_row)
            V_new.append(v_row)
            W_new.append(w_row)
    return (U_new, V_new, W_new)

def _split_p_decompose(m, n, p, p1):
    """
    Decompose m×n×p by splitting p into p1 and p2=p-p1.
    Uses 4 sub-problems (not 8) since we only split p.
    """
    p2 = p - p1
    U1, V1, W1 = _decompose(m, n, p1)
    U2, V2, W2 = _decompose(m, n, p2)
    
    U_new = []
    V_new = []
    W_new = []
    for r in range(len(U1)):
        u_row = [0] * (m * n)
        v_row = [0] * (n * p)
        w_row = [0] * (p * m)
        for ii in range(m):
            for jj in range(n):
                u_row[ii * n + jj] = U1[r][ii * n + jj]
        for jj in range(n):
            for kk in range(p1):
                v_row[jj * p + kk] = V1[r][jj * p1 + kk]
        for kk in range(p1):
            for ii in range(m):
                w_row[kk * m + ii] = W1[r][kk * m + ii]
        U_new.append(u_row)
        V_new.append(v_row)
        W_new.append(w_row)
    for r in range(len(U2)):
        u_row = [0] * (m * n)
        v_row = [0] * (n * p)
        w_row = [0] * (p * m)
        for ii in range(m):
            for jj in range(n):
                u_row[ii * n + jj] = U2[r][ii * n + jj]
        for jj in range(n):
            for kk in range(p2):
                v_row[jj * p + (p1 + kk)] = V2[r][jj * p2 + kk]
        for kk in range(p2):
            for ii in range(m):
                w_row[(p1 + kk) * m + ii] = W2[r][kk * m + ii]
        U_new.append(u_row)
        V_new.append(v_row)
        W_new.append(w_row)
    return (U_new, V_new, W_new)

def _split_n_decompose(m, n, p, n1):
    """
    Decompose m×n×p by splitting n into n1 and n2=n-n1.
    Uses 4 sub-problems.
    """
    n2 = n - n1
    U1, V1, W1 = _decompose(m, n1, p)
    U2, V2, W2 = _decompose(m, n2, p)
    
    U_new = []
    V_new = []
    W_new = []
    for r in range(len(U1)):
        u_row = [0] * (m * n)
        v_row = [0] * (n * p)
        w_row = [0] * (p * m)
        for ii in range(m):
            for jj in range(n1):
                u_row[ii * n + jj] = U1[r][ii * n1 + jj]
        for jj in range(n1):
            for kk in range(p):
                v_row[jj * p + kk] = V1[r][jj * p + kk]
        for kk in range(p):
            for ii in range(m):
                w_row[kk * m + ii] = W1[r][kk * m + ii]
        U_new.append(u_row)
        V_new.append(v_row)
        W_new.append(w_row)
    for r in range(len(U2)):
        u_row = [0] * (m * n)
        v_row = [0] * (n * p)
        w_row = [0] * (p * m)
        for ii in range(m):
            for jj in range(n2):
                u_row[ii * n + (n1 + jj)] = U2[r][ii * n2 + jj]
        for jj in range(n2):
            for kk in range(p):
                v_row[(n1 + jj) * p + kk] = V2[r][jj * p + kk]
        for kk in range(p):
            for ii in range(m):
                w_row[kk * m + ii] = W2[r][kk * m + ii]
        U_new.append(u_row)
        V_new.append(v_row)
        W_new.append(w_row)
    return (U_new, V_new, W_new)

def _split_m_decompose(m, n, p, m1):
    """
    Decompose m×n×p by splitting m into m1 and m2=m-m1.
    Uses 4 sub-problems.
    """
    m2 = m - m1
    U1, V1, W1 = _decompose(m1, n, p)
    U2, V2, W2 = _decompose(m2, n, p)
    
    U_new = []
    V_new = []
    W_new = []
    for r in range(len(U1)):
        u_row = [0] * (m * n)
        v_row = [0] * (n * p)
        w_row = [0] * (p * m)
        for ii in range(m1):
            for jj in range(n):
                u_row[ii * n + jj] = U1[r][ii * n + jj]
        for jj in range(n):
            for kk in range(p):
                v_row[jj * p + kk] = V1[r][jj * p + kk]
        for kk in range(p):
            for ii in range(m1):
                w_row[kk * m + ii] = W1[r][kk * m + ii]
        U_new.append(u_row)
        V_new.append(v_row)
        W_new.append(w_row)
    for r in range(len(U2)):
        u_row = [0] * (m * n)
        v_row = [0] * (n * p)
        w_row = [0] * (p * m)
        for ii in range(m2):
            for jj in range(n):
                u_row[(m1 + ii) * n + jj] = U2[r][ii * n + jj]
        for jj in range(n):
            for kk in range(p):
                v_row[jj * p + kk] = V2[r][jj * p + kk]
        for kk in range(p):
            for ii in range(m2):
                w_row[kk * m + (m1 + ii)] = W2[r][kk * m + ii]
        U_new.append(u_row)
        V_new.append(v_row)
        W_new.append(w_row)
    return (U_new, V_new, W_new)

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
    
    # Base case: empty
    if m == 0 or n == 0 or p == 0:
        return ([], [], [])
    
    # Degenerate cases
    if m == 1 and n == 1 and p == 1:
        return ([[1]], [[1]], [[1]])
        
    if m == 1:
        # Rank n*p
        rank = n * p
        U = [[0] * rank for _ in range(n)]
        V = [[0] * rank for _ in range(n * p)]
        W = [[0] * rank for _ in range(p)]
        col = 0
        for j in range(n):
            for k in range(p):
                U[j][col] = 1
                V[j * p + k][col] = 1
                W[k][col] = 1
                col += 1
        return (U, V, W)
        
    if p == 1:
        # Rank m*n
        rank = m * n
        U = [[0] * rank for _ in range(m * n)]
        V = [[0] * rank for _ in range(n)]
        W = [[0] * rank for _ in range(m)]
        col = 0
        for i in range(m):
            for j in range(n):
                U[i * n + j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
                col += 1
        return (U, V, W)
        
    if n == 1:
        # Rank m*p
        rank = m * p
        U = [[0] * rank for _ in range(m)]
        V = [[0] * rank for _ in range(p)]
        W = [[0] * rank for _ in range(m * p)]
        col = 0
        for i in range(m):
            for k in range(p):
                U[i][col] = 1
                V[k][col] = 1
                W[k * m + i][col] = 1
                col += 1
        return (U, V, W)

    # Special case: Strassen for 2×2 * 2×2
    if m == 2 and n == 2 and p == 2:
        U = [
            [1, 0, 0, 1],   # a+d
            [0, 0, 1, 1],   # c+d
            [1, 0, 0, 0],   # a
            [0, 0, 0, 1],   # d
            [1, 1, 0, 0],   # a+b
            [-1, 0, 1, 0],  # c-a
            [0, 1, 0, -1]   # b-d
        ]
        V = [
            [1, 0, 0, 1],   # e+h
            [1, 0, 0, 0],   # e
            [0, 1, 0, -1],  # f-h
            [-1, 0, 1, 0],  # g-e
            [0, 0, 0, 1],   # h
            [1, 1, 0, 0],   # e+f
            [0, 0, 1, 1]    # g+h
        ]
        W = [
            [1, 0, 0, 1],   # C11 and C22
            [0, -1, 0, -1], # C21 and C22
            [0, 1, 0, 1],   # C12 and C22
            [1, 0, 1, 0],   # C11 and C21
            [-1, 0, 0, 1],  # C11 and C12
            [0, 0, 0, 1],   # C22
            [1, 0, 0, 0]    # C11
        ]
        return (U, V, W)
    
    # Special case: 2×2×3 via splitting p=2+1
    if m == 2 and n == 2 and p == 3:
        return _split_p_decompose(2, 2, 3, 2)
    
    # Special case: 2×3×2 via splitting n=2+1
    if m == 2 and n == 3 and p == 2:
        return _split_n_decompose(2, 3, 2, 2)
    
    # Special case: 3×2×2 via splitting m=2+1
    if m == 3 and n == 2 and p == 2:
        return _split_m_decompose(3, 2, 2, 2)
    
    # Special case: 2×3×3 via splitting p=2+1
    if m == 2 and n == 3 and p == 3:
        return _split_p_decompose(2, 3, 3, 2)
    
    # Special case: 3×2×3 via splitting n=2+1
    if m == 3 and n == 2 and p == 3:
        return _split_n_decompose(3, 2, 3, 2)
    
    # Special case: 3×3×2 via splitting m=2+1
    if m == 3 and n == 3 and p == 2:
        return _split_m_decompose(3, 3, 2, 2)
    
    # Special case: 2×2×4 via splitting p=2+2
    if m == 2 and n == 2 and p == 4:
        return _split_p_decompose(2, 2, 4, 2)
    
    # Special case: 2×4×2 via splitting n=2+2
    if m == 2 and n == 4 and p == 2:
        return _split_n_decompose(2, 4, 2, 2)
    
    # Special case: 4×2×2 via splitting m=2+2
    if m == 4 and n == 2 and p == 2:
        return _split_m_decompose(4, 2, 2, 2)
    
    # Try all possible splits and pick the best
    best = None
    best_rank = m * n * p  # schoolbook
    
    # Generate candidate splits - prioritize balanced splits
    def get_splits(dim):
        splits = []
        for i in range(1, dim):
            splits.append(i)
        # Sort by how balanced the split is (more balanced first)
        splits.sort(key=lambda x: abs(dim - 2*x))
        return splits

    for m1 in get_splits(m):
        m2 = m - m1
        if time.time() - _start_time > TIME_LIMIT:
            break
        for n1 in get_splits(n):
            n2 = n - n1
            if time.time() - _start_time > TIME_LIMIT:
                break
            for p1 in get_splits(p):
                p2 = p - p1
                if time.time() - _start_time > TIME_LIMIT:
                    break
                try:
                    # 8 sub-decompositions (cached)
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
                        sub_decomps = [
                            (U1, V1, W1, m1, n1, p1, 0, 0, 0),
                            (U2, V2, W2, m1, n1, p2, 0, 0, p1),
                            (U3, V3, W3, m1, n2, p1, 0, n1, 0),
                            (U4, V4, W4, m1, n2, p2, 0, n1, p1),
                            (U5, V5, W5, m2, n1, p1, m1, 0, 0),
                            (U6, V6, W6, m2, n1, p2, m1, 0, p1),
                            (U7, V7, W7, m2, n2, p1, m1, n1, 0),
                            (U8, V8, W8, m2, n2, p2, m1, n1, p1),
                        ]
                        best = _combine_decomps(sub_decomps, m, n, p)
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
    # Clear cache to start fresh
    _decompose.cache_clear()
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    U, V, W = _decompose(n, m, p)
    return {"U": U, "V": V, "W": W}
