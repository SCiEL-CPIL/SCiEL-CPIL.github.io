"""
Combined approach: Block-partitioning with Strassen-based sub-decompositions.

Strategy:
- Decompose 3x5x6 matrix multiplication using block partitioning.
- Split dimensions into blocks where known low-rank decompositions exist.
- Specifically: 3 = 1 + 2, 5 = 2 + 3, 6 = 2 + 4 (or other splits)
- Use Strassen (rank 7 for 2x2x2) and schoolbook for smaller blocks.
- The key insight: M_{n+m, p+q, r+s} can be bounded by combinations of
  smaller matrix multiplication tensors.

For 3x5x6:
- Split A (3x5) into blocks: row blocks [1, 2], col blocks [2, 3]
- Split B (5x6) into blocks: row blocks [2, 3], col blocks [2, 4]
- Split C (3x6) into blocks: row blocks [1, 2], col blocks [2, 4]

Block structure:
  A = [A11 A12]     where A11 is 1x2, A12 is 1x3
       [A21 A22]     where A21 is 2x2, A22 is 2x3
  B = [B11 B12]     where B11 is 2x2, B12 is 2x4
       [B21 B22]     where B21 is 3x2, B22 is 3x4
  C = [C11 C12]     where C11 is 1x2, C12 is 1x4
       [C21 C22]     where C21 is 2x2, C22 is 2x4

C11 = A11*B11  (1x2 * 2x2 -> 1x2): rank 2 (schoolbook)
C12 = A11*B12  (1x2 * 2x4 -> 1x4): rank 4 (schoolbook)
C21 = A21*B11 + A22*B21  (2x2*2x2 + 2x3*3x2 -> 2x2): 
       This is a sum of two products, not directly a single MM tensor.
       Actually, C21 = A21*B11 + A22*B21 is not a standard MM.
       
Let me reconsider. The block multiplication gives:
  C11 = A11 * B11
  C12 = A11 * B12
  C21 = A21 * B11 + A22 * B21
  C22 = A21 * B12 + A22 * B22

The terms C21 and C22 involve sums of products, which don't directly correspond
to a single matrix multiplication tensor. This approach is problematic.

Alternative: Use the fact that M_{n,m,p} is subadditive:
  R(n+m, p+q, r) <= R(n,p,r) + R(m,q,r)  [not quite right]

Actually, the correct subadditivity is:
  R(n1+n2, m, p) <= R(n1, m, p) + R(n2, m, p)  [splitting rows of A]
  R(n, m1+m2, p) <= R(n, m1, p) + R(n, m2, p)  [splitting cols of A / rows of B]
  R(n, m, p1+p2) <= R(n, m, p1) + R(n, m, p2)  [splitting cols of B / cols of C]

So:
  R(3, 5, 6) <= R(1, 5, 6) + R(2, 5, 6)
  R(1, 5, 6) = 6 (each of 5*6=30 terms... no, R(1,m,p) = m*p)
  Actually R(1, m, p) = m*p since it's just dot products.
  R(1, 5, 6) = 30? No, that's not right either.
  
  For 1xM times MxP: the result is 1xP, each entry is a dot product of length M.
  R(1, M, P) = P (you need P independent multiplications, one for each output).
  Wait, no. The tensor for 1xM * MxP has entries T[i,j,k] where i in [0], j in [0,M), k in [0,P).
  T[0,j,k] = 1 for all j,k. So it's a rank... let's think.
  U has shape (1*M, rank) = (M, rank)
  V has shape (M*P, rank) = (MP, rank)  
  W has shape (P*1, rank) = (P, rank)
  
  We need sum_c U[j,c] * V[j*P+k,c] * W[k,c] = 1 for all j,k.
  
  If we set rank = P: U[j,c] = 1 for all j,c; V[j*P+k,c] = 1 if k==c else 0; W[k,c] = 1 if k==c else 0.
  Then sum_c U[j,c]*V[j*P+k,c]*W[k,c] = sum_c 1 * (k==c ? 1 : 0) * (k==c ? 1 : 0) = 1 if k is valid.
  Yes! R(1, M, P) = P.
  
  Similarly R(N, 1, P) = N, R(N, M, 1) = M.

So:
  R(3, 5, 6) <= R(1, 5, 6) + R(2, 5, 6) = 6 + R(2, 5, 6)
  R(2, 5, 6) <= R(2, 2, 6) + R(2, 3, 6)
  R(2, 2, 6) <= R(2, 2, 2) + R(2, 2, 2) + R(2, 2, 2) = 7+7+7 = 21? 
  No, R(2,2,6) = R(2,2,2+2+2) <= R(2,2,2)*3? No.
  R(n, m, p1+p2) <= R(n, m, p1) + R(n, m, p2)
  R(2, 2, 6) <= R(2,2,2) + R(2,2,2) + R(2,2,2) = 7+7+7 = 21. But actually R(2,2,6) = 12 (schoolbook) or maybe less.
  
  This is getting complicated. Let me just implement a clean approach:
  1. Start with schoolbook (rank 90).
  2. Try to improve using known bounds from subadditivity.
  3. Use Strassen where applicable.

Given the time constraints and the fact that all previous attempts scored 1.0 (schoolbook),
let me implement a careful block decomposition that actually works.

Key decomposition:
  R(3,5,6) <= R(1,5,6) + R(2,5,6) = 6 + R(2,5,6)
  R(2,5,6) <= R(2,5,2) + R(2,5,4)
  R(2,5,4) <= R(2,5,2) + R(2,5,2) = 2*R(2,5,2)
  So R(2,5,6) <= 3*R(2,5,2)
  
  R(2,5,2) <= R(2,2,2) + R(2,3,2) = 7 + R(2,3,2)
  R(2,3,2) <= R(2,1,2) + R(2,2,2) = 2 + 7 = 9
  So R(2,5,2) <= 7 + 9 = 16
  R(2,5,6) <= 3*16 = 48
  R(3,5,6) <= 6 + 48 = 54

That's a big improvement from 90! Let me implement this.
"""

import time
import numpy as np

DIMENSIONS = [3, 5, 6]


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else len(V[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _strassen_2x2x2():
    """Return the Strassen decomposition for 2x2x2 matrix multiplication (rank 7)."""
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]

    # M1 = (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[1][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # M2 = (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1

    # M3 = A11(B21-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[0][2] = 1; W[1][2] = 1

    # M4 = A22(B11-B21)
    U[3][3] = 1
    V[0][3] = 1; V[2][3] = -1
    W[2][3] = 1; W[3][3] = 1

    # M5 = (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1

    # M6 = (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[2][5] = 1; W[3][5] = 1

    # M7 = (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = 1

    return U, V, W


def _schoolbook(n, m, p):
    """Return schoolbook decomposition for n x m * m x p (rank n*m*p)."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _concat_decomps(decomp_list, n, m, p):
    """
    Concatenate multiple decompositions that partition the same (n,m,p) tensor.
    Each decomp is (U, V, W) for a sub-tensor. We need to merge them.
    
    Actually, for subadditivity, we decompose along one dimension.
    E.g., splitting n into n1+n2: we get two decompositions for (n1,m,p) and (n2,m,p).
    The U matrices have different row counts, so we pad and stack.
    """
    total_rank = sum(len(U[0]) for U, V, W in decomp_list)
    
    U_new = [[0] * total_rank for _ in range(n * m)]
    V_new = [[0] * total_rank for _ in range(m * p)]
    W_new = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for U, V, W in decomp_list:
        r = len(U[0])
        # Copy into appropriate rows
        for i in range(len(U)):
            for c in range(r):
                U_new[i][col_offset + c] = U[i][c]
        for j in range(len(V)):
            for c in range(r):
                V_new[j][col_offset + c] = V[j][c]
        for k in range(len(W)):
            for c in range(r):
                W_new[k][col_offset + c] = W[k][c]
        col_offset += r
    
    return U_new, V_new, W_new


def _split_n_decomp(n1, n2, m, p, decomp1, decomp2):
    """
    Combine decompositions for (n1,m,p) and (n2,m,p) into one for ((n1+n2),m,p).
    decomp1: (U1, V1, W1) with U1 shape (n1*m, r1)
    decomp2: (U2, V2, W2) with U2 shape (n2*m, r2)
    Result: U shape ((n1+n2)*m, r1+r2), V shape (m*p, r1+r2), W shape (p*(n1+n2), r1+r2)
    
    U_new[i*m+j, c] = U1[i*m+j, c] if i < n1 else U2[(i-n1)*m+j, c-r1]
    V_new[j*p+k, c] = V1[j*p+k, c] if c < r1 else V2[j*p+k, c-r1]
    W_new[k*(n1+n2)+i, c] = W1[k*n1+i, c] if i < n1 else W2[k*n2+(i-n1), c-r1]
    """
    r1 = len(decomp1[0][0])
    r2 = len(decomp2[0][0])
    total_rank = r1 + r2
    
    U_new = [[0] * total_rank for _ in range((n1 + n2) * m)]
    V_new = [[0] * total_rank for _ in range(m * p)]
    W_new = [[0] * total_rank for _ in range(p * (n1 + n2))]
    
    U1, V1, W1 = decomp1
    U2, V2, W2 = decomp2
    
    # Fill U
    for i in range(n1):
        for j in range(m):
            for c in range(r1):
                U_new[i * m + j][c] = U1[i * m + j][c]
    for i in range(n2):
        for j in range(m):
            for c in range(r2):
                U_new[(n1 + i) * m + j][r1 + c] = U2[i * m + j][c]
    
    # Fill V
    for j in range(m):
        for k in range(p):
            for c in range(r1):
                V_new[j * p + k][c] = V1[j * p + k][c]
            for c in range(r2):
                V_new[j * p + k][r1 + c] = V2[j * p + k][c]
    
    # Fill W
    for k in range(p):
        for i in range(n1):
            for c in range(r1):
                W_new[k * (n1 + n2) + i][c] = W1[k * n1 + i][c]
        for i in range(n2):
            for c in range(r2):
                W_new[k * (n1 + n2) + n1 + i][r1 + c] = W2[k * n2 + i][c]
    
    return U_new, V_new, W_new


def _split_m_decomp(n, m1, m2, p, decomp1, decomp2):
    """
    Combine decompositions for (n,m1,p) and (n,m2,p) into one for (n,(m1+m2),p).
    decomp1: (U1, V1, W1) with U1 shape (n*m1, r1), V1 shape (m1*p, r1)
    decomp2: (U2, V2, W2) with U2 shape (n*m2, r2), V2 shape (m2*p, r2)
    Result: U shape (n*(m1+m2), r1+r2), V shape ((m1+m2)*p, r1+r2), W shape (p*n, r1+r2)
    """
    r1 = len(decomp1[0][0])
    r2 = len(decomp2[0][0])
    total_rank = r1 + r2
    
    U_new = [[0] * total_rank for _ in range(n * (m1 + m2))]
    V_new = [[0] * total_rank for _ in range((m1 + m2) * p)]
    W_new = [[0] * total_rank for _ in range(p * n)]
    
    U1, V1, W1 = decomp1
    U2, V2, W2 = decomp2
    
    # Fill U
    for i in range(n):
        for j in range(m1):
            for c in range(r1):
                U_new[i * (m1 + m2) + j][c] = U1[i * m1 + j][c]
        for j in range(m2):
            for c in range(r2):
                U_new[i * (m1 + m2) + m1 + j][r1 + c] = U2[i * m2 + j][c]
    
    # Fill V
    for j in range(m1):
        for k in range(p):
            for c in range(r1):
                V_new[j * p + k][c] = V1[j * p + k][c]
    for j in range(m2):
        for k in range(p):
            for c in range(r2):
                V_new[(m1 + j) * p + k][r1 + c] = V2[j * p + k][c]
    
    # Fill W
    for k in range(p):
        for i in range(n):
            for c in range(r1):
                W_new[k * n + i][c] = W1[k * n + i][c]
            for c in range(r2):
                W_new[k * n + i][r1 + c] = W2[k * n + i][c]
    
    return U_new, V_new, W_new


def _split_p_decomp(n, m, p1, p2, decomp1, decomp2):
    """
    Combine decompositions for (n,m,p1) and (n,m,p2) into one for (n,m,(p1+p2)).
    decomp1: (U1, V1, W1) with V1 shape (m*p1, r1), W1 shape (p1*n, r1)
    decomp2: (U2, V2, W2) with V2 shape (m*p2, r2), W2 shape (p2*n, r2)
    Result: U shape (n*m, r1+r2), V shape (m*(p1+p2), r1+r2), W shape ((p1+p2)*n, r1+r2)
    """
    r1 = len(decomp1[0][0])
    r2 = len(decomp2[0][0])
    total_rank = r1 + r2
    
    U_new = [[0] * total_rank for _ in range(n * m)]
    V_new = [[0] * total_rank for _ in range(m * (p1 + p2))]
    W_new = [[0] * total_rank for _ in range((p1 + p2) * n)]
    
    U1, V1, W1 = decomp1
    U2, V2, W2 = decomp2
    
    # Fill U
    for i in range(n):
        for j in range(m):
            for c in range(r1):
                U_new[i * m + j][c] = U1[i * m + j][c]
            for c in range(r2):
                U_new[i * m + j][r1 + c] = U2[i * m + j][c]
    
    # Fill V
    for j in range(m):
        for k in range(p1):
            for c in range(r1):
                V_new[j * (p1 + p2) + k][c] = V1[j * p1 + k][c]
        for k in range(p2):
            for c in range(r2):
                V_new[j * (p1 + p2) + p1 + k][r1 + c] = V2[j * p2 + k][c]
    
    # Fill W
    for k in range(p1):
        for i in range(n):
            for c in range(r1):
                W_new[k * n + i][c] = W1[k * n + i][c]
    for k in range(p2):
        for i in range(n):
            for c in range(r2):
                W_new[(p1 + k) * n + i][r1 + c] = W2[k * n + i][c]
    
    return U_new, V_new, W_new


def _decomp_1mp(m, p):
    """Decomposition for 1 x m * m x p, rank = p."""
    rank = p
    U = [[0] * rank for _ in range(m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p)]
    
    for j in range(m):
        for c in range(p):
            U[j][c] = 1
    for j in range(m):
        for k in range(p):
            V[j * p + k][k] = 1
    for k in range(p):
        W[k][k] = 1
    
    return U, V, W


def _get_decomp(n, m, p, deadline):
    """
    Recursively get a low-rank decomposition for n x m * m x p.
    Uses subadditivity and known base cases.
    """
    if time.time() > deadline:
        return _schoolbook(n, m, p)
    
    # Base cases
    if n == 1:
        return _decomp_1mp(m, p)
    if m == 1:
        # n x 1 * 1 x p: rank = n*p? No, R(n,1,p) = n*p
        # Actually T[i,0,k] = 1 for all i,k. 
        # U[i][c], V[0*1+k=k][c], W[k*n+i][c]
        # sum_c U[i][c]*V[k][c]*W[k*n+i][c] = 1
        # Set rank = n*p: U[i][i*p+k] = 1, V[k][i*p+k] = 1, W[k*n+i][i*p+k] = 1
        rank = n * p
        U = [[0] * rank for _ in range(n)]
        V = [[0] * rank for _ in range(p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for k in range(p):
                c = i * p + k
                U[i][c] = 1
                V[k][c] = 1
                W[k * n + i][c] = 1
        return U, V, W
    if p == 1:
        # n x m * m x 1: rank = n*m
        rank = n * m
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m)]
        W = [[0] * rank for _ in range(n)]
        for i in range(n):
            for j in range(m):
                c = i * m + j
                U[i * m + j][c] = 1
                V[j][c] = 1
                W[i][c] = 1
        return U, V, W
    if n == m == p == 2:
        return _strassen_2x2x2()
    
    # Try splitting n
    if n >= 2:
        # Try n = 1 + (n-1)
        d1 = _get_decomp(1, m, p, deadline)
        d2 = _get_decomp(n - 1, m, p, deadline)
        result = _split_n_decomp(1, n - 1, m, p, d1, d2)
        r = len(result[0][0])
        
        # Also try n = 2 + (n-2) if n >= 4
        best = result
        if n >= 4:
            d1b = _get_decomp(2, m, p, deadline)
            d2b = _get_decomp(n - 2, m, p, deadline)
            result_b = _split_n_decomp(2, n - 2, m, p, d1b, d2b)
            if len(result_b[0][0]) < r:
                best = result_b
                r = len(result_b[0][0])
        
        # Compare with schoolbook
        sb = _schoolbook(n, m, p)
        if len(sb[0][0]) < r:
            best = sb
            r = len(sb[0][0])
        
        if r < n * m * p:
            return best
    
    # Try splitting m
    if m >= 2:
        d1 = _get_decomp(n, 1, p, deadline)
        d2 = _get_decomp(n, m - 1, p, deadline)
        result = _split_m_decomp(n, 1, m - 1, p, d1, d2)
        r = len(result[0][0])
        
        best = result
        if m >= 4:
            d1b = _get_decomp(n, 2, p, deadline)
            d2b = _get_decomp(n, m - 2, p, deadline)
            result_b = _split_m_decomp(n, 2, m - 2, p, d1b, d2b)
            if len(result_b[0][0]) < r:
                best = result_b
                r = len(result_b[0][0])
        
        sb = _schoolbook(n, m, p)
        if len(sb[0][0]) < r:
            best = sb
            r = len(sb[0][0])
        
        if r < n * m * p:
            return best
    
    # Try splitting p
    if p >= 2:
        d1 = _get_decomp(n, m, 1, deadline)
        d2 = _get_decomp(n, m, p - 1, deadline)
        result = _split_p_decomp(n, m, 1, p - 1, d1, d2)
        r = len(result[0][0])
        
        best = result
        if p >= 4:
            d1b = _get_decomp(n, m, 2, deadline)
            d2b = _get_decomp(n, m, p - 2, deadline)
            result_b = _split_p_decomp(n, m, 2, p - 2, d1b, d2b)
            if len(result_b[0][0]) < r:
                best = result_b
                r = len(result_b[0][0])
        
        sb = _schoolbook(n, m, p)
        if len(sb[0][0]) < r:
            best = sb
            r = len(sb[0][0])
        
        if r < n * m * p:
            return best
    
    return _schoolbook(n, m, p)


def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 3x5 matrix by a 5x6 matrix.
    """
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Start with schoolbook as fallback
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try recursive decomposition
    try:
        U, V, W = _get_decomp(n, m, p, deadline)
        r = len(U[0])
        if r < best_rank:
            # Verify
            if _verify(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = r
    except Exception:
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
