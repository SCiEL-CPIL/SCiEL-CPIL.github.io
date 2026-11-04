import time
from functools import lru_cache
import numpy as np

DIMENSIONS = [4, 5, 5]
_start_time = time.time()
TIME_LIMIT = 145  # Safety margin

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
        
    # Specific optimization for 4x5x5 using block decomposition
    # Split M=4 into 2,2; N=5 into 2,3; P=5 into 2,3
    if m == 4 and n == 5 and p == 5:
        # We build the decomposition by combining sub-decompositions
        # Sub-problems:
        # 1. (2, 2, 2) -> Strassen (Rank 7)
        # 2. (2, 2, 3) -> Schoolbook (Rank 12)
        # 3. (2, 3, 2) -> Schoolbook (Rank 12)
        # 4. (2, 3, 3) -> Schoolbook (Rank 18)
        # Total Rank for one 2x5x5 block: 7 + 12 + 12 + 18 = 49
        # Two blocks (M split 2+2): Total Rank 98
        
        def get_sub_decomp(sm, sn, sp):
            return _decompose(sm, sn, sp)
            
        # Get sub-decompositions
        U1, V1, W1 = get_sub_decomp(2, 2, 2)
        U2, V2, W2 = get_sub_decomp(2, 2, 3)
        U3, V3, W3 = get_sub_decomp(2, 3, 2)
        U4, V4, W4 = get_sub_decomp(2, 3, 3)
        
        # We need to construct the full U, V, W for 4x5x5
        # The structure is:
        # C11 (2x2) = A11(2x2) * B11(2x2)  -> Uses U1, V1, W1
        # C12 (2x3) = A11(2x2) * B12(2x3)  -> Uses U2, V2, W2
        # C21 (2x2) = A21(2x3) * B21(3x2)  -> Uses U3, V3, W3
        # C22 (2x3) = A21(2x3) * B22(3x3)  -> Uses U4, V4, W4
        
        # Note: The indices in U, V, W correspond to the global matrices
        # U has rows for A elements (m*n = 20)
        # V has rows for B elements (n*p = 25)
        # W has rows for C elements (p*m = 20)
        
        total_rank = len(U1) + len(U2) + len(U3) + len(U4)
        # We have two sets of these (for M1 and M2)
        # M1 is rows 0-1 of A, M2 is rows 2-3 of A
        # N1 is cols 0-1 of A, N2 is cols 2-4 of A
        # P1 is cols 0-1 of B, P2 is cols 2-4 of B
        
        U_full = []
        V_full = []
        W_full = []
        
        def add_block(U_s, V_s, W_s, sm, sn, sp, offset_m, offset_n, offset_p):
            for r in range(len(U_s)):
                u_row = [0] * (m * n)
                v_row = [0] * (n * p)
                w_row = [0] * (p * m)
                
                # Map U_s indices (sm*sn) to global A indices (m*n)
                for ii in range(sm):
                    for jj in range(sn):
                        # Local index in U_s: ii*sn + jj
                        # Global index in U: (offset_m + ii)*n + (offset_n + jj)
                        u_val = U_s[r][ii * sn + jj]
                        if u_val != 0:
                            u_row[(offset_m + ii) * n + (offset_n + jj)] = u_val
                            
                # Map V_s indices (sn*sp) to global B indices (n*p)
                for jj in range(sn):
                    for kk in range(sp):
                        # Local index in V_s: jj*sp + kk
                        # Global index in V: (offset_n + jj)*p + (offset_p + kk)
                        v_val = V_s[r][jj * sp + kk]
                        if v_val != 0:
                            v_row[(offset_n + jj) * p + (offset_p + kk)] = v_val
                            
                # Map W_s indices (sp*sm) to global C indices (p*m)
                for kk in range(sp):
                    for ii in range(sm):
                        # Local index in W_s: kk*sm + ii
                        # Global index in W: (offset_p + kk)*m + (offset_m + ii)
                        w_val = W_s[r][kk * sm + ii]
                        if w_val != 0:
                            w_row[(offset_p + kk) * m + (offset_m + ii)] = w_val
                            
                U_full.append(u_row)
                V_full.append(v_row)
                W_full.append(w_row)
                
        # Block 1: A11(2x2) * B11(2x2) -> C11(2x2)
        # M offset 0, N offset 0, P offset 0
        add_block(U1, V1, W1, 2, 2, 2, 0, 0, 0)
        
        # Block 2: A11(2x2) * B12(2x3) -> C12(2x3)
        # M offset 0, N offset 0, P offset 2
        add_block(U2, V2, W2, 2, 2, 3, 0, 0, 2)
        
        # Block 3: A21(2x3) * B21(3x2) -> C21(2x2)
        # M offset 0, N offset 2, P offset 0
        add_block(U3, V3, W3, 2, 3, 2, 0, 2, 0)
        
        # Block 4: A21(2x3) * B22(3x3) -> C22(2x3)
        # M offset 0, N offset 2, P offset 2
        add_block(U4, V4, W4, 2, 3, 3, 0, 2, 2)
        
        # Now for the second half of M (rows 2-3)
        # Block 5: A21(2x2) * B11(2x2) -> C21(2x2)
        # M offset 2, N offset 0, P offset 0
        add_block(U1, V1, W1, 2, 2, 2, 2, 0, 0)
        
        # Block 6: A21(2x2) * B12(2x3) -> C22(2x3)
        # M offset 2, N offset 0, P offset 2
        add_block(U2, V2, W2, 2, 2, 3, 2, 0, 2)
        
        # Block 7: A22(2x3) * B21(3x2) -> C23(2x2) -- Wait, C indices are (P, M)
        # C is 5x4.
        # C rows 0-1 (P1) come from B1 (N1, N2)
        # C rows 2-4 (P2) come from B2 (N1, N2)
        
        # Let's re-verify the C indexing.
        # W has shape (p*m, rank). Index is k*m + i.
        # k is row of C (0..p-1), i is col of C (0..m-1).
        
        # For M offset 2 (i=2,3 in C):
        # Block 7: A22(2x3) * B21(3x2) -> C_part(2x2) where C rows are P1(0-1), C cols are M2(2-3)
        # So k is 0-1, i is 2-3.
        add_block(U3, V3, W3, 2, 3, 2, 2, 2, 0)
        
        # Block 8: A22(2x3) * B22(3x3) -> C_part(2x3) where C rows are P2(2-4), C cols are M2(2-3)
        # So k is 2-4, i is 2-3.
        add_block(U4, V4, W4, 2, 3, 3, 2, 2, 2)
        
        return (U_full, V_full, W_full)

    # Fallback to schoolbook for other dimensions
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
    # Note: The function signature in _decompose is (m, n, p)
    # But DIMENSIONS is [4, 5, 5] which is m, n, p?
    # Task: "multiplying a 4 by 5 matrix by a 5 by 5 matrix"
    # A is 4x5, B is 5x5.
    # So m=4, n=5, p=5.
    # The variable names in construct are n, m, p = DIMENSIONS.
    # So n=4, m=5, p=5.
    # But _decompose takes (m, n, p).
    # So we should call _decompose(4, 5, 5).
    # Let's check the variable assignment in construct.
    # n, m, p = DIMENSIONS -> n=4, m=5, p=5.
    # _decompose(n, m, p) -> _decompose(4, 5, 5).
    # This matches m=4, n=5, p=5.
    U, V, W = _decompose(n, m, p)
    return {"U": U, "V": V, "W": W}
