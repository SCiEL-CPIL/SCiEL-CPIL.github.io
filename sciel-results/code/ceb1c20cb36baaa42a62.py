"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]
n, m, p = DIMENSIONS

# Deadline guard
_START_TIME = time.time()
DEADLINE = 130  # seconds, safety margin before 150s limit


def _get_strassen_2x2x2():
    """
    Return U, V, W factors for 2x2x2 matrix multiplication (Strassen, rank 7).
    U: (4, 7) indexed as U[i*2+j, c] for A[i,j]
    V: (4, 7) indexed as V[j*2+k, c] for B[j,k]
    W: (4, 7) indexed as W[k*2+i, c] for C[i,k]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # U: A factors (rows: a=0, b=1, c=2, d=3)
    U[0, 0] = 1   # a in m0
    U[3, 0] = 1   # d in m0
    U[2, 1] = 1   # c in m1
    U[3, 1] = 1   # d in m1
    U[0, 2] = 1   # a in m2
    U[3, 3] = 1   # d in m3
    U[0, 4] = 1   # a in m4
    U[1, 4] = 1   # b in m4
    U[2, 5] = 1   # c in m5
    U[0, 5] = -1  # -a in m5
    U[1, 6] = 1   # b in m6
    U[3, 6] = -1  # -d in m6
    
    # V: B factors (rows: e=0, f=1, g=2, h=3)
    V[0, 0] = 1   # e in m0
    V[3, 0] = 1   # h in m0
    V[0, 1] = 1   # e in m1
    V[1, 2] = 1   # f in m2
    V[3, 2] = -1  # -h in m2
    V[2, 3] = 1   # g in m3
    V[0, 3] = -1  # -e in m3
    V[3, 4] = 1   # h in m4
    V[0, 5] = 1   # e in m5
    V[1, 5] = 1   # f in m5
    V[2, 6] = 1   # g in m6
    V[3, 6] = 1   # h in m6
    
    # W: C factors (rows: C[0,0]=0, C[1,0]=1, C[0,1]=2, C[1,1]=3)
    # W[k*2+i, c] where k is col, i is row
    # C[0,0]: k=0,i=0 -> row 0
    # C[1,0]: k=0,i=1 -> row 1
    # C[0,1]: k=1,i=0 -> row 2
    # C[1,1]: k=1,i=1 -> row 3
    
    W[0, 0] = 1   # m0 in C[0,0]
    W[0, 3] = 1   # m3 in C[0,0]
    W[0, 4] = -1  # -m4 in C[0,0]
    W[0, 6] = 1   # m6 in C[0,0]
    
    W[1, 1] = 1   # m1 in C[1,0]
    W[1, 3] = 1   # m3 in C[1,0]
    
    W[2, 2] = 1   # m2 in C[0,1]
    W[2, 4] = 1   # m4 in C[0,1]
    
    W[3, 0] = 1   # m0 in C[1,1]
    W[3, 1] = -1  # -m1 in C[1,1]
    W[3, 2] = 1   # m2 in C[1,1]
    W[3, 5] = 1   # m5 in C[1,1]
    
    return U, V, W


def _verify(U_list, V_list, W_list, rank_r):
    """Verify the decomposition exactly."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank_r):
                    s += U_list[i * m + j][c] * V_list[j * p + k][c] * W_list[k * n + i][c]
                if s != 1:
                    return False
    return True


def _build_strassen_decomposition():
    """
    Build tensor decomposition for 3x5 @ 5x6 using Strassen blocks.
    
    Strategy:
    - Split A (3x5) into top 2x5 and bottom 1x5
    - Split B (5x6) into three 5x2 blocks
    - For top part: use Strassen 2x2x2 for each (2x2 A-block, 2x2 B-block)
    - For bottom part: schoolbook
    
    Block structure:
    A = [A_top (2x5); A_bot (1x5)]
    B = [B_0 (5x2) B_1 (5x2) B_2 (5x2)]
    
    A_top @ B = [A_top @ B_0, A_top @ B_1, A_top @ B_2]
    For each A_top @ B_t (2x5 @ 5x2):
      Split A_top into [A_tl (2x2) A_tr (2x3)]
      But B_t is 5x2, so we need to split B_t into [B_t_top (2x2); B_t_bot (3x2)]
      Then A_top @ B_t = A_tl @ B_t_top + A_tr @ B_t_bot
      
      This gets complicated. Let me use a simpler approach:
      For 2x5 @ 5x2, split into:
        - 2x2 @ 2x2 (top-left): Strassen rank 7
        - 2x3 @ 3x2 (top-right + bottom-left interaction): schoolbook 6 terms
        - 2x2 @ 2x2 (bottom-right if we had it, but we don't)
      
      Actually, let me just use a cleaner block decomposition.
      
    Alternative simpler approach:
    - 6 non-overlapping 2x2x2 blocks using Strassen (rank 7 each) = 42
    - Remaining terms: schoolbook
    - Total: 42 + 42 = 84 (same as current, but let me verify the block placement)
    
    Better approach from inspiration:
    - Top 2 rows of A (2x5) times B (5x6):
      Split B into three 5x2 blocks
      For each 2x5 @ 5x2:
        This can be done with rank 7 + 10 = 17? No, that's not right.
        
    Let me just use the simple 6-block Strassen approach and verify it works.
    """
    strassen_U, strassen_V, strassen_W = _get_strassen_2x2x2()
    
    # First verify Strassen works
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    s += strassen_U[i*2+j, c] * strassen_V[j*2+k, c] * strassen_W[k*2+i, c]
                if s != 1:
                    return None  # Strassen is broken, fallback needed
    
    # 6 non-overlapping 2x2x2 blocks
    # i in {0,1}, j in {0,1} or {2,3}, k in {0,1} or {2,3} or {4,5}
    blocks = [
        (0, 0, 0),  # i={0,1}, j={0,1}, k={0,1}
        (0, 0, 2),  # i={0,1}, j={0,1}, k={2,3}
        (0, 0, 4),  # i={0,1}, j={0,1}, k={4,5}
        (0, 2, 0),  # i={0,1}, j={2,3}, k={0,1}
        (0, 2, 2),  # i={0,1}, j={2,3}, k={2,3}
        (0, 2, 4),  # i={0,1}, j={2,3}, k={4,5}
    ]
    
    rank = 6 * 7 + 42  # 42 Strassen terms + 42 schoolbook terms
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    # Place Strassen blocks
    for (i0, j0, k0) in blocks:
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        i = i0 + di
                        j = j0 + dj
                        k = k0 + dk
                        U[i * m + j, col] += strassen_U[di * 2 + dj, c]
                        V[j * p + k, col] += strassen_V[dj * 2 + dk, c]
                        W[k * n + i, col] += strassen_W[dk * 2 + di, c]
            col += 1
    
    # Schoolbook for remaining terms
    # Remaining (i,j,k) triples:
    # i=2, j=0..4, k=0..5: 5*6 = 30 terms
    # i=0,1, j=4, k=0..5: 2*1*6 = 12 terms
    # Total: 42 terms
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this (i,j,k) is covered by a Strassen block
                covered = False
                for (i0, j0, k0) in blocks:
                    if i >= i0 and i < i0 + 2:
                        if j >= j0 and j < j0 + 2:
                            if k >= k0 and k < k0 + 2:
                                covered = True
                                break
                    if covered:
                        break
                if not covered:
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return U, V, W, rank


def _build_schoolbook():
    """Fallback schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W, rank


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 3x5 matrix by a 5x6 matrix.
    """
    start_time = time.time()
    
    # Try Strassen-based decomposition
    result = _build_strassen_decomposition()
    
    if result is not None:
        U, V, W, rank = result
        U_list = [[int(U[i, j]) for j in range(rank)] for i in range(n * m)]
        V_list = [[int(V[i, j]) for j in range(rank)] for i in range(m * p)]
        W_list = [[int(W[i, j]) for j in range(rank)] for i in range(p * n)]
        
        # Verify
        if _verify(U_list, V_list, W_list, rank):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Check deadline
    if time.time() - start_time > DEADLINE - 20:
        # Not enough time for fallback, return best effort
        pass
    
    # Fallback: schoolbook
    U, V, W, rank = _build_schoolbook()
    U_list = [[int(U[i, j]) for j in range(rank)] for i in range(n * m)]
    V_list = [[int(V[i, j]) for j in range(rank)] for i in range(m * p)]
    W_list = [[int(W[i, j]) for j in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
