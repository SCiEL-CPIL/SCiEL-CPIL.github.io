"""
Rank-minimal tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
Uses a block-based Strassen approach with rank 42.
"""
import time

DIMENSIONS = [2, 4, 6]
N, M, P = 2, 4, 6
NM = N * M   # 8
MP = M * P   # 24
PN = P * N   # 12


def _build_rank42():
    """
    Build the rank-42 decomposition using block Strassen.
    Split A (2x4) into two 2x2 blocks horizontally.
    Split B (4x6) into two 2x6 blocks vertically, then each into three 2x2 blocks.
    This gives 6 2x2x2 sub-problems, each solved with Strassen (rank 7).
    Total rank = 6 * 7 = 42.
    """
    rank = 42
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    # Define the 6 blocks: (a_row_range, a_col_range, b_row_range, b_col_range)
    # A is 2x4: rows 0-1, cols 0-3
    # B is 4x6: rows 0-3, cols 0-5
    # C is 2x6: rows 0-1, cols 0-5
    #
    # Block decomposition:
    # A = [A1 | A2] where A1 is 2x2 (cols 0-1), A2 is 2x2 (cols 2-3)
    # B = [B1; B2] where B1 is 2x6 (rows 0-1), B2 is 2x6 (rows 2-3)
    # B1 = [B11 B12 B13], B2 = [B21 B22 B23] where each Bij is 2x2
    #
    # C = A*B = A1*B1 + A2*B2
    # C1 (cols 0-1) = A1*B11 + A2*B21
    # C2 (cols 2-3) = A1*B12 + A2*B22
    # C3 (cols 4-5) = A1*B13 + A2*B23
    
    blocks = [
        # (a_rows, a_cols, b_rows, b_cols)
        ([0, 1], [0, 1], [0, 1], [0, 1]),   # A1*B11 -> C1
        ([0, 1], [0, 1], [0, 1], [2, 3]),   # A1*B12 -> C2
        ([0, 1], [0, 1], [0, 1], [4, 5]),   # A1*B13 -> C3
        ([0, 1], [2, 3], [2, 3], [0, 1]),   # A2*B21 -> C1
        ([0, 1], [2, 3], [2, 3], [2, 3]),   # A2*B22 -> C2
        ([0, 1], [2, 3], [2, 3], [4, 5]),   # A2*B23 -> C3
    ]
    
    # Strassen 2x2x2 decomposition (7 terms)
    # For matrices A (2x2), B (2x2), C = A*B (2x2)
    # Local indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    #                B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    #                C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    strassen = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    col = 0
    for block in blocks:
        a_rows, a_cols, b_rows, b_cols = block
        
        # Map local 2x2 indices to global indices
        a_map = {
            0: a_rows[0] * M + a_cols[0],
            1: a_rows[0] * M + a_cols[1],
            2: a_rows[1] * M + a_cols[0],
            3: a_rows[1] * M + a_cols[1]
        }
        b_map = {
            0: b_rows[0] * P + b_cols[0],
            1: b_rows[0] * P + b_cols[1],
            2: b_rows[1] * P + b_cols[0],
            3: b_rows[1] * P + b_cols[1]
        }
        # W index: k*N + i where k is col, i is row
        w_map = {
            0: b_cols[0] * N + a_rows[0],
            1: b_cols[1] * N + a_rows[0],
            2: b_cols[0] * N + a_rows[1],
            3: b_cols[1] * N + a_rows[1]
        }
        
        for u_coeffs, v_coeffs, w_coeffs in strassen:
            for local_idx, coeff in u_coeffs:
                U[a_map[local_idx]][col] += coeff
            for local_idx, coeff in v_coeffs:
                V[b_map[local_idx]][col] += coeff
            for local_idx, coeff in w_coeffs:
                W[w_map[local_idx]][col] += coeff
            col += 1
    
    return U, V, W


def _verify(U, V, W, tol=1e-9):
    """Verify the decomposition is exact."""
    rank = len(U[0])
    for u_idx in range(NM):
        for v_idx in range(MP):
            for w_idx in range(PN):
                val = 0
                for r in range(rank):
                    val += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                i = u_idx // M
                j = u_idx % M
                j2 = v_idx // P
                k = v_idx % P
                k2 = w_idx // N
                i2 = w_idx % N
                if i == i2 and j == j2 and k == k2:
                    if abs(val - 1) > tol:
                        return False
                else:
                    if abs(val) > tol:
                        return False
    return True


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x6 matrix.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Build the rank-42 block-Strassen decomposition
    U, V, W = _build_rank42()
    
    # Verify correctness
    if not _verify(U, V, W):
        # Fallback to schoolbook rank 48
        rank = N * M * P
        U = [[0] * rank for _ in range(NM)]
        V = [[0] * rank for _ in range(MP)]
        W = [[0] * rank for _ in range(PN)]
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j][col] = 1
                    V[j * P + k][col] = 1
                    W[k * N + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
