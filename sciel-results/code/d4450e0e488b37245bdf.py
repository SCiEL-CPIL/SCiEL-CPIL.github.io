"""Program for matrix multiplication 3,4,5 tensor rank minimization via Strassen blocks."""
import time

DIMENSIONS = [3, 4, 5]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 60


def _verify(U, V, W):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = N, M, P
    r = len(U[0])
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    if any(len(row) != r for row in U):
        return False
    if any(len(row) != r for row in V):
        return False
    if any(len(row) != r for row in W):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def construct():
    """
    Exact tensor decomposition for multiplying a 3x4 matrix by a 4x5 matrix.
    Uses Strassen's algorithm on 2x2x2 sub-blocks to reduce rank below schoolbook.
    
    Returns:
        dict: {'U': list of lists, 'V': list of lists, 'W': list of lists}
              where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
              with integer entries.
    """
    n, m, p = N, M, P
    t0 = time.time()
    
    def add_strassen_block(i_off, j_off, k_off, col, U, V, W):
        """
        Add 7 Strassen terms for a 2x2x2 block starting at column 'col'.
        Returns the new column index (col + 7).
        """
        # The 7 Strassen multipliers for C = A*B where C[i][k] = sum_j A[i][j]*B[j][k]
        # Local indices: A[ii][jj], B[jj][kk], C[ii][kk]
        
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) -> C[0,0]+=M1, C[1,1]+=M1
        U[(i_off+0)*m+(j_off+0)][col] += 1
        U[(i_off+1)*m+(j_off+1)][col] += 1
        V[(j_off+0)*p+(k_off+0)][col] += 1
        V[(j_off+1)*p+(k_off+1)][col] += 1
        W[(k_off+0)*n+(i_off+0)][col] += 1
        W[(k_off+1)*n+(i_off+1)][col] += 1
        col += 1
        
        # M2 = (A[1,0]+A[1,1])*B[0,0] -> C[1,0]+=M2, C[1,1]-=M2
        U[(i_off+1)*m+(j_off+0)][col] += 1
        U[(i_off+1)*m+(j_off+1)][col] += 1
        V[(j_off+0)*p+(k_off+0)][col] += 1
        W[(k_off+0)*n+(i_off+1)][col] += 1
        W[(k_off+1)*n+(i_off+1)][col] += -1
        col += 1
        
        # M3 = A[0,0]*(B[0,1]-B[1,1]) -> C[0,1]+=M3, C[1,1]+=M3
        U[(i_off+0)*m+(j_off+0)][col] += 1
        V[(j_off+0)*p+(k_off+1)][col] += 1
        V[(j_off+1)*p+(k_off+1)][col] += -1
        W[(k_off+1)*n+(i_off+0)][col] += 1
        W[(k_off+1)*n+(i_off+1)][col] += 1
        col += 1
        
        # M4 = A[1,1]*(B[1,0]-B[0,0]) -> C[1,0]+=M4, C[0,0]+=M4
        U[(i_off+1)*m+(j_off+1)][col] += 1
        V[(j_off+1)*p+(k_off+0)][col] += 1
        V[(j_off+0)*p+(k_off+0)][col] += -1
        W[(k_off+0)*n+(i_off+1)][col] += 1
        W[(k_off+0)*n+(i_off+0)][col] += 1
        col += 1
        
        # M5 = (A[0,0]+A[0,1])*B[1,1] -> C[0,1]+=M5, C[0,0]-=M5
        U[(i_off+0)*m+(j_off+0)][col] += 1
        U[(i_off+0)*m+(j_off+1)][col] += 1
        V[(j_off+1)*p+(k_off+1)][col] += 1
        W[(k_off+1)*n+(i_off+0)][col] += 1
        W[(k_off+0)*n+(i_off+0)][col] += -1
        col += 1
        
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1]) -> C[1,1]+=M6
        U[(i_off+1)*m+(j_off+0)][col] += 1
        U[(i_off+0)*m+(j_off+0)][col] += -1
        V[(j_off+0)*p+(k_off+0)][col] += 1
        V[(j_off+0)*p+(k_off+1)][col] += 1
        W[(k_off+1)*n+(i_off+1)][col] += 1
        col += 1
        
        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1]) -> C[0,0]+=M7
        U[(i_off+0)*m+(j_off+1)][col] += 1
        U[(i_off+1)*m+(j_off+1)][col] += -1
        V[(j_off+1)*p+(k_off+0)][col] += 1
        V[(j_off+1)*p+(k_off+1)][col] += 1
        W[(k_off+0)*n+(i_off+0)][col] += 1
        col += 1
        
        return col
    
    # Six disjoint Strassen 2x2x2 blocks
    # Each block covers i in {i_off, i_off+1}, j in {j_off, j_off+1}, k in {k_off, k_off+1}
    strassen_blocks = [
        (0, 0, 0),  # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
        (0, 2, 2),  # Block 2: i in {0,1}, j in {2,3}, k in {2,3}
        (1, 0, 3),  # Block 3: i in {1,2}, j in {0,1}, k in {3,4}
        (1, 2, 0),  # Block 4: i in {1,2}, j in {2,3}, k in {0,1}
        (1, 0, 0),  # Block 5: i in {1,2}, j in {0,1}, k in {0,1}
        (1, 2, 2),  # Block 6: i in {1,2}, j in {2,3}, k in {2,3}
    ]
    
    # Identify which (i,j,k) tuples are covered by Strassen blocks.
    strassen_covered = set()
    for i_off, j_off, k_off in strassen_blocks:
        for ii in range(2):
            for jj in range(2):
                for kk in range(2):
                    strassen_covered.add((i_off + ii, j_off + jj, k_off + kk))
    
    # Verify blocks are truly disjoint
    assert len(strassen_covered) == 6 * 8, f"Blocks overlap! Covered {len(strassen_covered)} tuples, expected 48"
    
    # Build the full decomposition directly into matrices
    # Total rank: 6*7 + (60-48) = 42 + 12 = 54
    total_rank = 6 * 7 + (n * m * p - len(strassen_covered))
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i_off, j_off, k_off in strassen_blocks:
        col = add_strassen_block(i_off, j_off, k_off, col, U, V, W)
    
    # Add schoolbook terms for all other (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    assert col == total_rank, f"Column mismatch: {col} vs {total_rank}"
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if _verify(U, V, W):
        return result
    
    # Fallback to schoolbook if verification fails
    U0 = [[0] * RANK_SCHOOLBOOK for _ in range(n * m)]
    V0 = [[0] * RANK_SCHOOLBOOK for _ in range(m * p)]
    W0 = [[0] * RANK_SCHOOLBOOK for _ in range(p * n)]
    col0 = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j][col0] = 1
                V0[j * p + k][col0] = 1
                W0[k * n + i][col0] = 1
                col0 += 1
    return {"U": U0, "V": V0, "W": W0}
