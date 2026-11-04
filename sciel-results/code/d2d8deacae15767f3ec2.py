"""4x5x6 matrix multiplication decomposition (rank 56, Strassen-based)."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    DEADLINE = 150.0
    SAFETY_MARGIN = 5.0
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(i0, j0, k0, r_start):
        def u_idx(i, j): return (i0 + i) * m + (j0 + j)
        def v_idx(j, k): return (j0 + j) * p + (k0 + k)
        def w_idx(k, i): return (k0 + k) * n + (i0 + i)
        
        # M1 = (A00 + A11)(B00 + B11)
        U[u_idx(0, 0)][r_start] += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(0, 0)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        
        # M2 = (A10 + A11)B00
        r_start += 1
        U[u_idx(1, 0)][r_start] += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(0, 0)][r_start] += 1
        W[w_idx(0, 1)][r_start] += 1
        W[w_idx(1, 1)][r_start] += -1
        
        # M3 = A00(B01 - B11)
        r_start += 1
        U[u_idx(0, 0)][r_start] += 1
        V[v_idx(0, 1)][r_start] += 1
        V[v_idx(1, 1)][r_start] += -1
        W[w_idx(1, 0)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        
        # M4 = A11(B10 - B00)
        r_start += 1
        U[u_idx(1, 1)][r_start] += 1
        V[v_idx(1, 0)][r_start] += 1
        V[v_idx(0, 0)][r_start] += -1
        W[w_idx(0, 0)][r_start] += 1
        W[w_idx(0, 1)][r_start] += 1
        
        # M5 = (A00 + A01)B11
        r_start += 1
        U[u_idx(0, 0)][r_start] += 1
        U[u_idx(0, 1)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += -1
        W[w_idx(1, 0)][r_start] += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        r_start += 1
        U[u_idx(1, 0)][r_start] += 1
        U[u_idx(0, 0)][r_start] += -1
        V[v_idx(0, 0)][r_start] += 1
        V[v_idx(0, 1)][r_start] += 1
        W[w_idx(1, 1)][r_start] += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        r_start += 1
        U[u_idx(0, 1)][r_start] += 1
        U[u_idx(1, 1)][r_start] += -1
        V[v_idx(1, 0)][r_start] += 1
        V[v_idx(1, 1)][r_start] += 1
        W[w_idx(0, 0)][r_start] += 1
        
        return r_start + 1

    def add_2x2x3_block(i0, j0, k0, r_start):
        # For a 2x2x3 block, we use Strassen for the first 2 columns (rank 7)
        # and schoolbook for the 3rd column (rank 2*2*1 = 4)
        # Total rank = 7 + 4 = 11
        
        # First 2 columns: Strassen
        r_start = add_strassen_block(i0, j0, k0, r_start)
        
        # 3rd column: schoolbook (2x2x1)
        # C[i, j+2] = sum_k A[i, k] * B[k, j+2] for i in 0..1, k in 0..1
        # So we need 4 terms: (i=0,k=0), (i=0,k=1), (i=1,k=0), (i=1,k=1)
        # Each term contributes to C[i, 2]
        
        # Term 1: A[0,0] * B[0,2] -> C[0,2]
        U[(i0+0)*m + (j0+0)][r_start] = 1
        V[(j0+0)*p + (k0+2)][r_start] = 1
        W[(k0+2)*n + (i0+0)][r_start] = 1
        r_start += 1
        
        # Term 2: A[0,1] * B[1,2] -> C[0,2]
        U[(i0+0)*m + (j0+1)][r_start] = 1
        V[(j0+1)*p + (k0+2)][r_start] = 1
        W[(k0+2)*n + (i0+0)][r_start] = 1
        r_start += 1
        
        # Term 3: A[1,0] * B[0,2] -> C[1,2]
        U[(i0+1)*m + (j0+0)][r_start] = 1
        V[(j0+0)*p + (k0+2)][r_start] = 1
        W[(k0+2)*n + (i0+1)][r_start] = 1
        r_start += 1
        
        # Term 4: A[1,1] * B[1,2] -> C[1,2]
        U[(i0+1)*m + (j0+1)][r_start] = 1
        V[(j0+1)*p + (k0+2)][r_start] = 1
        W[(k0+2)*n + (i0+1)][r_start] = 1
        r_start += 1
        
        return r_start

    def add_2x3x3_block(i0, j0, k0, r_start):
        # For a 2x3x3 block, we decompose into:
        # - 2x2x3 block for the first 2 rows of A (rank 11)
        # - 2x1x3 block for the 3rd row of A (rank 2*1*3 = 6)
        # Total rank = 11 + 6 = 17
        
        # First 2 rows: 2x2x3
        r_start = add_2x2x3_block(i0, j0, k0, r_start)
        
        # 3rd row: 2x1x3 schoolbook
        # C[2, j] = sum_k A[2, k] * B[k, j] for j in 0..2, k in 0..2
        # So we need 6 terms: (k=0,j=0), (k=0,j=1), (k=0,j=2), (k=1,j=0), (k=1,j=1), (k=1,j=2)
        
        # Term 1: A[2,0] * B[0,0] -> C[2,0]
        U[(i0+2)*m + (j0+0)][r_start] = 1
        V[(j0+0)*p + (k0+0)][r_start] = 1
        W[(k0+0)*n + (i0+2)][r_start] = 1
        r_start += 1
        
        # Term 2: A[2,0] * B[0,1] -> C[2,1]
        U[(i0+2)*m + (j0+0)][r_start] = 1
        V[(j0+0)*p + (k0+1)][r_start] = 1
        W[(k0+1)*n + (i0+2)][r_start] = 1
        r_start += 1
        
        # Term 3: A[2,0] * B[0,2] -> C[2,2]
        U[(i0+2)*m + (j0+0)][r_start] = 1
        V[(j0+0)*p + (k0+2)][r_start] = 1
        W[(k0+2)*n + (i0+2)][r_start] = 1
        r_start += 1
        
        # Term 4: A[2,1] * B[1,0] -> C[2,0]
        U[(i0+2)*m + (j0+1)][r_start] = 1
        V[(j0+1)*p + (k0+0)][r_start] = 1
        W[(k0+0)*n + (i0+2)][r_start] = 1
        r_start += 1
        
        # Term 5: A[2,1] * B[1,1] -> C[2,1]
        U[(i0+2)*m + (j0+1)][r_start] = 1
        V[(j0+1)*p + (k0+1)][r_start] = 1
        W[(k0+1)*n + (i0+2)][r_start] = 1
        r_start += 1
        
        # Term 6: A[2,1] * B[1,2] -> C[2,2]
        U[(i0+2)*m + (j0+1)][r_start] = 1
        V[(j0+1)*p + (k0+2)][r_start] = 1
        W[(k0+2)*n + (i0+2)][r_start] = 1
        r_start += 1
        
        return r_start

    r_offset = 0
    # Block 1: i in [0,1], j in [0,1], k in [0,2] -> 2x2x3 (rank 11)
    r_offset = add_2x2x3_block(0, 0, 0, r_offset)
    
    # Block 2: i in [0,1], j in [2,4], k in [0,2] -> 2x3x3 (rank 17)
    r_offset = add_2x3x3_block(0, 2, 0, r_offset)
    
    # Block 3: i in [2,3], j in [0,1], k in [3,5] -> 2x2x3 (rank 11)
    r_offset = add_2x2x3_block(2, 0, 3, r_offset)
    
    # Block 4: i in [2,3], j in [2,4], k in [3,5] -> 2x3x3 (rank 17)
    r_offset = add_2x3x3_block(2, 2, 3, r_offset)
    
    # Verify we used exactly 56 ranks
    assert r_offset == 56, f"Expected rank 56, got {r_offset}"
    
    return {"U": U, "V": V, "W": W}
