import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 120

    n, m, p = DIMENSIONS

    # Rank 50 decomposition
    # Structure:
    # Two independent 2x2x7 blocks (one for j in {0,1}, one for j in {2,3}).
    # Each 2x2x7 block is decomposed into three 2x2x2 Strassen blocks (rank 7 each)
    # and one 2x2x1 Schoolbook block (rank 4).
    # Total rank per 2x2x7 block: 3*7 + 4 = 25.
    # Total rank: 2 * 25 = 50.
    
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0

    def emit_strassen(i0, j0, k0):
        """
        Emits a 2x2x2 Strassen block.
        A rows i0, i0+1; cols j0, j0+1.
        B rows j0, j0+1; cols k0, k0+1.
        C rows i0, i0+1; cols k0, k0+1.
        """
        nonlocal col
        
        # M1 = (A11 + A22) * (B11 + B22)
        U[i0*m + j0, col] = 1.0
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[j0*p + k0, col] = 1.0
        V[(j0+1)*p + (k0+1), col] = 1.0
        W[k0*n + i0, col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        
        # M2 = (A21 + A22) * B11
        U[(i0+1)*m + j0, col] = 1.0
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[j0*p + k0, col] = 1.0
        W[k0*n + (i0+1), col] = 1.0
        W[(k0+1)*n + (i0+1), col] = -1.0
        col += 1
        
        # M3 = A11 * (B12 - B22)
        U[i0*m + j0, col] = 1.0
        V[j0*p + (k0+1), col] = 1.0
        V[(j0+1)*p + (k0+1), col] = -1.0
        W[(k0+1)*n + i0, col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        
        # M4 = A22 * (B21 - B11)
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[(j0+1)*p + k0, col] = 1.0
        V[j0*p + k0, col] = -1.0
        W[k0*n + (i0+1), col] = 1.0
        W[k0*n + i0, col] = 1.0
        col += 1
        
        # M5 = (A11 + A12) * B22
        U[i0*m + j0, col] = 1.0
        U[i0*m + (j0+1), col] = 1.0
        V[(j0+1)*p + (k0+1), col] = 1.0
        W[k0*n + i0, col] = -1.0
        W[(k0+1)*n + i0, col] = 1.0
        col += 1
        
        # M6 = (A21 - A11) * (B11 + B12)
        U[(i0+1)*m + j0, col] = 1.0
        U[i0*m + j0, col] = -1.0
        V[j0*p + k0, col] = 1.0
        V[j0*p + (k0+1), col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        
        # M7 = (A12 - A22) * (B21 + B22)
        U[i0*m + (j0+1), col] = 1.0
        U[(i0+1)*m + (j0+1), col] = -1.0
        V[(j0+1)*p + k0, col] = 1.0
        V[(j0+1)*p + (k0+1), col] = 1.0
        W[k0*n + i0, col] = 1.0
        col += 1

    def emit_schoolbook_2x2x1(j0, k0):
        """
        Emits a 2x2x1 schoolbook block.
        A rows 0, 1; cols j0, j0+1.
        B rows j0, j0+1; col k0.
        C rows 0, 1; col k0.
        """
        nonlocal col
        i0 = 0
        
        # T1: A11 * B11 -> C11
        U[i0*m + j0, col] = 1.0
        V[j0*p + k0, col] = 1.0
        W[k0*n + i0, col] = 1.0
        col += 1
        
        # T2: A12 * B21 -> C11
        U[i0*m + (j0+1), col] = 1.0
        V[(j0+1)*p + k0, col] = 1.0
        W[k0*n + i0, col] = 1.0
        col += 1
        
        # T3: A21 * B11 -> C21
        U[(i0+1)*m + j0, col] = 1.0
        V[j0*p + k0, col] = 1.0
        W[k0*n + (i0+1), col] = 1.0
        col += 1
        
        # T4: A22 * B21 -> C21
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[(j0+1)*p + k0, col] = 1.0
        W[k0*n + (i0+1), col] = 1.0
        col += 1

    # Part 1: j in {0, 1} (A cols 0,1)
    # B rows 0,1 (B top half)
    # Output k in {0, 1, 2, 3, 4, 5, 6}
    
    # Block 1.1: 2x2x2 for k=0,1
    emit_strassen(0, 0, 0)
    # Block 1.2: 2x2x2 for k=2,3
    emit_strassen(0, 0, 2)
    # Block 1.3: 2x2x2 for k=4,5
    emit_strassen(0, 0, 4)
    # Block 1.4: 2x2x1 for k=6
    emit_schoolbook_2x2x1(0, 6)
    
    # Part 2: j in {2, 3} (A cols 2,3)
    # B rows 2,3 (B bottom half)
    # Output k in {0, 1, 2, 3, 4, 5, 6}
    
    # Block 2.1: 2x2x2 for k=0,1
    # Note: j0 is now 2 (cols 2,3 of A, rows 2,3 of B)
    emit_strassen(0, 2, 0)
    # Block 2.2: 2x2x2 for k=2,3
    emit_strassen(0, 2, 2)
    # Block 2.3: 2x2x2 for k=4,5
    emit_strassen(0, 2, 4)
    # Block 2.4: 2x2x1 for k=6
    emit_schoolbook_2x2x1(2, 6)
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                if abs(val - 1.0) > 1e-9:
                    return None

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
