"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    n, m, p = DIMENSIONS

    def _build_block_decomposition(n1, m1, p1, offset_i, offset_j, offset_k, U, V, W, col_start):
        """Build schoolbook decomposition for a sub-block.
        
        n1: number of rows in A sub-block
        m1: number of cols in A sub-block (rows in B sub-block)  
        p1: number of cols in B sub-block
        offset_i: offset in i dimension
        offset_j: offset in j dimension
        offset_k: offset in k dimension
        Returns new column index.
        """
        col = col_start
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    U[(offset_i + i) * m + (offset_j + j), col] = 1
                    V[(offset_j + j) * p + (offset_k + k), col] = 1
                    W[(offset_k + k) * n + (offset_i + i), col] = 1
                    col += 1
        return col

    # Strategy: Block decomposition
    # Split m=5 into 2+3, p=6 into 3+3
    # A (2x5) = [A1 (2x2) | A2 (2x3)]
    # B (5x6) = [B1 (2x3); B2 (3x3)]
    # C = A1 @ B1 + A2 @ B2
    # Sub-problems: 2x2x3 (rank 12) and 2x3x3 (rank 18)
    # Total rank = 30
    
    R_total = 2*2*3 + 2*3*3  # 12 + 18 = 30
    
    U = np.zeros((n * m, R_total), dtype=np.float64)
    V = np.zeros((m * p, R_total), dtype=np.float64)
    W = np.zeros((p * n, R_total), dtype=np.float64)
    
    # Block 1: A1 (2x2) @ B1 (2x3) -> 2x2x3
    # i: 0-1, j: 0-1, k: 0-2
    col = _build_block_decomposition(2, 2, 3, 0, 0, 0, U, V, W, 0)
    
    # Block 2: A2 (2x3) @ B2 (3x3) -> 2x3x3
    # i: 0-1, j: 2-4, k: 3-5
    col = _build_block_decomposition(2, 3, 3, 0, 2, 3, U, V, W, col)
    
    # Validate
    def _validate(U_arr, V_arr, W_arr, R):
        for _ in range(50):
            A = np.random.randint(-3, 4, (n, m))
            B = np.random.randint(-3, 4, (m, p))
            C_expected = A @ B
            C_calc = np.zeros((n, p), dtype=np.float64)
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        for r in range(R):
                            s += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                    C_calc[i, k] = s
            if not np.allclose(C_calc, C_expected, atol=1e-9):
                return False
        return True
    
    if not _validate(U, V, W, R_total):
        # Fallback to schoolbook
        schoolbook_rank = n * m * p
        U = np.zeros((n * m, schoolbook_rank), dtype=np.float64)
        V = np.zeros((m * p, schoolbook_rank), dtype=np.float64)
        W = np.zeros((p * n, schoolbook_rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        R_total = schoolbook_rank
    
    return {"U": [list(map(float, row)) for row in U],
            "V": [list(map(float, row)) for row in V],
            "W": [list(map(float, row)) for row in W]}
# EVOLVE-BLOCK-END
