import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses 6 Strassen 2x2x2 blocks (rank 7 each) plus schoolbook for the rest.
    Total rank: 6*7 + (105 - 6*8) = 42 + 57 = 99.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        """Add a rank-1 term. u_entries, v_entries, w_entries are dicts: {index: value}"""
        U_cols.append([0] * U_rows)
        V_cols.append([0] * V_rows)
        W_cols.append([0] * W_rows)
        for idx, val in u_entries.items():
            U_cols[-1][idx] = val
        for idx, val in v_entries.items():
            V_cols[-1][idx] = val
        for idx, val in w_entries.items():
            W_cols[-1][idx] = val

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        """Add 7 Strassen rank-1 terms for a 2x2x2 block.
        
        A submatrix: rows i1,i2, cols j1,j2
        B submatrix: rows j1,j2, cols k1,k2
        C submatrix: rows i1,i2, cols k1,k2
        
        U indices: i*m+j
        V indices: j*p+k
        W indices: k*n+i
        """
        # A positions
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        # B positions
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        # C positions
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        # M1 = (a11+a22)*(b11+b22) -> W: +1 at c11, +1 at c22
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        
        # M2 = (a21+a22)*b11 -> W: +1 at c21, -1 at c22
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        
        # M3 = a11*(b12-b22) -> W: +1 at c12, +1 at c22
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        
        # M4 = a22*(b21-b11) -> W: +1 at c11, +1 at c21
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        
        # M5 = (a11+a12)*b22 -> W: -1 at c11, +1 at c12
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        
        # M6 = (a21-a11)*(b11+b12) -> W: +1 at c22
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        
        # M7 = (a12-a22)*(b21+b22) -> W: +1 at c11
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    # Define 6 Strassen blocks
    blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]

    # Track which (i,j,k) triples are covered by Strassen blocks
    covered = set()
    for (i1, i2, j1, j2, k1, k2) in blocks:
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Add schoolbook terms for uncovered (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)
    
    # Build matrices
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c in range(rank):
        for r in range(U_rows):
            U[r][c] = U_cols[c][r]
        for r in range(V_rows):
            V[r][c] = V_cols[c][r]
        for r in range(W_rows):
            W[r][c] = W_cols[c][r]

    # Quick verification: check that the tensor decomposition is correct
    # For each (i,j,k), sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    # We only check a few entries to save time
    if time.time() < deadline - 5:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check a sample of entries
        sample_indices = [(0,0,0), (1,2,3), (2,4,6), (0,3,5), (1,1,4)]
        for (i, j, k) in sample_indices:
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            # Compute sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
            prod = U_np[u_idx] * V_np[v_idx] * W_np[w_idx]
            s = np.sum(prod)
            if abs(s - 1.0) > 1e-10:
                # Fallback: return schoolbook if verification fails
                U_fb = [[0] * (n*m*p) for _ in range(n*m)]
                V_fb = [[0] * (n*m*p) for _ in range(m*p)]
                W_fb = [[0] * (n*m*p) for _ in range(p*n)]
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            idx = i2*m*p + j2*p + k2
                            U_fb[i2*m+j2][idx] = 1
                            V_fb[j2*p+k2][idx] = 1
                            W_fb[k2*n+i2][idx] = 1
                return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
