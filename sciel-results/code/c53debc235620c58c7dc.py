"""
Strategy: Explicitly construct a rank-17 decomposition for 2x3x3 matrix multiplication.

The 2x3x3 multiplication tensor can be viewed as a block matrix multiplication:
C (2x3) = A (2x3) * B (3x3)

We partition the indices to isolate a 2x2x2 core sub-problem and the remaining boundary terms.
1. Core (i in {0,1}, j in {0,1}, k in {0,1}): Use Strassen's algorithm (rank 7).
2. Boundary (j=2 OR k=2): Use schoolbook terms (rank 10).
Total rank = 17.

This decomposition is exact and uses integer coefficients.
"""

import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix.
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    n, m, p = DIMENSIONS
    # n=2, m=3, p=3
    # U shape: (6, R)
    # V shape: (9, R)
    # W shape: (6, R)
    
    rank = 17
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # --- Part 1: Strassen for the 2x2 core (i in {0,1}, j in {0,1}, k in {0,1}) ---
    # Indices:
    # U[i*m + j, c]
    # V[j*p + k, c]
    # W[k*n + i, c]
    
    # Helper to set a term
    def set_term(c, u_vals, v_vals, w_vals):
        for idx, val in u_vals.items():
            if val != 0:
                U[idx, c] += val
        for idx, val in v_vals.items():
            if val != 0:
                V[idx, c] += val
        for idx, val in w_vals.items():
            if val != 0:
                W[idx, c] += val

    # Strassen Formulas for 2x2 matrices:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Output reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Index mappings for 2x2 core (local indices 0,1):
    # A00 -> U[0, c] (i=0, j=0)
    # A01 -> U[1, c] (i=0, j=1)
    # A10 -> U[3, c] (i=1, j=0)
    # A11 -> U[4, c] (i=1, j=1)
    
    # B00 -> V[0, c] (j=0, k=0)
    # B01 -> V[1, c] (j=0, k=1)
    # B10 -> V[3, c] (j=1, k=0)
    # B11 -> V[4, c] (j=1, k=1)
    
    # C00 -> W[0, c] (k=0, i=0)
    # C01 -> W[2, c] (k=1, i=0)
    # C10 -> W[1, c] (k=0, i=1)
    # C11 -> W[3, c] (k=1, i=1)
    
    # Term 0: M1
    set_term(0,
             {0: 1, 4: 1},  # A00, A11
             {0: 1, 4: 1},  # B00, B11
             {0: 1, 3: 1}   # C00, C11
            )
            
    # Term 1: M2
    set_term(1,
             {3: 1, 4: 1},  # A10, A11
             {0: 1},        # B00
             {1: 1, 3: -1}  # C10, -C11
            )
            
    # Term 2: M3
    set_term(2,
             {0: 1},        # A00
             {1: 1, 4: -1}, # B01, -B11
             {2: 1, 3: 1}   # C01, C11
            )
            
    # Term 3: M4
    set_term(3,
             {4: 1},        # A11
             {3: 1, 0: -1}, # B10, -B00
             {0: 1, 1: 1}   # C00, C10
            )
            
    # Term 4: M5
    set_term(4,
             {0: 1, 1: 1},  # A00, A01
             {4: 1},        # B11
             {0: -1, 2: 1}  # -C00, C01
            )
            
    # Term 5: M6
    set_term(5,
             {3: 1, 0: -1}, # A10, -A00
             {0: 1, 1: 1},  # B00, B01
             {3: 1}         # C11
            )
            
    # Term 6: M7
    set_term(6,
             {1: 1, 4: -1}, # A01, -A11
             {3: 1, 4: 1},  # B10, B11
             {0: 1}         # C00
            )
            
    # --- Part 2: Schoolbook for the boundary ---
    # Boundary triples (i, j, k) where j=2 OR k=2.
    # i in {0,1}
    
    boundary_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j == 2 or k == 2:
                    boundary_triples.append((i, j, k))
                    
    # There are 10 such triples.
    # j=2, k=0: (0,2,0), (1,2,0)
    # j=2, k=1: (0,2,1), (1,2,1)
    # j=2, k=2: (0,2,2), (1,2,2)
    # j=0, k=2: (0,0,2), (1,0,2)
    # j=1, k=2: (0,1,2), (1,1,2)
    
    for idx, (i, j, k) in enumerate(boundary_triples):
        c = 7 + idx
        U[i * m + j, c] = 1
        V[j * p + k, c] = 1
        W[k * n + i, c] = 1
        
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails (should not happen)
        U_sb, V_sb, W_sb = _schoolbook(n, m, p)
        return {
            "U": _to_list(U_sb),
            "V": _to_list(V_sb),
            "W": _to_list(W_sb)
        }
        
    return {
        "U": _to_list(U),
        "V": _to_list(V),
        "W": _to_list(W)
    }

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition exactly reconstructs the multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition as numpy arrays."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _to_list(M):
    result = []
    for row in M:
        new_row = []
        for x in row:
            rx = round(x * 2) / 2
            if abs(rx - round(rx)) < 1e-9:
                new_row.append(int(round(rx)))
            else:
                new_row.append(rx)
        result.append(new_row)
    return result

# EVOLVE-BLOCK-END
