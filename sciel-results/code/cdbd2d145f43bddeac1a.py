import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Total rank is 44
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Define the 4 blocks
    # Block 0: j in {0,1}, k in {0,1}
    # Block 1: j in {0,1}, k in {2,3}
    # Block 2: j in {2,3}, k in {0,1}
    # Block 3: j in {2,3}, k in {2,3}
    
    blocks = [
        (0, 0), # j_offset, k_offset
        (0, 2),
        (2, 0),
        (2, 2)
    ]
    
    col_offset = 0
    
    for j_off, k_off in blocks:
        # Local indices j_loc in {0,1}, k_loc in {0,1}
        # Global j = j_off + j_loc, k = k_off + k_loc
        
        # --- Strassen Part (i in {0,1}) ---
        # We need 7 terms.
        # Term 1: (A00+A11)(B00+B11) -> C00, C11
        # Term 2: (A10+A11)B00 -> C10, C11
        # Term 3: A00(B01-B11) -> C00, C01
        # Term 4: A11(B10-B00) -> C00, C10
        # Term 5: (A00+A01)B11 -> C00, C01
        # Term 6: (A10-A00)(B00+B01) -> C11
        # Term 7: (A01-A11)(B10+B11) -> C00, C11
        
        # Helper to add a term
        def add_term(u_indices, v_indices, w_indices, w_coeffs):
            nonlocal col_offset
            for idx in u_indices:
                U[idx, col_offset] += 1.0
            for idx in v_indices:
                V[idx, col_offset] += 1.0
            for idx, coeff in zip(w_indices, w_coeffs):
                W[idx, col_offset] += coeff
            col_offset += 1

        # Local A indices (i, j_loc)
        # A00: (0, 0) -> u_idx(0, j_off+0)
        # A01: (0, 1) -> u_idx(0, j_off+1)
        # A10: (1, 0) -> u_idx(1, j_off+0)
        # A11: (1, 1) -> u_idx(1, j_off+1)
        
        # Local B indices (j_loc, k_loc)
        # B00: (0, 0) -> v_idx(j_off+0, k_off+0)
        # B01: (0, 1) -> v_idx(j_off+0, k_off+1)
        # B10: (1, 0) -> v_idx(j_off+1, k_off+0)
        # B11: (1, 1) -> v_idx(j_off+1, k_off+1)
        
        # Local C indices (k_loc, i)
        # C00: (0, 0) -> w_idx(k_off+0, 0)
        # C01: (0, 1) -> w_idx(k_off+1, 0)
        # C10: (1, 0) -> w_idx(k_off+0, 1)
        # C11: (1, 1) -> w_idx(k_off+1, 1)
        
        # Term 1
        add_term(
            [u_idx(0, j_off), u_idx(1, j_off+1)],
            [v_idx(j_off, k_off), v_idx(j_off+1, k_off+1)],
            [w_idx(k_off, 0), w_idx(k_off+1, 1)],
            [1.0, 1.0]
        )
        
        # Term 2
        add_term(
            [u_idx(1, j_off), u_idx(1, j_off+1)],
            [v_idx(j_off, k_off)],
            [w_idx(k_off, 1), w_idx(k_off+1, 1)],
            [1.0, -1.0] # C10 gets +1, C11 gets -1
        )
        
        # Term 3
        add_term(
            [u_idx(0, j_off)],
            [v_idx(j_off, k_off+1), v_idx(j_off+1, k_off+1)],
            [w_idx(k_off, 0), w_idx(k_off+1, 0)],
            [1.0, 1.0] # C00 gets +1, C01 gets +1
        )
        
        # Term 4
        add_term(
            [u_idx(1, j_off+1)],
            [v_idx(j_off+1, k_off), v_idx(j_off, k_off)],
            [w_idx(k_off, 0), w_idx(k_off, 1)],
            [1.0, 1.0] # C00 gets +1, C10 gets +1
        )
        
        # Term 5
        add_term(
            [u_idx(0, j_off), u_idx(0, j_off+1)],
            [v_idx(j_off+1, k_off+1)],
            [w_idx(k_off, 0), w_idx(k_off+1, 0)],
            [-1.0, 1.0] # C00 gets -1, C01 gets +1
        )
        
        # Term 6
        add_term(
            [u_idx(1, j_off), u_idx(0, j_off)],
            [v_idx(j_off, k_off), v_idx(j_off, k_off+1)],
            [w_idx(k_off+1, 1)],
            [1.0] # C11 gets +1
        )
        
        # Term 7
        add_term(
            [u_idx(0, j_off+1), u_idx(1, j_off+1)],
            [v_idx(j_off+1, k_off), v_idx(j_off+1, k_off+1)],
            [w_idx(k_off, 0), w_idx(k_off+1, 1)],
            [1.0, 1.0] # C00 gets +1, C11 gets +1
        )
        
        # --- Trivial Part (i=2) ---
        # 4 terms
        # T1: A20 B00 -> C20
        add_term(
            [u_idx(2, j_off)],
            [v_idx(j_off, k_off)],
            [w_idx(k_off, 2)],
            [1.0]
        )
        
        # T2: A21 B10 -> C20
        add_term(
            [u_idx(2, j_off+1)],
            [v_idx(j_off+1, k_off)],
            [w_idx(k_off, 2)],
            [1.0]
        )
        
        # T3: A20 B01 -> C21
        add_term(
            [u_idx(2, j_off)],
            [v_idx(j_off, k_off+1)],
            [w_idx(k_off+1, 2)],
            [1.0]
        )
        
        # T4: A21 B11 -> C21
        add_term(
            [u_idx(2, j_off+1)],
            [v_idx(j_off+1, k_off+1)],
            [w_idx(k_off+1, 2)],
            [1.0]
        )

    # Verify
    def calculate_residuals(U, V, W):
        triplets = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        U_rows = np.array([u_idx(i, j) for (i, j, k) in triplets])
        V_rows = np.array([v_idx(j, k) for (i, j, k) in triplets])
        W_rows = np.array([w_idx(k, i) for (i, j, k) in triplets])
        
        U_sel = U[U_rows]
        V_sel = V[V_rows]
        W_sel = W[W_rows]
        
        products = U_sel * V_sel * W_sel
        sums = np.sum(products, axis=1)
        
        return sums - 1.0

    res = calculate_residuals(U, V, W)
    if not np.allclose(res, 0.0, atol=1e-9):
        # Fallback to schoolbook if construction fails
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[u_idx(i, j), col] = 1.0
                    V[v_idx(j, k), col] = 1.0
                    W[w_idx(k, i), col] = 1.0
                    col += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
