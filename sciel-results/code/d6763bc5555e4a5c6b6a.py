import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    total_eq = n * m * p  # 64

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

    # Precompute index arrays
    eq_indices = np.arange(total_eq)
    i_idx = (eq_indices // (m * p)).astype(int)
    j_idx = ((eq_indices // p) % m).astype(int)
    k_idx = (eq_indices % p).astype(int)

    u_row_idx = i_idx * m + j_idx
    v_row_idx = j_idx * p + k_idx
    w_row_idx = k_idx * n + i_idx

    def verify_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        U_eq = U[u_row_idx, :]
        V_eq = V[v_row_idx, :]
        W_eq = W[w_row_idx, :]
        vals = np.sum(U_eq * V_eq * W_eq, axis=1)
        return np.all(np.abs(vals - 1) < tol)

    def snap_to_half_int(X):
        return np.round(2 * X) / 2.0

    def compute_residual_snapped(U, V, W):
        U_s = snap_to_half_int(U)
        V_s = snap_to_half_int(V)
        W_s = snap_to_half_int(W)
        U_eq = U_s[u_row_idx, :]
        V_eq = V_s[v_row_idx, :]
        W_eq = W_s[w_row_idx, :]
        vals = np.sum(U_eq * V_eq * W_eq, axis=1)
        return np.sum((vals - 1)**2)

    # Strassen 2x2x2 decomposition
    # For 2x2 * 2x2, the 7 products are:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    def strassen_2x2_block():
        """
        Return the 7 Strassen terms for a 2x2 * 2x2 block.
        Each term is a tuple (u_coeffs, v_coeffs, w_coeffs)
        where u_coeffs: dict of (local_a_row, local_a_col) -> coeff
                        v_coeffs: dict of (local_b_row, local_b_col) -> coeff
                        w_coeffs: dict of (local_c_row, local_c_col) -> coeff
        Local indices: A is 2x2, B is 2x2, C is 2x2
        """
        terms = [
            # M1 = (a00+a11)(b00+b11)
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # M2 = (a10+a11)b00
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            # M3 = a00(b01-b11)
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            # M4 = a11(b10-b00)
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            # M5 = (a00+a01)b11
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            # M6 = (a10-a00)(b00+b01)
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            # M7 = (a01-a11)(b10+b11)
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]
        return terms

    # Build the full Strassen decomposition for 2x4 * 4x8
    # A is 2x4, B is 4x8
    # We can think of B as 2x2 blocks of 2x4 each, or 4x2 blocks of 2x2
    # Let's partition B into 4 column blocks of 2 columns each: B[:, 0:2], B[:, 2:4], B[:, 4:6], B[:, 6:8]
    # And A has 2 rows.
    # For each column block of B (2 columns), we have a 4x2 submatrix.
    # We can further partition each 4x2 into two 2x2 blocks (rows 0-1 and rows 2-3).
    # So for each of the 4 column blocks, we have 2 row-blocks of B, giving 8 2x2 blocks total.
    # For each 2x2 block of B, we apply Strassen with the corresponding 2x2 block of A.
    # But A is only 2x4, so for a given column block of B (say columns 2v to 2v+1),
    # the corresponding A block is A[:, 2v:2v+2] which is 2x2.
    # Wait, that's not right. Let me reconsider.

    # Actually, the standard way to apply Strassen to 2x4 * 4x8:
    # Partition A into 1x2 blocks of 2x2 (since A is 2x4, we have 2 column blocks of 2x2)
    # Partition B into 2x4 blocks of 2x2 (since B is 4x8, we have 2 row blocks x 4 col blocks of 2x2)
    # For each pair of A column block (2x2) and B row block (2x2) and B column block (2x2),
    # we get a 2x2 product that contributes to C.
    
    # More precisely:
    # A = [A0 | A1] where A0, A1 are 2x2
    # B = [B00 B01 B02 B03; B10 B11 B12 B13] where each Bij is 2x2
    # C = A0*B00 + A1*B10, A0*B01 + A1*B11, A0*B02 + A1*B12, A0*B03 + A1*B13
    # Each term like A0*B00 is a 2x2 * 2x2 product, which can be done with 7 Strassen multiplications.
    # There are 2 (A blocks) * 4 (B col blocks) = 8 such 2x2*2x2 products.
    # Total rank = 8 * 7 = 56.

    # Let's build this decomposition.
    def build_strassen_decomposition():
        rank = 56
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        strassen_terms = strassen_2x2_block()
        
        col = 0
        # a_block: which 2x2 block of A (column block), 0 or 1
        # b_row_block: which 2x2 row block of B, 0 or 1
        # b_col_block: which 2x2 column block of B, 0 to 3
        for a_block in range(2):  # A column block
            for b_row_block in range(2):  # B row block
                for b_col_block in range(4):  # B column block
                    # For this combination, we compute A[a_block] * B[b_row_block, b_col_block]
                    # This contributes to C[:, b_col_block]
                    
                    # A block: rows 0-1, columns 2*a_block to 2*a_block+1
                    # B block: rows 2*b_row_block to 2*b_row_block+1, columns 2*b_col_block to 2*b_col_block+1
                    # C block: rows 0-1, columns 2*b_col_block to 2*b_col_block+1
                    
                    for t in range(7):
                        u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
                        
                        # U coefficients: A[i, j] where i in {0,1}, j in {2*a_block, 2*a_block+1}
                        for (li, lj), coeff in u_coeffs.items():
                            u_idx = li * m + (2 * a_block + lj)
                            U[u_idx, col] += coeff
                        
                        # V coefficients: B[j, k] where j in {2*b_row_block, 2*b_row_block+1}, k in {2*b_col_block, 2*b_col_block+1}
                        for (lj, lk), coeff in v_coeffs.items():
                            v_idx = (2 * b_row_block + lj) * p + (2 * b_col_block + lk)
                            V[v_idx, col] += coeff
                        
                        # W coefficients: C[i, k] where i in {0,1}, k in {2*b_col_block, 2*b_col_block+1}
                        # W index = k * n + i
                        for (li, lk), coeff in w_coeffs.items():
                            w_idx = (2 * b_col_block + lk) * n + li
                            W[w_idx, col] += coeff
                        
                        col += 1
        
        return U, V, W, rank

    U, V, W, rank = build_strassen_decomposition()
    
    # Verify the Strassen decomposition
    if verify_decomposition(U, V, W):
        # Try to snap to half-integers
        U_s = snap_to_half_int(U)
        V_s = snap_to_half_int(V)
        W_s = snap_to_half_int(W)
        if verify_decomposition(U_s, V_s, W_s):
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        else:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fall back to schoolbook
        U, V, W, rank = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
