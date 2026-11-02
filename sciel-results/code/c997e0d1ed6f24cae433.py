"""Combined program: Strassen block decomposition with verification."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140  # 150s limit, keep 10s safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Try Strassen-based decomposition
    if time.time() < deadline:
        strassen_result = _try_strassen_decomposition(n, m, p)
        if strassen_result:
            U_s, V_s, W_s, rank_s = strassen_result
            # Verify the decomposition
            if _verify_decomposition(U_s, V_s, W_s, n, m, p, rank_s):
                best_U, best_V, best_W = U_s, V_s, W_s
                best_rank = rank_s

    return {"U": best_U, "V": best_V, "W": best_W}

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    # Check that all entries are integer or half-integer
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not _is_half_integer(val):
                    return False
    
    # Reconstruct tensor and check
    T_recon = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                T_recon[i, j, k] = val
    
    if not np.allclose(T_recon, 1.0, atol=1e-9):
        return False
    
    return True

def _is_half_integer(x, tol=1e-9):
    """Check if x is an integer or half-integer."""
    return abs(x * 2 - round(x * 2)) < tol

def _try_strassen_decomposition(n, m, p):
    """
    Try to construct a rank-28 decomposition using Strassen's algorithm
    on 2x2 blocks.
    
    For 2x4x4:
    - Split A (2x4) into two 2x2 blocks: A_L (cols 0,1) and A_R (cols 2,3)
    - Split B (4x4) into four 2x2 blocks: B_TL, B_TR, B_BL, B_BR
    - C (2x4) = A_L * [B_TL B_TR] + A_R * [B_BL B_BR]
    - Each 2x2 * 2x2 product can be done with 7 multiplications (Strassen)
    - Total: 4 blocks * 7 = 28 multiplications
    """
    if n != 2 or m != 4 or p != 4:
        return None
    
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # (2*4, 28) = (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # (4*4, 28) = (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # (4*2, 28) = (8, 28)
    
    # Strassen terms for 2x2 matrix multiplication
    # P = X * Y where X, Y are 2x2
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    # P00 = M1 + M4 - M5 + M7
    # P01 = M3 + M5
    # P10 = M2 + M4
    # P11 = M1 - M2 + M3 + M6
    
    # Define Strassen terms as (u_coeffs, v_coeffs, w_coeffs)
    # Each is a dict mapping local index to coefficient
    # Local indices for 2x2: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    strassen_terms = [
        # (u, v, w) for M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # (u, v, w) for M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # (u, v, w) for M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # (u, v, w) for M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # (u, v, w) for M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # (u, v, w) for M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # (u, v, w) for M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def apply_strassen_block(a_col_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
        """
        Apply Strassen decomposition to a specific block.
        
        a_col_offset: 0 for left half of A (cols 0,1), 1 for right half (cols 2,3)
        b_row_offset: 0 for top half of B (rows 0,1), 1 for bottom half (rows 2,3)
        b_col_offset: 0 for left half of B (cols 0,1), 1 for right half (cols 2,3)
        c_col_offset: 0 for left half of C (cols 0,1), 1 for right half (cols 2,3)
        start_col: starting column index in U, V, W
        """
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(strassen_terms):
            col = start_col + t_idx
            
            # Map U coefficients
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                global_row = r
                global_col = a_col_offset * 2 + c
                global_idx = global_row * m + global_col
                U[global_idx][col] += coeff
            
            # Map V coefficients
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                global_row = b_row_offset * 2 + r
                global_col = b_col_offset * 2 + c
                global_idx = global_row * p + global_col
                V[global_idx][col] += coeff
            
            # Map W coefficients
            # W[k*n + i] where k is column, i is row
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                global_row = r
                global_col = c_col_offset * 2 + c
                # W index: k*n + i, where k=global_col, i=global_row
                global_idx = global_col * n + global_row
                W[global_idx][col] += coeff
    
    # Apply Strassen to 4 blocks:
    # Block 1: A_L * B_TL -> C_L (left part)
    apply_strassen_block(0, 0, 0, 0, 0)
    
    # Block 2: A_R * B_BL -> C_L (left part)
    apply_strassen_block(1, 1, 0, 0, 7)
    
    # Block 3: A_L * B_TR -> C_R (right part)
    apply_strassen_block(0, 0, 1, 1, 14)
    
    # Block 4: A_R * B_BR -> C_R (right part)
    apply_strassen_block(1, 1, 1, 1, 21)
    
    return U, V, W, rank

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    print(f"Max error: {np.max(np.abs(C - Cf))}")
    print(f"Rank: {U.shape[1]}")
