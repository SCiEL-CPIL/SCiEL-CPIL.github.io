"""Rewritten program for 3x4x4 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 4, 4]

# Deadline guard: kill before 150s wall-clock
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin

def _time_remaining():
    return _DEADLINE - time.time()

def _strassen_2x2(U, V, W, i_off, j_off, k_off, U_idx, V_idx, W_idx, col):
    """
    Add a Strassen rank-7 decomposition for a 2x2x2 block.
    U_idx(i, j) -> global row index in U for A element at local (i,j)
    V_idx(j, k) -> global row index in V for B element at local (j,k)
    W_idx(i, k) -> global row index in W for C element at local (i,k)
    i, j, k are local indices (0 or 1)
    Returns new col offset.
    """
    # Strassen products:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    
    # Each product M_r corresponds to a rank-1 term in the tensor.
    # For product (sum_{i,j} alpha[i,j]*A[i,j]) * (sum_{j,k} beta[j,k]*B[j,k]),
    # we set U[i*m+j, col] = alpha[i,j], V[j*p+k, col] = beta[j,k], and
    # W gets the output coefficients.
    
    # Actually, for Strassen, each M_r is a product of two linear forms.
    # The tensor term is: (sum_{i,j} alpha[i,j] A[i,j]) * (sum_{j,k} beta[j,k] B[j,k])
    # which contributes to C[i,k] with coefficient... wait, no.
    
    # Let me reconsider. In Strassen, we compute:
    # M1 = (a00+a11)(b00+b11)
    # This is a scalar. In the tensor decomposition, this corresponds to a rank-1 term
    # where U has nonzero entries at positions corresponding to a00 and a11,
    # V has nonzero entries at positions corresponding to b00 and b11,
    # and W has nonzero entries at positions corresponding to the C elements that M1 contributes to.
    
    # M1 contributes to: C00 += M1, C11 -= M1 (actually C11 += M1 in some formulations)
    # Let's use the standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So M1 contributes +1 to C00 and +1 to C11.
    # M2 contributes +1 to C10 and -1 to C11.
    # M3 contributes +1 to C01 and +1 to C11.
    # M4 contributes +1 to C00 and +1 to C10.
    # M5 contributes -1 to C00 and +1 to C01.
    # M6 contributes +1 to C11.
    # M7 contributes +1 to C00.
    
    # For each Strassen product M_r, we define:
    # U coefficients for A elements
    # V coefficients for B elements
    # W coefficients for C elements
    
    strassen_products = [
        # (U_coeffs: dict (i,j)->val, V_coeffs: dict (j,k)->val, W_coeffs: dict (i,k)->val)
        # M1 = (a00+a11)(b00+b11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2 = (a10+a11)*b00
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3 = a00*(b01-b11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4 = a11*(b10-b00)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5 = (a00+a01)*b11
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6 = (a10-a00)*(b00+b01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7 = (a01-a11)*(b10+b11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    for u_coeffs, v_coeffs, w_coeffs in strassen_products:
        for (li, lj), val_u in u_coeffs.items():
            gi = li + i_off
            gj = lj + j_off
            U[U_idx(gi, gj), col] = val_u
        for (lj, lk), val_v in v_coeffs.items():
            gj = lj + j_off
            gk = lk + k_off
            V[V_idx(gj, gk), col] = val_v
        for (li, lk), val_w in w_coeffs.items():
            gi = li + i_off
            gk = lk + k_off
            W[W_idx(gi, gk), col] = val_w
        col += 1
    
    return col

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 48
    
    # Initialize with schoolbook
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_result = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
    
    # Try Strassen-based decomposition
    # Split A (3x4) into two overlapping 2x4 blocks:
    # Block 1: rows 0-1 of A
    # Block 2: rows 1-2 of A
    # 
    # For each 2x4 * 4x4 -> 2x4, split B into two 4x2 halves:
    # Left half: cols 0-1, Right half: cols 2-3
    # 
    # Each 2x2 * 2x2 -> 2x2 uses Strassen rank 7.
    # So each 2x4 * 4x4 -> 2x4 uses 7+7 = 14 rank.
    # Two blocks: 14+14 = 28 rank.
    # But row 1 of A is shared between blocks, so we need to combine carefully.
    
    # Actually, the two blocks are independent in terms of output:
    # Block 1 produces C rows 0-1
    # Block 2 produces C rows 1-2
    # Row 1 of C gets contributions from both blocks.
    
    # Total rank: 14 + 14 = 28
    
    rank_strassen = 28
    U_s = np.zeros((n * m, rank_strassen))
    V_s = np.zeros((m * p, rank_strassen))
    W_s = np.zeros((p * n, rank_strassen))
    
    col = 0
    
    # Block 1: A rows 0-1, B cols 0-1 (left half)
    # A local (0,0)->global (0,0), (0,1)->global (0,1), (1,0)->global (1,0), (1,1)->global (1,1)
    # B local (0,0)->global (0,0), (0,1)->global (0,1), (1,0)->global (1,0), (1,1)->global (1,1)
    # C local (0,0)->global (0,0), (0,1)->global (0,1), (1,0)->global (1,0), (1,1)->global (1,1)
    
    def U_idx_block1_left(i, j):
        return i * m + j
    
    def V_idx_block1_left(j, k):
        return j * p + k
    
    def W_idx_block1_left(i, k):
        return k * n + i
    
    col = _strassen_2x2(
        U_s, V_s, W_s,
        0, 0, 0,
        U_idx_block1_left, V_idx_block1_left, W_idx_block1_left,
        col
    )
    
    # Block 1: A rows 0-1, B cols 2-3 (right half)
    def U_idx_block1_right(i, j):
        return i * m + j
    
    def V_idx_block1_right(j, k):
        return j * p + (k + 2)
    
    def W_idx_block1_right(i, k):
        return (k + 2) * n + i
    
    col = _strassen_2x2(
        U_s, V_s, W_s,
        0, 0, 2,
        U_idx_block1_right, V_idx_block1_right, W_idx_block1_right,
        col
    )
    
    # Block 2: A rows 1-2, B cols 0-1 (left half)
    def U_idx_block2_left(i, j):
        return (i + 1) * m + j
    
    def V_idx_block2_left(j, k):
        return j * p + k
    
    def W_idx_block2_left(i, k):
        return k * n + (i + 1)
    
    col = _strassen_2x2(
        U_s, V_s, W_s,
        1, 0, 0,
        U_idx_block2_left, V_idx_block2_left, W_idx_block2_left,
        col
    )
    
    # Block 2: A rows 1-2, B cols 2-3 (right half)
    def U_idx_block2_right(i, j):
        return (i + 1) * m + j
    
    def V_idx_block2_right(j, k):
        return j * p + (k + 2)
    
    def W_idx_block2_right(i, k):
        return (k + 2) * n + (i + 1)
    
    col = _strassen_2x2(
        U_s, V_s, W_s,
        1, 0, 2,
        U_idx_block2_right, V_idx_block2_right, W_idx_block2_right,
        col
    )
    
    # Verify the decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank_strassen):
                    s += U_s[i*m+j, r] * V_s[j*p+k, r] * W_s[k*n+i, r]
                if abs(s - 1) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
    
    if valid:
        return {
            "U": U_s.tolist(),
            "V": V_s.tolist(),
            "W": W_s.tolist()
        }
    
    # Fallback to schoolbook
    return best_result
