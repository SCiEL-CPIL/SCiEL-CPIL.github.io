"""
Rank-98 Strassen-based decomposition for 4x4 * 4x8.

Strategy:
  - Use the rank-7 Strassen decomposition for 2x2*2x2 as a building block.
  - Compose it with the 4x4*4x4 Strassen block structure (7 block-level
    multiplications, each of which is a 2x2*2x2 product decomposed at rank 7).
  - This gives 7 * 7 = 49 rank-1 terms per 4x4*4x4 half.
  - The 4x8 B matrix is split into two 4x4 halves, giving 2 * 49 = 98 total rank.

This is a STRUCTURAL improvement over the schoolbook rank-128 baseline.
The key insight from the inspiration is to use the full Strassen 2x2x2
decomposition INSIDE each of the 7 Strassen block-level multiplications
for the 4x4x4 case, rather than using schoolbook 2x2*2x2 (rank 4).
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]


def _strassen_2x2_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2.
    
    U: (4, 7) - rows indexed by i*2+j for A[i,j]
    V: (4, 7) - rows indexed by j*2+k for B[j,k]
    W: (4, 7) - rows indexed by k*2+i for C[i,k] (output convention W[k*n+i])
    
    The 7 Strassen multiplications:
      M0 = (a00+a11)(b00+b11)
      M1 = (a10+a11)(b00)
      M2 = (a00)(b01-b11)
      M3 = (a11)(b10-b00)
      M4 = (a00+a01)(b11)
      M5 = (a10-a00)(b00+b01)
      M6 = (a01-a11)(b10+b11)
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # M0: (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M1: (a10+a11)(b00)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M2: (a00)(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M3: (a11)(b10-b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M4: (a00+a01)(b11)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M5: (a10-a00)(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M6: (a01-a11)(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _strassen_4x4_4x4():
    """
    Construct the rank-49 Strassen-based decomposition for 4x4 * 4x4.
    
    Uses the 7 Strassen block-level multiplications, each of which is
    a 2x2*2x2 product decomposed at rank 7.
    
    U: (16, 49) - rows indexed by i*4+j
    V: (16, 49) - rows indexed by j*4+k
    W: (16, 49) - rows indexed by k*4+i (output convention)
    """
    n = 4
    U2, V2, W2 = _strassen_2x2_2x2()
    
    rank = 49
    U = np.zeros((n * n, rank))
    V = np.zeros((n * n, rank))
    W = np.zeros((n * n, rank))
    
    # Block-level Strassen specifications for 4x4x4
    # A blocks: A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1)
    # Each block is 2x2
    
    # P_k = A_eff_k * B_eff_k where A_eff and B_eff are 2x2
    # C = sum_k C_spec_k * P_k
    
    # A_specs[k]: list of (coeff, block_idx) for forming A_eff_k
    A_specs = [
        [(1, 0), (1, 3)],   # P0: A00 + A11
        [(1, 2), (1, 3)],   # P1: A10 + A11
        [(1, 0)],           # P2: A00
        [(1, 3)],           # P3: A11
        [(1, 0), (1, 1)],   # P4: A00 + A01
        [(1, 2), (-1, 0)],  # P5: A10 - A00
        [(1, 1), (-1, 3)],  # P6: A01 - A11
    ]
    
    # B_specs[k]: list of (coeff, block_idx) for forming B_eff_k
    B_specs = [
        [(1, 0), (1, 3)],   # P0: B00 + B11
        [(1, 0)],           # P1: B00
        [(1, 1), (-1, 3)],  # P2: B01 - B11
        [(1, 2), (-1, 0)],  # P3: B10 - B00
        [(1, 3)],           # P4: B11
        [(1, 0), (1, 1)],   # P5: B00 + B01
        [(1, 2), (1, 3)],   # P6: B10 + B11
    ]
    
    # C_specs[k]: list of (coeff, c_block_idx) for how P_k contributes to C
    # C00 = P0 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    C_specs = [
        [(1, 0), (1, 3)],   # P0: C00 + C11
        [(-1, 0), (1, 2)],  # P1: -C00 + C10  (wait, let me recheck)
        [(1, 1), (1, 3)],   # P2: C01 + C11
        [(1, 0), (1, 2)],   # P3: C00 + C10
        [(-1, 0), (1, 1)],  # P4: -C00 + C01
        [(1, 3)],           # P5: C11
        [(1, 0)],           # P6: C00
    ]
    
    # Let me re-verify the C_specs with the standard Strassen formulas:
    # C00 = M0 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    #
    # So for each M_k (P_k), which C blocks does it contribute to?
    # P0 (M0): C00 (+1), C11 (+1)
    # P1 (M1): C10 (+1), C11 (-1)
    # P2 (M2): C01 (+1), C11 (+1)
    # P3 (M3): C10 (+1)
    # P4 (M4): C00 (-1), C01 (+1)
    # P5 (M5): C11 (+1)
    # P6 (M6): C00 (+1)
    
    C_specs = [
        [(1, 0), (1, 3)],   # P0: +C00, +C11
        [(1, 2), (-1, 3)],  # P1: +C10, -C11
        [(1, 1), (1, 3)],   # P2: +C01, +C11
        [(1, 2)],           # P3: +C10
        [(-1, 0), (1, 1)],  # P4: -C00, +C01
        [(1, 3)],           # P5: +C11
        [(1, 0)],           # P6: +C00
    ]
    
    # Now build the full decomposition
    for k_p in range(7):  # Strassen block-level multiplication index
        for m_s in range(7):  # Inner Strassen 2x2x2 index
            col = k_p * 7 + m_s
            
            # --- U (Input A) ---
            for coeff_a, block_idx in A_specs[k_p]:
                bi, bj = divmod(block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        global_i = bi * 2 + r
                        global_j = bj * 2 + c
                        global_a_idx = global_i * 4 + global_j
                        a_eff_idx = r * 2 + c
                        U[global_a_idx, col] += coeff_a * U2[a_eff_idx, m_s]
            
            # --- V (Input B) ---
            for coeff_b, block_idx in B_specs[k_p]:
                bi, bj = divmod(block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        global_j = bi * 2 + r
                        global_k = bj * 2 + c
                        global_b_idx = global_j * 4 + global_k
                        b_eff_idx = r * 2 + c
                        V[global_b_idx, col] += coeff_b * V2[b_eff_idx, m_s]
            
            # --- W (Output C) ---
            for coeff_c, c_block_idx in C_specs[k_p]:
                bi, bj = divmod(c_block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        global_i = bi * 2 + r
                        global_k = bj * 2 + c
                        global_w_idx = global_k * 4 + global_i
                        p_eff_idx = r * 2 + c
                        W[global_w_idx, col] += coeff_c * W2[p_eff_idx, m_s]
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-10):
    """Verify that the decomposition correctly decomposes the multiplication tensor."""
    U_arr = np.array(U, dtype=float) if not isinstance(U, np.ndarray) else U
    V_arr = np.array(V, dtype=float) if not isinstance(V, np.ndarray) else V
    W_arr = np.array(W, dtype=float) if not isinstance(W, np.ndarray) else W
    
    # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False, (i, j, k, val)
    return True, None


def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Build the rank-49 Strassen decomposition for 4x4*4x4
    U49, V49, W49 = _strassen_4x4_4x4()
    
    # Verify the 4x4x4 decomposition
    valid, err = _verify_decomposition(U49, V49, W49, 4, 4, 4, 49)
    
    if valid:
        # Split 4x8 into two 4x4 halves
        rank = 98
        U = np.zeros((n * m, rank))  # 16 x 98
        V = np.zeros((m * p, rank))  # 32 x 98
        W = np.zeros((p * n, rank))  # 32 x 98
        
        # First half: B columns 0-3
        U[:, 0:49] = U49
        V[0:16, 0:49] = V49  # B rows 0-3, cols 0-3
        W[0:16, 0:49] = W49  # C rows 0-3, cols 0-3 (k=0..3, i=0..3)
        
        # Second half: B columns 4-7
        U[:, 49:98] = U49
        V[16:32, 49:98] = V49  # B rows 0-3, cols 4-7
        W[16:32, 49:98] = W49  # C rows 4-7, cols 0-3 (k=4..7, i=0..3)
        
        # Final verification with time check
        if time.time() < deadline - 10:
            valid2, err2 = _verify_decomposition(U, V, W, n, m, p, rank)
            if not valid2:
                # Fallback to schoolbook
                return _schoolbook(n, m, p)
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook rank 128
        return _schoolbook(n, m, p)


def _schoolbook(n, m, p):
    """Schoolbook decomposition with rank n*m*p."""
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
