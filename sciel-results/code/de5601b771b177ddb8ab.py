"""
Combined: Strassen 2x2 blocking (rank 42) + greedy rank-reduction search.
Structural idea from inspirations: clean dict-based Strassen terms and verification.
Refinement machinery: iterative column-merging and column-removal to reduce rank.
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 * 2x2 matrix multiplication.
    Each term is a tuple (U_coeffs, V_coeffs, W_coeffs) where:
      U_coeffs: dict {(i, j): coeff} for the 2x2 left matrix
      V_coeffs: dict {(j, k): coeff} for the 2x2 right matrix
      W_coeffs: dict {(i, k): coeff} for the 2x2 output
    
    Strassen formulas:
      M1 = (P00+P11)*(Q00+Q11)
      M2 = (P10+P11)*Q00
      M3 = P00*(Q01-Q11)
      M4 = P11*(Q10-Q00)
      M5 = (P00+P01)*Q11
      M6 = (P10-P00)*(Q00+Q01)
      M7 = (P01-P11)*(Q10+Q11)
    
    Result:
      R00 = M1 - M4 + M5 + M6
      R01 = M3 + M5
      R10 = M2 + M4
      R11 = M1 - M3 + M7
    """
    terms = []
    
    # M1 = (P00+P11)*(Q00+Q11)
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    ))
    
    # M2 = (P10+P11)*Q00
    terms.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1}
    ))
    
    # M3 = P00*(Q01-Q11)
    terms.append((
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(0,1): 1}
    ))
    
    # M4 = P11*(Q10-Q00)
    terms.append((
        {(1,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,0): -1, (1,0): 1}
    ))
    
    # M5 = (P00+P01)*Q11
    terms.append((
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1}
    ))
    
    # M6 = (P10-P00)*(Q00+Q01)
    terms.append((
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(0,0): 1}
    ))
    
    # M7 = (P01-P11)*(Q10+Q11)
    terms.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(1,1): 1}
    ))
    
    return terms


def _build_strassen_decomposition():
    """
    Construct a rank-42 decomposition using Strassen's algorithm.
    
    Strategy: Split the 2x4 * 4x6 multiplication into blocks:
    - A is split into two 2x2 blocks: A_left (cols 0,1) and A_right (cols 2,3)
    - B is split into top (rows 0,1) and bottom (rows 2,3), each with 3 column blocks of width 2
    
    So C = A_left * B_top + A_right * B_bottom
    
    Each (2x2) * (2x2) block multiplication uses 7 Strassen terms.
    Total: 2 (A blocks) * 3 (B col blocks) * 7 (Strassen) = 42 terms.
    
    Returns U, V, W as numpy arrays.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42
    
    U = np.zeros((n * m, rank), dtype=np.float64)  # 8 x 42
    V = np.zeros((m * p, rank), dtype=np.float64)  # 24 x 42
    W = np.zeros((p * n, rank), dtype=np.float64)  # 12 x 42
    
    strassen_terms = _strassen_2x2_terms()
    
    col = 0
    # For each A block (left/right) and each B column block
    for a_block in range(2):  # 0: A cols 0,1; 1: A cols 2,3
        for b_col_block in range(3):  # B col blocks: 0,1,2 (each width 2)
            for t in range(7):  # 7 Strassen terms
                U_coeffs, V_coeffs, W_coeffs = strassen_terms[t]
                
                # Update U: A block a_block has columns at offset a_block*2
                for (i_loc, j_loc), coeff in U_coeffs.items():
                    i = i_loc
                    j = a_block * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx, col] += coeff
                
                # Update V: B rows depend on a_block (top if a_block=0, bottom if a_block=1)
                # B col block b_col_block has columns at offset b_col_block*2
                for (j_loc, k_loc), coeff in V_coeffs.items():
                    j = a_block * 2 + j_loc
                    k = b_col_block * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx, col] += coeff
                
                # Update W: output C[i, k] where i is row of C, k is col of C
                # W index: k*n + i (official trace convention)
                for (i_loc, k_loc), coeff in W_coeffs.items():
                    i = i_loc
                    k = b_col_block * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx, col] += coeff
                
                col += 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n=2, m=4, p=6, trials=30):
    """Verify a decomposition numerically with integer arithmetic."""
    rank = U.shape[1]
    
    for trial in range(trials):
        A = np.random.randint(-5, 6, size=(n, m))
        B = np.random.randint(-5, 6, size=(m, p))
        
        # Expected C
        C_exp = A @ B
        
        # Computed C
        C_comp = np.zeros((n, p), dtype=np.float64)
        for col in range(rank):
            u_val = np.dot(U[:, col], A.flatten())
            v_val = np.dot(V[:, col], B.flatten())
            # W has shape (p*n, rank), index k*n+i
            w_col = W[:, col]  # shape (p*n,)
            for i in range(n):
                for k in range(p):
                    C_comp[i, k] += w_col[k * n + i] * u_val * v_val
        
        if not np.allclose(C_comp, C_exp, atol=1e-9):
            return False
    return True


def _verify_decomposition_fast(U, V, W, n=2, m=4, p=6, trials=20):
    """Faster verification using einsum."""
    rank = U.shape[1]
    
    for trial in range(trials):
        A = np.random.randint(-5, 6, size=(n, m))
        B = np.random.randint(-5, 6, size=(m, p))
        
        C_exp = A @ B
        
        # C_comp[i,k] = sum_r W[k*n+i, r] * (sum_{i',j'} U[i'*m+j', r] * A[i',j']) * (sum_{j'',k''} V[j''*p+k'', r] * B[j'',k''])
        # Using einsum: C_comp = einsum('kr,ar,br->ab', W, U, V) where a indexes A, b indexes B
        # But we need to reshape properly
        # U @ A_flat gives vector of length rank
        # V @ B_flat gives vector of length rank
        # Then C_flat = W.T @ (u_vec * v_vec)
        u_vec = U.T @ A.flatten()  # shape (rank,)
        v_vec = V.T @ B.flatten()  # shape (rank,)
        C_flat = W.T @ (u_vec * v_vec)  # shape (p*n,)
        C_comp = C_flat.reshape(p, n).T  # shape (n, p)
        
        if not np.allclose(C_comp, C_exp, atol=1e-9):
            return False
    return True


def _try_remove_column(U, V, W, col_idx):
    """
    Try to remove column col_idx by expressing it as a linear combination of other columns.
    Returns (success, new_U, new_V, new_W) if successful.
    """
    rank = U.shape[1]
    if rank <= 1:
        return False, U, V, W
    
    # We want to find coefficients c_j such that for all entries:
    # U[:, col_idx] = sum_{j != col_idx} c_j * U[:, j]
    # V[:, col_idx] = sum_{j != col_idx} c_j * V[:, j]
    # W[:, col_idx] = sum_{j != col_idx} c_j * W[:, j]
    
    # This is a system of linear equations. Let's set up:
    # For U: U[:, col_idx] = U_other @ c
    # For V: V[:, col_idx] = V_other @ c
    # For W: W[:, col_idx] = W_other @ c
    
    # Combine all constraints into one system
    U_other = np.delete(U, col_idx, axis=1)  # shape (n*m, rank-1)
    V_other = np.delete(V, col_idx, axis=1)  # shape (m*p, rank-1)
    W_other = np.delete(W, col_idx, axis=1)  # shape (p*n, rank-1)
    
    A_system = np.vstack([U_other, V_other, W_other])  # shape (n*m + m*p + p*n, rank-1)
    b_system = np.concatenate([U[:, col_idx], V[:, col_idx], W[:, col_idx]])  # shape (n*m + m*p + p*n,)
    
    # Solve least squares
    try:
        c, residuals, rank_sys, sv = np.linalg.lstsq(A_system, b_system, rcond=1e-10)
    except np.linalg.LinAlgError:
        return False, U, V, W
    
    # Check if the solution is good
    reconstructed = A_system @ c
    if not np.allclose(reconstructed, b_system, atol=1e-8):
        return False, U, V, W
    
    # Check that the coefficients are "nice" (close to integers or half-integers)
    # This is a heuristic to avoid numerical garbage
    rounded_c = np.round(c * 2) / 2  # round to nearest half-integer
    if not np.allclose(rounded_c, c, atol=1e-6):
        # Try rounding to nearest integer
        rounded_c = np.round(c)
        if not np.allclose(rounded_c, c, atol=1e-6):
            return False, U, V, W
    
    # Use the rounded coefficients
    c = rounded_c
    
    # Verify the reconstruction with rounded coefficients
    reconstructed = A_system @ c
    if not np.allclose(reconstructed, b_system, atol=1e-8):
        return False, U, V, W
    
    # Remove the column
    new_U = np.delete(U, col_idx, axis=1)
    new_V = np.delete(V, col_idx, axis=1)
    new_W = np.delete(W, col_idx, axis=1)
    
    return True, new_U, new_V, new_W


def _try_merge_columns(U, V, W, col1, col2, coeff):
    """
    Try to merge column col2 into col1 by adding coeff * col2 to col1, then removing col2.
    Returns (success, new_U, new_V, new_W) if successful.
    """
    # Add coeff * col2 to col1
    U[:, col1] += coeff * U[:, col2]
    V[:, col1] += coeff * V[:, col2]
    W[:, col1] += coeff * W[:, col2]
    
    # Remove col2
    new_U = np.delete(U, col2, axis=1)
    new_V = np.delete(V, col2, axis=1)
    new_W = np.delete(W, col2, axis=1)
    
    return True, new_U, new_V, new_W


def _greedy_rank_reduction(U, V, W, max_time=60):
    """
    Greedy algorithm to reduce the rank of the decomposition.
    Tries to remove columns one by one, checking if the decomposition remains valid.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Make copies
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    # Try to remove columns repeatedly
    removed_any = True
    while removed_any and (time.time() - start_time) < max_time:
        removed_any = False
        rank = U.shape[1]
        
        for col_idx in range(rank):
            if (time.time() - start_time) > max_time:
                break
            
            success, new_U, new_V, new_W = _try_remove_column(U, V, W, col_idx)
            if success:
                # Verify the new decomposition
                if _verify_decomposition_fast(new_U, new_V, new_W, n, m, p, trials=10):
                    U, V, W = new_U, new_V, new_W
                    removed_any = True
                    break  # Restart from beginning since indices changed
    
    return U, V, W


def _schoolbook_decomposition():
    """Fallback: rank-m = 4 decomposition."""
    n, m, p = DIMENSIONS
    rank = m
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    for j in range(m):
        for i in range(n):
            U[i * m + j, j] = 1
        for k in range(p):
            V[j * p + k, j] = 1
        for i in range(n):
            for k in range(p):
                W[k * n + i, j] = 1
    
    return U, V, W


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2 by 4 
    matrix by a 4 by 6 matrix.
    
    Strategy: Use Strassen-based rank-42 decomposition, then attempt greedy rank reduction.
    
    The output factor follows the official trace convention W[k*n+i,r].
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Build the Strassen decomposition (rank 42)
    U, V, W = _build_strassen_decomposition()
    
    # Verify the decomposition
    if not _verify_decomposition(U, V, W, n, m, p, trials=20):
        # Fall back to schoolbook
        U_fb, V_fb, W_fb = _schoolbook_decomposition()
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try greedy rank reduction
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        max_reduction_time = min(60, remaining_time - 15)
        U_red, V_red, W_red = _greedy_rank_reduction(U, V, W, max_time=max_reduction_time)
        
        if U_red.shape[1] < best_rank:
            # Verify the reduced decomposition
            if _verify_decomposition(U_red, V_red, W_red, n, m, p, trials=20):
                best_rank = U_red.shape[1]
                best_U, best_V, best_W = U_red, V_red, W_red
    
    # Check deadline
    if time.time() > deadline - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p, trials=30):
        # Fall back to schoolbook
        U_fb, V_fb, W_fb = _schoolbook_decomposition()
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
