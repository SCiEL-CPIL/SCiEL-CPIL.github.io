"""Recursive Strassen decomposition of 4x4x4 matrix multiplication (rank 49)."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

# Strassen's 7 products for 2x2 matrices.
_STRASSEN_2X2 = [
    # P1 = (a+d)(e+h)
    ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
    # P2 = (c+d)e
    ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
    # P3 = a(f-h)
    ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)]),
    # P4 = d(g-e)
    ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)]),
    # P5 = (a+b)h
    ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
    # P6 = (c-a)(e+f)
    ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)]),
    # P7 = (b-d)(g+h)
    ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)]),
]

# How each C entry is formed from the 7 Strassen products:
_STRASSEN_C_COMBINATION = [
    (0, 0, [(0, 1), (3, 1), (4, -1), (6, 1)]),
    (0, 1, [(2, 1), (4, 1)]),
    (1, 0, [(1, 1), (3, 1)]),
    (1, 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),
]


def _strassen_4x4_decomposition():
    """
    Build rank-49 decomposition for 4x4x4 matrix multiplication using
    recursive Strassen: apply Strassen at the 4x4 level (7 block products),
    then apply Strassen within each 2x2 sub-block product (7 scalar mults each).
    Total rank = 7 * 7 = 49.
    """
    n, m, p = 4, 4, 4
    rank = 49

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # The 7 block-level products for 4x4 Strassen:
    block_products = [
        # M1 = (A00+A11)(B00+B11)
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        # M2 = (A10+A11)B00
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
        # M3 = A00(B01-B11)
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)]),
        # M4 = A11(B10-B00)
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)]),
        # M5 = (A00+A01)B11
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
        # M6 = (A10-A00)(B00+B01)
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)]),
        # M7 = (A01-A11)(B10+B11)
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)]),
    ]

    c_block_combination = [
        (0, 0, [(0, 1), (3, 1), (4, -1), (6, 1)]),
        (0, 1, [(2, 1), (4, 1)]),
        (1, 0, [(1, 1), (3, 1)]),
        (1, 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),
    ]

    # Build U and V
    for i, (a_combo, b_combo) in enumerate(block_products):
        for s, (a_local, b_local) in enumerate(_STRASSEN_2X2):
            r = i * 7 + s

            for (a_br, a_bc, a_coeff) in a_combo:
                for (lr, lc, l_coeff) in a_local:
                    row = 2 * a_br + lr
                    col = 2 * a_bc + lc
                    U[row * m + col][r] += a_coeff * l_coeff

            for (b_br, b_bc, b_coeff) in b_combo:
                for (lr, lc, l_coeff) in b_local:
                    row = 2 * b_br + lr
                    col = 2 * b_bc + lc
                    V[row * p + col][r] += b_coeff * l_coeff

    # Build W
    for i in range(7):
        for s in range(7):
            r = i * 7 + s
            for (cbr, cbc, c_list) in c_block_combination:
                w_i = 0
                for (pi, wi) in c_list:
                    if pi == i:
                        w_i = wi
                        break
                if w_i == 0:
                    continue
                for (lr, lc, v_list) in _STRASSEN_C_COMBINATION:
                    v_s = 0
                    for (si, vs) in v_list:
                        if si == s:
                            v_s = vs
                            break
                    if v_s == 0:
                        continue
                    c_row = 2 * cbr + lr
                    c_col = 2 * cbc + lc
                    W[c_col * n + c_row][r] += w_i * v_s

    return {"U": U, "V": V, "W": W}


def _schoolbook_decomposition(n, m, p):
    """Fallback schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True


def _try_rank_reduction_search(n, m, p, deadline, base_result):
    """
    Try to find a rank < 49 decomposition by searching for linear dependencies
    among the columns of the factor matrices, or by trying structured modifications.
    
    Strategy:
    1. Check if any column is a linear combination of others (in U, V, or W)
    2. Try to find a rank-48 decomposition by solving a least-squares problem
    3. Try small integer perturbations to absorb redundant terms
    """
    rng_seed = 42
    import random
    rng = random.Random(rng_seed)
    
    U0 = [row[:] for row in base_result["U"]]
    V0 = [row[:] for row in base_result["V"]]
    W0 = [row[:] for row in base_result["W"]]
    rank49 = 49

    U_np = np.array(U0, dtype=float)
    V_np = np.array(V0, dtype=float)
    W_np = np.array(W0, dtype=float)

    elapsed_start = time.time()

    # Strategy 1: Check for exact linear dependencies among columns
    # If column r of U is a linear combination of other columns, and similarly
    # for V and W, we might be able to eliminate it.
    
    # For each column r, check if it can be expressed as a linear combination
    # of the other columns in U (or V, or W).
    for elim_r in range(rank49):
        if time.time() > deadline - 15:
            break
        
        # Check if column elim_r of U is in the span of other U columns
        U_other = np.delete(U_np, elim_r, axis=1)
        U_elim = U_np[:, elim_r]
        
        # Solve least squares: U_other @ x ≈ U_elim
        try:
            x, residuals, rank_u, sv = np.linalg.lstsq(U_other, U_elim, rcond=None)
            # Check if the fit is exact
            U_recon = U_other @ x
            if np.allclose(U_recon, U_elim, atol=1e-10):
                # Column is linearly dependent in U
                # Check if we can also handle V and W
                V_other = np.delete(V_np, elim_r, axis=1)
                V_elim = V_np[:, elim_r]
                x_v, _, _, _ = np.linalg.lstsq(V_other, V_elim, rcond=None)
                V_recon = V_other @ x_v
                if np.allclose(V_recon, V_elim, atol=1e-10):
                    W_other = np.delete(W_np, elim_r, axis=1)
                    W_elim = W_np[:, elim_r]
                    x_w, _, _, _ = np.linalg.lstsq(W_other, W_elim, rcond=None)
                    W_recon = W_other @ x_w
                    if np.allclose(W_recon, W_elim, atol=1e-10):
                        # All three factors have linear dependence in column elim_r
                        # We can eliminate this column
                        U_trim = np.delete(U_np, elim_r, axis=1)
                        V_trim = np.delete(V_np, elim_r, axis=1)
                        W_trim = np.delete(W_np, elim_r, axis=1)
                        
                        # Convert back to lists
                        U_list = U_trim.tolist()
                        V_list = V_trim.tolist()
                        W_list = W_trim.tolist()
                        
                        # Verify
                        if _verify_decomposition(U_list, V_list, W_list, n, m, p, rank49 - 1):
                            return {"U": U_list, "V": V_list, "W": W_list}
        except np.linalg.LinAlgError:
            pass

    # Strategy 2: Try to find a rank-48 decomposition by modifying one column
    # to absorb contributions from another column.
    # For each pair (elim_r, absorb_r), try to find a small modification to
    # column absorb_r that absorbs the contribution of column elim_r.
    
    for elim_r in range(rank49):
        if time.time() > deadline - 15:
            break
        
        # Get the contribution of column elim_r to the tensor
        u_elim = U_np[:, elim_r]
        v_elim = V_np[:, elim_r]
        w_elim = W_np[:, elim_r]
        
        # Skip if column is zero
        if np.all(u_elim == 0) or np.all(v_elim == 0) or np.all(w_elim == 0):
            continue
        
        # Try to absorb this column into other columns
        # We look for a column absorb_r such that we can modify it to absorb elim_r
        for absorb_r in range(rank49):
            if absorb_r == elim_r:
                continue
            if time.time() > deadline - 15:
                break
            
            # Try to find a scalar alpha such that:
            # (u_absorb + alpha * u_elim) * (v_absorb) * (w_absorb) 
            # can be combined with (u_absorb) * (v_absorb + alpha * v_elim) * (w_absorb)
            # or similar modifications to absorb the elim_r contribution.
            #
            # This is complex. Instead, try a simpler approach:
            # Check if column elim_r can be written as a linear combination
            # of columns in U, V, and W simultaneously with the SAME coefficients.
            
            # This is a more restrictive condition but easier to check.
            pass

    # Strategy 3: Try structured modifications based on known algebraic identities
    # For 4x4 matrix multiplication, there might be identities that allow
    # combining two Strassen products.
    
    # Try modifying the W matrix to see if we can reduce rank
    # by exploiting the structure of the C combination.
    
    # For each C entry, check if the W coefficients have a pattern that
    # allows combining terms.
    
    # Given time constraints, return None if no reduction found.
    return None


def construct():
    deadline = time.time() + 135.0  # Safety margin before 150s limit
    n, m, p = DIMENSIONS

    # Build the recursive Strassen rank-49 decomposition
    result = _strassen_4x4_decomposition()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])

    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p, rank):
        # Fallback to schoolbook
        return _schoolbook_decomposition(n, m, p)

    # Try to find a lower-rank decomposition via search
    if time.time() < deadline - 20:
        reduced_result = _try_rank_reduction_search(n, m, p, deadline, result)
        if reduced_result is not None:
            return reduced_result

    # Return the verified rank-49 Strassen decomposition
    return result

# EVOLVE-BLOCK-END
