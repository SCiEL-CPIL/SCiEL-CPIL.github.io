import time
import numpy as np

DIMENSIONS = [2, 4, 5]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0


def _remaining():
    return _DEADLINE - time.time()


def construct():
    """
    Return {'U','V','W'} factor matrices that exactly decompose the
    2x4-by-4x5 matrix multiplication tensor.

    Strategy (block decomposition):
      - Split the 4x5 B into four 2x2 blocks plus a final 4x1 column.
      - Each 2x2 (A-block) * (B-block) -> 2x2 C block uses Strassen (rank 7).
      - The final 2x4 * 4x1 -> 2x1 C column uses schoolbook (rank 8).
      Total rank = 4*7 + 8 = 36.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    rank = 36

    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36

    col = 0

    def add_strassen_222(i0, j0, k0, col_start):
        """
        Add a Strassen 2x2x2 decomposition for:
        C[i0:i0+2, k0:k0+2] = A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        """
        nonlocal col

        # M1 = (A00 + A11) * (B00 + B11)
        c = col_start
        U[i0 * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        col += 1

        # M2 = (A10 + A11) * B00
        c = col_start + 1
        U[(i0 + 1) * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] -= 1
        col += 1

        # M3 = A00 * (B01 - B11)
        c = col_start + 2
        U[i0 * m + j0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] -= 1
        W[(k0 + 1) * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        col += 1

        # M4 = A11 * (B10 - B00)
        c = col_start + 3
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + k0][c] += 1
        V[j0 * p + k0][c] -= 1
        W[k0 * n + i0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        col += 1

        # M5 = (A00 + A01) * B11
        c = col_start + 4
        U[i0 * m + j0][c] += 1
        U[i0 * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] -= 1
        W[(k0 + 1) * n + i0][c] += 1
        col += 1

        # M6 = (A10 - A00) * (B00 + B01)
        c = col_start + 5
        U[(i0 + 1) * m + j0][c] += 1
        U[i0 * m + j0][c] -= 1
        V[j0 * p + k0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        col += 1

        # M7 = (A01 - A11) * (B10 + B11)
        c = col_start + 6
        U[i0 * m + (j0 + 1)][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] -= 1
        V[(j0 + 1) * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        col += 1

        return col_start + 7

    def add_schoolbook_2x4x1(i0, j0, k0, col_start):
        """
        Add schoolbook decomposition for:
        C[i0:i0+2, k0:k0+1] = A[i0:i0+2, j0:j0+4] * B[j0:j0+4, k0:k0+1]
        """
        c = col_start
        for di in range(2):
            for dj in range(4):
                U[(i0 + di) * m + (j0 + dj)][c] = 1
                V[(j0 + dj) * p + k0][c] = 1
                W[k0 * n + (i0 + di)][c] = 1
                c += 1
        return c

    # Block 1: C[:, 0:2] from A[:, 0:2] * B[0:2, 0:2]
    col = add_strassen_222(0, 0, 0, 0)
    # Block 2: C[:, 0:2] from A[:, 2:4] * B[2:4, 0:2]
    col = add_strassen_222(0, 2, 0, col)
    # Block 3: C[:, 2:4] from A[:, 0:2] * B[0:2, 2:4]
    col = add_strassen_222(0, 0, 2, col)
    # Block 4: C[:, 2:4] from A[:, 2:4] * B[2:4, 2:4]
    col = add_strassen_222(0, 2, 2, col)
    # Block 5: C[:, 4] from A * B[:, 4] (schoolbook 2x4 * 4x1)
    col = add_schoolbook_2x4x1(0, 0, 4, col)

    assert col == rank, f"Expected rank {rank}, got {col}"

    # ---- Exact verification ----------------------------------------------
    if _remaining() > 0.5:
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    acc = 0
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    for r in range(rank):
                        acc += U[ui][r] * V[vj][r] * W[wk][r]
                    assert acc == 1, f"Verification failed at i={i},j={j},k={k}: acc={acc}"

    # ---- Local search: try to reduce rank --------------------------------
    # Convert to numpy for vectorized operations
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)

    # Build the multiplication tensor T[i,j,k] = 1 if (i,j,k) is a valid entry
    # T has shape (n*m, m*p, p*n) = (8, 20, 10)
    # T[ui, vj, wk] = 1 if ui=i*m+j, vj=j*p+k, wk=k*n+i
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                T[ui, vj, wk] = 1.0

    def _verify_decomp(U_arr, V_arr, W_arr, tol=1e-9):
        """Check if U@V@W reconstructs T."""
        # For each rank component r, we have U[:,r], V[:,r], W[:,r]
        # The tensor is sum_r outer(U[:,r], V[:,r], W[:,r])
        # Check: for all (ui,vj,wk): sum_r U[ui,r]*V[vj,r]*W[wk,r] == T[ui,vj,wk]
        # Vectorized: einsum
        recon = np.einsum('ur,vr,wr->uvw', U_arr, V_arr, W_arr)
        return np.allclose(recon, T, atol=tol, rtol=0)

    def _try_reduce_rank(U_arr, V_arr, W_arr, max_reductions=5):
        """
        Try to reduce rank by finding linear dependencies.
        Strategy: for each pair of rank components, check if one can be expressed
        as a linear combination of others by solving for coefficients.
        """
        current_U = U_arr.copy()
        current_V = V_arr.copy()
        current_W = W_arr.copy()
        current_rank = current_U.shape[1]
        
        reduced = 0
        for attempt in range(max_reductions):
            if _remaining() < 10.0:
                break
                
            # Try to find a rank component that is linearly dependent on others
            # For each component r, check if we can find alpha_r such that
            # the contribution of component r can be absorbed by modifying others.
            
            # A simpler approach: check if any column in U is linearly dependent
            # on the others, AND the same holds for V and W simultaneously.
            
            # For each candidate component to remove:
            best_r = -1
            best_coeff = None
            
            for r in range(current_rank):
                if _remaining() < 10.0:
                    break
                # Check if U[:,r] is in span of U[:, others]
                U_others = np.delete(current_U, r, axis=1)
                V_others = np.delete(current_V, r, axis=1)
                W_others = np.delete(current_W, r, axis=1)
                
                # Solve for coefficients c such that U[:,r] ≈ U_others @ c
                # Use least squares
                try:
                    c_U, _, _, _ = np.linalg.lstsq(U_others, current_U[:, r], rcond=None)
                    U_resid = current_U[:, r] - U_others @ c_U
                    
                    c_V, _, _, _ = np.linalg.lstsq(V_others, current_V[:, r], rcond=None)
                    V_resid = current_V[:, r] - V_others @ c_V
                    
                    c_W, _, _, _ = np.linalg.lstsq(W_others, current_W[:, r], rcond=None)
                    W_resid = current_W[:, r] - W_others @ c_W
                    
                    # Check if the same coefficients work for all three
                    if (np.allclose(c_U, c_V, atol=1e-8) and 
                        np.allclose(c_U, c_W, atol=1e-8)):
                        # Check residual
                        if (np.linalg.norm(U_resid) < 1e-8 and
                            np.linalg.norm(V_resid) < 1e-8 and
                            np.linalg.norm(W_resid) < 1e-8):
                            best_r = r
                            best_coeff = c_U
                            break
                except:
                    continue
            
            if best_r >= 0:
                # Remove component best_r
                current_U = np.delete(current_U, best_r, axis=1)
                current_V = np.delete(current_V, best_r, axis=1)
                current_W = np.delete(current_W, best_r, axis=1)
                current_rank -= 1
                reduced += 1
            else:
                # No more simple reductions; try a different approach
                # Check if we can merge two components
                break
        
        return current_U, current_V, current_W, current_rank

    # Attempt rank reduction
    if _remaining() > 5.0:
        U_reduced, V_reduced, W_reduced, new_rank = _try_reduce_rank(
            U_np, V_np, W_np, max_reductions=3
        )
        
        # Verify the reduced decomposition
        if _verify_decomp(U_reduced, V_reduced, W_reduced):
            # Convert back to lists with integer/half-integer entries
            U_final = np.round(U_reduced * 2) / 2
            V_final = np.round(V_reduced * 2) / 2
            W_final = np.round(W_reduced * 2) / 2
            
            # Final verification with rounded values
            if _verify_decomp(U_final, V_final, W_final):
                U = U_final.tolist()
                V = V_final.tolist()
                W = W_final.tolist()
                rank = new_rank

    return {"U": U, "V": V, "W": W}
