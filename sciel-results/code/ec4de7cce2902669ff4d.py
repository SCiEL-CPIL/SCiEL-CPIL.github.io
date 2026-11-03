import time
import numpy as np

def construct():
    _t0 = time.time()
    deadline = _t0 + 130.0  # safety margin before 150s kill

    n, m, p = 3, 4, 6  # C is n x p, A is n x m, B is m x p

    def _build_block_decomp():
        """Build the rank-66 block-structured decomposition."""
        RANK = 66
        U = [[0] * RANK for _ in range(n * m)]
        V = [[0] * RANK for _ in range(m * p)]
        W = [[0] * RANK for _ in range(p * n)]
        col = 0

        def _idx_u(i, k): return i * m + k
        def _idx_v(k, j): return k * p + j
        def _idx_w(j, i): return j * n + i

        def _add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A0 = _idx_u(a_r, a_c); A1 = _idx_u(a_r, a_c + 1)
            B00 = _idx_v(b_r, b_c); B01 = _idx_v(b_r, b_c + 1)
            B10 = _idx_v(b_r + 1, b_c); B11 = _idx_v(b_r + 1, b_c + 1)
            C0 = _idx_w(c_c, c_r); C1 = _idx_w(c_c + 1, c_r)
            U[A0][col] = 1; V[B00][col] = 1; W[C0][col] = 1; col += 1
            U[A1][col] = 1; V[B10][col] = 1; W[C0][col] = 1; col += 1
            U[A0][col] = 1; V[B01][col] = 1; W[C1][col] = 1; col += 1
            U[A1][col] = 1; V[B11][col] = 1; W[C1][col] = 1; col += 1

        def _add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A00 = _idx_u(a_r, a_c); A01 = _idx_u(a_r, a_c + 1)
            A10 = _idx_u(a_r + 1, a_c); A11 = _idx_u(a_r + 1, a_c + 1)
            B00 = _idx_v(b_r, b_c); B01 = _idx_v(b_r, b_c + 1)
            B10 = _idx_v(b_r + 1, b_c); B11 = _idx_v(b_r + 1, b_c + 1)
            C00 = _idx_w(c_c, c_r); C01 = _idx_w(c_c + 1, c_r)
            C10 = _idx_w(c_c, c_r + 1); C11 = _idx_w(c_c + 1, c_r + 1)

            U[A00][col]=1; U[A11][col]=1; V[B00][col]=1; V[B11][col]=1
            W[C00][col]=1; W[C11][col]=1; col += 1
            U[A10][col]=1; U[A11][col]=1; V[B00][col]=1
            W[C10][col]=1; W[C11][col]=-1; col += 1
            U[A00][col]=1; V[B01][col]=1; V[B11][col]=-1
            W[C01][col]=1; W[C11][col]=1; col += 1
            U[A11][col]=1; V[B10][col]=1; V[B00][col]=-1
            W[C00][col]=1; W[C10][col]=1; col += 1
            U[A00][col]=1; U[A01][col]=1; V[B11][col]=1
            W[C00][col]=-1; W[C01][col]=1; col += 1
            U[A10][col]=1; U[A00][col]=-1; V[B00][col]=1; V[B01][col]=1
            W[C11][col]=1; col += 1
            U[A01][col]=1; U[A11][col]=-1; V[B10][col]=1; V[B11][col]=1
            W[C00][col]=1; col += 1

        for j in range(3):
            bc = 2 * j
            _add_1x2_2x2(0, 0, 0, bc, 0, 2 * j)
            _add_1x2_2x2(0, 2, 2, bc, 0, 2 * j)
            _add_strassen(1, 0, 0, bc, 1, 2 * j)
            _add_strassen(1, 2, 2, bc, 1, 2 * j)

        return U, V, W, col

    U, V, W, rank = _build_block_decomp()

    # Convert to numpy for tensor operations
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Build the true tensor T[i,k,j] = delta_{i,k,j} for matmul
    # T is indexed: A[i,k] * B[k,j] -> C[i,j]
    # In our convention: U row = i*m+k, V row = k*p+j, W row = j*n+i
    # T[U_idx, V_idx, W_idx] = 1 if (i,k,j) forms valid matmul, else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T[i*m+k, k*p+j, j*n+i] = 1.0

    def _residual(Ua, Va, Wa):
        """Compute Frobenius norm of residual T - sum_r Ua[:,r]⊗Va[:,r]⊗Wa[:,r]."""
        recon = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
        return np.linalg.norm(T - recon)

    # Verify our decomposition is exact
    assert _residual(U_np, V_np, W_np) < 1e-10, "Block decomposition not exact!"

    # Try to reduce rank by removing redundant columns
    # A column r is redundant if removing it can be compensated by adjusting others
    # Strategy: try removing each column and check if residual stays 0
    # (This only works if there's linear dependence in the rank-1 terms)

    # More powerful: try pairwise column merging
    # If columns r1 and r2 have U,V,W that are linearly related,
    # we might merge them into one column with a scaled entry.

    best_U, best_V, best_W = U_np, V_np, W_np
    best_rank = rank

    # Try to find if any single column can be removed (zero residual after removal)
    for r in range(rank):
        if time.time() > deadline - 20:
            break
        Ua = U_np[:, :r] + U_np[:, r+1:] if r < rank-1 else U_np[:, :r]
        Va = V_np[:, :r] + V_np[:, r+1:] if r < rank-1 else V_np[:, :r]
        Wa = W_np[:, :r] + W_np[:, r+1:] if r < rank-1 else W_np[:, :r]
        # Actually need to concatenate properly
        if r < rank - 1:
            Ua = np.concatenate([U_np[:, :r], U_np[:, r+1:]], axis=1)
            Va = np.concatenate([V_np[:, :r], V_np[:, r+1:]], axis=1)
            Wa = np.concatenate([W_np[:, :r], W_np[:, r+1:]], axis=1)
        else:
            Ua = U_np[:, :r]
            Va = V_np[:, :r]
            Wa = W_np[:, :r]
        res = _residual(Ua, Va, Wa)
        if res < 1e-10:
            best_U, best_V, best_W = Ua, Va, Wa
            best_rank = rank - 1
            U_np, V_np, W_np = Ua, Va, Wa
            break  # restart from reduced

    # Try HALS-style: for a target rank R < best_rank,
    # initialize from the best_rank decomposition (drop columns) and
    # run alternating least squares to see if we can reach zero residual.
    for target_rank in range(best_rank - 1, max(best_rank - 8, 55), -1):
        if time.time() > deadline - 30:
            break

        # Initialize: take first target_rank columns
        Ua = U_np[:, :target_rank].copy()
        Va = V_np[:, :target_rank].copy()
        Wa = W_np[:, :target_rank].copy()

        # Run HALS iterations
        converged = False
        for it in range(200):
            if time.time() > deadline - 30:
                break

            # Update U: for each row i, solve for U[i,:] given V,W
            # T[i,:,:] = sum_r U[i,r] * V[:,r] ⊗ W[:,r]
            # Flatten: T[i] (vec) = sum_r U[i,r] * (V[:,r] ⊗ W[:,r])
            # So U[i,:] = (V⊗W)^T * T[i]  (least squares)

            # Build design matrix: each column r is V[:,r] ⊗ W[:,r]
            # Shape: (m*p * p*n, target_rank) = (24*18, target_rank) = (432, target_rank)
            design = np.zeros((m*p * p*n, target_rank))
            for r in range(target_rank):
                design[:, r] = np.kron(Va[:, r], Wa[:, r])

            # For each i in U rows:
            for i in range(n*m):
                Ti = T[i, :, :].flatten()
                # Solve design^T design * U[i,:] = design^T Ti
                try:
                    sol = np.linalg.lstsq(design, Ti, rcond=None)[0]
                    Ua[i, :] = sol
                except:
                    pass

            # Update V similarly
            design_v = np.zeros((n*m * p*n, target_rank))
            for r in range(target_rank):
                design_v[:, r] = np.kron(Ua[:, r], Wa[:, r])
            for j in range(m*p):
                Tj = T[:, j, :].flatten()
                try:
                    sol = np.linalg.lstsq(design_v, Tj, rcond=None)[0]
                    Va[j, :] = sol
                except:
                    pass

            # Update W
            design_w = np.zeros((n*m * m*p, target_rank))
            for r in range(target_rank):
                design_w[:, r] = np.kron(Ua[:, r], Va[:, r])
            for k in range(p*n):
                Tk = T[:, :, k].flatten()
                try:
                    sol = np.linalg.lstsq(design_w, Tk, rcond=None)[0]
                    Wa[k, :] = sol
                except:
                    pass

            res = _residual(Ua, Va, Wa)
            if res < 1e-10:
                converged = True
                break

        if converged:
            best_U, best_V, best_W = Ua, Va, Wa
            best_rank = target_rank
            break

    # Convert back to lists with integer/half-integer rounding
    def _round_matrix(M):
        """Round to nearest integer or half-integer."""
        M_out = []
        for row in M:
            M_out.append([int(round(2*x)/2) for x in row])
        return M_out

    U_list = _round_matrix(best_U)
    V_list = _round_matrix(best_V)
    W_list = _round_matrix(best_W)

    # Verify exactness after rounding
    U_chk = np.array(U_list, dtype=float)
    V_chk = np.array(V_list, dtype=float)
    W_chk = np.array(W_list, dtype=float)
    res = _residual(U_chk, V_chk, W_chk)

    if res < 1e-10:
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to original block decomposition
        U_orig, V_orig, W_orig, _ = _build_block_decomp()
        return {"U": U_orig, "V": V_orig, "W": W_orig}
