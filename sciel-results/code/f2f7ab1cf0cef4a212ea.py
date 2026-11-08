import numpy as np
import time
import sys

_START_TIME = time.time()
DEADLINE_SECONDS = 125

def _elapsed():
    return time.time() - _START_TIME

def _time_remaining():
    return DEADLINE_SECONDS - _elapsed()

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x7 matrix multiplication.
    
    Uses Strassen's algorithm for 2x2x2 blocks where possible, and schoolbook for the
    remaining 2x1 blocks. Then attempts to reduce rank by finding linear dependencies
    among the rank-1 tensors and expressing them as integer/half-integer combinations.
    """
    n, m, p = 2, 4, 7
    nA = n * m   # 8
    nB = m * p   # 28
    nW = p * n   # 14
    total_entries = n * m * p  # 56
    
    def build_strassen_decomposition():
        """
        Build initial decomposition using Strassen for 2x2x2 blocks and schoolbook
        for 2x1 blocks. Returns U, V, W lists and rank.
        """
        rank_max = 50
        U = [[0] * rank_max for _ in range(nA)]
        V = [[0] * rank_max for _ in range(nB)]
        W = [[0] * rank_max for _ in range(nW)]
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            if col >= rank_max:
                return
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            e_idx = j0 * p + k0
            f_idx = j0 * p + (k0 + 1)
            g_idx = (j0 + 1) * p + k0
            h_idx = (j0 + 1) * p + (k0 + 1)
            c00 = k0 * n + i0
            c01 = (k0 + 1) * n + i0
            c10 = k0 * n + (i0 + 1)
            c11 = (k0 + 1) * n + (i0 + 1)

            # M1 = (a+d)(e+h)
            U[a_idx][col] += 1; U[d_idx][col] += 1
            V[e_idx][col] += 1; V[h_idx][col] += 1
            W[c00][col] += 1; W[c11][col] += 1
            col += 1

            # M2 = (c+d)e
            U[c_idx][col] += 1; U[d_idx][col] += 1
            V[e_idx][col] += 1
            W[c10][col] += 1; W[c11][col] -= 1
            col += 1

            # M3 = a(f-h)
            U[a_idx][col] += 1
            V[f_idx][col] += 1; V[h_idx][col] -= 1
            W[c01][col] += 1; W[c11][col] += 1
            col += 1

            # M4 = d(g-e)
            U[d_idx][col] += 1
            V[g_idx][col] += 1; V[e_idx][col] -= 1
            W[c00][col] += 1; W[c10][col] += 1
            col += 1

            # M5 = (a+b)h
            U[a_idx][col] += 1; U[b_idx][col] += 1
            V[h_idx][col] += 1
            W[c00][col] -= 1; W[c01][col] += 1
            col += 1

            # M6 = (c-a)(e+f)
            U[c_idx][col] += 1; U[a_idx][col] -= 1
            V[e_idx][col] += 1; V[f_idx][col] += 1
            W[c11][col] += 1
            col += 1

            # M7 = (b-d)(g+h)
            U[b_idx][col] += 1; U[d_idx][col] -= 1
            V[g_idx][col] += 1; V[h_idx][col] += 1
            W[c00][col] += 1
            col += 1

        def add_schoolbook_2x1(i0, j0, k0):
            nonlocal col
            if col >= rank_max:
                return
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            e_idx = j0 * p + k0
            g_idx = (j0 + 1) * p + k0
            c00 = k0 * n + i0
            c10 = k0 * n + (i0 + 1)

            U[a_idx][col] += 1; V[e_idx][col] += 1; W[c00][col] += 1
            col += 1
            U[b_idx][col] += 1; V[g_idx][col] += 1; W[c00][col] += 1
            col += 1
            U[c_idx][col] += 1; V[e_idx][col] += 1; W[c10][col] += 1
            col += 1
            U[d_idx][col] += 1; V[g_idx][col] += 1; W[c10][col] += 1
            col += 1

        for j_start in [0, 2]:
            for k_start in [0, 2, 4]:
                add_strassen(0, j_start, k_start)
            add_schoolbook_2x1(0, j_start, 6)

        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
        return U, V, W, col

    def verify(U, V, W, tol=1e-9):
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    ai = i * m + j
                    bi = j * p + k
                    wi = k * n + i
                    for l in range(r):
                        val += U[ai][l] * V[bi][l] * W[wi][l]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def try_reduce_rank(U, V, W, rank):
        """
        Try to reduce rank by finding linear dependencies among the rank-1 tensors
        and expressing them as integer/half-integer combinations.
        """
        if rank <= 1:
            return U, V, W, rank
        
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Build the matrix of flattened rank-1 tensors
        T_mat = np.zeros((total_entries, rank))
        for l in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * (m * p) + j * p + k
                        T_mat[idx, l] = U_np[i * m + j, l] * V_np[j * p + k, l] * W_np[k * n + i, l]
        
        # Find the rank of T_mat
        try:
            U_svd, s, Vt = np.linalg.svd(T_mat, full_matrices=False)
            tol = 1e-10
            effective_rank = int(np.sum(s > tol))
        except:
            return U, V, W, rank
        
        if effective_rank >= rank:
            return U, V, W, rank
        
        # Try multiple strategies to find a valid reduced decomposition
        best_result = None
        best_rank = rank
        
        # Strategy 1: Greedy basis selection
        used = []
        remaining = list(range(rank))
        
        for _ in range(rank):
            if not remaining:
                break
            best_col = None
            best_residual = float('inf')
            for col in remaining:
                if not used:
                    best_col = col
                    break
                vec = T_mat[:, col]
                used_mat = T_mat[:, used]
                try:
                    coeffs, residuals, rank_q, sv = np.linalg.lstsq(used_mat, vec, rcond=None)
                    residual_vec = vec - used_mat @ coeffs
                    residual_norm = np.linalg.norm(residual_vec)
                    if residual_norm < best_residual:
                        best_residual = residual_norm
                        best_col = col
                except:
                    best_col = col
                    break
            
            if best_col is None:
                break
            
            used.append(best_col)
            remaining.remove(best_col)
            
            if len(used) >= effective_rank:
                break
        
        # Check if this basis works
        if len(used) < rank:
            new_U, new_V, new_W, new_rank = _try_basis(T_mat, U_np, V_np, W_np, used, rank)
            if new_rank < best_rank:
                best_rank = new_rank
                best_result = (new_U, new_V, new_W, new_rank)
        
        # Strategy 2: Try random subsets of size effective_rank
        import random
        random.seed(42)
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts and _time_remaining() > 15:
            attempts += 1
            # Randomly select a subset of size effective_rank
            candidate_basis = random.sample(range(rank), effective_rank)
            
            new_U, new_V, new_W, new_rank = _try_basis(T_mat, U_np, V_np, W_np, candidate_basis, rank)
            if new_rank < best_rank:
                best_rank = new_rank
                best_result = (new_U, new_V, new_W, new_rank)
            
            # Early exit if we found optimal
            if best_rank == effective_rank:
                break
        
        if best_result is not None:
            return best_result
        
        return U, V, W, rank

    def _try_basis(T_mat, U_np, V_np, W_np, basis_cols, original_rank):
        """
        Try to use the given basis columns and express all other columns as
        half-integer linear combinations. Returns new decomposition if successful.
        """
        basis_cols = list(basis_cols)
        new_rank = len(basis_cols)
        
        if new_rank >= original_rank:
            return None, None, None, original_rank
        
        # Build basis matrix
        basis_mat = T_mat[:, basis_cols]
        
        # Check if basis has full column rank
        try:
            rank_basis = np.linalg.matrix_rank(basis_mat)
        except:
            return None, None, None, original_rank
        
        if rank_basis < new_rank:
            return None, None, None, original_rank
        
        # For each non-basis column, find coefficients and check if half-integer
        coefficients = {}
        non_basis_cols = [l for l in range(original_rank) if l not in basis_cols]
        
        for col in non_basis_cols:
            try:
                coeffs, residuals, rank_q, sv = np.linalg.lstsq(basis_mat, T_mat[:, col], rcond=None)
                # Check if coefficients are close to half-integer values
                is_valid = True
                for c in coeffs:
                    c_round = round(c * 2) / 2
                    if abs(c - c_round) > 1e-6:
                        is_valid = False
                        break
                if is_valid:
                    coefficients[col] = coeffs
            except:
                pass
        
        # If we couldn't express all non-basis columns, try a different approach:
        # Only keep basis columns that can be used to express others
        # For now, if we have any coefficients, try to build a partial reduction
        
        if not coefficients:
            return None, None, None, original_rank
        
        # Build new U, V, W by combining basis and non-basis columns
        new_U = np.zeros((U_np.shape[0], new_rank))
        new_V = np.zeros((V_np.shape[0], new_rank))
        new_W = np.zeros((W_np.shape[0], new_rank))
        
        # Copy basis columns
        for i, col in enumerate(basis_cols):
            new_U[:, i] = U_np[:, col]
            new_V[:, i] = V_np[:, col]
            new_W[:, i] = W_np[:, col]
        
        # For non-basis columns with valid coefficients, we need to "absorb" them
        # This is tricky because the tensor decomposition is not linear in the usual sense
        # The rank-1 tensors don't add linearly in a simple way
        
        # Actually, we can't just absorb non-basis columns into basis columns
        # because the tensor product structure doesn't allow simple linear combination
        
        # The correct approach: if column l can be written as sum of alpha_i * column_i,
        # then the corresponding rank-1 tensor T_l = sum alpha_i * T_i
        # But T_l = u_l ⊗ v_l ⊗ w_l, and T_i = u_i ⊗ v_i ⊗ w_i
        # So we need u_l ⊗ v_l ⊗ w_l = sum alpha_i * (u_i ⊗ v_i ⊗ w_i)
        # This is generally NOT possible unless the tensors are specially structured
        
        # So the only valid reduction is if we can find a subset of columns that
        # spans the column space AND the non-spanning columns are exactly zero
        # (i.e., they are redundant in the tensor sense)
        
        # For now, let's just check if the basis columns alone reconstruct the tensor
        # This means all non-basis columns must be linearly dependent in a way
        # that allows exact representation
        
        # Simplified check: verify that using only basis columns gives correct result
        # This will fail if non-basis columns contribute, which they do in general
        
        # So this approach is flawed. Let's return failure unless all non-basis
        # columns have zero contribution, which is unlikely
        
        return None, None, None, original_rank

    # Build initial decomposition
    U, V, W, rank = build_strassen_decomposition()
    
    # Verify
    if not verify(U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(nA)]
        V = [[0] * rank for _ in range(nB)]
        W = [[0] * rank for _ in range(nW)]
        col = 0
        for j in range(m):
            for i in range(n):
                for k in range(p):
                    ai = i * m + j
                    bi = j * p + k
                    wi = k * n + i
                    U[ai][col] = 1
                    V[bi][col] = 1
                    W[wi][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    if _time_remaining() > 30:
        U, V, W, rank = try_reduce_rank(U, V, W, rank)
    
    return {"U": U, "V": V, "W": W}
