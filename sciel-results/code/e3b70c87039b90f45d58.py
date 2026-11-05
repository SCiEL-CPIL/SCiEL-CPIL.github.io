import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    n, m, p = DIMENSIONS  # 4, 4, 7

    def build_schoolbook():
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

    def strassen_2x2_block(i_off, j_off, k_off, a_scale=1, b_scale=1, c_scale=1):
        terms = []
        
        # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        terms.append((u, v, w))

        # M2 = (A21+A22)*B11 -> C21 += M2, C22 -= M2
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a_scale
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = c_scale
        w[(k_off+1) * n + (i_off+1)] = -c_scale
        terms.append((u, v, w))

        # M3 = A11*(B12-B22) -> C12 += M3, C22 += M3
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        v = [0] * (m * p)
        v[j_off * p + (k_off+1)] = b_scale
        v[(j_off+1) * p + (k_off+1)] = -b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = c_scale
        w[(k_off+1) * n + (i_off+1)] = c_scale
        terms.append((u, v, w))

        # M4 = A22*(B21-B11) -> C11 += M4, C21 += M4
        u = [0] * (n * m)
        u[(i_off+1) * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = b_scale
        v[j_off * p + k_off] = -b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        w[k_off * n + (i_off+1)] = c_scale
        terms.append((u, v, w))

        # M5 = (A11+A12)*B22 -> C12 += M5, C11 -= M5
        u = [0] * (n * m)
        u[i_off * m + j_off] = a_scale
        u[i_off * m + (j_off+1)] = a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = c_scale
        w[k_off * n + i_off] = -c_scale
        terms.append((u, v, w))

        # M6 = (A21-A11)*(B11+B12) -> C22 += M6
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = a_scale
        u[i_off * m + j_off] = -a_scale
        v = [0] * (m * p)
        v[j_off * p + k_off] = b_scale
        v[j_off * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[(k_off+1) * n + (i_off+1)] = c_scale
        terms.append((u, v, w))

        # M7 = (A12-A22)*(B21+B22) -> C11 += M7
        u = [0] * (n * m)
        u[i_off * m + (j_off+1)] = a_scale
        u[(i_off+1) * m + (j_off+1)] = -a_scale
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = b_scale
        v[(j_off+1) * p + (k_off+1)] = b_scale
        w = [0] * (p * n)
        w[k_off * n + i_off] = c_scale
        terms.append((u, v, w))

        return terms

    def build_4x4x7_optimized():
        all_terms = []
        
        # Part 1: 4x4 * 4x4 (columns 0-3 of B) - 49 terms via recursive Strassen
        outer_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]

        inner_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]

        def get_block_indices(block_idx, local_idx, dim_a, offset_k=0):
            br = block_idx // 2
            bc = block_idx % 2
            lr = local_idx // 2
            lc = local_idx % 2
            row = br * 2 + lr
            col = bc * 2 + lc + offset_k
            if dim_a == 'A':
                return row * m + col
            elif dim_a == 'B':
                return row * p + col
            elif dim_a == 'C':
                return col * n + row

        for outer_a, outer_b, outer_c in outer_terms:
            for inner_x, inner_y, inner_z in inner_terms:
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                for x_local, x_coeff in inner_x:
                    for a_block, a_coeff in outer_a:
                        idx = get_block_indices(a_block, x_local, 'A')
                        u[idx] += a_coeff * x_coeff
                for y_local, y_coeff in inner_y:
                    for b_block, b_coeff in outer_b:
                        idx = get_block_indices(b_block, y_local, 'B')
                        v[idx] += b_coeff * y_coeff
                for z_local, z_coeff in inner_z:
                    for c_block, c_coeff in outer_c:
                        idx = get_block_indices(c_block, z_local, 'C')
                        w[idx] += c_coeff * z_coeff
                all_terms.append((u, v, w))

        # Part 2: 4x4 * 4x2 (columns 4-5 of B) - 28 terms
        all_terms.extend(strassen_2x2_block(0, 0, 4))
        all_terms.extend(strassen_2x2_block(0, 2, 4))
        all_terms.extend(strassen_2x2_block(2, 0, 4))
        all_terms.extend(strassen_2x2_block(2, 2, 4))
        
        # Part 3: 4x4 * 4x1 (column 6 of B) - 16 terms
        for i in range(n):
            for j in range(m):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + 6] = 1
                w = [0] * (p * n)
                w[6 * n + i] = 1
                all_terms.append((u, v, w))

        rank = len(all_terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for c, (u, v, w) in enumerate(all_terms):
            for r in range(n * m):
                if u[r] != 0:
                    U[r][c] = u[r]
            for r in range(m * p):
                if v[r] != 0:
                    V[r][c] = v[r]
            for r in range(p * n):
                if w[r] != 0:
                    W[r][c] = w[r]
        
        return {"U": U, "V": V, "W": W}

    def try_rank_reduction(U_list, V_list, W_list):
        """
        Attempt to reduce rank by finding linear dependencies.
        Uses vectorized residual computation for speed.
        Returns (U_reduced, V_reduced, W_reduced) or None if no reduction found.
        """
        if time.time() - start_time > DEADLINE - 15:
            return None
        
        U = np.array(U_list, dtype=float)
        V = np.array(V_list, dtype=float)
        W = np.array(W_list, dtype=float)
        rank = U.shape[1]
        
        target = np.ones((n, m, p), dtype=float)
        
        def compute_residual(U_arr, V_arr, W_arr):
            """Compute the residual tensor = target - decomposition."""
            r = U_arr.shape[1]
            if r == 0:
                return target.copy()
            U_r = U_arr.reshape(n, m, r)
            V_r = V_arr.reshape(m, p, r)
            W_r = W_arr.reshape(p, n, r)
            tensor_decomp = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            return target - tensor_decomp
        
        def term_contribution(idx, U_arr, V_arr, W_arr):
            """Compute the contribution of a single term to the tensor."""
            U_r = U_arr[:, idx].reshape(n, m)
            V_r = V_arr[:, idx].reshape(m, p)
            W_r = W_arr[:, idx].reshape(p, n)
            return U_r[None, :, :] * V_r[:, None, :] * W_r[None, None, :]
        
        improved = False
        max_iters = 50
        iter_count = 0
        
        # Multiple passes of greedy removal using residual tracking
        for _pass in range(5):
            if time.time() - start_time > DEADLINE - 5:
                break
            if rank <= 1:
                break
                
            # Compute current residual
            residual = compute_residual(U, V, W)
            if np.allclose(residual, 0, atol=1e-9):
                # Try to remove terms one by one
                removed_in_pass = 0
                for remove_idx in range(rank - 1, -1, -1):
                    if time.time() - start_time > DEADLINE - 5:
                        return None
                    if iter_count >= max_iters:
                        break
                    iter_count += 1
                    
                    # Compute contribution of this term
                    contrib = term_contribution(remove_idx, U, V, W)
                    
                    # If removing this term, new residual = residual + contrib
                    new_residual = residual + contrib
                    if np.allclose(new_residual, 0, atol=1e-9):
                        # Safe to remove
                        U = np.delete(U, remove_idx, axis=1)
                        V = np.delete(V, remove_idx, axis=1)
                        W = np.delete(W, remove_idx, axis=1)
                        rank -= 1
                        improved = True
                        removed_in_pass += 1
                        residual = new_residual
                        # Shift indices: all subsequent indices are now shifted down
                        # But we're iterating backwards, so this is fine
                        break  # Restart the pass since indices changed
                
                if removed_in_pass > 0:
                    continue  # Start a new pass since indices changed
                else:
                    break  # No more removable terms in this pass
        
        if improved and rank < U_list[0].__len__():
            U_new = [[0] * rank for _ in range(n * m)]
            V_new = [[0] * rank for _ in range(m * p)]
            W_new = [[0] * rank for _ in range(p * n)]
            
            for c in range(rank):
                for r in range(n * m):
                    if U[r, c] != 0:
                        U_new[r][c] = U[r, c]
                for r in range(m * p):
                    if V[r, c] != 0:
                        V_new[r][c] = V[r, c]
                for r in range(p * n):
                    if W[r, c] != 0:
                        W_new[r][c] = W[r, c]
            
            return U_new, V_new, W_new
        
        return None

    if time.time() - start_time > DEADLINE:
        return build_schoolbook()

    try:
        result = build_4x4x7_optimized()
        
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        rank = U.shape[1]
        
        # Verify the decomposition
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        tensor_decomp = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        
        target = np.ones((n, m, p), dtype=float)
        if not np.allclose(tensor_decomp, target, atol=1e-9):
            return build_schoolbook()
        
        # Try to reduce rank
        reduced = try_rank_reduction(result["U"], result["V"], result["W"])
        
        if reduced is not None:
            U_red, V_red, W_red = reduced
            # Verify reduced decomposition
            U_red_arr = np.array(U_red, dtype=float)
            V_red_arr = np.array(V_red, dtype=float)
            W_red_arr = np.array(W_red, dtype=float)
            rank_red = U_red_arr.shape[1]
            
            U_red_r = U_red_arr.reshape(n, m, rank_red)
            V_red_r = V_red_arr.reshape(m, p, rank_red)
            W_red_r = W_red_arr.reshape(p, n, rank_red)
            
            tensor_decomp_red = np.einsum('ijc,jkc,kic->ijk', U_red_r, V_red_r, W_red_r)
            
            if np.allclose(tensor_decomp_red, target, atol=1e-9):
                return {"U": U_red, "V": V_red, "W": W_red}
        
        return result
    except Exception:
        return build_schoolbook()
