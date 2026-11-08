import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 140  # seconds, safety margin from 150s

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_baseline = 50
    U = [[0] * rank_baseline for _ in range(n * m)]
    V = [[0] * rank_baseline for _ in range(m * p)]
    W = [[0] * rank_baseline for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h)
        U[a_idx][col] += 1; U[d_idx][col] += 1
        V[e_idx][col] += 1; V[h_idx][col] += 1
        W[c00_idx][col] += 1; W[c11_idx][col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[c_idx][col] += 1; U[d_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1; W[c11_idx][col] -= 1
        col += 1
        
        # M3 = a(f-h)
        U[a_idx][col] += 1
        V[f_idx][col] += 1; V[h_idx][col] -= 1
        W[c01_idx][col] += 1; W[c11_idx][col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[d_idx][col] += 1
        V[g_idx][col] += 1; V[e_idx][col] -= 1
        W[c00_idx][col] += 1; W[c10_idx][col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[a_idx][col] += 1; U[b_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] -= 1; W[c01_idx][col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[c_idx][col] += 1; U[a_idx][col] -= 1
        V[e_idx][col] += 1; V[f_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[b_idx][col] += 1; U[d_idx][col] -= 1
        V[g_idx][col] += 1; V[h_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        U[a_idx][col] += 1; V[e_idx][col] += 1; W[c00_idx][col] += 1
        col += 1
        U[b_idx][col] += 1; V[g_idx][col] += 1; W[c00_idx][col] += 1
        col += 1
        U[c_idx][col] += 1; V[e_idx][col] += 1; W[c10_idx][col] += 1
        col += 1
        U[d_idx][col] += 1; V[g_idx][col] += 1; W[c10_idx][col] += 1
        col += 1

    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)
    
    assert col == rank_baseline, f"Expected rank {rank_baseline}, got {col}"
    
    # Convert to numpy arrays for numerical rank reduction
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Verify the baseline is correct
    def verify_decomposition(U_arr, V_arr, W_arr):
        # T[i,j,k] should equal sum_r U[i,r]*V[j,r]*W[k,r]
        # For matrix mult: T[i,j,k] = 1 if j is the inner index connecting A[i,j] and B[j,k]
        # Actually T[i,j,k] = delta_{i,j,k} where the tensor is T[i,j,k] = 1 if the multiplication
        # C[i,k] = sum_j A[i,j] * B[j,k] has term A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for all valid i,j,k (it's the identity-like tensor for matmul)
        # Wait, the matrix multiplication tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # No, that's not right. The tensor is T[i,j,k] where C[i,k] = sum_j A[i,j] B[j,k]
        # So T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
        # The tensor has shape (n*m, m*p, p*n) in the flattened index convention
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # The decomposition is: for each (i,j,k) in the tensor, T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Wait, let me re-read the convention.
        # U[i*m+j, r] corresponds to A[i,j]
        # V[j*p+k, r] corresponds to B[j,k]  
        # W[k*n+i, r] corresponds to C[k,i]... wait, W[k*n+i, r]
        # So W index is k*n + i, meaning W is indexed by (k, i) where k is col of C, i is row of C
        # The tensor T has entries T[a_idx, b_idx, c_idx] where:
        # a_idx = i*m + j (A[i,j])
        # b_idx = j*p + k (B[j,k])
        # c_idx = k*n + i (C[i,k] stored as W[k*n+i])
        # T[a_idx, b_idx, c_idx] = 1 if a_idx, b_idx, c_idx are consistent (same j connects them)
        # i.e., a_idx = i*m+j, b_idx = j*p+k, c_idx = k*n+i
        # For each (i,j,k): a_idx = i*m+j, b_idx = j*p+k, c_idx = k*n+i
        # T should have 1 at (i*m+j, j*p+k, k*n+i)
        
        n_val, m_val, p_val = n, m, p
        T_expected = np.zeros((n_val * m_val, m_val * p_val, p_val * n_val))
        for i in range(n_val):
            for j in range(m_val):
                for k in range(p_val):
                    a_idx = i * m_val + j
                    b_idx = j * p_val + k
                    c_idx = k * n_val + i
                    T_expected[a_idx, b_idx, c_idx] = 1
        
        # Compute T_decomposed
        # T_decomposed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is expensive for large dimensions, but our dims are small
        T_decomposed = np.zeros((n_val * m_val, m_val * p_val, p_val * n_val))
        for r in range(U_arr.shape[1]):
            u_col = U_arr[:, r]
            v_col = V_arr[:, r]
            w_col = W_arr[:, r]
            # Outer product of three vectors
            # T_decomposed[a,b,c] += u_col[a] * v_col[b] * w_col[c]
            T_decomposed += np.einsum('a,b,c->abc', u_col, v_col, w_col)
        
        return np.allclose(T_decomposed, T_expected, atol=1e-10)
    
    assert verify_decomposition(U_np, V_np, W_np), "Baseline decomposition is incorrect!"
    
    # Try to reduce rank by finding linear dependencies
    # The rank of the tensor is the minimum number of terms needed
    # We can try to find if any column is a linear combination of others
    
    # Approach: For each subset of columns, check if removing one column
    # still gives a valid decomposition (i.e., the removed column's contribution
    # can be absorbed by others)
    
    # More practical: Use the fact that if U, V, W have linearly dependent columns,
    # we can reduce rank. But the tensor rank is not simply the matrix rank.
    
    # Let's try a different approach: enumerate all possible 3-tensor decompositions
    # with rank < 50 and check if any is valid.
    
    # Actually, let's try to use the structure. The tensor T for (2,4,7) matmul
    # has a specific structure. Let me think about lower bounds.
    
    # For (n,m,p) matmul, the tensor rank is at least max(n*m, m*p, p*n) / something
    # Actually, a simple lower bound is max(n*m, m*p, p*n) since each term can
    # contribute to at most one entry in each mode.
    
    # n*m = 8, m*p = 28, p*n = 14
    # So rank >= 28? No, that's not a valid lower bound.
    
    # Let me try a greedy approach: start with rank 50 and try to remove terms
    # while maintaining correctness.
    
    best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
    best_rank = rank_baseline
    
    # Try removing each column and see if the decomposition still works
    # This is O(rank * verification_cost) which might be too slow
    # Let's try a smarter approach
    
    # Check if any two columns can be merged (their sum is zero or one is a scalar multiple)
    for r in range(rank_baseline):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        # Try removing column r
        U_test = np.delete(U_np, r, axis=1)
        V_test = np.delete(V_np, r, axis=1)
        W_test = np.delete(W_np, r, axis=1)
        
        if verify_decomposition(U_test, V_test, W_test):
            best_U, best_V, best_W = U_test, V_test, W_test
            best_rank = rank_baseline - 1
            U_np, V_np, W_np = U_test, V_test, W_test
            rank_baseline -= 1
            # Continue trying to remove more
            r = -1  # Reset to start checking from beginning
    
    # Try merging pairs of columns that are linearly dependent
    # If col_i = alpha * col_j in all three matrices, we can merge them
    for i in range(U_np.shape[1]):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        for j in range(i + 1, U_np.shape[1]):
            # Check if U[:,i] = alpha * U[:,j] for some alpha
            # and similarly for V and W
            try:
                # Find alpha from U
                mask_u = np.abs(U_np[:, j]) > 1e-10
                if np.any(mask_u):
                    alpha_u = U_np[mask_u, i] / U_np[mask_u, j]
                    if np.allclose(alpha_u, alpha_u[0], atol=1e-10):
                        alpha = alpha_u[0]
                        # Check V
                        mask_v = np.abs(V_np[:, j]) > 1e-10
                        if np.any(mask_v):
                            alpha_v = V_np[mask_v, i] / V_np[mask_v, j]
                            if np.allclose(alpha_v, alpha_v[0], atol=1e-10) and np.isclose(alpha_v[0], alpha, atol=1e-10):
                                # Check W
                                mask_w = np.abs(W_np[:, j]) > 1e-10
                                if np.any(mask_w):
                                    alpha_w = W_np[mask_w, i] / W_np[mask_w, j]
                                    if np.allclose(alpha_w, alpha_w[0], atol=1e-10) and np.isclose(alpha_w[0], alpha, atol=1e-10):
                                        # Columns i and j are proportional, can merge
                                        # Remove column i (keep j)
                                        U_test = np.delete(U_np, i, axis=1)
                                        V_test = np.delete(V_np, i, axis=1)
                                        W_test = np.delete(W_np, i, axis=1)
                                        if verify_decomposition(U_test, V_test, W_test):
                                            best_U, best_V, best_W = U_test, V_test, W_test
                                            best_rank = U_np.shape[1] - 1
                                            U_np, V_np, W_np = U_test, V_test, W_test
                                            break
            except:
                pass
        else:
            continue
        break
    
    # Try to reduce further by checking if we can find a rank-deficient
    # decomposition using SVD on the tensor slices
    
    # For each pair of modes, compute the matrix and check its rank
    # T_{(a,b)}[c, r] = sum over... this is getting complex
    
    # Let's try a final approach: use the fact that we can perturb the
    # decomposition slightly and see if a lower rank works
    
    # Convert back to list format
    U_final = best_U.tolist()
    V_final = best_V.tolist()
    W_final = best_W.tolist()
    
    # Round to nearest integer or half-integer if close
    for arr in [U_final, V_final, W_final]:
        for row in arr:
            for idx in range(len(row)):
                val = row[idx]
                if abs(val - round(val)) < 1e-10:
                    row[idx] = int(round(val))
                elif abs(val * 2 - round(val * 2)) < 1e-10:
                    row[idx] = round(val * 2) / 2
    
    return {"U": U_final, "V": V_final, "W": W_final}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
