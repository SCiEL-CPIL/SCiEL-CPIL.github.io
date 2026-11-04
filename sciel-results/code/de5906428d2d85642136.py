import time
import numpy as np
from fractions import Fraction
import scipy.linalg

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12

    start_time = time.time()
    deadline = start_time + 110

    def build_rank44():
        rank = 44
        U = [[Fraction(0)] * rank for _ in range(dim_u)]
        V = [[Fraction(0)] * rank for _ in range(dim_v)]
        W = [[Fraction(0)] * rank for _ in range(dim_w)]

        col = 0
        def add_strassen(X_u, Y_v, Z_w, col):
            U[X_u[0]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            U[X_u[2]][col] += 1; U[X_u[3]][col] += 1
            V[Y_v[0]][col] += 1
            W[Z_w[2]][col] += 1; W[Z_w[3]][col] -= 1
            col += 1
            U[X_u[0]][col] += 1
            V[Y_v[1]][col] += 1; V[Y_v[3]][col] -= 1
            W[Z_w[1]][col] += 1; W[Z_w[3]][col] += 1
            col += 1
            U[X_u[3]][col] += 1
            V[Y_v[2]][col] += 1; V[Y_v[0]][col] -= 1
            W[Z_w[0]][col] += 1; W[Z_w[2]][col] += 1
            col += 1
            U[X_u[0]][col] += 1; U[X_u[1]][col] += 1
            V[Y_v[3]][col] += 1
            W[Z_w[0]][col] -= 1; W[Z_w[1]][col] += 1
            col += 1
            U[X_u[2]][col] += 1; U[X_u[0]][col] -= 1
            V[Y_v[0]][col] += 1; V[Y_v[1]][col] += 1
            W[Z_w[3]][col] += 1
            col += 1
            U[X_u[1]][col] += 1; U[X_u[3]][col] -= 1
            V[Y_v[2]][col] += 1; V[Y_v[3]][col] += 1
            W[Z_w[0]][col] += 1
            col += 1
            return col

        # Four 2x2x2 Strassen blocks
        col = add_strassen([0, 1, 4, 5], [0, 1, 4, 5], [0, 3, 1, 4], col)
        col = add_strassen([2, 3, 6, 7], [8, 9, 12, 13], [0, 3, 1, 4], col)
        col = add_strassen([0, 1, 4, 5], [2, 3, 6, 7], [6, 9, 7, 10], col)
        col = add_strassen([2, 3, 6, 7], [10, 11, 14, 15], [6, 9, 7, 10], col)
        
        # Schoolbook for the 3rd row of A
        for j in range(4):
            U[8+j][col] += 1; V[j*4+0][col] += 1; W[2][col] += 1
            col += 1
            U[8+j][col] += 1; V[j*4+1][col] += 1; W[5][col] += 1
            col += 1
        for j in range(4):
            U[8+j][col] += 1; V[j*4+2][col] += 1; W[8][col] += 1
            col += 1
            U[8+j][col] += 1; V[j*4+3][col] += 1; W[11][col] += 1
            col += 1

        return U, V, W, col

    # Precompute expected tensor
    T_expected = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1.0

    def verify_frac(U, V, W):
        U_arr = np.array([[float(x) for x in row] for row in U], dtype=np.float64)
        V_arr = np.array([[float(x) for x in row] for row in V], dtype=np.float64)
        W_arr = np.array([[float(x) for x in row] for row in W], dtype=np.float64)
        
        if U_arr.size == 0:
            return np.allclose(np.zeros_like(T_expected), T_expected, atol=1e-10, rtol=1e-10)
            
        T_actual = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
        return np.allclose(T_actual, T_expected, atol=1e-10, rtol=1e-10)

    U, V, W, actual_rank = build_rank44()
    
    if not verify_frac(U, V, W):
        # Fallback to schoolbook
        rank = 48
        U = [[0.0]*rank for _ in range(dim_u)]
        V = [[0.0]*rank for _ in range(dim_v)]
        W = [[0.0]*rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Convert to float arrays for linear algebra
    U_np = np.array([[float(x) for x in row] for row in U], dtype=np.float64)
    V_np = np.array([[float(x) for x in row] for row in V], dtype=np.float64)
    W_np = np.array([[float(x) for x in row] for row in W], dtype=np.float64)
    
    # Check for linear dependencies among the rank terms
    # We want to find c such that sum_r c_r U_r V_r W_r = 0
    # This is equivalent to finding the null space of the matrix M where M_{(u,v,w), r} = U_{u,r} V_{v,r} W_{w,r}
    
    if actual_rank > 1:
        # Reshape U, V, W to (dim, rank)
        # M is (dim_u * dim_v * dim_w) x rank
        # To save memory, we can sample or use SVD on the tensor directly?
        # No, the tensor is small (2304 x 44).
        
        U_flat = U_np.flatten() # No, U is (12, 44)
        # We need to construct M
        # M[i, r] where i is index in (u,v,w) space
        
        # Efficient construction:
        # M = np.einsum('ur,vr,wr->uvwr', U_np, V_np, W_np).reshape(-1, actual_rank)
        
        if time.time() < deadline:
            M = np.einsum('ur,vr,wr->uvwr', U_np, V_np, W_np).reshape(-1, actual_rank)
            
            # Find null space
            # SVD is more robust
            try:
                _, s, vh = np.linalg.svd(M, full_matrices=False)
                # Null space corresponds to singular values close to 0
                if len(s) > 0 and s[0] > 0:
                    tol = 1e-10 * s[0]
                    rank_M = np.sum(s > tol)
                else:
                    rank_M = 0
                
                if rank_M < actual_rank and rank_M > 0:
                    # The rank-1 tensors T_r are linearly dependent.
                    # We can find a basis for their span by selecting independent columns.
                    # Use the right singular vectors: the first rank_M rows of vh span the column space.
                    # New factor matrices: U_new = U_np @ vh[:rank_M].T, etc.
                    # Then T_new = sum_{r'<rank_M} U_new[:,r'] V_new[:,r'] W_new[:,r']
                    #           = sum_{r'<rank_M} sum_r U[:,r] vh[r',r] * sum_r' V[:,r'] vh[r',r'] * ... 
                    # Actually: U_new[:,r'] = sum_r U[:,r] vh[r',r]
                    # So sum_{r'} U_new[:,r'] V_new[:,r'] W_new[:,r'] 
                    #   = sum_{r'} (sum_r U_r vh[r',r]) (sum_s V_s vh[r',s]) (sum_t W_t vh[r',t])
                    # This is NOT the same as sum_r U_r V_r W_r in general!
                    # The SVD of M gives a basis for the TENSOR space, not a valid factorization.
                    # We need a different approach for factor reduction.
                    pass
            except:
                pass

    # Perform greedy removal, which is safe and will now find the dependencies faster if we just loop
    # But we already know there are dependencies.
    
    current_U = U_np
    current_V = V_np
    current_W = W_np
    current_rank = actual_rank
    
    # Phase 1: Greedy single-term removal
    max_removals = 15
    removed = 0
    while removed < max_removals and current_rank > 1:
        if time.time() > deadline:
            break
        found = False
        for r_idx in range(current_rank):
            keep = [i for i in range(current_rank) if i != r_idx]
            
            U_test = current_U[:, keep]
            V_test = current_V[:, keep]
            W_test = current_W[:, keep]
            
            T_test = np.einsum('ir,jr,kr->ijk', U_test, V_test, W_test)
            if np.allclose(T_test, T_expected, atol=1e-10, rtol=1e-10):
                current_U = U_test
                current_V = V_test
                current_W = W_test
                current_rank -= 1
                removed += 1
                found = True
                break
        if not found:
            break
    
    # Phase 2: Try to merge pairs of terms into a single term
    # If T_a + T_b can be expressed as a single rank-1 tensor, we can reduce rank by 1.
    # More generally, if there exist coefficients c_a, c_b, c_c such that
    # c_a*T_a + c_b*T_b = c_c*T_c (for some existing or new term), we can merge.
    # A simpler heuristic: try all pairs (a,b) and check if T_a + T_b - T_c = 0 for some c != a,b.
    # This is O(n^3) but n is small.
    if current_rank >= 3 and time.time() < deadline:
        merge_found = True
        while merge_found and current_rank >= 3:
            if time.time() > deadline:
                break
            merge_found = False
            for a in range(current_rank):
                if time.time() > deadline:
                    break
                for b in range(a + 1, current_rank):
                    if time.time() > deadline:
                        break
                    # Compute T_a + T_b
                    T_ab = np.einsum('ir,jr,kr->ijk', 
                                     current_U[:, [a, b]], 
                                     current_V[:, [a, b]], 
                                     current_W[:, [a, b]])
                    # Try to subtract each other term c
                    for c in range(current_rank):
                        if c == a or c == b:
                            continue
                        T_ab_minus_c = T_ab - np.einsum('ir,jr,kr->ijk',
                                                        current_U[:, [c]],
                                                        current_V[:, [c]],
                                                        current_W[:, [c]])
                        if np.allclose(T_ab_minus_c, np.zeros_like(T_expected), atol=1e-10, rtol=1e-10):
                            # T_a + T_b = T_c, so we can remove a and b, keep c
                            keep = [i for i in range(current_rank) if i != a and i != b]
                            current_U = current_U[:, keep]
                            current_V = current_V[:, keep]
                            current_W = current_W[:, keep]
                            current_rank -= 2
                            merge_found = True
                            break
                    if merge_found:
                        break
                if merge_found:
                    break
    
    # Phase 3: Try linear combinations of two terms to replace one term
    # If T_a + alpha*T_b can be expressed as beta*T_c for some c, we can merge.
    # This is more expensive, so only do it if we have time.
    if current_rank >= 3 and time.time() < deadline - 20:
        # Try small integer/rational coefficients
        for alpha_num in range(-4, 5):
            if alpha_num == 0:
                continue
            for a in range(current_rank):
                if time.time() > deadline:
                    break
                for b in range(current_rank):
                    if b == a:
                        continue
                    if time.time() > deadline:
                        break
                    # T_a + alpha_num * T_b
                    T_combo = (np.einsum('ir,jr,kr->ijk', 
                                         current_U[:, [a]], 
                                         current_V[:, [a]], 
                                         current_W[:, [a]])
                               + alpha_num * np.einsum('ir,jr,kr->ijk',
                                                       current_U[:, [b]],
                                                       current_V[:, [b]],
                                                       current_W[:, [b]]))
                    # Check if this equals beta * T_c for some c and beta in {-4..4}
                    for c in range(current_rank):
                        if c == a or c == b:
                            continue
                        T_c = np.einsum('ir,jr,kr->ijk',
                                        current_U[:, [c]],
                                        current_V[:, [c]],
                                        current_W[:, [c]])
                        for beta_num in range(-4, 5):
                            if beta_num == 0:
                                continue
                            if np.allclose(T_combo - beta_num * T_c, np.zeros_like(T_expected), atol=1e-10, rtol=1e-10):
                                # Replace term c with T_a + alpha*T_b, remove a and b
                                # New term c: U_c' = U_a + alpha*U_b, V_c' = V_a + alpha*V_b, W_c' = W_a + alpha*W_b
                                # But we need beta*T_c = T_a + alpha*T_b, so:
                                # U_c_new = (1/beta)*(U_a + alpha*U_b), etc.
                                # Only valid if beta divides evenly (for half-integer constraint)
                                keep = [i for i in range(current_rank) if i != a and i != b]
                                c_idx_in_keep = keep.index(c)
                                new_U = current_U[:, keep].copy()
                                new_V = current_V[:, keep].copy()
                                new_W = current_W[:, keep].copy()
                                new_U[:, c_idx_in_keep] = (current_U[:, a] + alpha_num * current_U[:, b]) / beta_num
                                new_V[:, c_idx_in_keep] = (current_V[:, a] + alpha_num * current_V[:, b]) / beta_num
                                new_W[:, c_idx_in_keep] = (current_W[:, a] + alpha_num * current_W[:, b]) / beta_num
                                # Verify
                                T_verify = np.einsum('ir,jr,kr->ijk', new_U, new_V, new_W)
                                if np.allclose(T_verify, T_expected, atol=1e-10, rtol=1e-10):
                                    current_U = new_U
                                    current_V = new_V
                                    current_W = new_W
                                    current_rank -= 2
                                    break
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
    
    return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}
