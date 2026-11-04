import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    nm, mp, pn = n * m, m * p, p * n

    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    def verify_solution(U, V, W, T, tol=1e-9):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T, atol=tol)

    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    deadline = time.time() + 135

    # Strategy: Try a structured decomposition based on the observation that
    # the 3x9 block can be viewed as three 3x3 blocks.
    # For each i in {0, 1}, the slice T[i, :, :] is a 3x9 matrix.
    # T[i, j, k] = 1 if k*n+i corresponds to the output index.
    # Specifically, for fixed i, the slice is a 3x9 matrix S_i where S_i[j, k] = 1 if (k*n+i) is the output row.
    # Wait, T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
    # For fixed i, T[i*m+j, j*p+k, k*n+i] = 1.
    # This means for each i, the "slice" of the tensor along the first dimension (for all j,k) is not a standard matrix multiplication slice.
    # However, we can try to decompose using the fact that m=3 is small.
    
    # Let's try a specific constructive decomposition.
    # We know that the rank of matrix multiplication M_{n,m,p} is at most n*m*p.
    # For n=2, m=3, p=9.
    # Can we do better?
    # The rank of M_{2,3,3} is 16.
    # If we decompose the 9 columns into three groups of 3, we might be able to reuse the M_{2,3,3} decomposition.
    # Let B be 3x9. We can write B = [B1, B2, B3] where Bj is 3x3.
    # A B = [A B1, A B2, A B3].
    # If we have a rank-16 decomposition for M_{2,3,3}, we can use it for each block.
    # Total rank would be 3 * 16 = 48.
    # This is less than 54.
    
    # Let's try to find a rank-16 decomposition for M_{2,3,3}.
    # Actually, it is known that rank(M_{2,3,3}) = 16.
    # We can search for this or use a known one.
    # Since we have time, we can try to find a low-rank decomposition for M_{2,3,3} and then combine them.
    
    # For now, let's just use the 48-rank construction as a solid improvement over 54.
    # We need a rank-16 decomposition for 2x3x3.
    # Let's try to find one using the same method as the original, but for 2,3,3.
    
    def find_decomposition(n_dim, m_dim, p_dim, target_rank, deadline_inner):
        nm_d, mp_d, pn_d = n_dim * m_dim, m_dim * p_dim, p_dim * n_dim
        T_d = np.zeros((nm_d, mp_d, pn_d))
        for i in range(n_dim):
            for j in range(m_dim):
                for k in range(p_dim):
                    T_d[i*m_dim+j, j*p_dim+k, k*n_dim+i] = 1.0
        
        def verify_d(U, V, W, T, tol=1e-9):
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            return np.allclose(T_approx, T, atol=tol)

        def is_half_integer(mat, tol=1e-6):
            scaled = 2.0 * mat
            rounded = np.round(scaled)
            if np.all(np.abs(scaled - rounded) < tol):
                return True
            return False

        for r in range(target_rank, n_dim * m_dim * p_dim + 1):
            if time.time() > deadline_inner:
                return None
            for trial in range(100):
                if time.time() > deadline_inner:
                    return None
                np.random.seed(42 + r * 1000 + trial)
                V = np.random.randint(-1, 2, size=(mp_d, r))
                W = np.random.randint(-1, 2, size=(pn_d, r))
                
                M = (V[:, None, :] * W[None, :, :]).reshape(mp_d * pn_d, r)
                
                if np.linalg.matrix_rank(M) < r:
                    continue
                    
                U = np.zeros((nm_d, r))
                valid = True
                for a in range(nm_d):
                    target = T_d[a, :, :].flatten()
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(M, target, rcond=None)
                        if residuals and residuals[0] > 1e-6:
                            valid = False
                            break
                        U[a, :] = sol
                    except:
                        valid = False
                        break
                        
                if not valid:
                    continue
                    
                if is_half_integer(U):
                    U_snapped = np.round(2.0 * U) / 2.0
                    if verify_d(U_snapped, V, W, T_d):
                        return U_snapped, V, W, r
        return None

    # Try to find a rank-16 decomposition for 2x3x3
    deadline_inner = time.time() + 10
    res_233 = find_decomposition(2, 3, 3, 16, deadline_inner)
    
    if res_233 is not None:
        U_233, V_233, W_233, r_233 = res_233
        # Construct rank 3 * r_233 = 48 decomposition for 2x3x9
        new_rank = 3 * r_233
        if new_rank < best_rank:
            U_new = np.zeros((nm, new_rank))
            V_new = np.zeros((mp, new_rank))
            W_new = np.zeros((pn, new_rank))
            
            for block in range(3):
                # The block corresponds to columns block*3 to block*3+2 of B
                # And rows block*3 to block*3+2 of the output C? No.
                # B is 3x9. Let's say B = [B1 B2 B3] where Bj is 3x3.
                # A B = [A B1, A B2, A B3].
                # The output C is 2x9.
                # The decomposition of A B_j uses the 2x3x3 tensor.
                # For block b (0,1,2):
                # Input A is 2x3.
                # Input B_b is 3x3.
                # Output C_b is 2x3.
                # The indices in the global tensor:
                # A index: i*m+j (i=0,1; j=0,1,2) -> 0..5
                # B_b index: j*3 + (k-b*3)? No.
                # The global B index is j*p+k. For block b, the columns are b*3 to b*3+2.
                # So k_global = b*3 + k_local.
                # B index = j*9 + (b*3 + k_local).
                # Output C index: k_global*n + i = (b*3 + k_local)*2 + i.
                
                # So for each local (i, j, k_local) in 2x3x3:
                # U index: i*3+j
                # V index: j*9 + (b*3 + k_local)
                # W index: (b*3 + k_local)*2 + i
                
                # The decomposition for 2x3x3 has rank r_233.
                # U_233 is 6 x r_233
                # V_233 is 9 x r_233  (for 3x3, mp=9? No, for 2x3x3, mp=3*3=9, pn=3*2=6)
                # Wait, for 2x3x3:
                # nm = 6, mp = 9, pn = 6.
                # V_233 is 9 x r_233.
                # W_233 is 6 x r_233.
                
                # We need to map these to the global indices.
                # For block b:
                # U is the same for all blocks? Yes, A is the same.
                # So U_new[:, b*r_233 : (b+1)*r_233] = U_233
                
                # V_233 has rows indexed by j*3 + k_local (since p_local=3).
                # We need to map this to j*9 + (b*3 + k_local).
                # Let's construct V_new.
                for r_local in range(r_233):
                    col_global = b * r_233 + r_local
                    U_new[:, col_global] = U_233[:, r_local]
                    
                    # Map V
                    for j in range(3):
                        for k_local in range(3):
                            v_idx_local = j * 3 + k_local
                            v_idx_global = j * 9 + (b * 3 + k_local)
                            V_new[v_idx_global, col_global] = V_233[v_idx_local, r_local]
                    
                    # Map W
                    for k_local in range(3):
                        for i in range(2):
                            w_idx_local = k_local * 2 + i
                            w_idx_global = (b * 3 + k_local) * 2 + i
                            W_new[w_idx_global, col_global] = W_233[w_idx_local, r_local]
                
            if verify_solution(U_new, V_new, W_new, T):
                best_rank = new_rank
                best_U, best_V, best_W = U_new, V_new, W_new
                print(f"Found decomposition with rank {best_rank}")

    # If we didn't find 48, or if we have time, try to search for even lower ranks directly on 2x3x9
    # using the random search method, but starting from a higher lower bound.
    # We know rank >= 54? No, we found 48.
    # Let's try to find something between 48 and 54, or lower than 48.
    # The random search is expensive. We have 120s.
    # We used ~10s for the 233 search.
    
    # Let's try the random search for 2x3x9 from rank 48 downwards? No, upwards from a lower bound.
    # Lower bound for 2x3x9 is not well known, but at least max(n*m, m*p, p*n) = 27?
    # Actually, the rank of matrix multiplication is at least max(n*m, m*p, p*n) / something?
    # Strassen's lower bound: rank(M_{n,m,p}) >= (n*m*p)^{log_2(7)/3}? No.
    # Let's just try to see if we can beat 48 with the random search.
    # But 48 is already a 12% improvement.
    
    # Let's try a few random trials for r=47, 46...
    # The search space is huge.
    # Maybe we can stick with 48 if it's valid.
    
    # Fallback to schoolbook if 48 construction failed or verification failed.
    if best_rank == rank_schoolbook:
        # Try random search for 2x3x9 directly
        for r in range(40, rank_schoolbook):
            if time.time() > deadline - 10:
                break
            if r >= best_rank:
                continue
            found = False
            for trial in range(50):
                if time.time() > deadline - 10:
                    break
                np.random.seed(100 + r * 1000 + trial)
                V = np.random.randint(-1, 2, size=(mp, r))
                W = np.random.randint(-1, 2, size=(pn, r))
                
                M = (V[:, None, :] * W[None, :, :]).reshape(mp * pn, r)
                
                if np.linalg.matrix_rank(M) < r:
                    continue
                    
                U = np.zeros((nm, r))
                valid = True
                for a in range(nm):
                    target = T[a, :, :].flatten()
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(M, target, rcond=None)
                        if residuals and residuals[0] > 1e-6:
                            valid = False
                            break
                        U[a, :] = sol
                    except:
                        valid = False
                        break
                        
                if not valid:
                    continue
                    
                def is_half_integer(mat, tol=1e-6):
                    scaled = 2.0 * mat
                    rounded = np.round(scaled)
                    return np.all(np.abs(scaled - rounded) < tol)
                    
                if is_half_integer(U):
                    U_snapped = np.round(2.0 * U) / 2.0
                    if verify_solution(U_snapped, V, W, T):
                        best_rank = r
                        best_U, best_V, best_W = U_snapped, V, W
                        found = True
                        break
            if found:
                break

    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
