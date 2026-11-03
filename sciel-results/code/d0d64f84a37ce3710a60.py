import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(i,k)
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook decomposition (Rank 60)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, col] = 1
                V_sb[b, col] = 1
                W_sb[c, col] = 1
                col += 1

    def check_exact(U, V, W, tol=1e-9):
        # T_approx[a, b, c] = sum_s U[a, s] V[b, s] W[c, s]
        T_approx = np.einsum('as,bs,cs->abc', U, V, W, optimize=True)
        diff = np.abs(T_approx - T)
        return np.max(diff) < tol

    def find_decomposition_als(r, max_iter=1000, tol=1e-10, seed=0):
        np.random.seed(seed)
        # Initialize with small random values
        U = np.random.randn(nm, r) * 0.1
        V = np.random.randn(mp, r) * 0.1
        W = np.random.randn(pn, r) * 0.1
        
        # ALS: Fix two, solve for the third
        # We want to minimize || T - sum_s U_s V_s W_s ||_F^2
        # For fixed V, W, the problem for U is:
        # sum_{b,c} (T[a,b,c] - sum_s U[a,s] V[b,s] W[c,s])^2
        # This is a linear least squares problem for U[:, s]
        # Let M_s be the matricization of T along mode 1, with the other modes contracted by V and W?
        # Actually, for a specific column s of U, the contribution is U[a,s] * (sum_{s' != s} ... + V[:,s] W[:,s]^T)
        # It's easier to view it as:
        # T_approx = sum_s U[:,s] V[:,s]^T W[:,s]^T (outer products)
        # The term for U[:,s] is coupled with V[:,s] and W[:,s].
        # Standard ALS for CP decomposition:
        # Update U: For each s, solve for U[:,s] given V[:,s] and W[:,s] and the rest of the sum.
        # Let R = T - sum_{s' != s} U[:,s'] V[:,s']^T W[:,s']^T
        # Then R[a,b,c] = U[a,s] V[b,s] W[c,s]
        # This is not a simple linear system for U[:,s] because V and W are fixed vectors.
        # We can rewrite the objective for U[:,s] as:
        # min || T - U_s (V_s \otimes W_s)^T - Rest ||
        # This is a linear least squares: U_s = argmin || T_mat - U_s (V_s \otimes W_s)^T ||
        # Where T_mat is the matricization of T (flattening the other two modes).
        
        # Let's use the matricization approach.
        # Matricize T into T_1 (nm x mp*pn), T_2 (mp x nm*pn), T_3 (pn x nm*mp)
        # T_1[a, b*pn+c] = T[a,b,c]
        # Update U:
        # T_1 - U_rest = U[:,s] * (V[:,s] \otimes W[:,s])^T
        # Let vec_s = np.kron(V[:,s], W[:,s])
        # U[:,s] = (vec_s vec_s^T)^+ (T_1[:,s_vec_index] - U_rest[:,s_vec_index])
        # This is a rank-1 update problem.
        
        # Actually, simpler:
        # min || T - sum_s U_s V_s^T W_s^T ||
        # Gradient of U is 2 * (T_approx - T) contracted with V, W.
        # But ALS is:
        # For each s, we want to find u_s, v_s, w_s such that they approximate the residual.
        # If we fix v_s and w_s, the optimal u_s is:
        # u_s = argmin || T - sum_{s' != s} u_{s'} v_{s'}^T w_{s'}^T - u_s v_s^T w_s^T ||
        # Let R_s = T - sum_{s' != s} u_{s'} v_{s'}^T w_{s'}^T
        # We want u_s v_s^T w_s^T to approximate R_s.
        # This is equivalent to finding the best rank-1 approximation to R_s with the constraint that the other two factors are fixed to v_s and w_s?
        # No, v_s and w_s are also variables, but in ALS we fix them.
        # So, given v_s and w_s, we want to find u_s to minimize || R_s - u_s v_s^T w_s^T ||.
        # This is a linear least squares problem:
        # R_s = u_s (v_s \otimes w_s)^T
        # Let q_s = v_s \otimes w_s.
        # R_s_mat (nm x mp*pn) = u_s (q_s)^T
        # u_s = (q_s q_s^T)^+ R_s_mat_col? No.
        # R_s_mat = u_s q_s^T.
        # Vectorizing: vec(R_s_mat) = (q_s \otimes I) u_s.
        # u_s = (q_s \otimes I)^+ vec(R_s_mat).
        # This is computationally heavy if done naively.
        
        # Efficient ALS for CP:
        # Update U:
        # U = T_1 @ (W * V) @ pinv((W * V)^T @ (W * V) + eps*I) ? No, that's for dense.
        # For sparse/rank-r:
        # Let's just use the gradient step with a good learning rate, or a simplified ALS.
        # Given the small size (12x20x15), we can afford some computation.
        
        # Let's stick to Gradient Descent but with Adaptive Learning Rate (Adam) or just a very careful step.
        # The previous attempt failed likely due to poor initialization or learning rate.
        # Let's try initializing with the schoolbook decomposition columns.
        # If r < 60, we can't use all 60.
        # We can use a random subset of the schoolbook columns? No, they are linearly independent in the tensor space?
        # Actually, the schoolbook columns are orthogonal in a certain sense?
        
        # Let's try a different strategy:
        # 1. Start with rank 60 (schoolbook).
        # 2. Try to merge columns or zero out entries.
        # 3. Or, use the "strassen" structure if applicable.
        
        # For 3x4x5, let's just do a robust random search with ALS-like updates.
        # We will use the "unfolding" method for ALS.
        
        # Precompute matricizations
        T1 = T.reshape(nm, mp * pn)
        T2 = T.reshape(mp, nm * pn)
        T3 = T.reshape(pn, nm * mp)
        
        # Helper to get the "contracted" tensor for updating one factor
        # For U: We need to contract T with V and W.
        # T_contracted[a, s] = sum_{b, c} T[a, b, c] V[b, s] W[c, s]
        # This is exactly the gradient term (without the error term).
        # In ALS, we update U using the current V and W.
        # The optimal U given V, W is:
        # U = (T contracted with V, W) * (V*W contracted with V, W)^{-1}
        # Let A = einsum('abc,bs,cs->as', T, V, W)
        # Let B = einsum('bs,cs->s', V, W) ? No.
        # The normal equations for U:
        # sum_{s'} U[a, s'] sum_{b,c} V[b, s'] W[c, s'] V[b, s] W[c, s] = sum_{b,c} T[a,b,c] V[b,s] W[c,s]
        # Let M[s, s'] = sum_{b,c} V[b, s'] W[c, s'] V[b, s] W[c, s]
        # M is r x r.
        # Let RHS[a, s] = sum_{b,c} T[a,b,c] V[b,s] W[c,s]
        # Then U @ M = RHS  =>  U = RHS @ M^{-1}
        
        # This is a very efficient ALS step!
        
        def update_U(V, W):
            # M[s, s'] = sum_{b,c} V[b,s] W[c,s] V[b,s'] W[c,s']
            # This is (V^T V) * (W^T W) ? No.
            # sum_b V[b,s] V[b,s'] is (V^T V)[s, s']
            # sum_c W[c,s] W[c,s'] is (W^T W)[s, s']
            # But the sum is over b AND c.
            # sum_{b,c} V[b,s] W[c,s] V[b,s'] W[c,s'] = (sum_b V[b,s] V[b,s']) * (sum_c W[c,s] W[c,s'])
            # So M = (V^T V) * (W^T W) (Hadamard product? No, element-wise product of the matrices? No.)
            # Let G_V = V^T V (r x r)
            # Let G_W = W^T W (r x r)
            # M[s, s'] = G_V[s, s'] * G_W[s, s'] ?
            # Yes, because the sums over b and c are independent.
            G_V = V.T @ V
            G_W = W.T @ W
            M = G_V * G_W # Element-wise product
            
            # RHS[a, s] = sum_{b,c} T[a,b,c] V[b,s] W[c,s]
            # This is einsum('abc,bs,cs->as', T, V, W)
            RHS = np.einsum('abc,bs,cs->as', T, V, W, optimize=True)
            
            # Solve U @ M = RHS
            # U = RHS @ M^{-1}
            try:
                U_new = RHS @ np.linalg.inv(M + 1e-10 * np.eye(r))
            except np.linalg.LinAlgError:
                U_new = RHS @ np.linalg.pinv(M)
            return U_new

        def update_V(U, W):
            G_U = U.T @ U
            G_W = W.T @ W
            M = G_U * G_W
            RHS = np.einsum('abc,as,cs->bs', T, U, W, optimize=True)
            try:
                V_new = RHS @ np.linalg.inv(M + 1e-10 * np.eye(r))
            except np.linalg.LinAlgError:
                V_new = RHS @ np.linalg.pinv(M)
            return V_new

        def update_W(U, V):
            G_U = U.T @ U
            G_V = V.T @ V
            M = G_U * G_V
            RHS = np.einsum('abc,as,bs->cs', T, U, V, optimize=True)
            try:
                W_new = RHS @ np.linalg.inv(M + 1e-10 * np.eye(r))
            except np.linalg.LinAlgError:
                W_new = RHS @ np.linalg.pinv(M)
            return W_new

        for it in range(max_iter):
            U = update_U(V, W)
            V = update_V(U, W)
            W = update_W(U, V)
            
            # Check convergence
            T_approx = np.einsum('as,bs,cs->abc', U, V, W, optimize=True)
            loss = np.sum((T_approx - T)**2)
            if loss < tol:
                break
                
        return U, V, W, loss

    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Try ranks from 59 down to 50
    for r in range(rank_sb - 1, 49, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Multiple restarts for this rank
        for seed in range(10):
            if time.time() > deadline:
                break
            U, V, W, loss = find_decomposition_als(r, max_iter=500, tol=1e-12, seed=seed)
            
            if loss < 1e-6:
                # Round to nearest 0.5
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if check_exact(U_r, V_r, W_r, tol=1e-9):
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    best_rank = r
                    found = True
                    break # Found for this rank, go to next lower rank
        
        if found:
            continue # Try to find even lower rank
        else:
            # If not found for this rank, we might still want to check if a higher rank was better?
            # No, we are iterating downwards. If we fail at 59, we try 58.
            # If we fail all the way, we return the schoolbook.
            pass

    return best_result
