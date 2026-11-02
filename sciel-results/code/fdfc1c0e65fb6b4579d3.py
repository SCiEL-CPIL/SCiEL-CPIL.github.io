import numpy as np
import time

DIMENSIONS = [2, 2, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5 second buffer before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Target Tensor Construction
    # A: n x m, B: m x p, C: p x n (Trace convention W[k*n+i,r])
    # T[a, b, c] = 1 if j_a == j_b and i_a == i_c and k_b == k_c
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    
    def build_tensor(n, m, p):
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    T[a, b, c] = 1
        return T

    T = build_tensor(n, m, p)
    
    best_solution = None
    best_rank = 11
    
    def verify_and_snap(U, V, W, rank, tol=1e-6):
        """Verify decomposition and try to snap to rational numbers."""
        T_rec = np.zeros((n*m, m*p, p*n))
        for rr in range(rank):
            T_rec += np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
        
        err = np.linalg.norm(T - T_rec)
        if err < tol:
            # Try to snap to rational numbers
            for denom in [1, 2, 4, 6, 8, 12, 3, 5, 10, 16, 20, 24]:
                U_s = np.round(U * denom) / denom
                V_s = np.round(V * denom) / denom
                W_s = np.round(W * denom) / denom
                
                T_rec_exact = np.zeros((n*m, m*p, p*n))
                for rr in range(rank):
                    T_rec_exact += np.einsum('a,b,c->abc', U_s[:,rr], V_s[:,rr], W_s[:,rr])
                
                if np.allclose(T_rec_exact, T, atol=1e-9):
                    return True, (U_s, V_s, W_s)
        return False, None
    
    def polish_decomposition(U, V, W, rank, max_iters=200, tol=1e-10):
        """High-precision polish pass using alternating least squares."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U
            X_u = np.zeros((m*p*p*n, rank))
            for rr in range(rank):
                X_u[:,rr] = np.outer(V[:,rr], W[:,rr]).flatten()
            try:
                U_new, _, _, _ = np.linalg.lstsq(X_u, T.reshape(n*m, -1).T, rcond=None)
                U = U_new.T
            except:
                break
                
            # Update V
            X_v = np.zeros((n*m*p*n, rank))
            for rr in range(rank):
                X_v[:,rr] = np.outer(U[:,rr], W[:,rr]).flatten()
            try:
                V_new, _, _, _ = np.linalg.lstsq(X_v, T.reshape(m*p, -1).T, rcond=None)
                V = V_new.T
            except:
                break
                
            # Update W
            X_w = np.zeros((n*m*m*p, rank))
            for rr in range(rank):
                X_w[:,rr] = np.outer(U[:,rr], V[:,rr]).flatten()
            try:
                W_new, _, _, _ = np.linalg.lstsq(X_w, T.reshape(p*n, -1).T, rcond=None)
                W = W_new.T
            except:
                break
        
        return U, V, W
    
    # Strategy 1: Structured parameterized search for rank 10
    # The 2x2x3 matrix multiplication can be viewed as 2x2x2 (Strassen) + 2x2x1
    # Try to find a rank-10 decomposition by parameterizing around known structure
    
    target_rank = 10
    num_structured_restarts = 50
    
    for r in range(num_structured_restarts):
        if time.time() > deadline:
            break
        
        # Initialize with structured perturbation of rank-11 solution
        # Start from the known rank-11 and try to merge two terms
        
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        # Use the 4 Strassen-like terms for the 2x2x2 part (indices 0,1,2)
        # and 6 terms for the third column (indices 3-8)
        # Plus 2 more for mixing
        
        # Random structured initialization
        if r % 5 == 0:
            # Near-Strassen initialization
            # Term 0: (A00+A11)(B00+B11)
            U[0,0] = 1; U[3,0] = 1
            V[0,0] = 1; V[4,0] = 1
            W[0,0] = 1; W[3,0] = 1
            
            # Term 1: (A10+A11)B00
            U[2,1] = 1; U[3,1] = 1
            V[0,1] = 1
            W[1,1] = 1; W[3,1] = -1
            
            # Term 2: A00(B01-B11)
            U[0,2] = 1
            V[1,2] = 1; V[4,2] = -1
            W[2,2] = 1; W[3,2] = 1
            
            # Terms 3-8: for the third column (k=2)
            # A00*B02, A01*B02, A10*B12, A11*B12, plus 2 mixed
            U[0,3] = 1
            V[2,3] = 1
            W[4,3] = 1
            
            U[1,4] = 1
            V[2,4] = 1
            W[4,4] = 1
            
            U[2,5] = 1
            V[5,5] = 1
            W[5,5] = 1
            
            U[3,6] = 1
            V[5,6] = 1
            W[5,6] = 1
            
            # Two additional mixed terms with small random perturbations
            U[0,7] = np.random.randn() * 0.5
            U[1,7] = np.random.randn() * 0.5
            U[2,7] = np.random.randn() * 0.5
            U[3,7] = np.random.randn() * 0.5
            V[0,7] = np.random.randn() * 0.5
            V[1,7] = np.random.randn() * 0.5
            V[2,7] = np.random.randn() * 0.5
            V[3,7] = np.random.randn() * 0.5
            V[4,7] = np.random.randn() * 0.5
            V[5,7] = np.random.randn() * 0.5
            W[0,7] = np.random.randn() * 0.5
            W[1,7] = np.random.randn() * 0.5
            W[2,7] = np.random.randn() * 0.5
            W[3,7] = np.random.randn() * 0.5
            W[4,7] = np.random.randn() * 0.5
            W[5,7] = np.random.randn() * 0.5
            
            U[0,8] = np.random.randn() * 0.5
            U[1,8] = np.random.randn() * 0.5
            U[2,8] = np.random.randn() * 0.5
            U[3,8] = np.random.randn() * 0.5
            V[0,8] = np.random.randn() * 0.5
            V[1,8] = np.random.randn() * 0.5
            V[2,8] = np.random.randn() * 0.5
            V[3,8] = np.random.randn() * 0.5
            V[4,8] = np.random.randn() * 0.5
            V[5,8] = np.random.randn() * 0.5
            W[0,8] = np.random.randn() * 0.5
            W[1,8] = np.random.randn() * 0.5
            W[2,8] = np.random.randn() * 0.5
            W[3,8] = np.random.randn() * 0.5
            W[4,8] = np.random.randn() * 0.5
            W[5,8] = np.random.randn() * 0.5
        else:
            # Fully random but scaled
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
        
        # Normalize
        U_norm = np.linalg.norm(U)
        V_norm = np.linalg.norm(V)
        W_norm = np.linalg.norm(W)
        if U_norm > 0: U /= U_norm
        if V_norm > 0: V /= V_norm
        if W_norm > 0: W /= W_norm
        
        # Alternating minimization
        max_iters = 300
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U
            X_u = np.zeros((m*p*p*n, target_rank))
            for rr in range(target_rank):
                X_u[:,rr] = np.outer(V[:,rr], W[:,rr]).flatten()
            try:
                U, _, _, _ = np.linalg.lstsq(X_u, T.reshape(n*m, -1).T, rcond=None)
                U = U.T
            except:
                break
                
            # Update V
            X_v = np.zeros((n*m*p*n, target_rank))
            for rr in range(target_rank):
                X_v[:,rr] = np.outer(U[:,rr], W[:,rr]).flatten()
            try:
                V, _, _, _ = np.linalg.lstsq(X_v, T.reshape(m*p, -1).T, rcond=None)
                V = V.T
            except:
                break
                
            # Update W
            X_w = np.zeros((n*m*m*p, target_rank))
            for rr in range(target_rank):
                X_w[:,rr] = np.outer(U[:,rr], V[:,rr]).flatten()
            try:
                W, _, _, _ = np.linalg.lstsq(X_w, T.reshape(p*n, -1).T, rcond=None)
                W = W.T
            except:
                break
            
            # Check convergence every 20 iterations
            if it % 20 == 0:
                T_rec = np.zeros((n*m, m*p, p*n))
                for rr in range(target_rank):
                    T_rec += np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
                err = np.linalg.norm(T - T_rec)
                if err < 1e-8:
                    break
        
        # Polish
        U, V, W = polish_decomposition(U, V, W, target_rank, max_iters=100, tol=1e-10)
        
        # Verify and snap
        success, solution = verify_and_snap(U, V, W, target_rank, tol=1e-6)
        if success:
            best_solution = solution
            best_rank = target_rank
            break
    
    # Strategy 2: If rank 10 not found, try with more random restarts
    if best_solution is None:
        num_random_restarts = 100
        for r in range(num_random_restarts):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            max_iters = 200
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                X_u = np.zeros((m*p*p*n, target_rank))
                for rr in range(target_rank):
                    X_u[:,rr] = np.outer(V[:,rr], W[:,rr]).flatten()
                try:
                    U, _, _, _ = np.linalg.lstsq(X_u, T.reshape(n*m, -1).T, rcond=None)
                    U = U.T
                except:
                    break
                    
                X_v = np.zeros((n*m*p*n, target_rank))
                for rr in range(target_rank):
                    X_v[:,rr] = np.outer(U[:,rr], W[:,rr]).flatten()
                try:
                    V, _, _, _ = np.linalg.lstsq(X_v, T.reshape(m*p, -1).T, rcond=None)
                    V = V.T
                except:
                    break
                    
                X_w = np.zeros((n*m*m*p, target_rank))
                for rr in range(target_rank):
                    X_w[:,rr] = np.outer(U[:,rr], V[:,rr]).flatten()
                try:
                    W, _, _, _ = np.linalg.lstsq(X_w, T.reshape(p*n, -1).T, rcond=None)
                    W = W.T
                except:
                    break
                
                if it % 25 == 0:
                    T_rec = np.zeros((n*m, m*p, p*n))
                    for rr in range(target_rank):
                        T_rec += np.einsum('a,b,c->abc', U[:,rr], V[:,rr], W[:,rr])
                    err = np.linalg.norm(T - T_rec)
                    if err < 1e-8:
                        break
            
            # Polish
            U, V, W = polish_decomposition(U, V, W, target_rank, max_iters=50, tol=1e-10)
            
            success, solution = verify_and_snap(U, V, W, target_rank, tol=1e-6)
            if success:
                best_solution = solution
                best_rank = target_rank
                break
    
    if best_solution is not None:
        U, V, W = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Return known rank 11 solution
        rank = 11
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def set_u(i, j, r, val): U[i*m+j][r] = val
        def set_v(j, k, r, val): V[j*p+k][r] = val
        def set_w(k, i, r, val): W[k*n+i][r] = val

        # 1. (A00+A11)(B00+B11) -> M1
        set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
        set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
        set_w(0, 0, 0, 1)
        set_w(1, 1, 0, 1)

        # 2. (A10+A11)B00 -> M2
        set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
        set_v(0, 0, 1, 1)
        set_w(0, 1, 1, 1)
        set_w(1, 1, 1, -1)

        # 3. A00(B01-B11) -> M3
        set_u(0, 0, 2, 1)
        set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
        set_w(1, 0, 2, 1)
        set_w(1, 1, 2, 1)

        # 4. A11(B10-B00) -> M4
        set_u(1, 1, 3, 1)
        set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
        set_w(0, 0, 3, 1)
        set_w(0, 1, 3, 1)

        # 5. (A00+A01)B11 -> M5
        set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
        set_v(1, 1, 4, 1)
        set_w(0, 0, 4, -1)
        set_w(1, 0, 4, 1)

        # 6. (A10-A00)(B00+B01) -> M6
        set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
        set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
        set_w(1, 1, 5, 1)

        # 7. (A01-A11)(B10+B11) -> M7
        set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
        set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
        set_w(0, 0, 6, 1)

        # 8. A00 B02 -> M8
        set_u(0, 0, 7, 1)
        set_v(0, 2, 7, 1)
        set_w(2, 0, 7, 1)

        # 9. A01 B12 -> M9
        set_u(0, 1, 8, 1)
        set_v(1, 2, 8, 1)
        set_w(2, 0, 8, 1)

        # 10. A10 B02 -> M10
        set_u(1, 0, 9, 1)
        set_v(0, 2, 9, 1)
        set_w(2, 1, 9, 1)

        # 11. A11 B12 -> M11
        set_u(1, 1, 10, 1)
        set_v(1, 2, 10, 1)
        set_w(2, 1, 10, 1)

        return {"U": U, "V": V, "W": W}
