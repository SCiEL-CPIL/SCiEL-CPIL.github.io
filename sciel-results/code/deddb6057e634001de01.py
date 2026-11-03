import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Leave 35s safety margin for 150s limit
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    nonzero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
                nonzero_indices.append((a, b, c))
    
    def verify_fast(U_int, V_int, W_int):
        """Fast verification: check only non-zero entries first, then full tensor."""
        r = U_int.shape[1]
        # Check non-zero entries
        for (a, b, c) in nonzero_indices:
            val = int(np.dot(U_int[a, :], V_int[b, :] * W_int[c, :]))
            if val != 8:
                return False
        # Full check
        T_calc = np.einsum('ak,bk,ck->abc', U_int, V_int, W_int)
        T_target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
        for (a, b, c) in nonzero_indices:
            T_target[a, b, c] = 8
        if not np.array_equal(T_calc, T_target):
            return False
        return True
    
    def verify(U, V, W):
        """Verify that U, V, W are valid half-integer decompositions."""
        U_int = np.round(U * 2)
        V_int = np.round(V * 2)
        W_int = np.round(W * 2)
        
        if not np.allclose(U_int, U * 2, atol=0.01):
            return False
        if not np.allclose(V_int, V * 2, atol=0.01):
            return False
        if not np.allclose(W_int, W * 2, atol=0.01):
            return False
        
        U_int = U_int.astype(np.int64)
        V_int = V_int.astype(np.int64)
        W_int = W_int.astype(np.int64)
        
        return verify_fast(U_int, V_int, W_int)
    
    def solve_rank(r, time_limit, init_type='random'):
        """Solve for rank-r decomposition using alternating least squares."""
        if init_type == 'schoolbook_merge':
            # Start from schoolbook rank-63 and try to merge two terms
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < r:
                            U[i*m + j, col] = 1
                            V[j*p + k, col] = 1
                            W[k*n + i, col] = 1
                            col += 1
                        else:
                            break
                    if col >= r:
                        break
                if col >= r:
                    break
            # Add some randomization to break symmetry
            U += np.random.randn(n*m, r) * 0.1
            V += np.random.randn(m*p, r) * 0.1
            W += np.random.randn(p*n, r) * 0.1
        elif init_type == 'random':
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
        else:
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
        
        best_residual = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(30):
            if time.time() > time_limit:
                break
                
            # Update U
            for a in range(n*m):
                V_rep = np.repeat(V, p*n, axis=0)
                W_rep = np.tile(W, (m*p, 1))
                M_a = V_rep * W_rep
                t_a = T[a, :, :].flatten()
                U[a, :], residuals, rank, sv = np.linalg.lstsq(M_a, t_a, rcond=None)
                
            # Update V
            for b in range(m*p):
                U_rep = np.repeat(U, p*n, axis=0)
                W_rep = np.tile(W, (n*m, 1))
                M_b = U_rep * W_rep
                t_b = T[:, b, :].flatten()
                V[b, :], residuals, rank, sv = np.linalg.lstsq(M_b, t_b, rcond=None)
                
            # Update W
            for c in range(p*n):
                U_rep = np.repeat(U, m*p, axis=0)
                V_rep = np.tile(V, (n*m, 1))
                M_c = U_rep * V_rep
                t_c = T[:, :, c].flatten()
                W[c, :], residuals, rank, sv = np.linalg.lstsq(M_c, t_c, rcond=None)
            
            # Check convergence and verify
            U_rounded = np.round(U * 2) / 2.0
            V_rounded = np.round(V * 2) / 2.0
            W_rounded = np.round(W * 2) / 2.0
            
            if verify(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded
            
            # Track best solution by residual
            if it % 3 == 0:
                T_calc = np.einsum('ak,bk,ck->abc', U, V, W)
                residual = np.linalg.norm(T_calc - T)
                if residual < best_residual:
                    best_residual = residual
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Final check
        U_rounded = np.round(U * 2) / 2.0
        V_rounded = np.round(V * 2) / 2.0
        W_rounded = np.round(W * 2) / 2.0
        if verify(U_rounded, V_rounded, W_rounded):
            return U_rounded, V_rounded, W_rounded
        
        return None
    
    # Try ranks from 62 down to 50
    for r in range(62, 49, -1):
        if time.time() > deadline:
            break
        
        # Try multiple initialization types
        for init_type in ['random', 'schoolbook_merge', 'random']:
            if time.time() > deadline:
                break
            res = solve_rank(r, deadline - 3, init_type)
            if res is not None:
                U, V, W = res
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank 63
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
