"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Known: schoolbook is 80. Try to find lower rank.
    # Use numerical tensor rank decomposition via Newton's method on polynomial system.
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # The constraint: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # We parameterize and use gradient descent / Newton on the residual.
    
    def make_initialization(r, seed):
        rng = np.random.RandomState(seed)
        # Small random initialization
        U = rng.randn(nm, r) * 0.5
        V = rng.randn(mp, r) * 0.5
        W = rng.randn(pn, r) * 0.5
        return U, V, W
    
    def compute_residual(U, V, W):
        """Compute the residual: T_approx - T_true for each (i,j,k)."""
        # For each c, compute outer product contribution
        # Residual[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] - 1
        residual = np.zeros((n, m, p))
        for c in range(U.shape[1]):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)  # W[k*n+i] -> reshape to (p, n)
            # Contribution: uc[i,j] * vc[j,k] * wc[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residual[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
        residual -= 1.0
        return residual
    
    def compute_residual_fast(U, V, W):
        """Faster residual computation using einsum."""
        r = U.shape[1]
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # For each c: uc (n,m), vc (m,p), wc (p,n)
        # Sum over c of uc[i,j] * vc[j,k] * wc[k,i]
        U3 = U.T.reshape(r, n, m)  # (r, n, m)
        V3 = V.T.reshape(r, m, p)  # (r, m, p)
        W3 = W.T.reshape(r, p, n)  # (r, p, n)
        # Result[i,k] = sum_j sum_c U3[c,i,j] * V3[c,j,k] * W3[c,k,i]
        # = sum_c sum_j U3[c,i,j] * V3[c,j,k] * W3[c,k,i]
        # For fixed c: sum_j U3[c,i,j] * V3[c,j,k] = (U3[c] @ V3[c])[i,k]
        # Then multiply by W3[c,k,i] and sum over c
        result = np.zeros((n, p))
        for c in range(r):
            # U3[c] is (n,m), V3[c] is (m,p), W3[c] is (p,n)
            prod = U3[c] @ V3[c]  # (n,p)
            # Multiply elementwise by W3[c].T which is (n,p) since W3[c][k,i] -> transpose gives [i,k]
            result += prod * W3[c].T
        # Residual for each (i,j,k) is the same for all j? No!
        # Wait, I made an error. The residual depends on j too.
        # Let me redo this.
        # T_approx[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # This is NOT separable in the simple way I wrote.
        # Let me just use the direct computation but vectorized over c.
        
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # For each (i,j,k): sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Build index arrays
        ii, jj, kk = np.indices((n, m, p))
        u_idx = ii * m + jj  # (n,m,p)
        v_idx = jj * p + kk  # (n,m,p)
        w_idx = kk * n + ii  # (n,m,p)
        
        # Gather values
        Ug = U[u_idx, :]  # (n,m,p,r)
        Vg = V[v_idx, :]  # (n,m,p,r)
        Wg = W[w_idx, :]  # (n,m,p,r)
        
        # Sum over c
        result = np.einsum('...c,...c,...c->...', Ug, Vg, Wg)
        return result - 1.0
    
    def compute_gradients(U, V, W, residual):
        """Compute gradients of sum(residual^2) w.r.t. U, V, W."""
        r = U.shape[1]
        ii, jj, kk = np.indices((n, m, p))
        u_idx = ii * m + jj
        v_idx = jj * p + kk
        w_idx = kk * n + ii
        
        Ug = U[u_idx, :]  # (n,m,p,r)
        Vg = V[v_idx, :]
        Wg = W[w_idx, :]
        
        # d/dU[i*m+j, c] = sum over (i',j',k') where i'*m+j' == i*m+j of 2*res[i',j',k']*Vg*Wg
        # This is complex. Let's use automatic differentiation or just do it carefully.
        
        # Gradient w.r.t. U:
        # d(L)/dU[a,c] = 2 * sum_{(i,j,k): i*m+j=a} residual[i,j,k] * V[j*p+k,c] * W[k*n+i,c]
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res = residual[i, j, k]
                    a = i * m + j
                    b = j * p + k
                    d = k * n + i
                    for c in range(r):
                        grad_U[a, c] += 2 * res * Vg[i, j, k, c] * Wg[i, j, k, c]
                        grad_V[b, c] += 2 * res * Ug[i, j, k, c] * Wg[i, j, k, c]
                        grad_W[d, c] += 2 * res * Ug[i, j, k, c] * Vg[i, j, k, c]
        
        return grad_U, grad_V, grad_W
    
    # Try different ranks, starting from a reasonable lower bound
    # Known lower bounds: rank >= max(n*m, m*p, n*p) = max(16, 20, 20) = 20
    # But we can do better. Let's try from 40 upward.
    
    best_result = None
    best_rank = None
    best_residual = float('inf')
    
    for rank in range(30, 81):
        if time.time() > deadline:
            break
            
        if best_result is not None and rank > best_rank:
            break
            
        # Try multiple random initializations
        for seed in range(5):
            if time.time() > deadline:
                break
                
            U, V, W = make_initialization(rank, seed + rank * 1000)
            
            # Use gradient descent with adaptive learning rate
            lr = 0.01
            best_local_res = float('inf')
            
            for it in range(200):
                if time.time() > deadline:
                    break
                    
                residual = compute_residual_fast(U, V, W)
                res_norm = np.linalg.norm(residual)
                
                if res_norm < best_local_res:
                    best_local_res = res_norm
                    if res_norm < 1e-8:
                        # Found a solution!
                        if best_result is None or rank < best_rank:
                            best_rank = rank
                            best_residual = res_norm
                            best_result = (U.copy(), V.copy(), W.copy())
                        break
                
                if res_norm < 1e-10:
                    if best_result is None or rank < best_rank:
                        best_rank = rank
                        best_residual = res_norm
                        best_result = (U.copy(), V.copy(), W.copy())
                    break
                
                grad_U, grad_V, grad_W = compute_gradients(U, V, W, residual)
                
                # Adaptive learning rate
                g_norm = np.linalg.norm(grad_U) + np.linalg.norm(grad_V) + np.linalg.norm(grad_W)
                if g_norm > 0:
                    lr = min(lr * 2, 0.1) if res_norm < best_local_res * 0.9 else max(lr * 0.5, 1e-6)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
            
            if time.time() > deadline:
                break
    
    # If we found a solution, round to nearest half-integer and verify
    if best_result is not None:
        U, V, W = best_result
        # Round to nearest 0.5
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify
        residual = compute_residual_fast(U_rounded, V_rounded, W_rounded)
        if np.max(np.abs(residual)) < 1e-6:
            # Convert to lists
            U_list = [[float(U_rounded[i, j]) for j in range(U_rounded.shape[1])] for i in range(U_rounded.shape[0])]
            V_list = [[float(V_rounded[i, j]) for j in range(V_rounded.shape[1])] for i in range(V_rounded.shape[0])]
            W_list = [[float(W_rounded[i, j]) for j in range(W_rounded.shape[1])] for i in range(W_rounded.shape[0])]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
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
# EVOLVE-BLOCK-END

