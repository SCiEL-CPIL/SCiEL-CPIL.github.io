import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        r = U_mat.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U_mat[i*m + j, c] * V_mat[jp*p + k, c] * W_mat[k*n + i, c]
                        expected = 1.0 if j == jp else 0.0
                        if abs(s - expected) > tol:
                            return False
        return True
    
    def build_rank50():
        """Build the current rank-50 Strassen+schoolbook decomposition."""
        rank = 50
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        col = 0
        
        def add_strassen_222(i0, j0, k0):
            nonlocal col
            a = i0 * m + j0
            b = i0 * m + (j0 + 1)
            c = (i0 + 1) * m + j0
            d = (i0 + 1) * m + (j0 + 1)
            e = j0 * p + k0
            f = j0 * p + (k0 + 1)
            g = (j0 + 1) * p + k0
            h = (j0 + 1) * p + (k0 + 1)
            w00 = k0 * n + i0
            w01 = (k0 + 1) * n + i0
            w10 = k0 * n + (i0 + 1)
            w11 = (k0 + 1) * n + (i0 + 1)
            
            U[a, col] += 1; U[d, col] += 1
            V[e, col] += 1; V[h, col] += 1
            W[w00, col] += 1; W[w11, col] += 1
            col += 1
            
            U[c, col] += 1; U[d, col] += 1
            V[e, col] += 1
            W[w10, col] += 1; W[w11, col] -= 1
            col += 1
            
            U[a, col] += 1
            V[f, col] += 1; V[h, col] -= 1
            W[w01, col] += 1; W[w11, col] += 1
            col += 1
            
            U[d, col] += 1
            V[g, col] += 1; V[e, col] -= 1
            W[w00, col] += 1; W[w10, col] += 1
            col += 1
            
            U[a, col] += 1; U[b, col] += 1
            V[h, col] += 1
            W[w00, col] -= 1; W[w01, col] += 1
            col += 1
            
            U[c, col] += 1; U[a, col] -= 1
            V[e, col] += 1; V[f, col] += 1
            W[w11, col] += 1
            col += 1
            
            U[b, col] += 1; U[d, col] -= 1
            V[g, col] += 1; V[h, col] += 1
            W[w00, col] += 1
            col += 1

        def add_schoolbook_221(i0, j0, k0):
            nonlocal col
            a = i0 * m + j0
            b = i0 * m + (j0 + 1)
            c = (i0 + 1) * m + j0
            d = (i0 + 1) * m + (j0 + 1)
            e = j0 * p + k0
            g = (j0 + 1) * p + k0
            w0 = k0 * n + i0
            w1 = k0 * n + (i0 + 1)
            
            U[a, col] += 1; V[e, col] += 1; W[w0, col] += 1
            col += 1
            U[b, col] += 1; V[g, col] += 1; W[w0, col] += 1
            col += 1
            U[c, col] += 1; V[e, col] += 1; W[w1, col] += 1
            col += 1
            U[d, col] += 1; V[g, col] += 1; W[w1, col] += 1
            col += 1

        for j_start in [0, 2]:
            for k_start in [0, 2, 4]:
                add_strassen_222(0, j_start, k_start)
            add_schoolbook_221(0, j_start, 6)
        
        assert col == rank
        return U, V, W
    
    def try_reduce(U, V, W, max_iter=50):
        """Try to reduce rank by removing redundant terms."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            removed = False
            for r in range(rank - 1, -1, -1):
                # Save the term
                u_save = U[:, r].copy()
                v_save = V[:, r].copy()
                w_save = W[:, r].copy()
                
                # Try removing term r
                U[:, r] = 0
                V[:, r] = 0
                W[:, r] = 0
                
                if verify(U, V, W):
                    # Term r is redundant, remove it
                    U = np.delete(U, r, axis=1)
                    V = np.delete(V, r, axis=1)
                    W = np.delete(W, r, axis=1)
                    rank -= 1
                    removed = True
                    break
                else:
                    # Restore
                    U[:, r] = u_save
                    V[:, r] = v_save
                    W[:, r] = w_save
            
            if not removed:
                break
        
        return U, V, W
    
    # Build initial rank-50 decomposition
    U, V, W = build_rank50()
    
    if not verify(U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1
                    V[j*p + k, idx] = 1
                    W[k*n + i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by finding redundant terms
    if time.time() < deadline - 10:
        U, V, W = try_reduce(U, V, W)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
