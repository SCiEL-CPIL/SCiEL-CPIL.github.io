import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    # Build the rank-50 decomposition (from current code)
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    u_a, u_b, u_c, u_d = 0, 1, 3, 4
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1
        
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1
        
        col += 7
    
    k8 = 8
    v_e = k8
    v_g = p + k8
    
    U[u_a][col] += 1
    V[v_e][col] += 1
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1
    
    col += 4
    
    u_ar0, u_ar1 = 2, 5
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        col += 2
    
    # Convert to numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the target tensor for verification
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n*m):
        ai = i // m
        am = i % m
        for j in range(m*p):
            bj = j // p
            bm = j % p
            if am == bj:
                k = bm * n + ai
                T[i, j, k] = 1.0
    
    def verify_decomp(Uv, Vv, Wv):
        """Verify that sum_c Uv[:,c]*Vv[:,c]*Wv[:,c] == T"""
        # Efficient: for each k, compute sum_c Uv[:,c]*Wv[k,c] then check against T[:,:,k]
        # T[:,:,k] = sum_c (Uv[:,c] * Wv[k,c]) * Vv[:,c]
        # This is a rank-r decomposition of each T[:,:,k]
        # Use: for each c, the contribution to T[:,:,k] is Uv[:,c]*Wv[k,c] outer Vv[:,c]
        # Equivalent: T[:,:,k] = (Uv * Wv[k,:]) @ Vv.T  where Uv*Wv[k,:] is element-wise per row
        # Actually: T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
        # For fixed k: T[:,:,k][i,j] = sum_c (U[i,c]*W[k,c]) * V[j,c]
        # = (U * W[k,:]) @ V.T
        max_err = 0.0
        for k in range(p*n):
            computed = (Uv * Wv[k, :]) @ Vv.T
            max_err = max(max_err, np.max(np.abs(computed - T[:, :, k])))
        return max_err
    
    # Check if current decomposition is valid
    err = verify_decomp(U_np, V_np, W_np)
    if err > 1e-8:
        # Fallback: return as-is (should not happen)
        return {"U": U, "V": V, "W": W}
    
    best_rank = rank
    best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
    
    # Strategy: find linear dependencies among the rank-1 terms
    # The "ar" block (last 18 terms) has structure:
    # For each k in range(9):
    #   col = 32 + 2*k:   U[2,col]=1, V[18+k,col]=1, W[k*2+0,col]=1  (row 0, col k)
    #   col = 33 + 2*k:   U[5,col]=1, V[18+k,col]=1, W[k*2+1,col]=1  (row 1, col k)
    # 
    # Key observation: V[18+k, col] = 1 for BOTH the row-0 and row-1 terms at the same k.
    # This means the V factors are shared between pairs.
    #
    # We can look for linear dependencies by checking if any rank-1 term can be
    # expressed as a linear combination of others.
    #
    # More systematically: form the matrix of flattened rank-1 tensors and find null space.
    # But 2916 x 50 SVD is fast enough.
    
    if time.time() < DEADLINE:
        # Form flattened tensor matrix: each column c is vec(u_c ⊗ v_c ⊗ w_c)
        # Shape: (n*m * m*p * p*n, rank) = (6*27*18, 50) = (2916, 50)
        mat = np.zeros((n*m * m*p * p*n, rank))
        for c in range(rank):
            # u_c ⊗ v_c ⊗ w_c flattened
            # Index: (i * m*p + j) * p*n + k
            # = U[i,c] * V[j,c] * W[k,c]
            # Use outer products for speed
            uv = np.outer(U_np[:, c], V_np[:, c])  # (n*m, m*p)
            # For each k, mat[flat_index, c] = uv[:, :] * W[k, c]
            # Actually: mat[(i*mp+j)*pn+k, c] = U[i,c]*V[j,c]*W[k,c]
            # = uv[i,j] * W[k,c]
            # Reshape: mat is (n*m, m*p, p*n, rank) flattened
            # mat.reshape(n*m, m*p, p*n, rank)[:,:, :, c] = uv[:, :, None] * W[:, c]
            # Let's do it as: for each k, the slice is uv * W[k,c]
            for k in range(p*n):
                if W_np[k, c] != 0:
                    mat[(np.arange(n*m) * m*p + np.arange(m*p)[:, None]).T.ravel() if False else None, c]  # skip
            # Faster: use broadcasting
            # mat as 4D: (n*m, m*p, p*n, rank)
            pass
        
        # Redo more efficiently using einsum-like approach
        # mat4d[i, j, k, c] = U[i,c] * V[j,c] * W[k,c]
        # Use: U[:, :, None, :] * V[None, :, :, None] * W[None, None, :, :]
        # Shape: (n*m, 1, 1, rank) * (1, m*p, 1, rank) * (1, 1, p*n, rank)
        # = (n*m, m*p, p*n, rank)
        mat4d = U_np[:, :, None, None] * V_np[None, :, :, None] * W_np[None, None, :, :]
        # mat4d[i, j, k, c] = U[i,c]*V[j,c]*W[k,c] ✓
        mat = mat4d.reshape(-1, rank)
        
        # SVD to find rank and null space
        _, s, Vt = np.linalg.svd(mat, full_matrices=False)
        tol = s[0] * rank * 1e-10
        num_rank = np.sum(s > tol)
        
        if num_rank < rank:
            # There are (rank - num_rank) linear dependencies
            # Null space vectors are the last (rank - num_rank) rows of Vt
            # Try dropping columns one at a time from the null space components
            
            null_rows = Vt[num_rank:]  # (rank - num_rank, rank)
            
            # For each null space vector, find the column with max |component|
            # and try dropping it
            candidates = []
            for nr in null_rows:
                drop_col = int(np.argmax(np.abs(nr)))
                candidates.append(drop_col)
            
            # Also try: for each column, check if dropping it still gives valid decomposition
            # (this handles cases where the null space vector has small components)
            
            # Try each candidate
            for drop_col in sorted(set(candidates)):
                if time.time() > DEADLINE:
                    break
                U_new = np.delete(U_np, drop_col, axis=1)
                V_new = np.delete(V_np, drop_col, axis=1)
                W_new = np.delete(W_np, drop_col, axis=1)
                
                err = verify_decomp(U_new, V_new, W_new)
                if err < 1e-8:
                    new_rank = rank - 1
                    if new_rank < best_rank:
                        best_rank = new_rank
                        best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
                    
                    # Try to drop more columns from the reduced decomposition
                    # Rebuild and check for further dependencies
                    U_np, V_np, W_np = U_new.copy(), V_new.copy(), W_new.copy()
                    rank = new_rank
                    # Re-check for more dependencies
                    mat4d2 = U_np[:, :, None, None] * V_np[None, :, :, None] * W_np[None, None, :, :]
                    mat2 = mat4d2.reshape(-1, rank)
                    if rank > 1:
                        _, s2, Vt2 = np.linalg.svd(mat2, full_matrices=False)
                        tol2 = s2[0] * rank * 1e-10
                        num_rank2 = np.sum(s2 > tol2)
                        if num_rank2 < rank:
                            null_rows2 = Vt2[num_rank2:]
                            for nr2 in null_rows2:
                                if time.time() > DEADLINE:
                                    break
                                dc2 = int(np.argmax(np.abs(nr2)))
                                U_new2 = np.delete(U_np, dc2, axis=1)
                                V_new2 = np.delete(V_np, dc2, axis=1)
                                W_new2 = np.delete(W_np, dc2, axis=1)
                                err2 = verify_decomp(U_new2, V_new2, W_new2)
                                if err2 < 1e-8:
                                    new_rank2 = rank - 1
                                    if new_rank2 < best_rank:
                                        best_rank = new_rank2
                                        best_U, best_V, best_W = U_new2.copy(), V_new2.copy(), W_new2.copy()
                                    # Reset for potential further reduction
                                    U_np, V_np, W_np = U_new2.copy(), V_new2.copy(), W_new2.copy()
                                    rank = new_rank2
                    break  # Only try one drop from the original
    
    # Return best found
    if best_rank < 50:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {"U": U, "V": V, "W": W}
