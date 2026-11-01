"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Build the multiplication tensor
    # T[i*n+j, j*p+k, k*n+i] = 1 for all i,j,k
    # Actually the tensor is: for each (i,j,k), T has 1 at position (i*m+j, j*p+k, k*n+i)
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i*m+j, j*p+k, k*n+i]
    
    # The tensor T is defined by: T[a, b, c] = 1 if there exist i,j,k such that
    # a = i*m+j, b = j*p+k, c = k*n+i, else 0
    # More precisely, for each triple (i,j,k), we set T[i*m+j, j*p+k, k*n+i] = 1
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound: at least max(n*p, n*m, m*p) but really the tensor rank lower bound
    # For this problem, let's try ranks from some reasonable minimum
    
    def try_rank(r, seed=42):
        if time.time() > deadline - 5:
            return None
        np.random.seed(seed)
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Alternating minimization
        for it in range(500):
            if time.time() > deadline - 5:
                return None
            # Update U: for each (a, c), U[a, c] = sum_b T[a,b,c] * V[b,c] * W[c,c] / (V[b,c]^2 * W[c,c]^2)
            # Actually, we want to minimize ||T - sum_c U[:,c] outer V[:,c] outer W[:,c]||
            # Fix V and W, solve for U:
            # T[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
            # For fixed b,c: T[a,b,c] = sum_d U[a,d] * (V[b,d] * W[c,d])
            # So U[a,:] = argmin sum_{b,c} (T[a,b,c] - sum_d U[a,d]*V[b,d]*W[c,d])^2
            # This is a linear system: U[a,:] * (V[b,:] * W[c,:])^T = T[a,b,c] for all b,c
            # Let M[a] be the matrix with entries M[a][idx, d] = V[b,d]*W[c,d] where idx = b*(p*n)+c
            # Then U[a,:] * M[a].T = T[a,:,:].reshape(-1)
            
            # More efficient: reshape
            # For each a, we have: sum_d U[a,d] * V[b,d] * W[c,d] = T[a,b,c]
            # Define for each a: the "gram" matrix G_a[d1, d2] = sum_{b,c} V[b,d1]*W[c,d1]*V[b,d2]*W[c,d2]
            # And h_a[d] = sum_{b,c} T[a,b,c] * V[b,d] * W[c,d]
            # Then G_a @ U[a,:] = h_a
            
            # Precompute for efficiency
            # V is (m*p, r), W is (p*n, r)
            # For each a in range(n*m):
            #   h_a[d] = sum_{b,c} T[a,b,c] * V[b,d] * W[c,d]
            #   G_a[d1,d2] = sum_{b,c} V[b,d1]*W[c,d1]*V[b,d2]*W[c,d2]
            
            # Let's vectorize
            # TW[b,c,d] = V[b,d] * W[c,d]  shape (m*p, p*n, r)
            TW = V[:, None, :] * W[None, :, :]  # (m*p, p*n, r)
            
            # h[a, d] = sum_{b,c} T[a,b,c] * TW[b,c,d]
            h = np.einsum('abc,bcd->ad', T, TW)  # (n*m, r)
            
            # G[d1,d2] = sum_{a,b,c} V[b,d1]*W[c,d1]*V[b,d2]*W[c,d2]
            # = sum_{b,c} (V[b,:]*W[c,:]) (V[b,:]*W[c,:])^T
            # = sum_{b,c} TW[b,c,:] TW[b,c,:]^T
            G = np.einsum('bcd,bce->de', TW, TW)  # (r, r)
            
            # Solve G @ U.T = h.T  =>  U = (G^-1 @ h.T).T
            try:
                U_new = np.linalg.solve(G, h.T).T
            except np.linalg.LinAlgError:
                U_new = h  # fallback
            
            U = U_new
            
            # Update V similarly
            # T[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
            # For fixed a,c: T[a,b,c] = sum_d V[b,d] * (U[a,d] * W[c,d])
            # h_v[b, d] = sum_{a,c} T[a,b,c] * U[a,d] * W[c,d]
            # G_v[d1,d2] = sum_{a,c} U[a,d1]*W[c,d1]*U[a,d2]*W[c,d2]
            
            UW = U[:, None, :] * W[None, :, :]  # (n*m, p*n, r)
            h_v = np.einsum('abc,acd->bd', T, UW)  # (m*p, r)
            G_v = np.einsum('acd,ace->de', UW, UW)  # (r, r)
            try:
                V_new = np.linalg.solve(G_v, h_v.T).T
            except np.linalg.LinAlgError:
                V_new = h_v
            V = V_new
            
            # Update W
            # T[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
            # For fixed a,b: T[a,b,c] = sum_d W[c,d] * (U[a,d] * V[b,d])
            # h_w[c, d] = sum_{a,b} T[a,b,c] * U[a,d] * V[b,d]
            # G_w[d1,d2] = sum_{a,b} U[a,d1]*V[b,d1]*U[a,d2]*V[b,d2]
            
            UV = U[:, None, :] * V[None, :, :]  # (n*m, m*p, r)
            h_w = np.einsum('abc,abd->cd', T, UV)  # (p*n, r)
            G_w = np.einsum('abd,abe->de', UV, UV)  # (r, r)
            try:
                W_new = np.linalg.solve(G_w, h_w.T).T
            except np.linalg.LinAlgError:
                W_new = h_w
            W = W_new
        
        # Round to nearest half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Check exactness
        # Reconstruct tensor
        T_recon = np.einsum('ad,bd,cd->abc', U_r, V_r, W_r)
        if np.allclose(T_recon, T, atol=1e-9):
            return (r, U_r, V_r, W_r)
        
        # Also try rounding to integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        T_recon_i = np.einsum('ad,bd,cd->abc', U_i, V_i, W_i)
        if np.allclose(T_recon_i, T, atol=1e-9):
            return (r, U_i, V_i, W_i)
        
        return None
    
    # Try various ranks
    # Theoretical lower bound considerations:
    # For matrix multiplication, rank >= n*p = 15
    # Let's try from 15 upward, but start with some heuristics
    
    ranks_to_try = list(range(15, rank_sb + 1))
    
    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
        if r >= best_rank:
            break
        
        # Try multiple seeds for each rank
        for seed in [42, 0, 1, 2, 3, 7, 13, 99]:
            if time.time() > deadline - 5:
                break
            result = try_rank(r, seed)
            if result is not None:
                r_found, U_f, V_f, W_f = result
                if r_found < best_rank:
                    best_rank = r_found
                    # Convert to lists
                    U_list = [[float(x) for x in row] for row in U_f]
                    V_list = [[float(x) for x in row] for row in V_f]
                    W_list = [[float(x) for x in row] for row in W_f]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                break  # Found valid for this rank, move to next rank
    
    return best_result
# EVOLVE-BLOCK-END
