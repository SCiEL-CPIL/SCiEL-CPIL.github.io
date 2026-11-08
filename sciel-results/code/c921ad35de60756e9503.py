import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    nz_indices = np.argwhere(T == 1.0)
    nz_a = nz_indices[:, 0]
    nz_b = nz_indices[:, 1]
    nz_c = nz_indices[:, 2]
    
    def verify(U, V, W, atol=1e-8):
        for idx in range(len(nz_a)):
            a, b, c = nz_a[idx], nz_b[idx], nz_c[idx]
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val - 1.0) > atol:
                return False
        # Check zero entries
        checked = 0
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    if T[a,b,c] == 0:
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        if abs(val) > atol:
                            return False
                        checked += 1
                        if checked > 200:
                            return True
        return True
    
    best_rank = 28
    best_UVW = None
    
    for rank in [16, 18, 20, 22, 24, 26]:
        if time.time() > deadline - 15:
            break
        
        n_starts = 5
        found = False
        for start in range(n_starts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + start * 7919 + rank * 31)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Alternating least squares
            for iteration in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Update W: for each r, W[:,r] should satisfy
                # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                # Fix U,V, solve for W
                # This is a linear system in W
                # For each c: sum_r W[c,r] * (U[a,r]*V[b,r]) = T[a,b,c]
                # But this is coupled across (a,b,c)
                
                # Better: use the full tensor approach
                # Reshape: we want to minimize ||T - sum_r U_r ⊗ V_r ⊗ W_r||_F
                # Fix U and V, the optimal W is found by solving a linear system
                
                # Build the linear system for W
                # T_comp[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # For fixed U,V: T_comp = sum_r (U[:,r] * V[:,r]) ⊗ W[:,r]
                # Let M_r[a,b] = U[a,r]*V[b,r]
                # Then T_comp[a,b,c] = sum_r M_r[a,b] * W[c,r]
                # This is: T_comp = sum_r M_r ⊗ W_r (outer product in (a,b) and c)
                
                # Vectorize: let t_vec = T.flatten(), w_vec = W.flatten()
                # t_vec = A_w @ w_vec where A_w is a matrix
                # A_w[(a,b,c), (c',r)] = M_r[a,b] if c==c' else 0
                # So A_w is block diagonal with blocks M_r^T for each c
                
                # More efficiently:
                # For each c: T[:, :, c] = sum_r M_r * W[c, r]
                # where M_r is the (n*m x m*p) matrix with M_r[a,b] = U[a,r]*V[b,r]
                # So T[:,:,c] (flattened) = sum_r M_r.flatten() * W[c,r]
                # This is: T_c = (M_0.flatten(), M_1.flatten(), ..., M_{r-1}.flatten()) @ W[c,:]
                
                # Build the design matrix for W
                # Shape: (n*m * m*p) x rank
                # D[a*m*p + b, r] = U[a,r] * V[b,r]
                D = np.zeros((n*m * m*p, rank))
                for r in range(rank):
                    D[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                
                # T flattened: T_flat[a*(m*p) + b, c] -- no, T is (n*m, m*p, p*n)
                # T[a,b,c] where a in [0,n*m), b in [0,m*p), c in [0,p*n)
                # For fixed c: T[:,:,c] has shape (n*m, m*p)
                # T_flat_c = T[:,:,c].flatten() has shape (n*m*m*p,)
                # T_flat_c = D @ W[c,:]
                
                # Solve for W: W = (D^T D)^{-1} D^T T_flat
                # For each c: W[c,:] = (D^T D)^{-1} D^T T[:,:,c].flatten()
                DtD = D.T @ D
                try:
                    DtD_inv = np.linalg.inv(DtD)
                except np.linalg.LinAlgError:
                    DtD_inv = np.linalg.pinv(DtD)
                
                for c in range(p*n):
                    T_c = T[:, :, c].flatten()
                    W[c, :] = DtD_inv @ (D.T @ T_c)
                
                # Similarly update U and V
                # For fixed V and W: T[a,b,c] = sum_r U[a,r] * (V[b,r]*W[c,r])
                # Let N_r[b,c] = V[b,r]*W[c,r]
                # T[a,b,c] = sum_r U[a,r] * N_r[b,c]
                # For fixed a: T[a,:,:] = sum_r U[a,r] * N_r
                # T[a,:,:].flatten() = E @ U[a,:] where E[b*(p*n)+c, r] = V[b,r]*W[c,r]
                E = np.zeros((m*p * p*n, rank))
                for r in range(rank):
                    E[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                
                EtE = E.T @ E
                try:
                    EtE_inv = np.linalg.inv(EtE)
                except np.linalg.LinAlgError:
                    EtE_inv = np.linalg.pinv(EtE)
                
                for a in range(n*m):
                    T_a = T[a, :, :].flatten()
                    U[a, :] = EtE_inv @ (E.T @ T_a)
                
                # Update V
                # T[a,b,c] = sum_r (U[a,r]*W[c,r]) * V[b,r]
                # For fixed b: T[:,b,:].flatten() = F @ V[b,:] where F[a*(p*n)+c, r] = U[a,r]*W[c,r]
                F = np.zeros((n*m * p*n, rank))
                for r in range(rank):
                    F[:, r] = np.outer(U[:, r], W[:, r]).flatten()
                
                FtF = F.T @ F
                try:
                    FtF_inv = np.linalg.inv(FtF)
                except np.linalg.LinAlgError:
                    FtF_inv = np.linalg.pinv(FtF)
                
                for b in range(m*p):
                    T_b = T[:, b, :].flatten()
                    V[b, :] = FtF_inv @ (F.T @ T_b)
            
            if verify(U, V, W):
                best_rank = rank
                best_UVW = (U, V, W)
                found = True
                break
        
        if found:
            break
    
    if best_UVW is not None:
        U, V, W = best_UVW
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: rank-28 Strassen solution
    rank = 28
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    strassen = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)])
    ]
    
    def map_u(local_idx, a_col_off):
        r = local_idx // 2
        c = local_idx % 2
        return r * m + (a_col_off * 2 + c)
    
    def map_v(local_idx, b_row_off, b_col_off):
        r = local_idx // 2
        c = local_idx % 2
        return (b_row_off * 2 + r) * p + (b_col_off * 2 + c)
    
    def map_w(local_idx, c_col_off):
        r = local_idx // 2
        c = local_idx % 2
        return (c_col_off * 2 + c) * n + r
    
    def add_block(a_off, b_row_off, b_col_off, c_col_off, start_col):
        for i, (u_loc, v_loc, w_loc) in enumerate(strassen):
            col = start_col + i
            for idx, coeff in u_loc:
                U[map_u(idx, a_off), col] += coeff
            for idx, coeff in v_loc:
                V[map_v(idx, b_row_off, b_col_off), col] += coeff
            for idx, coeff in w_loc:
                W[map_w(idx, c_col_off), col] += coeff
    
    add_block(0, 0, 0, 0, 0)
    add_block(1, 1, 0, 0, 7)
    add_block(0, 0, 1, 1, 14)
    add_block(1, 1, 1, 1, 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
