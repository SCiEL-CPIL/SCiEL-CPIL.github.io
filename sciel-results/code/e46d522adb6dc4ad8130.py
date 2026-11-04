import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s margin
    
    n, m, p = 3, 3, 7
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    # Build the target tensor T as a 3D array T[a, b, c]
    # T[(i,j), (j',k), (k',i')] = 1 if i=i', j=j', k=k' else 0
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try ranks from low to high
    # Lower bound from matricization: 21
    best_rank = None
    best_U = None
    best_V = None
    best_W = None
    
    for R in range(21, 64):
        if time.time() > deadline - 20:
            break
        
        # Try to find a decomposition of rank R
        success = False
        for attempt in range(10):
            if time.time() > deadline - 20:
                break
            
            # Random initialization
            np.random.seed(attempt * 1000 + R)
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(pn, R)
            
            # Iterative refinement
            for it in range(200):
                if time.time() > deadline - 10:
                    break
                
                # Fix V, W, solve for U
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # For fixed a: T[a,b,c] = sum_r U[a,r] (V[b,r] W[c,r])
                # Let's form the system: for each a, we have a 21x21 matrix
                # T[a,:,:] = sum_r U[a,r] (V[:,r] W[:,r]^T)
                
                # Build the linear system for U
                # Unknowns: U[a,r] for r=0..R-1
                # Equations: for each (b,c), sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
                
                # This is: U[a,:] * (V W^T)[b,c] ... no
                # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
                # Let M_bc = V[:,r] W[:,r] for each r, so M is mp x pn x R
                # Then sum_r U[a,r] M[b,c,r] = T[a,b,c]
                
                # For fixed a: U[a,:] @ M[a,:,:] ... no
                # sum_r U[a,r] M[b,c,r] = T[a,b,c]
                # This is: (U[a,:] as row) contracted with M[b,c,:] = T[a,b,c]
                
                # Let's reshape: for fixed a, we have a system in U[a,:]
                # Flatten M to (mp*pn, R), flatten T[a] to (mp*pn,)
                # Then: M_flat @ U[a,:] = T[a].flat
                
                M = np.einsum('br,cr->bcr', V, W)  # (mp, pn, R)
                M_flat = M.transpose(2, 0, 1).reshape(R, mp * pn)  # (R, mp*pn)
                
                # For each a: M_flat.T @ U[a,:] = T[a].flat
                # i.e., (mp*pn x R) system
                A_mat = M_flat.T  # (mp*pn, R)
                b_vec = T[a].flat  # but this varies with a
                
                # Solve least squares for all a at once
                # U = (A_mat^T A_mat)^{-1} A_mat^T T_flat
                # where T_flat is (nm, mp*pn)
                T_flat = T.reshape(nm, mp * pn)
                
                # U = lstsq(A_mat, T_flat.T).T
                try:
                    U_new, _, _, _ = np.linalg.lstsq(A_mat, T_flat.T, rcond=None)
                    U = U_new.T  # (nm, R)
                except:
                    break
                
                # Fix U, W, solve for V
                M = np.einsum('ar,cr->acr', U, W)  # (nm, pn, R)
                M_flat = M.transpose(2, 0, 1).reshape(R, nm * pn)
                A_mat = M_flat.T  # (nm*pn, R)
                T_flat = T.transpose(1, 0, 2).reshape(mp, nm * pn)  # T[b,a,c]
                try:
                    V_new, _, _, _ = np.linalg.lstsq(A_mat, T_flat.T, rcond=None)
                    V = V_new.T
                except:
                    break
                
                # Fix U, V, solve for W
                M = np.einsum('ar,br->abr', U, V)  # (nm, mp, R)
                M_flat = M.transpose(2, 0, 1).reshape(R, nm * mp)
                A_mat = M_flat.T  # (nm*mp, R)
                T_flat = T.transpose(2, 0, 1).reshape(pn, nm * mp)  # T[c,a,b]
                try:
                    W_new, _, _, _ = np.linalg.lstsq(A_mat, T_flat.T, rcond=None)
                    W = W_new.T
                except:
                    break
                
                # Check convergence
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T_recon - T) / np.linalg.norm(T)
                if err < 1e-10:
                    success = True
                    break
            
            if success:
                # Round to nearest half-integer
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify
                T_recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                if np.allclose(T_recon, T, atol=1e-9):
                    best_rank = R
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    break
        
        if best_rank is not None:
            break
    
    if best_rank is None:
        # Fallback to schoolbook
        R = nm * mp  # 63
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
        best_rank = R
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
