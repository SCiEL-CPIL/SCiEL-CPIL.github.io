"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=3 rows of A, m=3 cols of A/rows of B, p=6 cols of B
    nm = n * m  # 9
    mp = m * p  # 18
    pn = p * n  # 18
    
    # Build the matrix multiplication tensor
    # T[a_idx, b_idx, c_idx] = 1 if A[a_idx] and B[b_idx] contribute to C[c_idx]
    # A is n×m: A[row, col], a_idx = row*m + col
    # B is m×p: B[row, col], b_idx = row*p + col
    # C is p×n: C[row, col], c_idx = col*n + row (W[k*n+i] where k=col, i=row)
    # C[c_row, c_col] = sum_j A[c_col, j] * B[j, c_row]
    # So: a_idx = c_col*m + j, b_idx = j*p + c_row, c_idx = c_row*n + c_col
    
    T = np.zeros((nm, mp, pn), dtype=float)
    for c_col in range(n):      # column of C (0..n-1)
        for c_row in range(p):  # row of C (0..p-1)
            for j in range(m):  # summation index (0..m-1)
                a_idx = c_col * m + j
                b_idx = j * p + c_row
                c_idx = c_row * n + c_col
                T[a_idx, b_idx, c_idx] = 1.0
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if T = sum_s U[:,s] ⊗ V[:,s] ⊗ W[:,s]"""
        T_approx = np.einsum('is,js,ks->ijk', U, V, W)
        return np.max(np.abs(T_approx - T)) < tol, np.max(np.abs(T_approx - T))
    
    def generate_schoolbook():
        """Generate the schoolbook rank-54 decomposition"""
        rank = n * m * p  # 54
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for c_col in range(n):
            for c_row in range(p):
                for j in range(m):
                    a_idx = c_col * m + j
                    b_idx = j * p + c_row
                    c_idx = c_row * n + c_col
                    U[a_idx, col] = 1.0
                    V[b_idx, col] = 1.0
                    W[c_idx, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    # Start with schoolbook
    U, V, W, schoolbook_rank = generate_schoolbook()
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), schoolbook_rank
    
    def als_optimize(rank, max_iter=2000, tol=1e-10, seed=42):
        """ALS optimization for rank-r decomposition"""
        if time.time() > deadline - 15:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with random factors
        U = rng.randn(nm, rank) * 0.1
        V = rng.randn(mp, rank) * 0.1
        W = rng.randn(pn, rank) * 0.1
        
        # Precompute for efficiency
        for it in range(max_iter):
            if time.time() > deadline - 15:
                return None
            
            # Update U: for each i, solve (V ⊙ W)^T (V ⊙ W) U[:,s] = (V ⊙ W)^T T[i,:,:]
            # T[i,j,k] = sum_s U[i,s] V[j,s] W[k,s]
            # For fixed i: T[i,:,:] = sum_s U[i,s] (V[:,s] ⊗ W[:,s])
            # Let M = V ⊗ W reshaped to (mp*pn, rank), M[j,k,s] = V[j,s]*W[k,s]
            # Then T_flat[i] = M @ U[i,:]
            
            # Build M: shape (mp*pn, rank)
            M = (V[:, :, np.newaxis] * W[np.newaxis, :, :]).reshape(mp * pn, rank)
            T_flat = T.reshape(nm, mp * pn)
            
            # Solve U[i,:] = (M^T M)^{-1} M^T T_flat[i,:]
            try:
                MTM = M.T @ M  # (rank, rank)
                MTM += 1e-10 * np.eye(rank)  # regularization
                MT = M.T @ T_flat  # (rank, nm)
                U_new = np.linalg.solve(MTM, MT).T  # (nm, rank)
            except np.linalg.LinAlgError:
                U_new = U
            
            # Update V
            # T[j,:,:] for fixed j: T[j,k,:] as (pn, nm)
            # T[j,k,i] = sum_s V[j,s] W[k,s] U[i,s]
            # Let N = W ⊗ U reshaped to (pn*nm, rank), N[k,i,s] = W[k,s]*U[i,s]
            # T_flat_v[j] = N @ V[j,:]
            
            N = (W[:, :, np.newaxis] * U[np.newaxis, :, :]).reshape(pn * nm, rank)
            T_flat_v = T.transpose(1, 0, 2).reshape(mp, nm * pn)  # T[j,i,k]
            # Actually T is (nm, mp, pn), so T.transpose(1,0,2) is (mp, nm, pn)
            # Reshape to (mp, nm*pn)
            T_flat_v = T.transpose(1, 0, 2).reshape(mp, nm * pn)
            
            try:
                NTM = N.T @ N
                NTM += 1e-10 * np.eye(rank)
                NT = N.T @ T_flat_v.T  # (rank, mp)
                V_new = np.linalg.solve(NTM, NT).T  # (mp, rank)
            except np.linalg.LinAlgError:
                V_new = V
            
            # Update W
            # T[k,:,:] for fixed k: T[k,j,i] as (mp, nm)
            # T[k,j,i] = sum_s W[k,s] V[j,s] U[i,s]
            # Let P = V ⊗ U reshaped to (mp*nm, rank), P[j,i,s] = V[j,s]*U[i,s]
            # T_flat_w[k] = P @ W[k,:]
            
            P = (V[:, :, np.newaxis] * U[np.newaxis, :, :]).reshape(mp * nm, rank)
            T_flat_w = T.transpose(2, 1, 0).reshape(pn, nm * mp)  # T[k,j,i]
            # T is (nm, mp, pn), transpose(2,1,0) gives (pn, mp, nm)
            T_flat_w = T.transpose(2, 1, 0).reshape(pn, mp * nm)
            
            try:
                PTM = P.T @ P
                PTM += 1e-10 * np.eye(rank)
                PT = P.T @ T_flat_w.T  # (rank, pn)
                W_new = np.linalg.solve(PTM, PT).T  # (pn, rank)
            except np.linalg.LinAlgError:
                W_new = W
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            if it % 50 == 0:
                T_approx = np.einsum('is,js,ks->ijk', U, V, W)
                max_err = np.max(np.abs(T_approx - T))
                if max_err < tol:
                    break
        
        # Final check
        T_approx = np.einsum('is,js,ks->ijk', U, V, W)
        max_err = np.max(np.abs(T_approx - T))
        
        if max_err < 1e-6:
            # Clean up small values
            U_clean = np.where(np.abs(U) < 1e-8, 0.0, U)
            V_clean = np.where(np.abs(V) < 1e-8, 0.0, V)
            W_clean = np.where(np.abs(W) < 1e-8, 0.0, W)
            
            T_clean = np.einsum('is,js,ks->ijk', U_clean, V_clean, W_clean)
            if np.max(np.abs(T_clean - T)) < 1e-4:
                return U_clean, V_clean, W_clean, max_err
        
        return None
    
    # Try ranks from low to high
    # Theoretical lower bound: max(nm, mp, pn) = 18, but realistically higher
    # Try from 20 up to 54
    for rank in range(20, schoolbook_rank + 1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random seeds for this rank
        found = False
        for seed in range(5):
            if time.time() > deadline - 20:
                break
            
            result = als_optimize(rank, max_iter=3000, tol=1e-10, seed=seed)
            if result is not None:
                U_new, V_new, W_new, err = result
                if err < 1e-6:
                    best_U, best_V, best_W, best_rank = U_new, V_new, W_new, rank
                    found = True
                    break
        
        if found:
            break
    
    # If we haven't found anything better, try a few more seeds at rank 30-40
    if best_rank == schoolbook_rank:
        for rank in [30, 35, 40]:
            if time.time() > deadline - 20:
                break
            for seed in range(10):
                if time.time() > deadline - 15:
                    break
                result = als_optimize(rank, max_iter=5000, tol=1e-10, seed=seed + 100)
                if result is not None:
                    U_new, V_new, W_new, err = result
                    if err < 1e-6:
                        best_U, best_V, best_W, best_rank = U_new, V_new, W_new, rank
                        break
            if best_rank < schoolbook_rank:
                break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

