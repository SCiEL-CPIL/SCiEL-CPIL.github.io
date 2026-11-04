"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24
    
    # Build the multiplication tensor T[i,j,k] = delta_{i,k} where:
    # U row index: i*m + j (i in 0..n-1, j in 0..m-1)
    # V row index: j*p + k (j in 0..m-1, k in 0..p-1)
    # W row index: k*n + i (k in 0..p-1, i in 0..n-1)
    # T[u,v,w] = 1 if there exist i,j,k such that u=i*m+j, v=j*p+k, w=k*n+i, else 0
    
    # The tensor entry T[u,v,w] = sum over all (i,j,k) with u=i*m+j, v=j*p+k, w=k*n+i of 1
    # Actually for matrix mult: C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i*m+j, j*p+k, k*n+i] = 1 for each (i,j,k)
    
    # Schoolbook: rank = n*m*p = 120
    rank_school = n * m * p
    
    # Try to find lower rank using alternating minimization
    # First, let's try a few candidate ranks below schoolbook
    
    best_result = None
    best_rank = rank_school
    
    # Schoolbook decomposition as baseline
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    
    # Try to find lower rank decompositions
    # Use alternating minimization with rational rounding
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Check if U,V,W exactly decompose the matrix multiplication tensor."""
        # Reconstruct tensor and check
        # T[u,v,w] should equal sum_c U[u,c]*V[v,c]*W[w,c]
        # For matrix mult: T[i*m+j, j*p+k, k*n+i] = 1, else 0
        
        # Build expected tensor (sparse)
        expected = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    expected[(u, v, w)] = 1
        
        # Compute reconstructed tensor entries
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Check all non-zero entries
        for (u, v, w), val in expected.items():
            recon = np.sum(U_arr[u, :] * V_arr[v, :] * W_arr[w, :])
            if abs(recon - val) > tol:
                return False
        
        # Check that no spurious non-zero entries exist
        # Sample some zero entries
        import random
        random.seed(42)
        for _ in range(200):
            u = random.randint(0, nm - 1)
            v = random.randint(0, mp - 1)
            w = random.randint(0, pn - 1)
            if (u, v, w) not in expected:
                recon = np.sum(U_arr[u, :] * V_arr[v, :] * W_arr[w, :])
                if abs(recon) > tol:
                    return False
        
        return True
    
    def round_to_half_integer(arr):
        """Round array entries to nearest half-integer."""
        return np.round(arr * 2) / 2
    
    def try_alternating_minimization(rank, max_iter=500, tol=1e-10):
        """Try to find a rank-r decomposition using alternating minimization."""
        
        # Initialize with random or structured initialization
        np.random.seed(rank)
        
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        # Normalize
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = float('inf')
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each u, U[u,:] = (V^T * W^T * T[:,u,:]) / ...
            # Actually, the tensor constraint is:
            # sum_c U[u,c]*V[v,c]*W[w,c] = T[u,v,w]
            
            # For fixed V,W, U[u,:] can be found by:
            # (V * W^T) @ U[u,:] = T[u,:,:] where (V*W^T)[v,w] = V[v,:]*W[w,:]
            
            # Let's do it more carefully
            # T[u,v,w] = sum_c U[u,c] * V[v,c] * W[w,c]
            # Define M[v,w] = sum_c V[v,c] * W[w,c]  (this is V @ W.T)
            # Then T[u,:,:] = U[u,:] @ M  ... no that's not right
            
            # Actually: T[u,v,w] = sum_c U[u,c] * (V[v,c] * W[w,c])
            # Let B_c[v,w] = V[v,c] * W[w,c] (outer product)
            # Then T[u,:,:] = sum_c U[u,c] * B_c
            
            # So we need to express T[u,:,:] as a linear combination of B_c
            # This is a least squares problem
            
            # Build the design matrix
            # For each c, B_c is a (mp * pn) vector
            # We want to solve: sum_c U[u,c] * B_c = T[u,:,:] (flattened)
            
            # This is expensive. Let's use a different approach.
            
            # Alternating: fix two, update one
            # Fix V and W, update U
            # For each u, we want: sum_c U[u,c] * V[:,c] * W[:,c] (as a 2D array) = T[u,:,:]
            
            # Let's flatten T[u,:,:] to a vector of length mp*pn
            # And for each c, let b_c = (V[:,c] * W[:,c]).flatten()
            # Then solve: U[u,:] @ B = t_u where B[c, idx] = b_c[idx]
            
            # Build B matrix (rank x mp*pn)
            B = np.zeros((rank, mp * pn))
            for c in range(rank):
                B[c, :] = (V[:, c] * W[:, c]).flatten()
            
            # For each u, solve: U[u,:] @ B = t_u
            # i.e., B.T @ U[u,:] = t_u
            t_u = np.zeros(mp * pn)
            # Fill t_u from the tensor
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if i * m + j == u:
                            v = j * p + k
                            w = k * n + i
                            t_u[v * pn + w] = 1
            
            # Solve least squares
            # U[u,:] = (B.T @ B)^{-1} @ B.T @ t_u
            # Use numpy lstsq
            sol, _, _, _ = np.linalg.lstsq(B.T, t_u, rcond=None)
            U[u, :] = sol
            
            # Update V similarly
            A = np.zeros((rank, nm * pn))
            for c in range(rank):
                A[c, :] = (U[:, c] * W[:, c]).flatten()
            
            for v in range(mp):
                t_v = np.zeros(nm * pn)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if j * p + k == v:
                                u = i * m + j
                                w = k * n + i
                                t_v[u * pn + w] = 1
                sol, _, _, _ = np.linalg.lstsq(A.T, t_v, rcond=None)
                V[v, :] = sol
            
            # Update W
            C = np.zeros((rank, nm * mp))
            for c in range(rank):
                C[c, :] = (U[:, c] * V[:, c]).flatten()
            
            for w in range(pn):
                t_w = np.zeros(nm * mp)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if k * n + i == w:
                                u = i * m + j
                                v = j * p + k
                                t_w[u * mp + v] = 1
                sol, _, _, _ = np.linalg.lstsq(C.T, t_w, rcond=None)
                W[w, :] = sol
            
            # Check error
            err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = i * m + j
                        v = j * p + k
                        w = k * n + i
                        recon = np.sum(U[u, :] * V[v, :] * W[w, :])
                        err += (recon - 1) ** 2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < tol:
                break
        
        # Round to half-integers and verify
        U_r = round_to_half_integer(best_U)
        V_r = round_to_half_integer(best_V)
        W_r = round_to_half_integer(best_W)
        
        if verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist()):
            return U_r, V_r, W_r
        
        return None
    
    # Try ranks from lower to higher
    # Based on known results, for 4x5 x 5x6, the rank might be around 100-110
    # Let's try a range
    
    for rank in range(90, rank_school + 1):
        if time.time() > deadline - 10:
            break
        
        result = try_alternating_minimization(rank, max_iter=200)
        if result is not None:
            U_r, V_r, W_r = result
            best_rank = rank
            best_result = {
                "U": U_r.tolist(),
                "V": V_r.tolist(),
                "W": W_r.tolist()
            }
            break  # Found a valid decomposition at this rank
    
    # If no lower rank found, return schoolbook
    return best_result
# EVOLVE-BLOCK-END

