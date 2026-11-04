"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.linalg import lstsq

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify that U, V, W form an exact decomposition of the 5x5x5 matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    # The tensor T[i,j,k] = 1 if and only if it's a valid matrix multiplication term
    # T is a 3D array where T[i,j,k] = 1 (for the standard decomposition)
    # Actually, the decomposition requires: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    # This is the standard matrix multiplication tensor decomposition.
    # We check: for each c, the contribution to C = A*B is: (U[:, :, c] outer product structure)
    # More precisely: C[i,k] = sum_j A[i,j] B[j,k]
    # The decomposition says: C[i,k] = sum_c (sum_j U[i,j,c] A[i,j]) * (sum_k V[j,k,c] B[j,k]) ... 
    # No, the standard form is:
    # C[i,k] = sum_c M_c[i,k] where M_c = (sum_j U[i,j,c] A[i,j]) * (sum_j V[j,k,c] B[j,k])
    # Actually the correct form for tensor decomposition of matrix mult:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_c (sum_j U[i,j,c] A[i,j]) * (sum_j V[j,k,c] B[j,k]) is wrong.
    # 
    # The correct formulation: We need U (n*m x r), V (m*p x r), W (p*n x r) such that
    # for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is because the matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k in the 
    # "standard" basis, and the decomposition is a sum of rank-1 tensors.
    
    # Let's verify by checking the tensor equality:
    # The matrix multiplication tensor T_{i,j,k} = 1 (it's just the all-ones tensor in the 
    # standard basis for this task? No...)
    
    # Actually, let me re-read the problem. The task says "exactly decompose the matrix 
    # multiplication tensor". The matrix multiplication tensor T is defined by:
    # T[i,j,k] = 1 if we're computing (AB)[i,k] = sum_j A[i,j]B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k (since each term A[i,j]B[j,k] contributes).
    # The rank-1 decomposition: T = sum_c u_c v_c w_c where
    # u_c is a vector of size n*m (indexed by i*m+j), v_c size m*p (indexed by j*p+k), 
    # w_c size p*n (indexed by k*n+i), and T[i,j,k] = sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
    
    # So we need: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    U_flat = U.reshape(n*m, rank)
    V_flat = V.reshape(m*p, rank)
    W_flat = W.reshape(p*n, rank)
    
    # Build the tensor and check
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_flat[i*m+j, c] * V_flat[j*p+k, c] * W_flat[k*n+i, c]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Start with schoolbook decomposition
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    # Try to reduce rank using randomized tensor decomposition
    # Strategy: try random rank r < 125 and see if we can find a solution
    # Use iterative refinement / alternating least squares on the tensor
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    deadline = time.time() + 130.0  # 10s margin
    
    # Try various target ranks
    target_ranks = [110, 108, 106, 104, 102, 100, 98, 96, 94, 92, 90]
    
    for target_r in target_ranks:
        if time.time() > deadline - 10:
            break
        
        # Multiple restarts
        num_restarts = max(1, int((deadline - time.time()) / 5.0))
        
        for restart in range(num_restarts):
            if time.time() > deadline - 2:
                break
            
            # Initialize with random projection of schoolbook
            # Pick target_r columns from the schoolbook decomposition with some perturbation
            U_r = np.zeros((n*m, target_r))
            V_r = np.zeros((m*p, target_r))
            W_r = np.zeros((p*n, target_r))
            
            # Start from schoolbook: pick target_r of the 125 basis vectors
            # But also add some random mixing to allow escape from local minima
            if target_r <= rank:
                idx = np.random.choice(rank, size=target_r, replace=False)
                U_r = U[:, idx].copy()
                V_r = V[:, idx].copy()
                W_r = W[:, idx].copy()
                # Add small random perturbation
                noise_scale = 0.1
                U_r += np.random.randn(n*m, target_r) * noise_scale
                V_r += np.random.randn(m*p, target_r) * noise_scale
                W_r += np.random.randn(p*n, target_r) * noise_scale
            else:
                U_r = np.random.randn(n*m, target_r)
                V_r = np.random.randn(m*p, target_r)
                W_r = np.random.randn(p*n, target_r)
            
            # Alternating least squares iterations
            # For each factor, fix the other two and solve least squares
            # This is the standard ALS approach for CP decomposition
            
            max_iters = 500
            tol = 1e-12
            
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    break
                
                # Update U: for each i,j, U[i*m+j, :] = V[j*p+k, :] * W[k*n+i, :] summed over k
                # More precisely: U[i*m+j, c] = argmin sum_{j,k} (target - sum_c U[i*m+j,c]V[j*p+k,c]W[k*n+i,c])^2
                # This is a linear system: for fixed V, W, the contribution to entry (i,j,k) from U is:
                # sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c])
                # So we need: sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
                # This is: U[i*m+j, :] @ (V[j*p+k, :] * W[k*n+i, :])^T = 1 for each k
                # i.e., U[i*m+j, :] @ M_k = 1 where M_k[c] = V[j*p+k,c] * W[k*n+i,c]
                
                # For each i,j: solve U[i*m+j, :] such that sum_{c} U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                # This is a linear system: A_U @ U[i*m+j, :]^T = b_U
                # where A_U[k, c] = V[j*p+k, c] * W[k*n+i, c], b_U[k] = 1
                
                for i in range(n):
                    for j in range(m):
                        A_U = np.outer(V[j*p:(j+1)*p, :], W[:, i].reshape(-1, 1).squeeze())
                        # A_U has shape (p, target_r)
                        # We want A_U @ U[i*m+j, :]^T = ones(p)
                        # i.e., A_U.T @ U[i*m+j, :] = ones(p)^T
                        # Solve: A_U.T @ x = ones(p)
                        b_U = np.ones(p)
                        try:
                            x, residuals, rank_s, sv = lstsq(A_U.T, b_U, rcond=None)
                            U_r[i*m+j, :] = x
                        except:
                            pass
                    
                    # Update V: for each j,k: sum_c V[j*p+k,c] * U[i*m+j,c] * W[k*n+i,c] = 1 for all i
                    for k in range(p):
                        A_V = np.outer(U[:, k].reshape(-1, 1).squeeze(), W[:, k].reshape(-1, 1).squeeze())
                        # A_V[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        # Wait, this isn't right. Let me reconsider.
                        # For fixed U, W: we need sum_c V[j*p+k,c] * U[i*m+j,c] * W[k*n+i,c] = 1 for all i
                        # i.e., V[j*p+k, :] @ (U[:, j] * W[:, k])^T = ones(n)
                        # where (U[:, j] * W[:, k])[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        A_V = np.zeros((n, target_r))
                        for ii in range(n):
                            A_V[ii, :] = U[ii*m+j, :] * W[k*n+ii, :]
                        b_V = np.ones(n)
                        try:
                            x, residuals, rank_s, sv = lstsq(A_V.T, b_V, rcond=None)
                            V_r[j*p+k, :] = x
                        except:
                            pass
                
                # Update W: for each k,i: sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1 for all j
                for k in range(p):
                    for i in range(n):
                        A_W = np.zeros((m, target_r))
                        for jj in range(m):
                            A_W[jj, :] = U[i*m+jj, :] * V[jj*p+k, :]
                        b_W = np.ones(m)
                        try:
                            x, residuals, rank_s, sv = lstsq(A_W.T, b_W, rcond=None)
                            W_r[k*n+i, :] = x
                        except:
                            pass
                
                # Check convergence
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(target_r):
                                s += U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                            err = abs(s - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < tol:
                    if target_r < best_rank:
                        best_U = U_r.copy()
                        best_V = V_r.copy()
                        best_W = W_r.copy()
                        best_rank = target_r
                    break
                
                # Periodic check
                if it % 100 == 0 and max_err < 1e-8:
                    if target_r < best_rank:
                        best_U = U_r.copy()
                        best_V = V_r.copy()
                        best_W = W_r.copy()
                        best_rank = target_r
    
    # Convert to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    # Round near-integers to integers for cleanliness
    U_out = [[round(v, 12) if abs(v) < 0.5 else round(v, 12) for v in row] for row in U_out]
    V_out = [[round(v, 12) if abs(v) < 0.5 else round(v, 12) for v in row] for row in V_out]
    W_out = [[round(v, 12) if abs(v) < 0.5 else round(v, 12) for v in row] for row in W_out]
    
    # Snap to exact integers or half-integers where possible
    def snap(arr):
        result = []
        for row in arr:
            new_row = []
            for v in row:
                # Check if close to integer
                iv = round(v)
                if abs(v - iv) < 1e-10:
                    new_row.append(int(iv))
                else:
                    # Check if close to half-integer
                    hv = round(v * 2) / 2.0
                    if abs(v - hv) < 1e-10:
                        if hv == int(hv):
                            new_row.append(int(hv))
                        else:
                            new_row.append(hv)
                    else:
                        new_row.append(v)
            result.append(new_row)
        return result
    
    U_out = snap(U_out)
    V_out = snap(V_out)
    W_out = snap(W_out)
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
