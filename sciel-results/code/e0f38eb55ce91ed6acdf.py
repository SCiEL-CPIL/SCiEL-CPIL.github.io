"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    # Build the full multiplication tensor
    # T[i, j, k] = 1 for the schoolbook multiplication tensor
    # The tensor has shape (n*m, m*p, p*n) in the factor convention
    # But let's work with the 3D tensor T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # and the decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # The target tensor in 3D form: T3[i, j, k] = 1 for all valid (i,j,k)
    # Actually the multiplication tensor T has T[i,j,k] = 1 for all i,j,k in their ranges
    # because (AB)[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is just all ones in 3D!
    
    # Wait, let me reconsider. The matrix multiplication tensor T is defined by:
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # So the 3D tensor T3[i,j,k] = 1 for all i,j,k
    T3 = np.ones((n, m, p), dtype=np.float64)
    
    # Rank-1 extraction via iterative method
    # A rank-1 component: u[i] * v[j] * w[k]
    # We want to find u, v, w that best fit the residual
    
    def extract_rank1(residual, max_iter=50, tol=1e-10):
        """Extract best rank-1 component from 3D tensor using power iteration."""
        # Initialize randomly
        np.random.seed(42)
        u = np.random.randn(n)
        v = np.random.randn(m)
        w = np.random.randn(p)
        
        for _ in range(max_iter):
            # Update w: w[k] = sum_{i,j} residual[i,j,k] * u[i] * v[j]
            w_new = np.einsum('ijk,i,j->k', residual, u, v)
            # Update u: u[i] = sum_{j,k} residual[i,j,k] * v[j] * w[k]
            u_new = np.einsum('ijk,j,k->i', residual, v, w_new)
            # Update v: v[j] = sum_{i,k} residual[i,j,k] * u[i] * w[k]
            v_new = np.einsum('ijk,i,k->j', residual, u_new, w_new)
            
            # Normalize to prevent blowup
            nu = np.linalg.norm(u_new)
            nv = np.linalg.norm(v_new)
            nw = np.linalg.norm(w_new)
            if nu < 1e-15 or nv < 1e-15 or nw < 1e-15:
                break
            u_new /= nu
            v_new /= nv
            w_new /= nw
            
            # Check convergence
            if (np.linalg.norm(u_new - u) < tol and 
                np.linalg.norm(v_new - v) < tol and 
                np.linalg.norm(w_new - w) < tol):
                u, v, w = u_new, v_new, w_new
                break
            u, v, w = u_new, v_new, w_new
        
        # Compute the scaling factor
        # The component is alpha * u[i] * v[j] * w[k]
        # alpha = sum(T * u*v*w) / sum(u^2 * v^2 * w^2)
        uvw = np.einsum('i,j,k->ijk', u, v, w)
        num = np.sum(residual * uvw)
        den = np.sum(uvw * uvw)
        if den < 1e-15:
            alpha = 0.0
        else:
            alpha = num / den
        
        return alpha, u, v, w
    
    # Greedy peeling algorithm
    residual = T3.copy()
    U_list = []
    V_list = []
    W_list = []
    
    best_rank = n * m * p  # schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook solution as fallback
    U_school = np.zeros((n*m, n*m*p))
    V_school = np.zeros((m*p, n*m*p))
    W_school = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()
    
    # Try greedy peeling
    for rank in range(1, best_rank + 1):
        if time.time() > deadline - 20:
            break
        
        # Extract rank-1 component
        alpha, u, v, w = extract_rank1(residual)
        
        if abs(alpha) < 1e-12:
            # Residual is essentially zero
            if rank - 1 < best_rank:
                best_rank = rank - 1
                # Build factor matrices
                if U_list:
                    U_mat = np.zeros((n*m, rank-1))
                    V_mat = np.zeros((m*p, rank-1))
                    W_mat = np.zeros((p*n, rank-1))
                    for c in range(rank-1):
                        a_c, u_c, v_c, w_c = U_list[c], V_list[c], W_list[c]
                        for i in range(n):
                            for j in range(m):
                                U_mat[i*m+j, c] = a_c * u_c[i]
                        for j in range(m):
                            for k in range(p):
                                V_mat[j*p+k, c] = v_c[j]
                        for k in range(p):
                            for i in range(n):
                                W_mat[k*n+i, c] = w_c[i]
                    best_U = U_mat.tolist()
                    best_V = V_mat.tolist()
                    best_W = W_mat.tolist()
            break
        
        # Subtract the component
        residual -= alpha * np.einsum('i,j,k->ijk', u, v, w)
        
        # Store for later reconstruction
        U_list.append((alpha, u, v, w))
        
        # Check if residual is close to zero
        if np.linalg.norm(residual) < 1e-10:
            if rank < best_rank:
                best_rank = rank
                U_mat = np.zeros((n*m, rank))
                V_mat = np.zeros((m*p, rank))
                W_mat = np.zeros((p*n, rank))
                for c in range(rank):
                    a_c, u_c, v_c, w_c = U_list[c]
                    for i in range(n):
                        for j in range(m):
                            U_mat[i*m+j, c] = a_c * u_c[i]
                    for j in range(m):
                        for k in range(p):
                            V_mat[j*p+k, c] = v_c[j]
                    for k in range(p):
                        for i in range(n):
                            W_mat[k*n+i, c] = w_c[i]
                best_U = U_mat.tolist()
                best_V = V_mat.tolist()
                best_W = W_mat.tolist()
            break
        
        # Verify the decomposition so far is still valid (should be by construction)
        # but check that we're making progress
    
    # If we didn't find improvement, try a different approach:
    # Use randomized initialization and multiple restarts
    if best_rank == n * m * p and time.time() < deadline - 30:
        # Try with different random seeds for the rank-1 extraction
        for seed in range(10):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(seed)
            residual2 = T3.copy()
            U_list2 = []
            
            for rank in range(1, best_rank + 1):
                if time.time() > deadline - 20:
                    break
                
                # Extract with random init
                u = np.random.randn(n)
                v = np.random.randn(m)
                w = np.random.randn(p)
                
                for _ in range(100):
                    w_new = np.einsum('ijk,i,j->k', residual2, u, v)
                    u_new = np.einsum('ijk,j,k->i', residual2, v, w_new)
                    v_new = np.einsum('ijk,i,k->j', residual2, u_new, w_new)
                    
                    nu = np.linalg.norm(u_new)
                    nv = np.linalg.norm(v_new)
                    nw = np.linalg.norm(w_new)
                    if nu < 1e-15 or nv < 1e-15 or nw < 1e-15:
                        break
                    u_new /= nu
                    v_new /= nv
                    w_new /= nw
                    
                    if (np.linalg.norm(u_new - u) < 1e-12 and 
                        np.linalg.norm(v_new - v) < 1e-12 and 
                        np.linalg.norm(w_new - w) < 1e-12):
                        u, v, w = u_new, v_new, w_new
                        break
                    u, v, w = u_new, v_new, w_new
                
                uvw = np.einsum('i,j,k->ijk', u, v, w)
                num = np.sum(residual2 * uvw)
                den = np.sum(uvw * uvw)
                if den < 1e-15:
                    alpha = 0.0
                else:
                    alpha = num / den
                
                if abs(alpha) < 1e-12:
                    if rank - 1 < best_rank:
                        best_rank = rank - 1
                        U_mat = np.zeros((n*m, rank-1))
                        V_mat = np.zeros((m*p, rank-1))
                        W_mat = np.zeros((p*n, rank-1))
                        for c in range(rank-1):
                            a_c, u_c, v_c, w_c = U_list2[c]
                            for i in range(n):
                                for j in range(m):
                                    U_mat[i*m+j, c] = a_c * u_c[i]
                            for j in range(m):
                                for k in range(p):
                                    V_mat[j*p+k, c] = v_c[j]
                            for k in range(p):
                                for i in range(n):
                                    W_mat[k*n+i, c] = w_c[i]
                        best_U = U_mat.tolist()
                        best_V = V_mat.tolist()
                        best_W = W_mat.tolist()
                    break
                
                residual2 -= alpha * np.einsum('i,j,k->ijk', u, v, w)
                U_list2.append((alpha, u, v, w))
                
                if np.linalg.norm(residual2) < 1e-10:
                    if rank < best_rank:
                        best_rank = rank
                        U_mat = np.zeros((n*m, rank))
                        V_mat = np.zeros((m*p, rank))
                        W_mat = np.zeros((p*n, rank))
                        for c in range(rank):
                            a_c, u_c, v_c, w_c = U_list2[c]
                            for i in range(n):
                                for j in range(m):
                                    U_mat[i*m+j, c] = a_c * u_c[i]
                            for j in range(m):
                                for k in range(p):
                                    V_mat[j*p+k, c] = v_c[j]
                            for k in range(p):
                                for i in range(n):
                                    W_mat[k*n+i, c] = w_c[i]
                        best_U = U_mat.tolist()
                        best_V = V_mat.tolist()
                        best_W = W_mat.tolist()
                    break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
