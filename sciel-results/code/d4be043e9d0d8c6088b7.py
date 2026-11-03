"""Matrix multiplication tensor decomposition for 4x4 * 4x7."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    rank_school = n * m * p  # 112
    
    # Schoolbook decomposition as fallback
    U_sb = np.zeros((n * m, rank_school))
    V_sb = np.zeros((m * p, rank_school))
    W_sb = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Try to find a lower-rank decomposition using a subspace/projection approach
    # Strategy: Try to find a rank-r decomposition where r < 112
    
    # For a 4x4*4x7 multiplication, we can try various ranks
    # The tensor is T[i,j,k] where i in [0,4), j in [0,4), k in [0,7)
    # We want U (16 x r), V (28 x r), W (28 x r) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta[i,j,k] pattern
    
    # Actually, the multiplication tensor has entries:
    # T[i, j, k] = 1 if the output element (i,k) depends on input (i,j) and (j,k)
    # In standard form: (AB)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i,j,k (it's the identity-like structure)
    # Wait, no. The tensor for matrix multiplication is:
    # T[i,j,k] = delta_{i,k} in the sense that (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # The "tensor" T is defined by T_{i,j,k} = 1, and the multiplication is
    # (A*B)_{i,k} = sum_{j,c} A_{i,j} * B_{j,k}
    # The decomposition is: sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] = A[i,j]*B[j,k] for all A,B
    
    # Let me think more carefully. The matrix multiplication tensor T_M has:
    # T_M[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
    # No wait, that's not right either.
    
    # The correct formulation: We want to compute C = A * B where A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The multiplication tensor T has T[i,j,k] = 1, and we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # This means U, V, W decompose the all-ones tensor? No...
    
    # Actually, I think the standard formulation is:
    # The multiplication map mu: A -> A*B is bilinear.
    # We can write mu(A,B) = sum_c (sum_{i,j} U[i,j,c] A[i,j]) * (sum_{j,k} V[j,k,c] B[j,k]) * W[...]
    # Hmm, let me reconsider.
    
    # The correct tensor decomposition:
    # We have tensors U in R^{nm x r}, V in R^{mp x r}, W in R^{pn x r}
    # such that for the multiplication tensor T (which is the identity in some sense):
    # T[i*m+j, j*p+k, k*n+i] ... 
    
    # Let me use the standard result: The rank of matrix multiplication M_{n,m,p} 
    # is the minimum r such that there exist U in F^{nm x r}, V in F^{mp x r}, W in F^{pn x r}
    # with sum_c U[i,j,c] V[j,k,c] W[k,i,c] = delta_{i,k} ... no.
    
    # OK let me just be concrete. The multiplication tensor is:
    # T = sum_{i,j,k} e_{i,j}^{(A)} ⊗ e_{j,k}^{(B)} ⊗ e_{k,i}^{(C)} ... 
    # No, C[i,k] = sum_j A[i,j] B[j,k], so:
    # T = sum_{i,j,k} (e_i ⊗ e_j) ⊗ (e_j ⊗ e_k) ⊗ (e_k ⊗ e_i)?? 
    # That doesn't work dimensionally.
    
    # Let me think of it as: the bilinear map mu: (A,B) -> AB
    # mu(A,B)[i,k] = sum_j A[i,j] B[j,k]
    # This is a bilinear map from (R^{nm}, R^{mp}) to R^{np}
    # Its tensor T_mu in R^{nm x mp x np} has T_mu[i*m+j, j*p+k, k*n+i] = 1
    # and 0 otherwise? No, that's not right either.
    
    # Actually: T_mu[i*m+j, j*p+k, k*n+i] should give the coefficient such that
    # (mu(A,B))[i,k] = sum_{i*m+j, j*p+k} A[i*m+j] B[j*p+k] T[i*m+j, j*p+k, k*n+i]
    # We need: sum_j A[i*m+j] B[j*p+k] = (AB)[i,k]
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise.
    # But wait, for a fixed (i,k), the sum is over j, and T[i*m+j, j*p+k, k*n+i] = 1.
    # For different (i',k'), T[i*m+j, j*p+k, k'*n+i'] = 0 unless (i',k') = (i,k).
    
    # So the tensor T has: T[a, b, c] = 1 if there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
    # Otherwise T[a,b,c] = 0.
    
    # Note: a = i*m+j means i = a//m, j = a%m
    # b = j*p+k means j = b//p, k = b%p
    # c = k*n+i means k = c//n, i = c%n
    # For consistency: a//m == c%n, a%m == b//p, b%p == c//n
    
    # Let's construct this tensor and try to find a low-rank decomposition.
    
    def build_tensor(n, m, p):
        """Build the matrix multiplication tensor as a 3D array."""
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T
    
    T = build_tensor(n, m, p)
    
    # The rank is the minimum r such that T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # i.e., T[a,b,c] = sum_c' U[a,c'] V[b,c'] W[c,c']
    
    # Try CP decomposition approach: use a simple greedy/greedy-like approach
    # Or use the fact that we can try specific structured decompositions
    
    # Let's try a few candidate ranks and see if we can find a valid decomposition
    # using a simple iterative approach
    
    def try_rank_decomposition(T, rank, max_iter=200, tol=1e-8):
        """Try to find a rank-r CP decomposition of T using iterative fitting.
        Returns (U, V, W) or None."""
        nm, mp, pn = T.shape
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        T_flat = T.flatten()
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Update U
            # T[a,b,c] = sum_c' U[a,c'] V[b,c'] W[c,c']
            # Fix V, W, update U:
            # For each a: T[a,:,:] = sum_c' U[a,c'] (V[:,c'] ⊗ W[:,c'])^T
            # This is a linear system: T[a,:,:].reshape(-1) = U[a,:] @ (V*W).T where V*W is mp*pn x rank
            VW = np.zeros((mp * pn, rank))
            for c_idx in range(rank):
                VW[:, c_idx] = (V[:, c_idx] * W[:, c_idx]).flatten()
            
            # For each a, solve: T[a,:,:].flatten() = VW.T @ U[a,:]
            # This is overdetermined in general, use least squares
            # Actually, T[a,:,:].flatten() is a vector in R^{mp*pn}, and VW is mp*pn x rank
            # So we want U[a,:] such that VW.T @ U[a,:] ≈ T[a,:,:].flatten()
            # This is: U[a,:] = (VW @ VW.T)^{-1} VW @ T[a,:,:].flatten() if rank <= mp*pn
            
            # More efficiently: compute U = T_reshaped @ (VW @ VW.T)^{-1} VW.T
            # where T_reshaped is nm x (mp*pn)
            T_reshaped = T.reshape(nm, mp * pn)
            G = VW.T @ VW  # rank x rank
            try:
                G_inv = np.linalg.pinv(G)
            except:
                return None
            U_new = T_reshaped @ VW @ G_inv
            U = U_new
            
            # Update V similarly
            UW = np.zeros((nm * pn, rank))
            for c_idx in range(rank):
                UW[:, c_idx] = (U[:, c_idx] * W[:, c_idx]).flatten()
            T_reshaped2 = T.reshape(mp, nm * pn)
            G2 = UW.T @ UW
            try:
                G2_inv = np.linalg.pinv(G2)
            except:
                return None
            V_new = T_reshaped2 @ UW @ G2_inv
            V = V_new
            
            # Update W
            UV = np.zeros((nm * mp, rank))
            for c_idx in range(rank):
                UV[:, c_idx] = (U[:, c_idx] * V[:, c_idx]).flatten()
            T_reshaped3 = T.reshape(pn, nm * mp)
            G3 = UV.T @ UV
            try:
                G3_inv = np.linalg.pinv(G3)
            except:
                return None
            W_new = T_reshaped3 @ UV @ G3_inv
            W = W_new
            
            # Check convergence
            T_approx = np.zeros_like(T)
            for c_idx in range(rank):
                T_approx += np.einsum('i,j,k->ijk', U[:, c_idx], V[:, c_idx], W[:, c_idx])
            err = np.linalg.norm(T_approx - T)
            if err < tol:
                return U, V, W
        
        # Check final error
        T_approx = np.zeros_like(T)
        for c_idx in range(rank):
            T_approx += np.einsum('i,j,k->ijk', U[:, c_idx], V[:, c_idx], W[:, c_idx])
        err = np.linalg.norm(T_approx - T)
        if err < tol:
            return U, V, W
        return None
    
    # Try ranks from lower to higher, but limit the search
    best_result = (U_sb.tolist(), V_sb.tolist(), W_sb.tolist())
    best_rank = rank_school
    
    # Try some candidate ranks
    # For 4x4*4x7, the schoolbook is 112. 
    # Known bounds: R(4,4,7) <= 104 (from literature) or similar
    # Let's try ranks: 100, 104, 108, 112
    candidate_ranks = [96, 100, 104, 108]
    
    for r in candidate_ranks:
        if time.time() > deadline - 10:
            break
        result = try_rank_decomposition(T, r, max_iter=50)
        if result is not None:
            U_r, V_r, W_r = result
            best_result = (U_r.tolist(), V_r.tolist(), W_r.tolist())
            best_rank = r
            break  # Found a valid one, use it
    
    # Convert to list format and return
    U_list, V_list, W_list = best_result
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
