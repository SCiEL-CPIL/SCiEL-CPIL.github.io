"""Matrix multiplication 4x5x5: find a rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 4x5 * 5x5 matrix multiplication.
    
    We first try to find a lower-rank decomposition using randomized numerical
    methods and integer/half-integer rounding, falling back to the schoolbook
    decomposition (rank 100) if no valid lower-rank solution is found.
    """
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 100
    
    # Try to find lower-rank decompositions
    best_rank = schoolbook_rank
    best_decomp = None
    
    # Build schoolbook decomposition as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(nm)]
    V_sb = [[0] * schoolbook_rank for _ in range(mp)]
    W_sb = [[0] * schoolbook_rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_decomp = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try various ranks from low to high
    # For 4x5x5, the lower bound is max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    # But known lower bounds for matrix multiplication rank are higher
    # Let's try ranks from 60 to 99 (below schoolbook)
    
    def try_rank(rank, time_budget):
        """Try to find a rank decomposition of given rank."""
        start_time = time.time()
        
        # The tensor T for matrix multiplication has T[i,j,k] = 1 if the 
        # (i,j) entry of A times (j,k) entry of B contributes to (k,i) entry of C
        # Actually, the tensor is: T[a,b,c] where a indexes (i,j) in A, 
        # b indexes (j,k) in B, c indexes (k,i) in C
        # T[a,b,c] = 1 if j matches (the shared dimension)
        
        # Let's define the tensor more carefully:
        # A is n x m, B is m x p, C = A * B is n x p
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # 
        # Tensor form: C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
        # where T[i,j,k] = 1 if the indices are consistent (which they always are
        # when j is the same index)
        
        # In the decomposition, we want:
        # C[i,k] = sum_r U[i,k,r] * V[i,j,r] * W[j,k,r]
        # Wait, let me re-read the contract:
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # W follows the trace convention W[k*n+i, r]
        
        # So the decomposition is:
        # (A*B)[i,k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
        # for all i, j, k... wait, that doesn't make sense.
        
        # Let me reconsider. The standard tensor decomposition for matrix mult:
        # (AB)[i,k] = sum_j A[i,j] B[j,k]
        # = sum_r (sum_j' U[i*m+j', r] A[i,j']) (sum_k' V[j'*p+k', r] B[j',k']) W[k*n+i, r]
        # 
        # No, that's not right either. Let me think again.
        #
        # The multiplication tensor M_{n,m,p} has entries M[i,j,k,l] where 
        # (i,j) indexes A, (k,l) indexes B, and the product is C[i,l] = sum_j A[i,j]B[j,l]
        #
        # Actually, the simpler way: think of it as a 3-tensor T with indices (i,k,j)
        # where T[i,k,j] = 1 (the j is the summation index)
        # And the decomposition is:
        # T[i,k,j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Let's verify: 
        # sum_j T[i,k,j] A[i,j] B[j,k] = sum_j sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] A[i,j] B[j,k]
        # = sum_r W[k*n+i,r] (sum_j U[i*m+j,r] A[i,j]) (sum_j V[j*p+k,r] B[j,k])
        #
        # For this to equal sum_j A[i,j] B[j,k] for all A, B, we need:
        # For each r, U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] must form a rank-1 term
        # in the tensor, and the sum over r must reconstruct T.
        
        # The tensor T[i,k,j] has T[i,k,j] = 1 for all i,k,j (since the 
        # multiplication involves all j values). Wait, no:
        # T[i,k,j] should be such that sum_j T[i,k,j] A[i,j] B[j,k] = (AB)[i,k]
        # = sum_j A[i,j] B[j,k]
        # So T[i,k,j] = 1 for all i, k, j.
        
        # Hmm, that means the tensor is just all ones? That can't be right for
        # a useful decomposition.
        
        # Let me reconsider the tensor structure.
        # The bilinear map (A,B) -> AB can be written as:
        # (AB)[i,k] = sum_{j} A[i,j] * B[j,k]
        #
        # This is a trilinear form in A, B, and a "output" tensor C:
        # C[i,k] = sum_{j} A[i,j] B[j,k]
        #
        # The tensor T has shape (n*p, n*m, m*p) with:
        # T[i*k, i*m+j, j*p+k] = 1 for valid i,j,k
        #
        # Or equivalently, T[a, b, c] where:
        # a = i*n + k (output index) -- but the contract says W has shape (p*n, rank)
        # with W[k*n + i, r], so the output index is k*n + i
        # b = i*m + j (input A index)
        # c = j*p + k (input B index)
        
        # So T[output, a_idx, b_idx] = 1 when:
        # output = k*n + i
        # a_idx = i*m + j
        # b_idx = j*p + k
        # for some i in [0,n), j in [0,m), k in [0,p)
        
        # For a decomposition of rank r:
        # T[output, a_idx, b_idx] = sum_{r=0}^{R-1} W[output, r] * U[a_idx, r] * V[b_idx, r]
        
        # We need to find U (nm x R), V (mp x R), W (pn x R) such that this holds.
        
        # Let's build the target tensor as a 3D array
        T = np.zeros((pn, nm, mp), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[k * n + i, i * m + j, j * p + k] = 1.0
        
        # Try random initializations
        for trial in range(50):
            if time.time() > start_time + time_budget:
                break
            if time.time() > deadline:
                break
            
            np.random.seed(42 + trial)
            
            U_mat = np.random.randn(nm, rank) * 0.5
            V_mat = np.random.randn(mp, rank) * 0.5
            W_mat = np.random.randn(pn, rank) * 0.5
            
            # Use ALS (Alternating Least Squares) to minimize ||T - sum_r U[:,r] V[:,r] W[:,r]||_F^2
            for it in range(50):
                if time.time() > start_time + time_budget:
                    break
                if time.time() > deadline:
                    break
                
                # Update U
                for r in range(rank):
                    # T[:,:,r] projected: for each a_idx, U[a_idx, r] should minimize
                    # sum_{output, b_idx} (T[output, a_idx, b_idx] - W[output,r]*V[b_idx,r]*U[a_idx,r])^2
                    # = sum_{output, b_idx} T[output, a_idx, b_idx] * W[output,r] * V[b_idx,r]
                    # divided by sum_{output, b_idx} (W[output,r] * V[b_idx,r])^2
                    numerator = np.sum(T[:, a_idx, :] * W_mat[:, r] * V_mat[:, r] if False else 
                                      T[:, :, :] * 0)  # placeholder
                    
                    # Actually, let's do this more carefully with matrix operations
                    pass
                
                # This is getting complex. Let me use a simpler approach.
            
            # Alternative: use the fact that for each fixed r, the term
            # W[:,r] * outer(U[:,r], V[:,r]) should approximate the tensor.
            
            # Let's try a different approach: use scipy or numpy for the optimization.
            # Actually, let's just try integer rounding on the schoolbook decomposition
            # and see if we can merge columns.
            
            break  # Skip complex optimization for now
        
        # Try a simpler approach: start from schoolbook and try to merge
        # columns by finding linear dependencies.
        
        return None, None  # Placeholder
    
    # For now, let's try a more direct approach:
    # Use the structure of the problem. The tensor T[k*n+i, i*m+j, j*p+k] = 1.
    # Note that this is a very structured tensor.
    
    # One simple improvement: if n=m=p, Strassen gives rank 7 for 2x2x2.
    # For 4x5x5, we can potentially use block-based algorithms.
    
    # Let's try to construct a decomposition using block multiplication ideas.
    # A is 4x5, B is 5x5. We can write A = [A1 | A2] where A1 is 4x3, A2 is 4x2.
    # B = [B1; B2] where B1 is 3x5, B2 is 2x5.
    # Then AB = A1*B1 + A2*B2.
    # This doesn't directly reduce rank unless we use Strassen-like tricks.
    
    # Given the complexity, let's just return the schoolbook decomposition.
    # But first, let's verify it's correct.
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that the decomposition is exact."""
        R = len(U[0])
        # Compute sum_r U[a,r] V[b,r] W[c,r] for all a,b,c
        # and check it equals T[c, a, b]
        
        # Build T
        T = np.zeros((pn, nm, mp), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[k * n + i, i * m + j, j * p + k] = 1.0
        
        # Compute the decomposed tensor
        # D[c, a, b] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Using broadcasting:
        U_arr = np.array(U, dtype=float)  # nm x R
        V_arr = np.array(V, dtype=float)  # mp x R
        W_arr = np.array(W, dtype=float)  # pn x R
        
        # D[c, a, b] = sum_r W_arr[c, r] * U_arr[a, r] * V_arr[b, r]
        # = sum_r (W_arr[:, r] * U_arr[:, r][:, None]) * V_arr[:, r][None, :]
        
        # More efficiently:
        # D = np.einsum('cr,ar,br->cab', W_arr, U_arr, V_arr)
        
        D = np.einsum('cr,ar,br->cab', W_arr, U_arr, V_arr, optimize=True)
        
        max_err = np.max(np.abs(D - T))
        return max_err < tol, max_err
    
    # Verify the schoolbook decomposition
    valid, err = verify_decomposition(U_sb, V_sb, W_sb)
    if not valid:
        # This shouldn't happen, but just in case
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Now try to find a better decomposition
    # Try ranks from 99 down to 50
    for target_rank in range(schoolbook_rank - 1, 49, -1):
        if time.time() > deadline - 5.0:
            break
        
        # Try random restarts for this rank
        for trial in range(20):
            if time.time() > deadline - 3.0:
                break
            
            np.random.seed(12345 + target_rank * 100 + trial)
            
            # Initialize with random matrices
            U_mat = np.random.randn(nm, target_rank) * 0.3
            V_mat = np.random.randn(mp, target_rank) * 0.3
            W_mat = np.random.randn(pn, target_rank) * 0.3
            
            # Use ALS to minimize ||T - D||_F where D[c,a,b] = sum_r W[c,r] U[a,r] V[b,r]
            # For each r, the term is W[:,r] * np.outer(U[:,r], V[:,r])
            
            # ALS updates:
            # Fix V, W: for each a, U[a,:] should minimize
            # sum_{c,b} (T[c,a,b] - sum_r W[c,r] U[a,r] V[b,r])^2
            # = sum_r (W[c,r] V[b,r]) U[a,r] summed over c,b
            # This is a linear system for U[a,:]
            
            # For each row a of U:
            # min_{u} sum_{c,b} (T[c,a,b] - sum_r W[c,r] V[b,r] u[r])^2
            # Let M[a] be the matrix with M[a][r, (c,b)] = W[c,r] * V[b,r]
            # Then we want u such that M[a] @ u ≈ T[:,a,:] flattened
            
            # Similarly for V and W.
            
            converged = False
            for it in range(30):
                if time.time() > deadline - 2.0:
                    break
                
                # Update U
                for a in range(nm):
                    # Build the coefficient matrix
                    # M[a][r, idx] = W[c,r] * V[b,r] where idx = c*mp + b
                    M_a = np.zeros((target_rank, pn * mp))
                    for c in range(pn):
                        for b in range(mp):
                            idx = c * mp + b
                            for r in range(target_rank):
                                M_a[r, idx] = W_mat[c, r] * V_mat[b, r]
                    
                    T_a = T[:, a, :].flatten()
                    
                    # Solve M_a @ u = T_a in least squares sense
                    try:
                        u, residuals, rank_a, sv = np.linalg.lstsq(M_a, T_a, rcond=None)
                        U_mat[a, :] = u
                    except:
                        pass
                
                # Update V
                for b in range(mp):
                    M_b = np.zeros((target_rank, pn * nm))
                    for c in range(pn):
                        for a in range(nm):
                            idx = c * nm + a
                            for r in range(target_rank):
                                M_b[r, idx] = W_mat[c, r] * U_mat[a, r]
                    
                    T_b = T[:, :, b].transpose().flatten()
                    
                    try:
                        v, residuals, rank_b, sv = np.linalg.lstsq(M_b, T_b, rcond=None)
                        V_mat[b, :] = v
                    except:
                        pass
                
                # Update W
                for c in range(pn):
                    M_c = np.zeros((target_rank, nm * mp))
                    for a in range(nm):
                        for b in range(mp):
                            idx = a * mp + b
                            for r in range(target_rank):
                                M_c[r, idx] = U_mat[a, r] * V_mat[b, r]
                    
                    T_c = T[c, :, :].flatten()
                    
                    try:
                        w, residuals, rank_c, sv = np.linalg.lstsq(M_c, T_c, rcond=None)
                        W_mat[c, :] = w
                    except:
                        pass
                
                # Check convergence
                D = np.einsum('cr,ar,br->cab', W_mat, U_mat, V_mat, optimize=True)
                err = np.max(np.abs(D - T))
                if err < 1e-8:
                    converged = True
                    break
            
            if converged:
                # Found a valid decomposition!
                # Round to integers or half-integers
                U_rounded = np.round(U_mat * 2) / 2
                V_rounded = np.round(V_mat * 2) / 2
                W_rounded = np.round(W_mat * 2) / 2
                
                # Verify the rounded version
                D_r = np.einsum('cr,ar,br->cab', W_rounded, U_rounded, V_rounded, optimize=True)
                err_r = np.max(np.abs(D_r - T))
                
                if err_r < 1e-10:
                    # Success!
                    best_rank = target_rank
                    best_decomp = {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    break
            elif time.time() > deadline - 1.0:
                break
        
        if best_rank < schoolbook_rank:
            break
    
    return best_decomp
# EVOLVE-BLOCK-END
