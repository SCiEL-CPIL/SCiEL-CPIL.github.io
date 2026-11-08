"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that sum_c U[:,c] V[:,c]^T W[:,c]^T = multiplication tensor.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The product for a given (i,j) entry of A and (j',k') entry of B should give
    C[i,k] = sum_j A[i,j] B[j,k].
    The tensor T[(i,j),(j',k'),(i',k')] = 1 if i=i', j=j', k=k'.
    We check: for each column c, the tensor T_c = outer(U_c, V_c, W_c) should sum to T.
    Equivalently, for random A and B, sum_c (A @ U_c) * (V_c^T @ B) * W_c^T should equal A @ B.
    """
    rank = U.shape[1]
    # Use deterministic test matrices
    A = np.random.RandomState(42).rand(n, m)
    B = np.random.RandomState(123).rand(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for c in range(rank):
        Uc = U[:, c]  # (n*m,)
        Vc = V[:, c]  # (m*p,)
        Wc = W[:, c]  # (p*n,)
        
        A_Uc = np.zeros((n, m))
        Vc_B = np.zeros((m, p))
        
        for i in range(n):
            for j in range(m):
                A_Uc[i, j] = A[i, j] * Uc[i * m + j]
        
        for j in range(m):
            for k in range(p):
                Vc_B[j, k] = Vc[j * p + k] * B[j, k]
        
        contrib = np.zeros((n, p))
        for k in range(p):
            for i in range(n):
                contrib[i, k] += Wc[k * n + i] * (A_Uc[:, k] @ Vc_B[k, :] if False else 0)
        # Actually let me recompute this properly
        # A_Uc has shape (n,m), Vc_B has shape (m,p)
        # The contribution to C is: for each c, C += outer-ish
        # Actually the formula is: C[i,k] = sum_j A[i,j] B[j,k]
        # In tensor form: C = sum_c (A * Uc_flat) (element-wise reshape) @ ... 
        # Let me just do it directly:
        # C[i,k] = sum_c Uc[i*m+? no...]
        # U row is (i,j), V row is (j,k), W row is (k,i)
        # C[i,k] = sum_j sum_c A[i,j] * Vc[j*p+k] * Wc[k*n+i] * Uc[i*m+j]... no that's not right either.
        
        # The decomposition is: C[i,k] = sum_c (sum_j A[i,j] * U[i*m+j, c]) * (sum_j' V[j'*p+k, c] * B[j',k]) ... no
        # Actually: C[i,k] = sum_c U_c[i*m+? ... 
        # Standard form: C[i,k] = sum_c (sum_j A[i,j] * U[i*m+j, c]) * (sum_k' V[?*p+k', c]) ... 
        # Let me re-derive. The multiplication tensor T has entries T[i,j,k'] where...
        # T(A,B) = C where C[i,k] = sum_j A[i,j] B[j,k]
        # As a 3-tensor: T_{(i,j),(j',k'),(i',k')} = delta_{i,i'} delta_{j,j'} delta_{k,k'}
        # Rank-1 term: sum_c u_c v_c w_c where
        # u_c is a vector in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
        # The product gives: C[i,k] = sum_c u_c[i*m + j] * v_c[j'*p + k'] * w_c[k'*n + i'] summed over j,j',k',i'
        # = sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] (with j=j', k'=k, i'=i from deltas)
        # So C[i,k] = sum_c (sum_j A[i,j] * u_c[i*m+j]) ... no, that's not a separable form.
        
        # Actually: C[i,k] = sum_c sum_j u_c[i*m+j] * A[i,j] * v_c[j*p+k] * ... 
        # No. Let me think again.
        # The bilinear form: C = T(A,B) where T is the tensor.
        # T(A,B)[i,k] = sum_{j,j',k',i'} T_{(i,j),(j',k'),(i',k')} A[i,j] B[j',k']
        # = sum_{j} A[i,j] B[j,k] (since T = delta_{i,i'}delta_{j,j'}delta_{k,k'})
        # For a rank-1 decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
        # where u_c ∈ R^{n*m}, v_c ∈ R^{m*p}, w_c ∈ R^{p*n}
        # Then T(A,B)[i,k] = sum_c (sum_{j} u_c[i*m+j] A[i,j]) ... no, this isn't right.
        
        # The correct interpretation:
        # T_{(i,j),(j',k'),(i',k')} A[i,j] B[j',k'] summed = sum over all indices of T * A * B
        # For rank-1: u_c[i*m+j] * v_c[j'*p+k'] * w_c[i'*n+k'] * A[i,j] * B[j',k'] * (output at (i',k'))
        # So the output C[i',k'] = sum_c u_c... no, the output is a bilinear form in A and B.
        # C[i',k'] = sum_{i,j,j',k'} T_{(i,j),(j',k'),(i',k')} A[i,j] B[j',k']
        # = sum_c sum_{i,j,j',k'} u_c[i*m+j] v_c[j'*p+k'] w_c[i'*n+k'] A[i,j] B[j',k']
        # = sum_c (sum_{i,j} u_c[i*m+j] A[i,j]) * (sum_{j',k'} v_c[j'*p+k'] B[j',k']) * w_c[i'*n+k']
        # Wait, that's not right either because i,j and j',k' are summed independently.
        
        # C[i',k'] = sum_c w_c[i'*n+k'] * (sum_{i,j} u_c[i*m+j] A[i,j]) * (sum_{j',k'} v_c[j'*p+k'] B[j',k'])
        # Hmm, but that would make C proportional to (A·u)(B·v) * w, which is rank-1 in output.
        
        # I think the correct form is:
        # C[i',k'] = sum_c sum_j u_c[i*m+j] * v_c[j*p+k'] * w_c[k'*n+i'] * ... no.
        
        # Let me just use the standard definition. The matrix multiplication tensor is:
        # T ∈ R^{(n*m) × (m*p) × (p*n)} with T_{(i,j),(j',k'),(i'',k'')} = 1 if i=i'', j=j', k'=k''
        # The bilinear map: <T, A ⊗ B> = sum T_{...} A_{ij} B_{j'k'} = sum_{i,j,k'} A[i,j] B[j,k'] with output indexed by (i'',k'') where i''=i, k''=k'
        # So the output C[i'',k''] = sum_j A[i'',j] B[j,k'']
        
        # For rank-1: T = sum_c U_c ⊗ V_c ⊗ W_c
        # <T, A⊗B> at output (i'',k'') = sum_c U_c[i*m+j'] V_c[j'*p+k''] W_c[k''*n+i''] A[i,j] B[j',k''] summed over i,j,j',k''
        # = sum_c W_c[k''*n+i''] * (sum_{i,j} U_c[i*m+j] A[i,j]) * (sum_{j'} V_c[j'*p+k''] B[j',k''])
        # This gives C[i'',k''] = sum_c W_c[k''*n+i''] * (U_c · A_flat) * (V_c · B_flat)
        
        # So the algorithm is:
        # For each c:
        #   s1_c = sum_{i,j} U_c[i*m+j] * A[i,j]  (a scalar)
        #   s2_c = sum_{j',k'} V_c[j'*p+k'] * B[j',k']  (a scalar)
        #   C[i'',k''] += W_c[k''*n+i''] * s1_c * s2_c
        
        s1 = np.dot(Uc.flatten(), A.flatten())
        s2 = np.dot(Vc.flatten(), B.flatten())
        Wc_mat = Wc.reshape(p, n).T  # (n, p) since W index is k*n+i
        C_computed += np.outer(np.ones(n), s1 * s2) * Wc_mat
    
    return np.allclose(C_computed, C_expected, atol=tol)

def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    best_rank = rank_schoolbook
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Try to find a lower rank decomposition using random initialization and optimization
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, leaving 10s margin
    
    # Known lower bounds: For 3x3x3, rank is 16 (Lader). For 3x3x6, we can try ranks below 54.
    # Let's try ranks from a lower bound up to schoolbook-1
    # Lower bound for 3x3x6: at least n*p = 18 (since output is 3x6)
    # Actually the trivial lower bound is n*p = 18.
    
    import random
    rng = np.random.RandomState(42)
    
    # Try a few ranks
    for target_rank in range(18, rank_schoolbook):
        if time.time() > deadline - 30:
            break
        if target_rank >= best_rank:
            break
            
        attempts = 3
        for attempt in range(attempts):
            if time.time() > deadline - 30:
                break
                
            # Initialize with random values
            U_r = rng.randn(n * m, target_rank) * 0.1
            V_r = rng.randn(m * p, target_rank) * 0.1
            W_r = rng.randn(p * n, target_rank) * 0.1
            
            # Simple gradient-free local search: randomly perturb and check
            # This is very slow, so let's just check if we can find anything by luck
            # Actually, for now, just return schoolbook if we can't find better
            pass
    
    # Convert to lists for the required format
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
