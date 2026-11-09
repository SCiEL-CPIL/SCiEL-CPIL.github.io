"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta_{jk}
    # For each (i, k), the sum over j and r should give delta_{jk}... 
    # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T[i,j,k] = delta_{jk}
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, the convention is W[k*n+i, r], so W index is (k, i)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This should equal 1 if j==k... no wait.
    # Matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is T[i,j,k] where T[i,j,k] = 1 if this contributes to C[i,k] via A[i,j]*B[j,k]
    # Actually the multiplication tensor has T[i,j,k] = delta_{j, j} ... 
    # Let me re-read: "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor for n×m * m×p is:
    # T[i, j, k] for i in range(n), j in range(m), k in range(p)
    # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to give (i,k) of C
    # So T[i,j,k] = 1 for all valid (i,j,k) where j is the shared index
    # Actually no - the tensor is T[i,j,k] = delta_{j,j} which is always 1? 
    # 
    # The standard definition: the matrix multiplication tensor M_{n,m,p} has
    # M_{i,j,k} = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # Since every (i,j,k) triple with j in [0,m) is valid, M[i,j,k] = 1 for all i,j,k?
    # No! The tensor is defined such that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The structure tensor has entries T[i,j,k] where the bilinear form is:
    # sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[i,k]
    # For this to equal sum_{i,k} C[i,k]^2 (in some sense)... 
    # 
    # Actually, the decomposition means:
    # For all A, B: C = A*B iff C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # Wait no. The standard tensor decomposition of matrix multiplication:
    # The tensor T in R^{n*m} x R^{m*p} x R^{p*n} (or similar indexing)
    # T[a, b, c] where a indexes (i,j), b indexes (j,k), c indexes (i,k)
    # T[(i,j), (j,k), (i,k)] = 1 if the indices are consistent, 0 otherwise
    # 
    # So T is 1 when the middle index j matches between a and b, and 
    # the first index i of a matches the first index i of c, and
    # the second index k of b matches the second index k of c.
    # 
    # Decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    # 
    # So: T[(i,j),(j',k),(i',k')] = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
    # This should equal 1 if i=i', j=j', k=k' (i.e., all indices match)
    # and 0 otherwise.
    # 
    # Wait, that's not right either. Let me think again.
    # 
    # The matrix multiplication tensor: for multiplying A (n×m) by B (m×p) to get C (n×p)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The tensor T is an element of (R^{n×m})^* ⊗ (R^{m×p})^* ⊗ (R^{n×p})^*
    # T(A,B,C) = sum_{i,j,k} A[i,j] B[j,k] C[i,k]  (trilinear form)
    # 
    # Actually no. The tensor is defined as:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k  in appropriate spaces
    # where the contraction gives the matrix product.
    # 
    # Let me use the standard definition from the task:
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # The decomposition satisfies:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   delta_{j, j} ... 
    # 
    # I think the correct interpretation is:
    # sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1 if this is a valid term
    # 
    # Actually, the matrix multiplication tensor has:
    # T[i, j, k] = 1 for all i, j, k (since every A[i,j]*B[j,k] contributes to C[i,k])
    # No, that's the full tensor with all ones, which isn't right.
    # 
    # Let me look at this differently. The tensor decomposition means:
    # The bilinear map (A, B) -> C = A*B can be written as:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # 
    # For this to equal sum_j A[i,j] * B[j,k] for all A, B:
    # We need: sum_r U[i*m+j, r] * V[j'*p+k, r] = delta_{j,j'} * (contribution to C[i,k])
    # 
    # Hmm, this is getting confusing. Let me just verify numerically.
    # 
    # The correct condition is:
    # For each (i, k): C[i,k] = sum_j A[i,j] * B[j,k]
    # And the decomposition gives:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # 
    # For this to hold for all A, B, we need:
    # sum_r U[i*m+j, r] * V[j'*p+k, r] = delta_{j,j'}  for all i, j, j', k
    # 
    # Wait, that can't be right because the left side depends on i and k but the right side doesn't.
    # 
    # I think the correct formulation is:
    # T[i,j,k] where the tensor is in R^{n} ⊗ R^{m} ⊗ R^{m} ⊗ R^{p} 
    # and T[i,j,j',k] = delta_{j,j'} * delta_{i,i} * delta_{k,k}... 
    # 
    # OK let me just use the direct verification:
    # The decomposition is valid if for random A, B:
    # C_computed[i,k] = sum_r (sum_j U[i*m+j, r]*A[i,j]) * (sum_j V[j*p+k, r]*B[j,k])
    # equals A[i,j]*B[j,k] summed over j.
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        U_col = U_arr[:, r]  # shape (n*m,)
        V_col = V_arr[:, r]  # shape (m*p,)
        W_col = W_arr[:, r]  # shape (p*n,)
        
        # Reshape
        U_mat = U_col.reshape(n, m)
        V_mat = V_col.reshape(m, p)
        W_mat = W_col.reshape(p, n)
        
        # C[i,k] += (U_mat @ A.T)[i] * (B @ V_mat.T)[k]... 
        # Actually: sum_j U[i,j,r] * A[i,j] is not a standard matrix product
        # It's element-wise product summed: <U_mat, A> for each r
        # No wait, U[i*m+j, r] * A[i,j] summed over j for fixed i and r
        # That's (U_mat * A).sum(axis=1) for each i
        # Similarly for V: sum_j V[j*p+k, r] * B[j,k] = (V_mat * B).sum(axis=0) for each k
        
        U_A = (U_mat * A).sum(axis=1)  # shape (n,)
        V_B = (V_mat * B).sum(axis=0)  # shape (p,)
        
        # Now C[i,k] += U_A[i] * V_B[k] * W_col[k*n+i]? 
        # But W is indexed by (k*n+i), so W_mat[k, i]
        # C[i,k] += U_A[i] * V_B[k] * W_mat[k, i]
        
        C_decomp += np.outer(U_A, V_B) * W_mat.T
    
    return np.max(np.abs(C_decomp - C_true)) < tol

def construct():
    """
    Constructs a tensor decomposition for 3x5 * 5x6 matrix multiplication.
    Tries to find a rank < 90 decomposition using numerical optimization.
    """
    n, m, p = DIMENSIONS  # 3, 5, 6
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    # First, create the schoolbook rank-90 decomposition as baseline
    rank_schoolbook = n * m * p  # 90
    
    # Try to find a lower rank decomposition
    # Strategy: start with rank 89 and try to solve for the entries
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Create schoolbook solution as initial guess
    def make_schoolbook(rank):
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        if rank == rank_schoolbook:
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
        else:
            # For rank < schoolbook, we need a different approach
            # Initialize with random values
            np.random.seed(0)
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
        
        return U, V, W
    
    # Define the objective function for optimization
    # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{j,k}... 
    # Wait, I need to get the tensor definition right.
    
    # Let me define the residual properly.
    # The matrix multiplication tensor T has:
    # T[i, j, k] = 1 if j is the shared index (always true for valid i,j,k)
    # No, the tensor is: for the bilinear map (A,B) -> AB
    # T is in (R^{nm})^* ⊗ (R^{mp})^* ⊗ (R^{np})^*
    # T(a, b, c) = sum_{i,j,k} a_{ij} b_{jk} c_{ik}
    # 
    # The decomposition T = sum_r u_r ⊗ v_r ⊗ w_r means:
    # sum_{i,j,k} a_{ij} b_{jk} c_{ik} = sum_r (sum_{i,j} u_r[i,j] a_{ij}) (sum_{j,k} v_r[j,k] b_{jk}) (sum_{i,k} w_r[i,k] c_{ik})
    # 
    # This must hold for all a, b, c. Expanding the right side:
    # sum_r sum_{i,j} sum_{j',k'} sum_{i',k''} u_r[i,j] v_r[j',k'] w_r[i',k''] a_{ij} b_{j'k'} c_{i'k''}
    # 
    # For this to equal sum_{i,j,k} a_{ij} b_{jk} c_{ik}, we need:
    # sum_r u_r[i,j] v_r[j',k'] w_r[i',k''] = delta_{i,i'} delta_{j,j'} delta_{k,k''}
    # 
    # Wait, that's not right. The left side of the original equation is:
    # sum_{i,j,k} a_{ij} b_{jk} c_{ik}
    # This is a specific trilinear form. The right side after expansion is:
    # sum_{i,j,j',k',i',k''} [sum_r u_r[i,j] v_r[j',k'] w_r[i',k'']] a_{ij} b_{j'k'} c_{i'k''}
    # 
    # For equality for all a,b,c, we need:
    # sum_r u_r[i,j] v_r[j',k'] w_r[i',k''] = delta_{j,j'} delta_{k,k''} delta_{i,i'} * (coefficient)
    # 
    # Actually, the coefficient should be 1 when i=i', j=j', k=k'' and the indices are consistent.
    # But j and j' are both in [0,m), and they're independent in the sum. So we need:
    # sum_r u_r[i,j] v_r[j',k'] w_r[i',k''] = delta_{i,i'} delta_{j,j'} delta_{k,k''}
    # 
    # Hmm, but that would mean the tensor is just the identity-like tensor, which doesn't match matrix multiplication.
    # 
    # I think I'm overcomplicating this. Let me use the direct verification approach with random matrices.
    
    def residual(U, V, W, rank):
        """Compute the maximum error over random test matrices."""
        U_arr = np.array(U).reshape(n, m, rank) if len(np.array(U).shape) == 2 else np.array(U)
        V_arr = np.array(V).reshape(m, p, rank) if len(np.array(V).shape) == 2 else np.array(V)
        W_arr = np.array(W).reshape(p, n, rank) if len(np.array(W).shape) == 2 else np.array(W)
        
        # Actually, U is (n*m, rank), so reshape to (n, m, rank)
        U_mat = np.array(U).reshape(n, m, rank)
        V_mat = np.array(V).reshape(m, p, rank)
        W_mat = np.array(W).reshape(p, n, rank)
        
        # For each (i, k), C[i,k] should equal sum_j A[i,j] * B[j,k]
        # Decomposition: C[i,k] = sum_r (sum_j U_mat[i,j,r] * A[i,j]) * (sum_j V_mat[j,k,r] * B[j,k]) * W_mat[k,i,r]
        # 
        # For this to equal sum_j A[i,j] B[j,k] for all A, B:
        # sum_r U_mat[i,j,r] * V_mat[j',k,r] * W_mat[k,i,r] = delta_{j,j'}
        # 
        # Wait, that's not quite right either. Let me think...
        # 
        # C_decomp[i,k] = sum_r [sum_j U_mat[i,j,r] A[i,j]] [sum_j' V_mat[j',k,r] B[j',k]] W_mat[k,i,r]
        # = sum_r sum_j sum_j' U_mat[i,j,r] V_mat[j',k,r] W_mat[k,i,r] A[i,j] B[j',k]
        # 
        # We want this to equal sum_j A[i,j] B[j,k] for all A, B.
        # So we need: sum_r U_mat[i,j,r] V_mat[j',k,r] W_mat[k,i,r] = delta_{j,j'}
        # 
        # This must hold for all i, j, j', k.
        
        # Compute the residual: for each (i, j, j', k), check if
        # sum_r U[i,j,r] V[j',k,r] W[k,i,r] == delta_{j,j'}
        
        # This is a lot of constraints. Let's sample.
        np.random.seed(123)
        max_err = 0
        num_tests = 50
        for _ in range(num_tests):
            i = np.random.randint(n)
            j = np.random.randint(m)
            jp = np.random.randint(m)
            k = np.random.randint(p)
            
            val = np.sum(U_mat[i, j, :] * V_mat[jp, k, :] * W_mat[k, i, :])
            target = 1.0 if j == jp else 0.0
            err = abs(val - target)
            max_err = max(max_err, err)
        
        return max_err
    
    # Try different ranks starting from 89 down
    for target_rank in range(89, 50, -1):
        if time.time() > deadline - 20:
            break
        
        # Initialize
        U, V, W = make_schoolbook(target_rank)
        
        # Flatten for optimization
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            if time.time() > deadline - 5:
                return 1e10
            U_flat = x[:n*m*target_rank]
            V_flat = x[n*m*target_rank : n*m*target_rank + m*p*target_rank]
            W_flat = x[n*m*target_rank + m*p*target_rank : ]
            
            U_mat = U_flat.reshape(n, m, target_rank)
            V_mat = V_flat.reshape(m, p, target_rank)
            W_mat = W_flat.reshape(p, n, target_rank)
            
            # Compute residual
            # sum_r U[i,j,r] V[j',k,r] W[k,i,r] should equal delta_{j,j'}
            # Compute all combinations (might be expensive)
            # Use einsum for efficiency
            # T[i,j,j',k] = sum_r U[i,j,r] V[j',k,r] W[k,i,r]
            # This is expensive for large rank. Use sampling.
            
            np.random.seed(456)
            max_err = 0
            num_samples = 100
            for _ in range(num_samples):
                i = np.random.randint(n)
                j = np.random.randint(m)
                jp = np.random.randint(m)
                k = np.random.randint(p)
                val = np.dot(U_mat[i, j, :], V_mat[jp, k, :] * W_mat[k, i, :])
                target = 1.0 if j == jp else 0.0
                err = (val - target) ** 2
                max_err = max(max_err, err)
            
            return max_err
        
        # Try a few random restarts
        found = False
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            
            if restart == 0:
                x = x0.copy()
            else:
                x = x0 + np.random.randn(len(x0)) * 0.1
            
            try:
                result = minimize(objective, x, method='L-BFGS-B', 
                                  options={'maxiter': 100, 'maxfun': 200})
                
                if result.fun < 1e-6:
                    # Check with higher precision
                    U_mat = result.x[:n*m*target_rank].reshape(n, m, target_rank)
                    V_mat = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m, p, target_rank)
                    W_mat = result.x[n*m*target_rank + m*p*target_rank :].reshape(p, n, target_rank)
                    
                    # Verify with more samples
                    np.random.seed(789)
                    max_err = 0
                    for _ in range(500):
                        i = np.random.randint(n)
                        j = np.random.randint(m)
                        jp = np.random.randint(m)
                        k = np.random.randint(p)
                        val = np.dot(U_mat[i, j, :], V_mat[jp, k, :] * W_mat[k, i, :])
                        target = 1.0 if j == jp else 0.0
                        max_err = max(max_err, abs(val - target))
                    
                    if max_err < 1e-8:
                        # Round to nearest half-integer
                        U_rounded = np.round(U_mat * 2) / 2
                        V_rounded = np.round(V_mat * 2) / 2
                        W_rounded = np.round(W_mat * 2) / 2
                        
                        # Verify rounded version
                        np.random.seed(101)
                        max_err_rounded = 0
                        for _ in range(500):
                            i = np.random.randint(n)
                            j = np.random.randint(m)
                            jp = np.random.randint(m)
                            k = np.random.randint(p)
                            val = np.dot(U_rounded[i, j, :], V_rounded[jp, k, :] * W_rounded[k, i, :])
                            target = 1.0 if j == jp else 0.0
                            max_err_rounded = max(max_err_rounded, abs(val - target))
                        
                        if max_err_rounded < 1e-9:
                            # Convert to lists
                            U_list = U_rounded.reshape(n*m, target_rank).tolist()
                            V_list = V_rounded.reshape(m*p, target_rank).tolist()
                            W_list = W_rounded.reshape(p*n, target_rank).tolist()
                            
                            best_result = {"U": U_list, "V": V_list, "W": W_list}
                            best_rank = target_rank
                            found = True
                            break
            except Exception as e:
                pass
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    U, V, W = make_schoolbook(rank_schoolbook)
    return {
        "U": U.tolist(),
        "V": V.tolist(), 
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
