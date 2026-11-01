"""Strategy: Numerical search for rank-25 decomposition using random restarts and local optimization."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin

    # 1. Schoolbook fallback (Rank 28)
    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k, col] = 1
                    V[k * p + j, col] = 1
                    W[j * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, r

    best_result, best_rank = schoolbook()

    # 2. Verification function
    def verify(U, V, W):
        """
        U: (n*m, r)
        V: (m*p, r)
        W: (p*n, r)
        Check if sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] == 1
        """
        r = U.shape[1]
        u_r = U.T.reshape(n, m, r)
        v_r = V.T.reshape(m, p, r)
        w_r = W.T.reshape(p, n, r)
        # T[i,j,k] should be 1 if it corresponds to the multiplication tensor
        # The tensor for matrix multiplication C = A * B is T[i, j, k] = 1 if A[i,k]*B[k,j]
        # Here indices are:
        # A is n x m (i, k)
        # B is m x p (k, j)
        # C is n x p (i, j)
        # Wait, the task says "2 by 2 matrix by a 2 by 7 matrix".
        # A: 2x2 (n=2, m=2)
        # B: 2x7 (m=2, p=7)
        # C: 2x7 (n=2, p=7)
        
        # The trace convention W[k*n+i, r] is mentioned in the prompt?
        # "output factor follows the official trace convention W[k*n+i,r]"
        # Let's re-read carefully: "W[k*n+i,r]"
        # In the schoolbook code above:
        # U[i*m+k, col] = 1  -> Index i*m+k. A is n x m? 
        # If A is n x m, then row is i, col is k. Index i*m + k is standard row-major for n rows, m cols.
        # V[k*p+j, col] = 1  -> Index k*p + j. B is m x p. Row k, col j.
        # W[j*n+i, col] = 1  -> Index j*n + i. C is p x n? 
        # The prompt says "W[k*n+i,r]". This is confusing.
        # Let's look at the dimensions in the prompt:
        # "2 by 2 matrix by a 2 by 7 matrix"
        # A: 2x2. B: 2x7. C: 2x7.
        # Usually C[i,j] = sum_k A[i,k] B[k,j].
        # If A is n x m, B is m x p, C is n x p.
        # n=2, m=2, p=7.
        # U rows: n*m = 4. Index i*m + k.
        # V rows: m*p = 14. Index k*p + j.
        # W rows: p*n = 14. Index ?
        # The prompt says "W[k*n+i,r]". 
        # If W index is j*n + i (where j is col of C, i is row of C), then for C[i,j], 
        # the term is W[j*n+i].
        # Let's check the schoolbook code in the prompt:
        # W[j * n + i][col] = 1.
        # So the index is j*n + i.
        # The tensor identity is:
        # sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1
        # This matches C[i,j] = sum_k A[i,k] B[k,j].
        
        T = np.einsum('ikr,kjr,jir->ij', u_r, v_r, w_r)
        # We want T[i,j] to be 1 for all i,j? 
        # No, the tensor for matrix multiplication has non-zero entries only for specific (i,j,k) triples?
        # No, the tensor T is the 3rd order tensor representing the bilinear map.
        # T[i, j, k] = 1 if the term A[i,k] B[k,j] contributes to C[i,j].
        # So for a fixed (i,j), C[i,j] = sum_k T[i,j,k] A[i,k] B[k,j].
        # In our decomposition, C[i,j] = sum_r (sum_k U[i*m+k,r] A[i,k]) (sum_k' V[k'*p+j,r] B[k',j])?
        # No, the bilinear form is sum_r U[i,m,k,r] V[k,m,j,r] W[j,n,i,r] A[i,k] B[k,j].
        # Wait, the standard decomposition is:
        # C[i,j] = sum_r (sum_k U[i,k,r] A[i,k]) (sum_k' V[k',j,r] B[k',j]) ? No.
        
        # Let's stick to the verification in the seed program which seemed to work for schoolbook.
        # Seed program verify:
        # T = np.einsum('ikc,kjc,jic->ijk', u_r, v_r, w_r)
        # return np.allclose(T, 1.0, atol=1e-9)
        
        # Let's trace the indices in the seed's verify:
        # u_r: (n, m, r) -> u_r[i,k,c] = U[i*m+k, c]
        # v_r: (m, p, r) -> v_r[k,j,c] = V[k*p+j, c]
        # w_r: (p, n, r) -> w_r[j,i,c] = W[j*n+i, c]
        
        # einsum 'ikc,kjc,jic->ijk'
        # Result T[i,j,k] = sum_c u_r[i,k,c] * v_r[k,j,c] * w_r[j,i,c]
        # This sums over c (rank).
        # The result is a tensor T of shape (n, p, m).
        # The seed checks if T is all 1s.
        # Does T[i,j,k] = 1 for all i,j,k?
        # For schoolbook:
        # U[i*m+k, c] is 1 only if c = i*m+k.
        # V[k*p+j, c] is 1 only if c = k*p+j.
        # W[j*n+i, c] is 1 only if c = j*n+i.
        # For the product to be non-zero, we need c = i*m+k = k*p+j = j*n+i.
        # This is NOT true for all i,j,k.
        # Example: n=2, m=2, p=7.
        # i=0, k=0, j=0.
        # c_U = 0*2+0 = 0.
        # c_V = 0*7+0 = 0.
        # c_W = 0*2+0 = 0.
        # Match. T[0,0,0] = 1.
        
        # i=0, k=0, j=1.
        # c_U = 0.
        # c_V = 1.
        # c_W = 1*2+0 = 2.
        # No match. T[0,1,0] = 0.
        
        # So the seed's verification `np.allclose(T, 1.0)` would FAIL for schoolbook?
        # Let's check the seed code again.
        # "T = np.einsum('ikc,kjc,jic->ijk', u_r, v_r, w_r)"
        # "return np.allclose(T, 1.0, atol=1e-9)"
        
        # If the seed scored 1.0, it means it returned the schoolbook result and the evaluator accepted it.
        # But if the verification function inside the seed was used to check the result, it would have failed?
        # The seed code defines `verify_fast` but never calls it on `best` before returning.
        # It just returns `best`.
        # The evaluator presumably has its own verification.
        # The evaluator checks if the decomposition is exact.
        # What is the "exact" condition?
        # The tensor for matrix multiplication A (n x m) * B (m x p) -> C (n x p) is:
        # T[i, j, k] = 1 if the term A[i,k] B[k,j] is part of the sum for C[i,j].
        # So T[i, j, k] = 1 for all valid i,j,k? 
        # Yes, because C[i,j] = sum_{k=0}^{m-1} A[i,k] B[k,j].
        # So the tensor is 1 everywhere in the (n, p, m) space?
        # No, the tensor is defined on indices (i, j, k) where i in 0..n-1, j in 0..p-1, k in 0..m-1.
        # And the value is 1 for all such triples.
        # So yes, the tensor is the all-ones tensor of shape (n, p, m).
        
        # Therefore, the condition is:
        # sum_c U[i*m+k, c] V[k*p+j, c] W[j*n+i, c] = 1 for all i, j, k.
        
        # The seed's einsum 'ikc,kjc,jic->ijk' produces T[i,j,k].
        # So `np.allclose(T, 1.0)` is the correct check.
        
        # My previous trace showed T[0,1,0] = 0 for schoolbook. Why?
        # i=0, j=1, k=0.
        # c_U = 0.
        # c_V = 1.
        # c_W = 2.
        # Product is 0.
        # But the tensor T[0,1,0] SHOULD be 1.
        # So schoolbook DOES NOT satisfy the condition?
        
        # Let's re-read the schoolbook construction.
        # U[i*m+k, col] = 1.
        # V[k*p+j, col] = 1.
        # W[j*n+i, col] = 1.
        # col is unique for each (i,k,j).
        # So for a specific (i,k,j), there is a column c such that:
        # U[i*m+k, c] = 1
        # V[k*p+j, c] = 1
        # W[j*n+i, c] = 1
        # And for this c, the product is 1.
        
        # Now consider the sum for T[i, j, k].
        # T[i, j, k] = sum_c U[i*m+k, c] V[k*p+j, c] W[j*n+i, c].
        # For a fixed i, j, k, how many columns c have U[i*m+k, c]=1 AND V[k*p+j, c]=1 AND W[j*n+i, c]=1?
        # In schoolbook, column c corresponds to a specific triple (i', k', j').
        # U[i*m+k, c] is 1 only if i'=i and k'=k.
        # V[k*p+j, c] is 1 only if k'=k and j'=j.
        # W[j*n+i, c] is 1 only if j'=j and i'=i.
        # So for a fixed (i,j,k), there is exactly ONE column c (corresponding to i',k',j' = i,k,j) where all three are 1.
        # So the sum is 1.
        
        # Why did my trace fail?
        # i=0, j=1, k=0.
        # We need c such that:
        # U[0*2+0, c] = 1 -> U[0, c] = 1. (i'=0, k'=0)
        # V[0*7+1, c] = 1 -> V[1, c] = 1. (k'=0, j'=1)
        # W[1*2+0, c] = 1 -> W[2, c] = 1. (j'=1, i'=0)
        
        # In schoolbook, is there a column for (i'=0, k'=0, j'=1)?
        # Yes, col = index of (0,0,1).
        # Let's check the indices.
        # i=0, k=0, j=1.
        # U[0*2+0, col] = U[0, col] = 1.
        # V[0*7+1, col] = V[1, col] = 1.
        # W[1*2+0, col] = W[2, col] = 1.
        # So yes, there is such a column.
        
        # So T[0,1,0] = 1.
        # My previous trace assumed c_U, c_V, c_W had to be the SAME index in a 1D array, but they are columns in different matrices.
        # The column index `col` is the same for U, V, W.
        # So yes, it works.
        
        return np.allclose(T, 1.0, atol=1e-9)

    # 3. Objective function for optimization
    # We want to minimize the error of the tensor identity.
    # Parameters are flattened U, V, W.
    
    def objective(params):
        r = 25  # Target rank
        U_flat = params[:n*m*r]
        V_flat = params[n*m*r:n*m*r + m*p*r]
        W_flat = params[n*m*r + m*p*r:]
        
        U = U_flat.reshape(n*m, r)
        V = V_flat.reshape(m*p, r)
        W = W_flat.reshape(p*n, r)
        
        u_r = U.T.reshape(n, m, r)
        v_r = V.T.reshape(m, p, r)
        w_r = W.T.reshape(p, n, r)
        
        T = np.einsum('ikc,kjc,jic->ijk', u_r, v_r, w_r)
        error = np.sum((T - 1.0)**2)
        return error

    # 4. Search for rank 25
    target_rank = 25
    found = False
    best_error = float('inf')
    
    # Random restarts
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 20: break
        
        # Initialize with small random values
        np.random.seed(42 + restart)
        init_U = np.random.randn(n*m, target_rank) * 0.1
        init_V = np.random.randn(m*p, target_rank) * 0.1
        init_W = np.random.randn(p*n, target_rank) * 0.1
        
        params0 = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
        
        try:
            res = minimize(objective, params0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_error:
                best_error = res.fun
                if res.fun < 1e-6:
                    # Found a solution
                    U_opt = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_opt = res.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_opt = res.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    # Round to half-integers
                    U_rounded = np.round(U_opt * 2) / 2.0
                    V_rounded = np.round(V_opt * 2) / 2.0
                    W_rounded = np.round(W_opt * 2) / 2.0
                    
                    if verify(U_rounded, V_rounded, W_rounded):
                        best_result = {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                        best_rank = target_rank
                        found = True
                        break
        except Exception:
            pass

    return best_result
# EVOLVE-BLOCK-END
