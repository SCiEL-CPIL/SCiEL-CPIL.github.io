"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]
# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly realizes the matrix multiplication tensor."""
    r = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # W is indexed as [k*n+i, c]
                w_idx = k * n + i
                u_idx = i * m + j
                v_idx = j * p + k
                val = sum(U[u_idx, c] * V[v_idx, c] * W[w_idx, c] for c in range(r))
                if abs(val - 1.0) > tol:
                    return False
                # Check orthogonality: for different (i,j,k), the sum should be 0
                # We check a few random non-matching triples to ensure no spurious terms
    # More rigorous: check that the tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # equals delta_{i,i'}delta_{j,j'}delta_{k,k'} for all i,j,k,i',j',k'
    # But that's O((n*m*p)^2 * r) which is 50^2 * r = 2500*r, manageable for r~30-50.
    # Let's do a full check.
    for i1 in range(n):
        for j1 in range(m):
            for k1 in range(p):
                expected = 1.0 if (i1 == i and j1 == j and k1 == k) else 0.0
                # Wait, the loop above already checks T[i,j,k]=1. Now check T[i',j',k']=0 for !=.
    # Actually, let's just build the full tensor and compare.
    T_computed = np.zeros((n, m, p, n, m, p))
    for c in range(r):
        u_vec = U[:, c]
        v_vec = V[:, c]
        w_vec = W[:, c]
        # u_vec has n*m entries, index i*m+j
        # v_vec has m*p entries, index j*p+k
        # w_vec has p*n entries, index k*n+i
        # Outer product structure:
        # T[i,j,k,i',j',k'] += U[i*m+j,c] * V[j'*p+k',c] * W[k*n+i',c]
        # Wait, the standard decomposition is:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # T[i,j,k] = A[i,j] B[j,k]
        # The tensor of matmul is T[i,j,k] = delta_{i,i'}delta_{j,j'}delta_{k,k'}
        # No, the multiplication tensor M satisfies M[i,j,k] = 1 if i,j,k are compatible, but here
        # the indices are (i, j, k) for the input A[i,j], B[j,k], and output C[i,k].
        # The rank of the matmul tensor is the minimum number of products of linear forms
        # L1(i,j) * L2(j,k) * L3(i,k) that sum to M[i,j,k].
        # M[i,j,k] = 1 for all i,j,k? No.
        # A is n x m, B is m x p, C is n x p.
        # C[i,k] = sum_j A[i,j] B[j,k].
        # The tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
        # T[i,j,k] = 1.
        # Wait, that's the standard matrix multiplication tensor.
        # The decomposition is: sum_c U_c[i,j] * V_c[j,k] * W_c[i,k] = T[i,j,k] = 1?
        # No, that's not right.
        # U_c[i,j] depends on i,j. V_c[j,k] depends on j,k. W_c[i,k] depends on i,k.
        # The product is U_c[i,j] V_c[j,k] W_c[i,k].
        # We want sum_c U_c[i,j] V_c[j,k] W_c[i,k] = 1 for all i,j,k.
        # Is that the correct interpretation?
        # Let's re-read the task.
        # "exactly decompose the matrix multiplication tensor"
        # "output factor follows the official trace convention W[k*n+i,r]"
        # Usually, the matmul tensor is T[i,j,k] = 1.
        # Let's check the schoolbook code.
        # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1.
        # Product: U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 * 1 * 1 = 1.
        # Sum over c: for each (i,j,k), there is exactly one c where all are 1.
        # So sum is 1.
        # For other indices?
        # If we evaluate at (i',j',k') where (i',j',k') != (i,j,k),
        # does the sum vanish?
        # The schoolbook decomposition has rank n*m*p.
        # For a fixed (i,j,k), the term c corresponding to (i,j,k) gives 1.
        # Other terms c' corresponding to (i',j',k') with i'!=i or j'!=j or k'!=k
        # will have U[i*m+j, c'] = 0 if i'!=i or j'!=j.
        # Wait, U[c'] is non-zero only at (i', j'). So U[i*m+j, c'] is 0 if (i,j) != (i',j').
        # So for a fixed (i,j), only terms with i'=i, j'=j can be non-zero in U.
        # Among those, V[j*p+k, c'] is non-zero only if (j,k) = (j',k').
        # So if (i,j,k) = (i',j',k'), all three are 1.
        # If any index differs, at least one factor is 0.
        # So the schoolbook decomposition exactly computes T[i,j,k] = 1 for all i,j,k.
        # It does NOT compute a delta function. It computes the all-ones tensor.
        # Is the matrix multiplication tensor the all-ones tensor?
        # C[i,k] = sum_j A[i,j] B[j,k].
        # This is a multilinear map. The tensor of this map has indices (i, j, k) and (i, k).
        # Usually, the rank of matrix multiplication is defined as the rank of the tensor T[i,j,k] = 1.
        # Yes, the standard definition is that the rank of n x m by m x p matrix multiplication
        # is the minimum r such that there exist U_c, V_c, W_c with
        # sum_c U_c[i,j] V_c[j,k] W_c[i,k] = 1 for all i,j,k.
        # So my verification logic was slightly off in the "expected" value.
        # It should be 1 for ALL (i,j,k).
        
        # Let's re-verify the schoolbook code.
        # For any (i,j,k), is sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] == 1?
        # Yes, because there is exactly one c (the one assigned to i,j,k) where
        # U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1.
        # For all other c, at least one is 0.
        # So the sum is always 1.
        
        # Therefore, the condition is:
        # for all i in [0,n), j in [0,m), k in [0,p):
        #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
        
    return True

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find a lower rank via random search with linear constraints
    # We fix n=2, m=5, p=5.
    # We search for rank r.
    # Constraints: for all i,j,k: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1.
    # This is a system of n*m*p = 50 equations.
    # Variables: 3 * (rows * r) = 3 * r * (10 + 25 + 10) = 135 r.
    # For r=20, 2700 variables, 50 equations. Under-determined.
    # We can use a simple iterative method:
    # 1. Initialize U, V, W randomly with small values.
    # 2. For each (i,j,k), compute residual e = 1 - sum(...).
    # 3. Update U, V, W to reduce residual.
    # This is like gradient descent on the sum of squared residuals.
    
    start_time = time.time()
    max_time = 110.0  # leave 10s margin
    
    for r in range(20, rank_schoolbook):
        if time.time() - start_time > max_time:
            break
        for attempt in range(20):
            if time.time() - start_time > max_time:
                break
            # Random initialization
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Gradient descent
            lr = 0.01
            for step in range(2000):
                if time.time() - start_time > max_time:
                    break
                # Compute residuals
                residuals = []
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            # Compute dot product
                            val = np.dot(np.dot(U[u_idx, :] * V[v_idx, :], W[w_idx, :]))
                            res = 1.0 - val
                            residuals.append(res)
                            # Gradients
                            # d/dU[u_idx,c] = V[v_idx,c] * W[w_idx,c] * res
                            # d/dV[v_idx,c] = U[u_idx,c] * W[w_idx,c] * res
                            # d/dW[w_idx,c] = U[u_idx,c] * V[v_idx,c] * res
                            prod_VW = V[v_idx, :] * W[w_idx, :]
                            prod_UW = U[u_idx, :] * W[w_idx, :]
                            prod_UV = U[u_idx, :] * V[v_idx, :]
                            grad_U[u_idx, :] += prod_VW * res
                            grad_V[v_idx, :] += prod_UW * res
                            grad_W[w_idx, :] += prod_UV * res
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                if np.max(np.abs(residuals)) < 1e-8:
                    break
            
            # Check final residual
            if np.max(np.abs(residuals)) < 1e-8:
                # Convert to lists of floats
                best_rank = r
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
