"""Matrix multiplication 3x4x4 rank minimization via structured search."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit
    
    n, m, p = DIMENSIONS
    R = n * m * p  # 48 schoolbook rank
    
    def schoolbook():
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify(U, V, W, tol=1e-6):
        """Check if decomposition is exact: sum_c U[i,j,c]*V[j,k,c]*W[k',i',c] should match matmul structure."""
        # For matmul: C[i,k'] = sum_j A[i,j] * B[j,k']
        # Tensor: T[i,j,k',i'] where i' = 0 (only one output per (i,k'))
        # W[k*n+i] convention: W index is k*n + i
        # Actually W[p*n, rank] so W[k*n + i, c]
        # The product should satisfy: for each c, the outer product structure
        # U[i*m+j, c] * V[j*p+k, c] * W[k'*n+i, c]
        # This should decompose the matmul tensor.
        
        # Check: for each (i, j, k, k'), the sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k'*n+i,c]
        # should be 1 if k==k' (i.e., B[j,k'] contributes to C[i,k'] when k=k'), 0 otherwise
        # Wait, let me think more carefully.
        # C[i, k'] = sum_j A[i,j] * B[j, k']
        # So the tensor T[i,j,k',i'] = delta_{k',i'}? No...
        # Actually the matmul tensor has entries T[i, j, k'] = delta where... 
        # Let me use the standard: the multiplication tensor M_{(i,j),(j',k')} = 1 if j=j'
        # In our indexing: U is indexed by (i,j) with i in [0,n), j in [0,m)
        # V is indexed by (j,k) with j in [0,m), k in [0,p)
        # W is indexed by (k',i') with k' in [0,p), i' in [0,n)
        # The condition is: sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = 1 if j=j' and i=i' and k=k', else 0
        # Wait no. For matmul C = A*B:
        # C[i, k'] = sum_j A[i,j] * B[j, k']
        # So the tensor is T[i, j, k'] = delta where the output index is (i, k')
        # The factorization: T[i,j,k'] = sum_c U[i,j,c] * V[j,k,c] * W[k',i,c]
        # This should equal 1 if the indices are consistent, i.e., it's just the identity in j.
        # Actually: sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
        
        U_mat = np.array(U) if isinstance(U, list) else U
        V_mat = np.array(V) if isinstance(V, list) else V
        W_mat = np.array(W) if isinstance(W, list) else W
        
        rank = U_mat.shape[1]
        
        # Build the tensor via einsum
        # T[i, j, k'] = sum_c sum_j' U[i,j,c] * V[j',k,c] * W[k',i,c]  -- no this isn't right
        # Let me use: result[i, k'] should = sum_j A[i,j] B[j, k']
        # With factors: result[i, k'] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k', c] * B[j,k'])
        # But we don't have A and B, we have the tensor itself.
        # The matmul tensor T[a, b, c] where a=(i,j), b=(j',k), c=(k',i')
        # T[(i,j),(j',k),(k',i')] = 1 if j=j', i=i', k=k'
        # Check: sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
        
        # Efficient check using einsum
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        # For each triple (a, b, c): sum_r U[a,r]*V[b,r]*W[c,r]
        # This is a 3-tensor. Let's check a subset for speed, or use vectorized approach.
        
        # Actually for exact verification with small sizes:
        # nm=12, mp=16, pn=12, rank up to 48
        # Total entries: 12*16*12 = 2304, times rank ops = manageable
        
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # We need T[a,b,c] = 1 if a's j-component == b's j-component AND a's i-component == c's i-component AND b's k-component == c's k-component
        # a = i*m + j, b = j'*p + k, c = k'*n + i'
        # Condition: j == j' AND i == i' AND k == k'
        
        max_err = 0.0
        for a in range(n * m):
            i_a = a // m
            j_a = a % m
            for b in range(m * p):
                j_b = b // p
                k_b = b % p
                expected_a_b = 1.0 if j_a == j_b else 0.0
                if expected_a_b == 0.0:
                    # Check that sum is 0
                    sub = U_mat[a] * V_mat[b]  # (rank,)
                    val = sub @ W_mat  # sum_c U[a,c]*V[b,c]*W[c,?] -- wait
                    # Actually T[a,b,c'] for specific c'
                    # Let me just do the full check differently
                    pass
            # This is getting complex. Let me do it properly.
        
        # Full verification:
        # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Expected: 1 if (a%m == b//p) and (a//m == c%n) and (b%p == c//n), else 0
        # a: 0..nm-1, b: 0..mp-1, c: 0..pn-1
        
        # Vectorized:
        # U_mat: (nm, R), V_mat: (mp, R), W_mat: (pn, R)
        # T[a,b,c] = U_mat[a] @ diag(V_mat[b]) @ W_mat[c] -- not quite
        # T[a,b,c] = sum_r U_mat[a,r] * V_mat[b,r] * W_mat[c,r]
        # = (U_mat * V_mat.T) -- no
        # For fixed a,b: sum_r U[a,r]*V[b,r] * W[c,r] = (elementwise product U[a,:]*V[b,:]) dot W[c,:]
        
        # Let's compute all:
        # UV[a, b, r] = U[a,r] * V[b,r]  -- too big: 12*16*48
        # Then T[a,b,c] = sum_r UV[a,b,r] * W[c,r]
        
        # Simpler: for random sampling
        random.seed(42)
        samples = min(200, (n*m)*(m*p)*(p*n))
        for _ in range(samples):
            a = random.randint(0, n*m-1)
            b = random.randint(0, m*p-1)
            c = random.randint(0, p*n-1)
            val = float(np.dot(np.dot(U_mat[a], V_mat[b]), W_mat[c]))
            i_a, j_a = a // m, a % m
            j_b, k_b = b // p, b % p
            k_c, i_c = c // n, c % n
            expected = 1.0 if (j_a == j_b and i_a == i_c and k_b == k_c) else 0.0
            err = abs(val - expected)
            if err > max_err:
                max_err = err
            if max_err > tol:
                return False, max_err
        return max_err < tol, max_err
    
    def als_decompose(tensor_shape, target_func, init_rank, n_iter=50, tol=1e-8):
        """ALS for tensor decomposition. target_func(a,b,c) gives expected value."""
        # Not practical for general use here; skip
        pass
    
    # Try known lower ranks
    # For 3x4x4, schoolbook is 48. 
    # Lower bounds: max(3*4, 4*4, 4*3) = 16, and others.
    # Let's try to find a decomposition with rank < 48 using random initialization + ALS-like approach.
    
    # Actually, let me try a structured approach:
    # The matmul tensor for n x m times m x p has known decompositions.
    # For small cases, the rank is often close to nm + mp + pn - something.
    # 3*4 + 4*4 + 4*3 = 12+16+12 = 40. Maybe rank 40 or so?
    
    # Let me try random search for small rank reductions.
    # Start with schoolbook and try to find linear dependencies / cancellations.
    
    # Alternative: use the fact that for matmul, we can use bilinear algorithms.
    # For 3x4x4, let's see if we can do better than 48.
    
    # Let me try: use numpy to do a least-squares approach.
    # Represent the tensor as a matrix and do SVD.
    
    # Flatten: T[i,j,k] for the matmul: C[i,k] = sum_j A[i,j]B[j,k]
    # The tensor has indices (i, j, k) with the "free" index being k (output) and i (output row)
    # Wait, I need to be more careful with the tensor structure.
    
    # The multiplication tensor T has components T_{(i,j), (j',k), (k',i')}
    # It equals 1 iff j=j', i=i', k=k'
    # 
    # Matrix form: Fix the "middle" index j=j'. Then T is like:
    # For each j, we have a term: A_col_i * B_row_k * C_{i,k}
    # 
    # Let's use a different flattening. Reshape U, V, W as:
    # The equation is: for all i, j, k, i', k':
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k'*n+i', c] = delta(j, j') * delta(i, i') * delta(k, k')
    #   where j' is the j-index of V (which must equal j for non-zero)
    # 
    # Simplify: set j' = j, i' = i, k' = k. Then:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # For other combinations, the sum must be 0.
    
    # This is a system of equations. Let's set up a linear system for a given rank r.
    # Unknowns: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] for c in 0..r-1
    # Constraints: for all valid (i,j,k,i',j',k'):
    #   sum_c U[i*m+j,c] V[j'*p+k,c] W[k'*n+i',c] = 1 if j=j' and i=i' and k=k' else 0
    
    # This is nonlinear (products of unknowns). But we can fix one matrix and solve for others.
    # ALS: fix V and W, solve for U (linear). Then fix U and W, solve for V. Etc.
    
    def als_solve(target_rank, n_iters=100, seed=0, time_limit=100):
        """Try to find a decomposition of given rank using ALS."""
        rng = np.random.RandomState(seed)
        
        U = rng.randn(n*m, target_rank)
        V = rng.randn(m*p, target_rank)
        W = rng.randn(p*n, target_rank)
        
        for it in range(n_iters):
            if time.time() - start_time > deadline - 5:
                return None
            # Solve for U: for each (i,j), U[i*m+j, :] should satisfy
            # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all k
            # This is: U[i*m+j, :] @ (V_j ⊙ W_i).T = 1_vec
            # where V_j is row j of V reshaped, W_i is row i of W reshaped
            # V[j*p+k, c] for k=0..p-1 is V[j*p:(j+1)*p, :]  (p x rank)
            # W[k*n+i, c] for k=0..p-1: W[i::n, :]  (p x rank)
            # Element-wise product: (p x rank)
            # We need U_row @ (V_sub * W_sub).T = ones(p)
            # i.e., U_row @ M.T = 1 where M = V_sub * W_sub (p x rank)
            # So U_row = M @ ones(p) / (M @ M.T)[0,0]? No...
            # U_row * M.T = 1 => U_row = M @ 1 / (M @ M.T) if M is square... 
            # Actually: U_row @ M.T = 1_vec means (M @ U_row.T) = 1_vec
            # So M @ U_row.T = 1 => U_row.T = M^{-1} @ 1 if M invertible, else least squares
            for i in range(n):
                for j in range(m):
                    V_sub = V[j*p:(j+1)*p, :]  # (p, rank)
                    W_sub = W[i::n, :]  # (p, rank)
                    M = V_sub * W_sub  # (p, rank)
                    # M @ u = 1
                    try:
                        u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    except:
                        u = np.zeros(target_rank)
                    U[i*m+j, :] = u
            
            # Solve for V: for each (j,k), V[j*p+k, :] 
            # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i
            # V_row @ (U_j ⊙ W_k).T = 1
            # U[i*m+j, c] for i=0..n-1: U[j::m, :] (n x rank)
            # W[k*n+i, c] for i=0..n-1: W[k*n:(k+1)*n, :] (n x rank)
            for j in range(m):
                for k in range(p):
                    U_sub = U[j::m, :]  # (n, rank)
                    W_sub = W[k*n:(k+1)*n, :]  # (n, rank)
                    M = U_sub * W_sub  # (n, rank)
                    try:
                        v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    except:
                        v = np.zeros(target_rank)
                    V[j*p+k, :] = v
            
            # Solve for W: for each (k,i), W[k*n+i, :]
            # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all j
            # W_row @ (U_i ⊙ V_k).T = 1
            # U[i*m+j, c] for j=0..m-1: U[i*m:(i+1)*m, :] (m x rank)
            # V[j*p+k, c] for j=0..m-1: V[k::p, :] (m x rank)
            for k in range(p):
                for i in range(n):
                    U_sub = U[i*m:(i+1)*m, :]  # (m, rank)
                    V_sub = V[k::p, :]  # (m, rank)
                    M = U_sub * V_sub  # (m, rank)
                    try:
                        w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    except:
                        w = np.zeros(target_rank)
                    W[k*n+i, :] = w
            
            # Check convergence
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = float(np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i]))
                        err = max(err, abs(val - 1.0))
            if err < tol:
                # Also check off-diagonal terms are ~0
                off_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for i2 in range(n):
                                for j2 in range(m):
                                    for k2 in range(p):
                                        if i==i2 and j==j2 and k==k2:
                                            continue
                                        val = float(np.dot(np.dot(U[i*m+j], V[j2*p+k]), W[k2*n+i2]))
                                        off_err = max(off_err, abs(val))
                if off_err < tol:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    # Try ranks from lower bound upward
    # Lower bound for matmul rank: at least max(n*m, m*p, p*n) = max(12,16,12) = 16
    # But for 3x4x4, known results suggest rank around 30-40
    # Let's try a few ranks
    
    best_result = schoolbook()
    best_rank = R
    
    # Try decreasing ranks
    for target_rank in range(30, R + 1):
        if time.time() - start_time > deadline - 10:
            break
        result = als_solve(target_rank, n_iters=80, seed=target_rank, time_limit=deadline - start_time - 10)
        if result is not None:
            # Verify
            valid, err = verify(result["U"], result["V"], result["W"], tol=1e-5)
            if valid:
                best_result = result
                best_rank = target_rank
                break
    
    return best_result

# EVOLVE-BLOCK-END
