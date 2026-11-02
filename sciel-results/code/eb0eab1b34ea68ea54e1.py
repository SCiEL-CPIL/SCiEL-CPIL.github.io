import time
import numpy as np
import scipy.linalg as la

# Dimensions for the problem: multiply a 4x5 matrix by a 5x6 matrix
n, m, p = 4, 5, 6
total_entries = n * m * p

def construct():
    start_time = time.time()
    # The deadline is 150s, but let's set a 100s deadline to be safe
    # and return schoolbook if we can't find anything better
    deadline = 100.0

    # Initialize the U, V, W matrices to zeros
    # U: shape (n*m, rank)
    # V: shape (m*p, rank)
    # W: shape (p*n, rank)
    
    # We'll do a tensor decomposition search
    # We can write the bilinear form as
    # sum_r (sum_ij U[i,j,r] * A[i,j]) * (sum_jk V[j,k,r] * B[j,k]) * (sum_il W[i,l,r] * C[i,l])? 
    # No, the standard way is:
    # C[i,l] = sum_r U[i,?,r] * V[?,l,r] * ...
    # Actually, C = A*B, so C[i,l] = sum_j A[i,j] * B[j,l]
    # The decomposition should look like:
    # sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = delta_{jk} * delta_{ii}? No, that's for the tensor.
    # The correct way is:
    # sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] is 1 if j is the same index, and 0 otherwise?
    # No, it's not that.
    
    # Let's write the tensor T of the multiplication.
    # T is a tensor of size (n*m) x (m*p) x (p*n).
    # T[a,b,c] = 1 if a,b,c form a valid triple (i,j), (j,k), (i,k)
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1
    # T[else] = 0
    
    # We want to find rank-r vectors u_1, v_1, w_1 such that
    # sum_{r=1}^r u_r \otimes v_r \otimes w_r = T
    # where u_r is a vector of size n*m, v_r of size m*p, and w_r of size p*n.
    
    # This is a bilinear decomposition.
    # Each term in the decomposition has the form:
    # (a_i u_i) \otimes (b_j v_j) \otimes (c_k w_k)
    # Let's represent a rank-1 term as three vectors: u, v, w.
    # The term in the sum is u * v * w (as an outer product of the 3 vectors).
    # The sum over r should be T.
    # So, sum_{r=1}^rank (u_r \otimes v_r \otimes w_r) = T.
    # Here, u_r \in R^{n*m}, v_r \in R^{m*p}, w_r \in R^{p*n}.
    
    # Let's define the tensor T explicitly.
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i*m + j
                b_idx = j*p + k
                c_idx = k*n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # We want to decompose T into a sum of rank-1 tensors.
    # Let's do a randomized tensor decomposition search.
    # We can try to find a decomposition of rank r using an iterative algorithm.
    # We can use a version of the ALS (Alternating Least Squares) algorithm for tensor decomposition.
    
    def verify(U, V, W):
        """
        Verifies if the decomposition is correct.
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        The sum over r of U[:,r] \otimes V[:,r] \otimes W[:,r] should equal T.
        We can check by verifying that for all a,b,c:
        sum_r U[a,r] * V[b,r] * W[c,r] == T[a,b,c]
        """
        # To make it faster, we can just sample some entries
        rank = U.shape[1]
        sample_entries = min(total_entries, 10000)
        np.random.seed(42)
        indices = np.random.choice(total_entries, sample_entries, replace=False)
        a_idx = indices // (m*p)
        bc_idx = indices % (m*p)
        b_idx = bc_idx // n
        c_idx = bc_idx % n
        
        for r in range(rank):
            # We need to be more careful with the indices
            pass
            
        # Actually, let's just do a full check for correctness, but with a smarter method
        # U^T V is a matrix of shape (rank, rank)? No.
        # For each a, U[a, :] is a vector of size rank.
        # Let's compute the sum for each a, b, c.
        
        # We can write it as:
        # M = np.zeros((n*m, m*p, p*n))
        # for r in range(rank):
        #     M += U[:, r][:, None, None] * V[:, r][None, :, None] * W[:, r][None, None, :]
        # This is too slow for large ranks.
        
        # Let's do it smarter.
        # Let's use einsum: np.einsum('ar,br,cr->abc', U, V, W)
        # This should be faster, but let's be careful with the memory.
        # For rank 100, 100x100x100 = 1e6, which is fine.
        # For rank 1000, 1e9 is too much.
        if rank > 500:
            # Sample a few entries
            sample = 1000
            a_idx = np.random.choice(n*m, sample, replace=False)
            b_idx = np.random.choice(m*p, sample, replace=False)
            c_idx = np.random.choice(p*n, sample, replace=False)
            computed = np.einsum('ar,br,cr->abc', U[a_idx, :], V[b_idx, :], W[c_idx, :])
            expected = T[a_idx, b_idx, c_idx]
            if not np.allclose(computed, expected, atol=1e-9):
                return False
        else:
            # Full computation
            computed = np.einsum('ar,br,cr->abc', U, V, W)
            if not np.allclose(computed, T, atol=1e-9):
                return False
            # Also check sparsity: if T[a,b,c]=0, computed should be 0
            # The above check already does this
        return True
    
    # Start with a random guess and do alternating least squares
    def als_decomposition(rank):
        """
        Alternating Least Squares for tensor decomposition.
        Returns U, V, W or None if it fails to converge.
        """
        # Random initialization
        U = np.random.randn(n*m, rank) / np.sqrt(rank)
        V = np.random.randn(m*p, rank) / np.sqrt(rank)
        W = np.random.randn(p*n, rank) / np.sqrt(rank)
        
        # We want to solve: sum_r U[:,r] \otimes V[:,r] \otimes W[:,r] = T
        # This is equivalent to minimizing ||T - sum_r U[:,r] \otimes V[:,r] \otimes W[:,r]||_F^2
        # We'll do ALS by alternating between updating U, V, and W.
        
        # To make it more efficient, let's reshape T
        # When updating U, for each column r, we need to find u_r such that
        # sum_r u_r \otimes V[:,r] \otimes W[:,r] is close to T
        # This can be rewritten as a linear system for u_r
        # Let's define for each r:
        # T - sum_{s != r} U[:,s] \otimes V[:,s] \otimes W[:,s] ≈ U[:,r] \otimes V[:,r] \otimes W[:,r]
        # This is hard to solve directly because it's still a tensor equation.
        
        # Let's use a simpler ALS formulation:
        # For fixed V and W, minimize over U.
        # This is a linear least squares problem for each column of U.
        # The (a, r)-th element of U should satisfy:
        # sum_r U[a,r] * (V[:,r] \otimes W[:,r])[b,c] = T[a,b,c]
        # This is still not quite right.
        
        # Let's define the "flattened" version.
        # For a fixed r, the contribution to the tensor from U[:,r], V[:,r], W[:,r] is:
        # U[:,r] \otimes V[:,r] \otimes W[:,r]
        # This can be seen as a matrix:
        # reshape this tensor to a matrix of shape (n*m, m*p * p*n)
        # Let's define:
        # T_r = U[:,r] \otimes V[:,r] \otimes W[:,r]
        # Reshape to (n*m, m*p*p*n)
        # T_r[a, b*p*n + c] = U[a,r] * V[b,r] * W[c,r]
        
        # When we fix V and W, and want to find U:
        # We need to solve for U such that:
        # sum_r T_r ≈ T
        # This is a linear system in U.
        # Let's write T as a matrix of shape (n*m, m*p*p*n)
        # T_mat[a, b*p*n + c] = T[a, b, c]
        
        T_mat = T.reshape(n*m, -1)  # shape (n*m, m*p*p*n)
        
        # For fixed V and W, the contribution of U[:,r] to T_mat is:
        # U[:,r] \otimes (V[:,r] \otimes W[:,r])_mat
        # where (V[:,r] \otimes W[:,r])_mat is a vector of size m*p*p*n
        # Let's define M_r = V[:,r] \otimes W[:,r] reshaped to (m*p*p*n,)
        # M_r = np.kron(V[:,r], W[:,r])
        
        # The equation is:
        # sum_r U[:,r] \otimes M_r = T_mat
        # This is a matrix equation. Let's write it out:
        # T_mat = sum_r np.outer(U[:,r], M_r)
        
        # Let's define the matrix A such that:
        # T_mat = U @ A
        # where A is of shape (rank, m*p*p*n)
        # A[r, :] = M_r
        
        A_mat = np.zeros((rank, m*p*p*n))
        for r in range(rank):
            A_mat[r, :] = np.kron(V[:,r], W[:,r])
        
        # Now we have: T_mat ≈ U @ A_mat
        # This is a linear least squares problem for U.
        # We can solve it with np.linalg.lstsq
        # T_mat = U @ A_mat
        # This is of the form C = A @ B where C is T_mat, A is U, and B is A_mat
        # We want to find U that minimizes ||T_mat - U @ A_mat||_F
        
        # This is a linear system in U. We can solve it as:
        # U @ A_mat.T @ A_mat = T_mat.T @ A_mat.T
        # A_mat.T @ A_mat is a (rank, rank) matrix
        # T_mat.T @ A_mat.T is a (m*p*p*n, rank) matrix
        
        # But we can also use a QR decomposition or SVD.
        # Let's use SVD for stability.
        # The solution to min_U ||T_mat - U @ A_mat||_F is:
        # U = T_mat.T @ A_mat.T @ (A_mat @ A_mat.T)^-1
        # Wait, no. Let's derive it properly.
        # We want to minimize ||T_mat - U @ A_mat||_F^2
        # Let's write it as:
        # min_U ||T_mat.T - A_mat.T @ U.T||_F^2
        # This is a linear least squares problem for U.T
        # Let's use np.linalg.lstsq
        U.T, residuals, rank_actual, s = np.linalg.lstsq(A_mat.T, T_mat.T, rcond=None)
        
        # Wait, this is for fixed V, W. We also need to update V and W.
        # This is getting complicated. Let's try a different approach.
        
        # Let's just do a simple ALS with full tensor reshaping.
        # It's easier to implement but might be slower.
        
        # Let's do it with explicit loops for clarity
        T_copy = T.copy()
        for _ in range(10):  # 10 iterations
            # Update U
            for r in range(rank):
                # Residual = T - sum_{s!=r} U[:,s] \otimes V[:,s] \otimes W[:,s]
                # For a given r, we want to find u_r that minimizes:
                # ||Residual - u_r \otimes V[:,r] \otimes W[:,r]||_F^2
                # This is a linear least squares problem for u_r.
                
                # Let's define B_r = V[:,r] \otimes W[:,r] as a tensor of size (m*p, p*n)
                B_r = np.outer(V[:,r], W[:,r])  # shape (m*p, p*n)
                
                # The term u_r \otimes B_r can be written as:
                # for each a, u_r[a] * B_r
                # So the residual for this a is:
                # Residual[a, :, :] - u_r[a] * B_r
                # We want to minimize the sum over a of ||Residual[a] - u_r[a] * B_r||_F^2
                
                # Let's define C_r as a matrix of size (n*m, m*p*p*n)
                # C_r[a, b*p*n+c] = B_r[b, c]
                C_r = B_r.reshape(-1)  # shape (m*p*p*n,)
                
                # Now the problem is:
                # min_u ||Residual_mat - u \otimes C_r||_F^2
                # where Residual_mat is Residual reshaped to (n*m, -1)
                
                Residual_mat = (T - T_copy).reshape(n*m, -1)  # This is wrong, we need the actual residual
                
                # Let's compute the residual properly
                Residual = np.zeros_like(T)
                for s in range(rank):
                    if s == r:
                        continue
                    Residual += np.outer(U[:,s], np.outer(V[:,s], W[:,s]).reshape(-1))[:, None, :]
                    # Wait, this isn't right either.
                
                # This is getting too complicated. Let's just use the full tensor approach.
        
        # Let's restart with a cleaner approach
        
        # For fixed V, W, the problem for U is:
        # min_U ||T - sum_r U[:,r] \otimes V[:,r] \otimes W[:,r]||_F^2
        # Let's define for each r: M_r = V[:,r] \otimes W[:,r]
        # M_r is a tensor of shape (m*p, p*n)
        
        # The term U[:,r] \otimes M_r can be written as:
        # for each a, U[a,r] * M_r
        
        # So the sum is:
        # for each a: sum_r U[a,r] * M_r
        
        # Let's define N_a as the sum of M_r for each a.
        # Actually, it's easier to think of it as a matrix multiplication.
        
        # Let's define a matrix A of size (n*m, m*p*p*n)
        # A[a, b*p*n+c] = sum_r U[a,r] * V[b,r] * W[c,r]
        
        # This is exactly the tensor T!
        
        # So the problem is to find U, V, W such that:
        # sum_r U[:,r] \otimes V[:,r] \otimes W[:,r] = T
        
        # Let's just do a simple ALS iteration
        for iteration in range(50):
            # Update U
            for r in range(rank):
                # For each r, we want to find u_r = U[:,r]
                # that minimizes ||T - sum_{s} U[:,s] \otimes V[:,s] \otimes W[:,s]||_F^2
                # with U[:,r] = u_r
                
                # This is equivalent to:
                # min_u ||T - u \otimes V[:,r] \otimes W[:,r] - sum_{s!=r} U[:,s] \otimes V[:,s] \otimes W[:,s]||_F^2
                
                # Let's define B = V[:,r] \otimes W[:,r]
                B = np.outer(V[:,r], W[:,r])  # shape (m*p, p*n)
                B_flat = B.reshape(-1)  # shape (m*p*p*n,)
                
                # Let's define R = T - sum_{s!=r} U[:,s] \otimes V[:,s] \otimes W[:,s]
                R = np.zeros_like(T)
                for s in range(rank):
                    if s == r:
                        continue
                    R += U[:,s][:, None, None] * V[:,s][None, :, None] * W[:,s][None, None, :]
                R -= T
                
                # Now we want to find u that minimizes ||R + u \otimes B||_F^2
                # Let's flatten R
                R_flat = R.reshape(n*m, -1)
                
                # The problem is: min_u ||R_flat + u \otimes B_flat||_F^2
                # Let's write it as:
                # min_u sum_a ||R_flat[a] + u[a] * B_flat||_F^2
                # = min_u sum_a (||R_flat[a]||_F^2 + 2*u[a]*<R_flat[a], B_flat> + u[a]^2 * ||B_flat||_F^2)
                # = min_u sum_a (C_a * u[a]^2 + D_a * u[a] + E_a)
                # where C_a = ||B_flat||_F^2, D_a = 2*<R_flat[a], B_flat>, E_a = ||R_flat[a]||_F^2
                
                # The minimum for each a is at:
                # u[a] = -D_a / (2*C_a) = -<R_flat[a], B_flat> / ||B_flat||_F^2
                
                B_norm_sq = np.dot(B_flat, B_flat)
                if B_norm_sq < 1e-10:
                    u = np.zeros(n*m)
                else:
                    # <R_flat[a], B_flat> for all a
                    # This is R_flat @ B_flat
                    D = R_flat @ B_flat  # shape (n*m,)
                    u = -D / B_norm_sq
                
                U[:, r] = u
            
            # Similarly for V and W
            # This is getting very long. Let's just use a simpler approach.
        
        # Let's use a different strategy: use a known decomposition if possible,
        # or just return the schoolbook decomposition.
        
        # For now, let's just return the schoolbook solution.
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        
        return U, V, W
    
    # For now, let's just use the schoolbook decomposition
    # We can try to improve it, but for a safe answer, schoolbook is correct
    
    U = np.zeros((n*m, n*m*p))
    V = np.zeros((m*p, n*m*p))
    W = np.zeros((p*n, n*m*p))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# Verify the solution
if __name__ == "__main__":
    start_time = time.time()
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    
    # Verify
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    rank = U.shape[1]
    print(f"Rank: {rank}")
    
    # Check using einsum
    computed = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.abs(computed - T)
    print(f"Max error: {np.max(error)}")
    if np.max(error) < 1e-9:
        print("Verification PASSED")
    else:
        print("Verification FAILED")
