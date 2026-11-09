"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Target rank to attempt. 
    # Schoolbook is 125. 
    # Strassen-based blocks often get close to 100-110.
    # We will try to find a decomposition with rank R < 125.
    # If we fail, we return the schoolbook decomposition (rank 125).
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def verify(U, V, W, tol=1e-9):
        """
        Check if U, V, W exactly decompose the matmul tensor.
        Tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        For matmul, T[i, j, k] is 1 if i, j, k are the standard indices? 
        Wait, the matmul tensor M has M[i, j, k] = 1 if we are computing (AB)_{ik} = sum_j A_{ij} B_{jk}.
        The tensor definition in the prompt:
        Output factor W follows official trace convention W[k*n+i, r].
        Standard tensor decomposition of matmul:
        M = sum_r u_r ⊗ v_r ⊗ w_r
        where M_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
        
        The matmul tensor M has M_{i,j,k} = 1 if and only if... 
        Actually, the matmul operation is linear. 
        (A B)_{i, k} = sum_j A_{i, j} B_{j, k}.
        This can be viewed as a tensor M with indices (i, j, k) where M_{i,j,k} = 1? 
        No, the tensor T such that T(A,B) = AB is defined by T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to (AB)_{i,k}.
        So T_{i,j,k} = 1 for all i, j, k? 
        Let's check: sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} = sum_{i,j,k} 1 * A_{i,j} B_{j,k} = sum_i (sum_j A_{i,j} B_{j,k}) e_i? 
        Yes, the standard matmul tensor has T_{i,j,k} = 1 for all i,j,k in the range.
        Wait, no. The output is a matrix C of size n x p.
        C_{i,k} = sum_j A_{i,j} B_{j,k}.
        The tensor M has 3 indices: row of A (i), col of A / row of B (j), col of B (k).
        M_{i,j,k} = 1.
        
        Let's verify with the schoolbook decomposition provided in the seed:
        U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for unique (i,j,k).
        Sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        For a fixed (i,j,k), only the term where c corresponds to (i,j,k) is non-zero.
        U=1, V=1, W=1. Sum = 1.
        So yes, the target tensor M has M_{i,j,k} = 1 for all i,j,k.
        """
        # Construct the full tensor M
        # M has shape (n, m, p) and all entries are 1.
        # We need to check if sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k.
        
        # Efficient check:
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        
        # We want T[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
        # Note the index permutation for W: W is indexed by (k, i) in the formula W[k*n+i].
        # So W_reshaped[k, i, r].
        
        # Let's compute the sum using einsum
        # U[i,j,r], V[j,k,r], W[k,i,r]
        # Sum over r.
        # Result T[i,j,k]
        
        # To avoid constructing the full tensor if R is large, we can check specific slices or use matrix multiplication.
        # However, n,m,p=5 is small. 5*5*5=125 elements.
        # R is around 100.
        # Computation is cheap.
        
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # We can fix i and k, and check if the vector V[j,k,:] dotted with (U[i,j,:] * W[k,i,:]) is 1?
        # Actually, let's just compute the full tensor.
        
        # T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        # This might be slow if done naively in a loop, but einsum is optimized.
        
        try:
            T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
        except Exception:
            return False
            
        # Check if T is all ones
        if not np.allclose(T, np.ones((n, m, p)), atol=tol, rtol=0):
            return False
        return True

    # Randomized search for lower rank
    # We will try to find a decomposition of rank R.
    # Strategy:
    # 1. Initialize U, V, W with random values.
    # 2. Use iterative projection to minimize || T(U,V,W) - M ||_F.
    # 3. If error drops to 0, we have a solution.
    
    def try_rank(R, max_iters=500, seed=42):
        np.random.seed(seed)
        
        # Initialize
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # Reshape for convenience
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        
        # Target M is all ones.
        # We want to minimize E = sum_{i,j,k} (sum_r U[i,j,r]V[j,k,r]W[k,i,r] - 1)^2
        
        # Alternating Least Squares (ALS)
        # Fix V, W, update U.
        # U[i,j,:] is a vector u.
        # Term for U[i,j,r] is u_r * V[j,k,r] * W[k,i,r].
        # For fixed i,j, we want to minimize sum_k (sum_r u_r V[j,k,r] W[k,i,r] - 1)^2.
        # Let A_{k,r} = V[j,k,r] W[k,i,r].
        # We want u such that A u ≈ 1 (vector of ones of size p).
        # This is a linear least squares problem for each (i,j).
        
        # Similarly for V and W.
        
        current_error = float('inf')
        
        for it in range(max_iters):
            # Update U
            for i in range(n):
                for j in range(m):
                    # Construct A (p x R)
                    # A[k, r] = V_r[j, k, r] * W_r[k, i, r]
                    A = V_r[j, :, :] * W_r[:, i, :]  # Shape (p, R)
                    b = np.ones(p)
                    # Solve A u = b
                    # Use lstsq
                    u, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    U_r[i, j, :] = u
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # Construct A (n x R)
                    # A[i, r] = U_r[i, j, r] * W_r[k, i, r]
                    A = U_r[:, j, :] * W_r[k, :, :]  # Shape (n, R)
                    b = np.ones(n)
                    v, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    V_r[j, k, :] = v
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # Construct A (m x R)
                    # A[j, r] = U_r[i, j, r] * V_r[j, k, r]
                    A = U_r[i, :, :] * V_r[:, k, :]  # Shape (m, R)
                    b = np.ones(m)
                    w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    W_r[k, i, :] = w
            
            # Check convergence
            if it % 10 == 0:
                T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
                current_error = np.linalg.norm(T - np.ones((n, m, p)))
                if current_error < 1e-9:
                    return U, V, W
            
            # Time check
            if time.time() - start_time > 140:
                return None

        # Final check
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
        if np.linalg.norm(T - np.ones((n, m, p))) < 1e-9:
            return U, V, W
        
        return None

    # Try a few ranks below 125
    # Known lower bounds for 5x5x5 are around 100-110.
    # We will try 110, 105, 100.
    # If we find one, we return it.
    
    for R in [110, 105, 100, 95, 90]:
        # Only try if we have time
        if time.time() - start_time > 120:
            break
            
        # Try a few seeds
        for seed in range(5):
            if time.time() - start_time > 140:
                break
            
            res = try_rank(R, max_iters=200, seed=seed)
            if res is not None:
                U, V, W = res
                # Final verification
                if verify(U, V, W):
                    return {"U": U, "V": V, "W": W}
        
        if time.time() - start_time > 140:
            break

    # Fallback to schoolbook
    U, V, W = get_schoolbook()
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
