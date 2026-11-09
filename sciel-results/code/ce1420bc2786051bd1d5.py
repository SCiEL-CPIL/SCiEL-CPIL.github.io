"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def schoolbook():
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
        return {"U": U, "V": V, "W": W}

    best = schoolbook()

    def validate(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        # The tensor T[i,j,k] = 1 if we're multiplying (i,j) of A with (j,k) of B to get (i,k)
        # Actually, the tensor for matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p
        # T[i, j, k] represents the contribution: C[i,k] += A[i,j] * B[j,k]
        # So the tensor is 1 for all valid (i,j,k) and 0 otherwise (it's all ones in the valid range)
        # Wait, the tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need to verify this identity holds for all i,j,k.
        
        # Build the tensor
        T = np.ones((n, m, p))
        
        # Compute the decomposition
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # We can do this efficiently using einsum or manual loops
        rank = U.shape[1]
        # Reshape for efficient computation
        # For each r, we have U[:, r] (n*m), V[:, r] (m*p), W[:, r] (p*n)
        # The product for a fixed r gives a 3D tensor of shape (n*m, m*p, p*n) which is too big
        # Instead, let's iterate over r and accumulate
        
        result = np.zeros((n, m, p))
        for r in range(rank):
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            # For each i, j, k: contribution is u_r[i,j] * v_r[j,k] * w_r[k,i]
            # This is like: for each (i,j,k), u_r[i,j] * v_r[j,k] * w_r[k,i]
            # We can compute this as: 
            # temp[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[k,i]
            # Let's do it step by step
            # First: a[i,j,k] = u_r[i,j] * v_r[j,k] -> shape (n, m, p)
            a = u_r[:, :, np.newaxis] * v_r[np.newaxis, :, :]  # (n, m, p)
            # Then multiply by w_r[k,i] which is (p, n) -> we need (n, m, p) with w_r[k,i]
            # w_r[k,i] for k in range(p), i in range(n) -> shape (p, n)
            # We need to index it as w_r[k, i] for each (i, j, k)
            # So we need a tensor of shape (n, m, p) where entry (i,j,k) = w_r[k, i]
            w_reshaped = np.zeros((n, m, p))
            for i in range(n):
                for k in range(p):
                    w_reshaped[i, :, k] = w_r[k, i]
            result += a * w_reshaped
        
        return np.allclose(result, T, atol=tol, rtol=0)

    def try_reinhard_strassen_like():
        """Try to find a lower rank decomposition using structured search."""
        # For 3x4 * 4x4, let's try ranks below 48
        # Use a simple approach: try to exploit structure
        
        # One idea: split the 4x4 multiplication into blocks
        # A is 3x4, B is 4x4
        # We can write B as [B1, B2] where B1 is 4x2, B2 is 4x2
        # Then C = A * [B1, B2] = [A*B1, A*B2]
        # A*B1 is 3x2, A*B2 is 3x2
        # Each 3x4 * 4x2 multiplication can be done with rank 3*4*2 = 24
        # Total rank: 48, same as schoolbook
        
        # Try a different split: use the fact that 4 = 2 + 2
        # This doesn't help directly.
        
        # Let's try a more sophisticated approach: use known results
        # For matrix multiplication, the Strassen algorithm gives rank 7 for 2x2
        # For larger matrices, we can use tensor product of Strassen
        
        # For 3x4 * 4x4:
        # We can pad to 4x4 * 4x4 and use Strassen-based methods
        # Or we can try to find a direct decomposition
        
        # Let's try a computational search for small rank improvements
        # Start with rank 47 and work down
        
        # Actually, let's try a different approach: use the fact that
        # we can combine some of the schoolbook terms
        
        # For now, let's just return the schoolbook as the best known
        return None

    # Try to find a better decomposition
    # Since this is a hard problem, let's try a few structured approaches
    
    # Approach 1: Try to use the fact that some entries can be shared
    # In the schoolbook, each term (i,j,k) has its own rank-1 component
    # Can we combine terms that share the same (i,j) or (j,k) or (k,i)?
    
    # For a fixed (i,j), we have p terms: (i,j,0), (i,j,1), ..., (i,j,p-1)
    # These all have the same U entry: U[i*m+j, :]
    # But different V and W entries
    
    # Let's try to see if we can reduce the rank by a small amount
    # by exploiting linear dependencies
    
    # For now, return the schoolbook decomposition
    # A proper search would take too long
    
    return best
# EVOLVE-BLOCK-END
