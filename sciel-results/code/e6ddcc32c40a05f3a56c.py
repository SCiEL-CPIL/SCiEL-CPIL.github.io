"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 115  # safety margin
    
    n, m, p = 4, 5, 6
    U_rows = n * m  # 20
    V_rows = m * p  # 30
    W_rows = p * n  # 24
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p  # 120
        U = [[0.0] * rank for _ in range(U_rows)]
        V = [[0.0] * rank for _ in range(V_rows)]
        W = [[0.0] * rank for _ in range(W_rows)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p
    
    # Try to find a better decomposition using random initialization and optimization
    # The tensor T[i,j,k] = delta[i,j,k] for matrix multiplication
    # We want U, V, W such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) is valid, 0 otherwise
    
    # For matrix multiplication, the tensor is:
    # T[i,j,k] = 1 for all i,j,k (it's the standard multiplication tensor)
    # Actually, for multiplying A (n x m) by B (m x p) to get C (n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is equivalent to finding a rank-r decomposition of the tensor
    # where the tensor is all ones in the (i,j,k) space
    
    # Let's try a simple approach: use random vectors and see if we can fit
    
    # Actually, let's think about this differently. The tensor for matrix multiplication
    # is a 3D tensor of size n x m x p where every entry is 1.
    # We need to decompose this as a sum of rank-1 tensors.
    
    # A rank-1 tensor is u * v * w where u is in R^(n*m), v in R^(m*p), w in R^(p*n)
    # But the constraint is that the product must equal 1 for all (i,j,k).
    
    # This is actually a very constrained problem. The standard result is that
    # the tensor rank of the matrix multiplication tensor for n x m x p is at least
    # max(n*m, m*p, p*n) and at most n*m*p.
    
    # For 4x5x6: lower bound is max(20, 30, 24) = 30, upper bound is 120.
    
    # Let's try a block decomposition approach.
    
    # Pad to 4x8x8 or similar to use known results?
    # Or use the fact that we can combine multiplications.
    
    # Let's try a simple optimization: start with random rank-r decomposition
    # and minimize the error.
    
    def try_random_rank(r, num_trials=50):
        if r >= best_rank:
            return None, None, None, best_rank
        
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = [[random.gauss(0, 1) for _ in range(r)] for _ in range(U_rows)]
            V = [[random.gauss(0, 1) for _ in range(r)] for _ in range(V_rows)]
            W = [[random.gauss(0, 1) for _ in range(r)] for _ in range(W_rows)]
            
            # Compute the tensor and check error
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # We want T[i,j,k] = 1 for all i,j,k
            
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r_idx in range(r):
                            val += U[i*m+j][r_idx] * V[j*p+k][r_idx] * W[k*n+i][r_idx]
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-6:
                return U, V, W, r
        
        return None, None, None, best_rank
    
    # Try various ranks
    for r in range(30, min(best_rank, 120)):
        if time.time() > deadline - 20:
            break
        U, V, W, rank = try_random_rank(r, num_trials=20)
        if U is not None:
            best_U, best_V, best_W = U, V, W
            best_rank = r
            break
    
    # If we didn't find anything better, try a structured approach
    # Use the fact that we can group columns
    
    # For now, return the best we found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
