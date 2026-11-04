"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    Uses a structured approach exploiting the tensor structure.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Schoolbook: rank = n*m*p = 2*4*5 = 40
    rank_schoolbook = n * m * p
    
    def schoolbook_decomposition():
        U = np.zeros((n*m, rank_schoolbook))
        V = np.zeros((m*p, rank_schoolbook))
        W = np.zeros((p*n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    def verify(U, V, W, tol=1e-9):
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find a rank-10 decomposition using the Kronecker product structure
    # The 2x4 matrix can be viewed as two 1x4 row vectors
    # Each row vector multiplied by the 4x5 matrix can use a rank-5 decomposition
    # Total rank should be 2*5 = 10
    
    def try_rank_10_decomposition():
        """
        Try to construct a rank-10 decomposition using the structure:
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        We can write this as:
        C[i,k] = sum_{j,k'} (A[i,j] * B[j,k']) * delta_{k',k}
        
        But we need to exploit that the 4x5 multiplication can be done with rank 5
        (which is the minimum possible for 1x4 * 4x5 since the output is 1x5).
        
        Actually, for 1x4 * 4x5, the rank is exactly 4*5/5 = 4? No.
        The rank of the 1x4 x 4x5 multiplication tensor is 4*5 = 20 in schoolbook.
        But wait, we can do better.
        
        Let me think again. For multiplying a 1x4 vector by a 4x5 matrix:
        C[k] = sum_j v[j] * M[j,k]
        
        This is a bilinear map. The tensor has entries T[j,k] = 1 for all j,k.
        The rank of this tensor is 4*5 = 20 in schoolbook.
        
        But actually, we can use the fact that this is just a matrix-vector product.
        The rank of the matrix multiplication tensor for 1x4 * 4x5 is min(4*5, ...) 
        Actually, the rank is exactly 4*5 = 20? No, that's the schoolbook rank.
        
        Let me reconsider. The tensor T[j,k] for the 1x4 * 4x5 case has 4*5 = 20 entries.
        Each entry T[j,k] = 1 means we need U[j,r]*V[k,r]*W[0,r] = 1 for all j,k.
        
        If we set W[0,r] = 1 for all r, then we need U[j,r]*V[k,r] = 1 for all j,k.
        This requires that the outer product U*V^T is all ones, which means U and V must be
        constant vectors. So U[j,r] = a_r and V[k,r] = b_r with a_r*b_r = 1.
        But then sum_r a_r*b_r = sum_r 1 = rank, so we need rank = 1? No, that's wrong.
        
        Wait, I'm confusing myself. Let me go back to the original problem.
        
        For the 2x4 * 4x5 case:
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        The tensor T[i,j,k] has entries T[i,j,k] = 1 for all i,j,k.
        We need to find U, V, W such that:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        One approach: Use the fact that the 2x4 matrix has 2 rows, and each row is a 1x4 vector.
        For each row i, we compute C[i,k] = sum_j A[i,j] * B[j,k].
        
        Now, for a fixed i, this is a 1x4 * 4x5 multiplication. The rank of this sub-problem
        is what we need to determine.
        
        Actually, let's try a different approach. Let's use the Strassen-like idea or
        just try to find a rank-10 decomposition directly.
        
        Let's assume rank = 10 and try to solve for U, V, W.
        """
        target_rank = 10
        
        # Initialize with random values and try to optimize
        np.random.seed(42)
        
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        # Normalize to avoid numerical issues
        U /= np.sqrt(target_rank)
        V /= np.sqrt(target_rank)
        W /= np.sqrt(target_rank)
        
        def tensor_residual(U_c, V_c, W_c):
            n, m, p = DIMENSIONS
            rank = U_c.shape[1]
            max_res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U_c[i*m+j, r] * V_c[j*p+k, r] * W_c[k*n+i, r]
                        max_res = max(max_res, abs(val - 1.0))
            return max_res
        
        # Try a few random initializations and local optimization
        best_res = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            U_cand = np.random.randn(n*m, target_rank)
            V_cand = np.random.randn(m*p, target_rank)
            W_cand = np.random.randn(p*n, target_rank)
            
            U_cand /= np.sqrt(target_rank)
            V_cand /= np.sqrt(target_rank)
            W_cand /= np.sqrt(target_rank)
            
            # Simple gradient descent
            lr = 0.01
            for iter in range(100):
                if time.time() > deadline - 10:
                    break
                
                # Compute gradient
                grad_U = np.zeros_like(U_cand)
                grad_V = np.zeros_like(V_cand)
                grad_W = np.zeros_like(W_cand)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Current value
                            val = 0.0
                            for r in range(target_rank):
                                val += U_cand[i*m+j, r] * V_cand[j*p+k, r] * W_cand[k*n+i, r]
                            # Error
                            err = val - 1.0
                            # Gradient contribution
                            for r in range(target_rank):
                                grad_U[i*m+j, r] += err * V_cand[j*p+k, r] * W_cand[k*n+i, r]
                                grad_V[j*p+k, r] += err * U_cand[i*m+j, r] * W_cand[k*n+i, r]
                                grad_W[k*n+i, r] += err * U_cand[i*m+j, r] * V_cand[j*p+k, r]
                
                # Update
                U_cand -= lr * grad_U
                V_cand -= lr * grad_V
                W_cand -= lr * grad_W
                
                # Normalize to prevent divergence
                U_cand /= np.sqrt(np.mean(U_cand**2) + 1e-10)
                V_cand /= np.sqrt(np.mean(V_cand**2) + 1e-10)
                W_cand /= np.sqrt(np.mean(W_cand**2) + 1e-10)
            
            res = tensor_residual(U_cand, V_cand, W_cand)
            if res < best_res:
                best_res = res
                best_U, best_V, best_W = U_cand.copy(), V_cand.copy(), W_cand.copy()
        
        if best_U is not None and best_res < 1e-6:
            return best_U, best_V, best_W, best_res
        
        return None, None, None, best_res
    
    # First, try the structured rank-10 approach
    U_r10, V_r10, W_r10, res_r10 = try_rank_10_decomposition()
    
    if U_r10 is not None and verify(U_r10, V_r10, W_r10, tol=1e-6):
        return {"U": U_r10.tolist(), "V": V_r10.tolist(), "W": W_r10.tolist()}
    
    # Fall back to schoolbook if rank-10 doesn't work
    U, V, W = schoolbook_decomposition()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
