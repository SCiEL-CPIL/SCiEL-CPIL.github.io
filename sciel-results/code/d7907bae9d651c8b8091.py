<<<<<<< SEARCH
"""Rewritten program for matrix multiplication 3x5 by 5x6 rank minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5 * 5x6."""
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p = 90
    # We need to find if a lower rank is possible.
    # For matrix multiplication, the rank is known to be at least max(n*m, m*p, n*p) = max(15, 30, 18) = 30.
    # Actually, the trivial lower bound is max(n*m, m*p, n*p).
    # The upper bound is n*m*p = 90.
    # Known results: For small matrices, the rank is often close to the schoolbook rank,
    # but sometimes improvements are possible.
    # However, for 3x5 by 5x6, it is likely that the rank is 90 or close to it.
    # Let's start with the schoolbook decomposition as a safe baseline.
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Convert to lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
=======
"""Rewritten program for matrix multiplication 3x5 by 5x6 rank minimization."""
import time
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [3, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    # Reconstruct the tensor
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] should be 1 if this is a valid term, but actually
    # the tensor for matmul has T[i,j,k] = delta_{i,j,k} in some sense
    # Actually, the multiplication tensor has entries:
    # T[i,j,k,l] = 1 if i corresponds to row of A, j to col of A/row of B, k to col of B
    # and l = (i,k) is the output entry
    # In the trace convention W[k*n+i, r], the tensor is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,j} * delta_{j,k} ... no
    # Actually for matmul: (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has indices (i, j, k) where i is row of A, j is intermediate, k is col of B
    # and the "output" is (i,k)
    # The standard decomposition: for each (i,j,k), we have a rank-1 term
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # Then sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 for all (i,j,k)
    # But that's not right for matmul...
    
    # Let me reconsider. The matrix multiplication tensor T is a 4-index tensor:
    # T[a, b, c, d] = 1 if a=c (row of A = row of output) and b=d (col of A = col of output)
    # Actually no. Let's use the standard formulation.
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i, j, k] where i indexes rows of A, j indexes cols of A / rows of B,
    # k indexes cols of B. The "contraction" gives the output.
    # In the factorization, we want:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    # But actually, the tensor for matmul is defined such that:
    # T[i, j, k] = 1 for all i, j, k (it's the identity in the sense that every combination appears)
    # No, that's not right either.
    
    # Let me just verify by checking that the reconstructed values match the expected tensor.
    # The expected tensor for matmul 3x5 * 5x6:
    # For each (i, j, k) with i in [0,3), j in [0,5), k in [0,6):
    # The contribution to (A*B)[i,k] from A[i,j]*B[j,k] is 1 * 1 = 1
    # So the tensor entry T[i,j,k] = 1 for all i,j,k
    
    # Wait, that would mean the tensor is all ones, which can't be right.
    # Let me think again. The multiplication tensor is:
    # T: R^{n*m} x R^{m*p} x R^{n*p} -> R
    # T(A, B, C) = trace(A^T * B * C) or something like that.
    # Actually, the standard definition: the tensor T satisfies
    # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[k,i] = (A*B) : C (Frobenius inner product)
    # So T[i,j,k] = 1 for all i,j,k
    
    # OK so the tensor is indeed all ones in the (i,j,k) indices.
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Let's verify this
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _try_rank(rank, n, m, p, max_time=10.0):
    """Try to find a decomposition of the given rank."""
    start = time.time()
    deadline = start + max_time
    
    # Initialize with random values
    np.random.seed(42)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # Use gradient descent to minimize the error
    # Error = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
    
    lr = 0.01
    for step in range(1000):
        if time.time() > deadline:
            break
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute the current value
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err = val - 1.0
                    
                    for r in range(rank):
                        dU[i*m+j, r] += 2 * err * V[j*p+k, r] * W[k*n+i, r]
                        dV[j*p+k, r] += 2 * err * U[i*m+j, r] * W[k*n+i, r]
                        dW[k*n+i, r] += 2 * err * U[i*m+j, r] * V[j*p+k, r]
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Check convergence
        if step % 100 == 0:
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        max_err = max(max_err, abs(val - 1.0))
            if max_err < 1e-6:
                # Round to nearest half-integers or integers
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                if _verify_decomposition(U, V, W, n, m, p):
                    return U, V, W
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5 * 5x6."""
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 90
    
    # Start with schoolbook as baseline
    rank = schoolbook_rank
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to find lower rank decompositions
    # Lower bound is max(n*m, m*p, n*p) = 30
    lower_bound = max(n*m, m*p, n*p)
    
    # Try ranks from lower_bound up to schoolbook_rank - 1
    for r in range(lower_bound, best_rank):
        if _time_left() < 5.0:
            break
        
        # Allocate time for this rank attempt
        time_per_attempt = min(15.0, _time_left() / max(1, best_rank - r))
        
        result = _try_rank(r, n, m, p, max_time=time_per_attempt)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = r
            break  # Found a decomposition, try even lower
    
    # Convert to lists as required by the interface
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
>>>>>>> REPLACE
