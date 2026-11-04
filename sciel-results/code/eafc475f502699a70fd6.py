<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

import numpy as np
import time

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if U, V, W form a valid decomposition of matrix multiplication."""
    rank = len(U[0])
    # Build the multiplication tensor T[i,j,k] = delta_{ik} (for C = A*B)
    # Actually, the tensor is T_{i,j,k} such that C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # The decomposition is: T_{i,j,k} = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Wait, the convention is W[k*n+i, r]. Let me re-read.
    # Output factor W[k*n+i, r]. So W index is k*n + i.
    # The tensor for matrix mult is T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Actually the tensor is defined as T[i,j,k] = 1 (it's just the identity in the "multiplication" sense)
    # More precisely: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i,j,k (it's the "multiplication tensor")
    # No wait. The tensor is T such that C[i,k] = sum_j A[i,j] * T[i,j,k] * B[j,k]? No.
    # The standard definition: The matrix multiplication tensor is T[i,j,k] where
    # C[i,k] = sum_j A[i,j] B[j,k]
    # As a trilinear form: T(A,B,C) = sum_{i,j,k} C[i,k] A[i,j] B[j,k]
    # The tensor components are T[i,j,k] = 1 if k==k (always 1) ... no.
    # Actually, for the decomposition to work:
    # C[i,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] ... no, that's not right either.
    
    # Let me think about this more carefully.
    # The standard tensor decomposition for matrix multiplication:
    # We want: C[i,k] = sum_j A[i,j] B[j,k]
    # This can be written as: C[i,k] = sum_c (sum_j U[i,j,c] A[i,j]) * (sum_j V[j,k,c] B[j,k]) ... no
    # 
    # Actually, the correct formulation is:
    # C[i,k] = sum_c U_c[i] * V_c[k] * W_c ... no.
    #
    # Let me use the standard definition. The multiplication tensor T has components:
    # T[i,j,k] = 1 if we're multiplying the (i,j) entry of A by the (j,k) entry of B
    # to get the (i,k) entry of C. So T[i,j,k] = 1 for all valid i,j,k.
    # 
    # Wait, that's not right either. The tensor is:
    # T_{i,j,k} = 1 (for all i, j, k) -- no, that would give sum_{i,j,k} A[i,j]B[j,k]C[k,i]
    #
    # The correct tensor is: T[i,j,k] = 1 if the indices match for multiplication.
    # Actually, the multiplication tensor is defined by:
    # <T, A ⊗ B ⊗ C> = sum_{i,j,k} A[i,j] B[j,k] C[k,i]  (with appropriate index conventions)
    # 
    # But for the decomposition, we need:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And T[i,j,k] should equal 1 for all i,j,k (since every A[i,j]*B[j,k] contributes to C[i,k])
    #
    # Wait no. Let me re-derive. 
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We want to write this as: C[i,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] ... this doesn't make sense
    #
    # The correct formulation: We're decomposing the tensor T where
    # T[i,j,k] = 1 for all i,j,k (this is the "full" tensor for matrix mult)
    # and we want T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # Hmm, but that would mean for each (i,j,k), the sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    #
    # Let me verify with the schoolbook: For each (i,j,k), there's exactly one c where all three are 1.
    # Yes! In the schoolbook, for each (i,j,k), we set U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1
    # for a unique c. So the product is 1 for that c and 0 for others. Sum = 1. Correct.
    
    # So the verification is: for all i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check all (i,j,k) combinations
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_random_rank(rank, n, m, p, max_iters=10000, tol=1e-6, seed=42):
    """Try to find a rank decomposition using random initialization and gradient descent."""
    np.random.seed(seed)
    
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Scale to get initial values closer to 1
    U *= 2.0
    V *= 2.0
    W *= 2.0
    
    lr = 0.01
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Compute loss and gradients
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] should be 1
                    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                    val = np.sum(prod)
                    loss += (val - 1.0) ** 2
                    
                    # d(loss)/d(U[i*m+j, c]) = 2*(val-1) * V[j*p+k, c] * W[k*n+i, c]
                    # d(loss)/d(V[j*p+k, c]) = 2*(val-1) * U[i*m+j, c] * W[k*n+i, c]
                    # d(loss)/d(W[k*n+i, c]) = 2*(val-1) * U[i*m+j, c] * V[j*p+k, c]
                    factor = 2.0 * (val - 1.0)
                    grad_U[i*m+j] += factor * V[j*p+k] * W[k*n+i]
                    grad_V[j*p+k] += factor * U[i*m+j] * W[k*n+i]
                    grad_W[k*n+i] += factor * U[i*m+j] * V[j*p+k]
        
        loss /= 2.0  # Because of the 0.5 in MSE
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < tol:
            break
        
        # Gradient descent step
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Adaptive learning rate
        if it % 100 == 0 and it > 0:
            lr *= 0.95
    
    # Check if we found a valid solution
    if best_loss < tol:
        # Convert to list of lists with rounded values
        U_list = [[round(x, 4) for x in row] for row in best_U]
        V_list = [[round(x, 4) for x in row] for row in best_V]
        W_list = [[round(x, 4) for x in row] for row in best_W]
        
        # Verify
        if _verify(U_list, V_list, W_list, n, m, p, tol=1e-4):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _structured_search(n, m, p, max_time=60):
    """Try structured decompositions with lower rank."""
    start_time = time.time()
    best_result = None
    best_rank = n * m * p
    
    # Try ranks from some lower bound up to schoolbook
    # Theoretical lower bound for 3x5x5x6 is at least max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But realistically, we need to try various ranks
    
    for rank in range(30, best_rank + 1, 2):
        if time.time() - start_time > max_time:
            break
        
        result = _try_random_rank(rank, n, m, p, max_iters=2000, tol=1e-8, seed=rank)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    return best_result, best_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # Must finish within 120s
    
    # First, get the schoolbook solution as a fallback
    schoolbook = _schoolbook()
    
    # Try to find a better decomposition
    result, rank = _structured_search(n, m, p, max_time=100)
    
    if time.time() > deadline - 5:
        return schoolbook
    
    if result is not None:
        return result
    
    return schoolbook
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
