"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose matrix multiplication."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matrix multiplication, T[i,j,k] should equal delta_{j,k} (i.e., 1 if j==k, 0 otherwise)
    # But actually for 2x3 * 3x5, the product C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is T[i,j,k] = 1 if the multiplication works, which means we need:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta_{j,k} for all i,k
    # Wait, that's not right. Let me think again.
    
    # The matrix multiplication tensor has entries T[i,j,k] where:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = 1 for all i,j,k (since each product term A[i,j]*B[j,k] contributes)
    # Actually no. The bilinear form is:
    # C[i,k] = sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * A[i,j] * B[j,k]
    # Wait, I need to be more careful.
    
    # Standard formulation: C = A * B where A is n x m, B is m x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # In the tensor decomposition framework:
    # We want: C[i,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] * A[i,j] * B[j,k]
    # This doesn't quite work. Let me use the standard approach.
    
    # Actually, the standard way is:
    # Define the multiplication tensor T with T[i,j,k] = 1 for all valid i,j,k
    # Then C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    
    # In rank-r decomposition:
    # T[i,j,k] = sum_{c=1}^r U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Create the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _random_decomp_attempt(n, m, p, rank, seed=None, iterations=100, tol=1e-9):
    """Try to find a rank-r decomposition using random initialization and gradient descent."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for attempt in range(iterations):
        # Random initialization with small values
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Simple iterative refinement
        lr = 0.01
        for it in range(200):
            # Compute current error
            err = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Current value
                        val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        err += (val - 1.0) ** 2
                        
                        # Gradients
                        for c in range(rank):
                            dU = (val - 1.0) * V[j*p+k][c] * W[k*n+i][c]
                            dV = (val - 1.0) * U[i*m+j][c] * W[k*n+i][c]
                            dW = (val - 1.0) * U[i*m+j][c] * V[j*p+k][c]
                            grad_U[i*m+j, c] += dU
                            grad_V[j*p+k, c] += dV
                            grad_W[k*n+i, c] += dW
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < tol:
                break
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        
        if best_err < tol:
            # Convert to list of lists with integer/half-integer check
            U_list = [[float(x) for x in row] for row in best_U]
            V_list = [[float(x) for x in row] for row in best_V]
            W_list = [[float(x) for x in row] for row in best_W]
            return U_list, V_list, W_list, best_err
    
    if best_err < tol:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return U_list, V_list, W_list, best_err
    
    return None, None, None, best_err

def _try_rank(n, m, p, rank, deadline, max_attempts=5):
    """Try to find a decomposition of given rank."""
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None, None, None
        U, V, W, err = _random_decomp_attempt(n, m, p, rank, seed=attempt, iterations=10, tol=1e-9)
        if U is not None:
            return U, V, W
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Safety margin before 120s limit
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p
    
    # Try lower ranks
    for rank in range(n * m * p - 1, max(n * m * p - 20, 0), -1):
        if time.time() > deadline:
            break
        U, V, W = _try_rank(n, m, p, rank, deadline, max_attempts=3)
        if U is not None:
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
            break  # Found a valid decomposition, stop
    
    return best_result
# EVOLVE-BLOCK-END
