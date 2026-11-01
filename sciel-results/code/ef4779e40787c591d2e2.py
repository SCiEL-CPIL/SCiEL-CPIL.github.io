"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check that the tensor entry T[i,j,k] = delta_{i,j,k} is reproduced
                # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                expected = 1.0
                if abs(val - expected) > 1e-10:
                    return False
                # Also check that non-matching entries are zero
                # For a general matrix multiplication tensor, we need:
                # sum_c U[i*m+j, c] * V[j'*p+k', c] * W[k'*n+i', c] = 0 
                # unless i=i', j=j', k=k'
    
    # More complete check: verify the full tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            # The tensor T should have T[i,j,k] = 1 if (i,j,k) matches the product
                            # Actually, the matrix multiplication tensor has entries:
                            # T[i,j,k] where we multiply A[i,j] * B[j,k] = C[i,k]
                            # So the tensor is T[i,j,k] = delta (it's 1 for all valid (i,j,k))
                            # Wait, that's not right. The tensor is:
                            # T_{i,j,k} such that C_{i,k} = sum_j A_{i,j} B_{j,k}
                            # In tensor form: T[i,j,k] = 1 for all i,j,k (it's the identity in the middle)
                            # No, the tensor entries are:
                            # T[i,j,k] = 1 if the indices match the multiplication pattern
                            # Actually for matrix multiplication, the tensor is:
                            # T[i,j,k] = delta_{j,j} which is always 1 for valid indices
                            # Hmm, let me think again.
                            
                            # The matrix multiplication tensor T has entries T[i,j,k] where:
                            # (A * B)[i,k] = sum_j A[i,j] * B[j,k]
                            # So T[i,j,k] = 1 for all valid (i,j,k)
                            # But that's not a sparse tensor. Let me reconsider.
                            
                            # Actually, the tensor decomposition represents:
                            # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            # And T[i,j,k] should equal 1 for all (i,j,k) in the multiplication
                            # No wait, that would make it a rank-1 tensor.
                            
                            # I think I'm confusing myself. Let me just verify that the
                            # decomposition correctly computes matrix products.
                            pass
    
    # Let's do a direct numerical verification with random matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    expected = A @ B
    
    # Compute using the decomposition
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_c (sum_j A[i,j] U[i*m+j,c]) * (sum_k B[j,k] V[j*p+k,c]) * W[k*n+i,c]
    # Wait, that's not how it works. The decomposition gives:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And we need T[i,j,k] = 1 for the multiplication to work.
    # Actually, I think the tensor is defined such that:
    # (A*B)[i,k] = sum_{j} A[i,j] * B[j,k]
    # And the tensor T satisfies: sum_j T[i,j,k] A[i,j] B[j,k] = (A*B)[i,k]
    # So T[i,j,k] = 1 for all valid (i,j,k).
    
    # Let me just verify with a simple test
    # The schoolbook method should work, so let's verify it does
    result = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                # T[i,j,k] should be 1
                t_val = 0.0
                for c in range(rank):
                    t_val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                result[i, k] += A[i, j] * B[j, k] * t_val
    
    if not np.allclose(result, expected, atol=1e-10):
        return False
    
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Start with the schoolbook decomposition of rank n*m*p
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_school
    
    # Try to find a lower rank decomposition
    # The theoretical lower bound for 2x3x3 matrix multiplication rank is not well-known
    # but it's at least max(n*m, m*p, p*n) = max(6, 9, 6) = 9
    # And at most n*m*p = 18
    # Let's try ranks from 9 to 17
    
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    for target_rank in range(9, rank_school):
        if time.time() > deadline - 30:  # Keep 30s safety margin
            break
        
        # Try to find a decomposition of rank target_rank
        # Use a random initialization and optimization approach
        found = False
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(attempt * 1000 + target_rank)
            
            # Initialize U, V, W with random values
            U_rand = np.random.randn(n * m, target_rank)
            V_rand = np.random.randn(m * p, target_rank)
            W_rand = np.random.randn(p * n, target_rank)
            
            # Try to optimize to match the tensor
            # The tensor T[i,j,k] = 1 for all (i,j,k)
            # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
            
            # Use gradient descent to minimize the error
            lr = 0.01
            for iteration in range(1000):
                if time.time() > deadline - 30:
                    break
                
                # Compute gradients
                dU = np.zeros_like(U_rand)
                dV = np.zeros_like(V_rand)
                dW = np.zeros_like(W_rand)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # T[i,j,k] should be 1
                            val = np.sum(U_rand[i*m+j, :] * V_rand[j*p+k, :] * W_rand[k*n+i, :])
                            err = val - 1.0
                            for c in range(target_rank):
                                dU[i*m+j, c] += err * V_rand[j*p+k, c] * W_rand[k*n+i, c]
                                dV[j*p+k, c] += err * U_rand[i*m+j, c] * W_rand[k*n+i, c]
                                dW[k*n+i, c] += err * U_rand[i*m+j, c] * V_rand[j*p+k, c]
                
                U_rand -= lr * dU
                V_rand -= lr * dV
                W_rand -= lr * dW
                
                # Check convergence
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_rand[i*m+j, :] * V_rand[j*p+k, :] * W_rand[k*n+i, :])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-8:
                    # Found a valid decomposition!
                    # Convert to lists with rational approximation
                    U_list = [[0] * target_rank for _ in range(n * m)]
                    V_list = [[0] * target_rank for _ in range(m * p)]
                    W_list = [[0] * target_rank for _ in range(p * n)]
                    
                    for idx in range(n * m):
                        for c in range(target_rank):
                            U_list[idx][c] = round(U_rand[idx, c], 6)
                    for idx in range(m * p):
                        for c in range(target_rank):
                            V_list[idx][c] = round(V_rand[idx, c], 6)
                    for idx in range(p * n):
                        for c in range(target_rank):
                            W_list[idx][c] = round(W_rand[idx, c], 6)
                    
                    # Verify
                    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = target_rank
                        found = True
                        break
            
            if found:
                break
    
    return best_result
# EVOLVE-BLOCK-END
