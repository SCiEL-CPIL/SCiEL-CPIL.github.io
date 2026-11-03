"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _validate(U, V, W, n, m, p, tol=1e-10):
    """Check if the decomposition is valid: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = delta_{i,k} for all i,j,k"""
    rank = len(U[0]) if U else 0
    # Build tensors as numpy arrays
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i,j,k), check sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == (1 if i==k else 0)
    # But wait, the convention is: C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T[i,j,k] = delta_{i,k} for the multiplication tensor
    # Decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Let's verify by computing the full tensor
    # T[i,j,k] should equal delta_{i,k}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                expected = 1.0 if (i == k) else 0.0
                if abs(s - expected) > tol:
                    return False
    return True

def _strassen_4x4x9():
    """
    Use Strassen's algorithm for 2x2 blocks.
    A is 4x4, B is 4x9.
    Split into 2x2 blocks:
    A = [[A11, A12], [A21, A22]] each 2x2
    B = [[B11, B12], [B21, B22]] each 2x9
    C = [[C11, C12], [C21, C22]] each 2x9
    
    Strassen's 7 products:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Each Mx is a 2x2 x 2x9 product.
    For a 2x2 x 2x9 product, the tensor has rank 4*9 = 36 (schoolbook).
    But we can use Strassen again for the 2x2 part? No, 2x2 is minimal.
    
    Actually, for 2x2 x 2x9:
    We can write this as a sum of rank-1 terms.
    The tensor T[i,j,k] for i in {0,1}, j in {0,1}, k in {0,...,8}
    T[i,j,k] = delta_{i,k} if k < 2, else 0
    
    Rank is at most 4*9 = 36, but can we do better?
    For 2x2 x 2xp, the rank is known. For p=9, let's think...
    
    Actually, let's just use schoolbook for each 2x2 x 2x9 block: rank 36 each.
    Total: 7 * 36 = 252. That's worse than 144.
    
    So Strassen at 2x2 level doesn't help here.
    
    Let me try a different approach: use the fact that we're multiplying 4x4 by 4x9.
    We can think of B as four 4x9 matrices... no.
    
    Alternative: Use a rank decomposition based on known results.
    The rank of the 4x4 x 4x4 multiplication tensor is at most 49 (Lader's result).
    For 4x4 x 4x9, we can write B = [B_left | B_right] where B_left is 4x4 and B_right is 4x5.
    Then A*B = A*B_left + A*B_right.
    
    Rank of 4x4 x 4x4 tensor: at most 49
    Rank of 4x4 x 4x5 tensor: at most 4*4*5 = 80 (schoolbook)
    Total: 49 + 80 = 129 < 144
    
    But we need actual decompositions. Let me try to find one.
    
    Actually, let's just try a randomized search for a valid decomposition at rank < 144.
    """
    n, m, p = 4, 4, 9
    # Start with schoolbook
    return _schoolbook(n, m, p)

def _try_rank_search(n, m, p, max_rank, time_budget=100):
    """
    Try to find a valid decomposition at rank <= max_rank.
    Uses randomized initialization and gradient-based refinement.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    # The tensor T[i,j,k] = delta_{i,k} for i in [0,n), j in [0,m), k in [0,p)
    # We need: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Start with schoolbook and try to reduce
    best = _schoolbook(n, m, p)
    best_rank = len(best['U'][0])
    
    # Try various ranks from min possible up to best_rank - 1
    # Theoretical lower bound: rank >= max(n*p, m*p, n*m) = max(36, 36, 16) = 36
    # But for matrix multiplication, the rank is at least n*p (for square case, it's at least n^2)
    # For 4x4 x 4x9, lower bound is at least 4*9 = 36
    
    for target_rank in range(max(36, best_rank - 20), best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            
            # Initialize with small random values
            np.random.seed(42 + trial)
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Optimize to minimize ||T - sum_c U V W||^2
            # T[i,j,k] = delta_{i,k}
            
            # Use simple gradient descent or coordinate descent
            # For speed, let's use a few iterations of gradient descent
            
            lr = 0.01
            for iteration in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Compute residual
                residual = np.zeros((n, m, p))
                for c in range(target_rank):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                residual[i,j,k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                
                # Target tensor
                target = np.zeros((n, m, p))
                for i in range(n):
                    for k in range(p):
                        if i == k:
                            for j in range(m):
                                target[i,j,k] = 1.0
                
                # Check if we're close enough
                err = np.max(np.abs(residual - target))
                if err < 1e-10:
                    # Convert to list format
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    result = {"U": U_list, "V": V_list, "W": W_list}
                    if _validate(U_list, V_list, W_list, n, m, p):
                        return result
                    break
                
                # Gradient computation (simplified)
                # dU[i*m+j, c] = sum_{k} (residual[i,j,k] - target[i,j,k]) * V[j*p+k, c] * W[k*n+i, c]
                # dV[j*p+k, c] = sum_{i} (residual[i,j,k] - target[i,j,k]) * U[i*m+j, c] * W[k*n+i, c]
                # dW[k*n+i, c] = sum_{j} (residual[i,j,k] - target[i,j,k]) * U[i*m+j, c] * V[j*p+k, c]
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            diff = residual[i,j,k] - target[i,j,k]
                            for c in range(target_rank):
                                dU[i*m+j, c] += diff * V[j*p+k, c] * W[k*n+i, c]
                                dV[j*p+k, c] += diff * U[i*m+j, c] * W[k*n+i, c]
                                dW[k*n+i, c] += diff * U[i*m+j, c] * V[j*p+k, c]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            # Check final result
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if _validate(U_list, V_list, W_list, n, m, p):
                return result
    
    return best

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # First, get the schoolbook decomposition as a fallback
    schoolbook = _schoolbook(n, m, p)
    
    # Try to find a better decomposition
    # Time budget: leave 20 seconds for safety
    time_budget = min(100, max(5, 120 - (time.time() - start_time) - 20))
    
    if time_budget > 10:
        result = _try_rank_search(n, m, p, 144, time_budget)
        if result is not None and len(result['U'][0]) < len(schoolbook['U'][0]):
            return result
    
    return schoolbook
# EVOLVE-BLOCK-END
