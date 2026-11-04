import numpy as np
from itertools import product
import time

def construct():
    n, m, p = 2, 3, 6
    
    # The current seed achieves rank 33 by treating the 3x6 block as three 3x2 blocks:
    # 3 * (2x2 Strassen + 2x1x2 standard) = 3 * (7 + 4) = 33
    
    # Let's try to find a lower rank by exploiting the full structure.
    # The tensor is T_{i,j,k} = delta_{i,j,k} for C = AB where A is 2x3, B is 3x6.
    # Actually, the multiplication tensor M_{n,m,p} has entries M_{(i,j),(j,k),(i,k)} = 1.
    # Here n=2, m=3, p=6.
    # Indices: U has shape (n*m, r) = (6, r), V has shape (m*p, r) = (18, r), W has shape (p*n, r) = (12, r).
    # The condition is: for all i in [0,2), j in [0,3), k in [0,6):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other combinations of (i,j) in U, (j',k) in V, (k',i') in W where the indices don't match:
    #   Actually, the tensor decomposition condition is:
    #   M_{(i,j),(j,k),(i,k)} = 1 for valid i,j,k
    #   and M = 0 otherwise. But in matrix multiplication, the "otherwise" entries are not present in the standard 
    #   tensor definition. The tensor is supported only on the "multiplication" indices.
    
    # Let's try a rank of 30, 31, 32 to see if we can find a valid decomposition.
    # Given the time constraint, let's try a randomized search for rank 30.
    
    # Actually, let's first verify the rank 33 solution is correct, then try to lower it.
    # The theoretical lower bound for 2x3x6 is at least max(n*m, m*p, p*n) / something? 
    # Actually, the rank is at least the max of the dimensions of the individual matrices? No.
    # For 2x3x6, the rank is at least 2*3 = 6? No, that's for 2x3x3.
    # The rank of M_{n,m,p} is at least max(n*m, m*p, p*n) is not a lower bound.
    # Actually, the rank is at least n*m if p >= 1, etc. But that's not tight.
    
    # Let's try a direct construction with rank 30.
    # We can try to use Strassen for the 2x2 part and then handle the 3rd column/row more efficiently.
    
    # Alternative approach: Use the fact that 6 = 2*3 and try to group the p dimension differently.
    
    # Let's try a randomized search with rank 30.
    start_time = time.time()
    deadline = 115  # seconds
    
    # Try rank 30
    for rank in [30, 31, 32]:
        if time.time() - start_time > deadline:
            break
            
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Use gradient descent to minimize the error
        # Error = sum over all (i,j,k) of (M_{i,j,k} - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r])^2
        # where M_{i,j,k} = 1 if i,j,k are valid multiplication indices, else 0.
        
        # Actually, the tensor is only defined on the multiplication indices.
        # So we only need to enforce:
        # for all i in [0,n), j in [0,m), k in [0,p):
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # Let's use a simple iterative method.
        lr = 0.01
        for it in range(2000):
            if time.time() - start_time > deadline:
                break
                
            # Compute gradient
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Compute current value
                        val = 0.0
                        for r in range(rank):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err = 1.0 - val
                        
                        for r in range(rank):
                            grad_U[i*m+j, r] += err * V[j*p+k, r] * W[k*n+i, r]
                            grad_V[j*p+k, r] += err * U[i*m+j, r] * W[k*n+i, r]
                            grad_W[k*n+i, r] += err * U[i*m+j, r] * V[j*p+k, r]
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            if it % 100 == 0:
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(rank):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            max_err = max(max_err, abs(1.0 - val))
                if max_err < 1e-6:
                    break
        
        # Check if we found a valid solution
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    max_err = max(max_err, abs(1.0 - val))
        
        if max_err < 1e-6:
            # Round to nearest integers if possible
            U_rounded = np.round(U)
            V_rounded = np.round(V)
            W_rounded = np.round(W)
            
            max_err_rounded = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U_rounded[i*m+j, r] * V_rounded[j*p+k, r] * W_rounded[k*n+i, r]
                        max_err_rounded = max(max_err_rounded, abs(1.0 - val))
            
            if max_err_rounded < 1e-6:
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
    
    # If we couldn't find a lower rank, return the seed solution (rank 33)
    rank = 33
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    # Strassen factors for 2x2x2
    S_U = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 0], [1, 1]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 1], [0, -1]])
    ]
    S_V = [
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 1], [0, -1]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[0, 0], [1, 1]])
    ]
    S_W = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 1], [0, -1]]),
        np.array([[0, 0], [1, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[-1, 0], [1, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 0], [0, 0]])
    ]
    
    # Standard factors for 2x1x2
    Std_U = [
        np.array([[1], [0]]),
        np.array([[1], [0]]),
        np.array([[0], [1]]),
        np.array([[0], [1]])
    ]
    Std_V = [
        np.array([[1, 0]]),
        np.array([[0, 1]]),
        np.array([[1, 0]]),
        np.array([[0, 1]])
    ]
    Std_W = [
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 0], [1, 0]]),
        np.array([[0, 1], [0, 0]]),
        np.array([[0, 0], [0, 1]])
    ]
    
    col = 0
    for b in range(3):
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * m + j, col] = S_U[r][i, j]
            
            for j in range(2):
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[j * p + k_global, col] = S_V[r][j, kp]
            
            for kp in range(2):
                for i in range(2):
                    k_global = 2 * b + kp
                    W[k_global * n + i, col] = S_W[r][kp, i]
            
            col += 1
            
        for r in range(4):
            for i in range(2):
                U[i * m + 2, col] = Std_U[r][i, 0]
            
            for kp in range(2):
                k_global = 2 * b + kp
                V[2 * p + k_global, col] = Std_V[r][0, kp]
            
            for kp in range(2):
                for i in range(2):
                    k_global = 2 * b + kp
                    W[k_global * n + i, col] = Std_W[r][kp, i]
            
            col += 1
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
