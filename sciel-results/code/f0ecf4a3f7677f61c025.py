import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    def get_schoolbook():
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
    
    def validate(U, V, W):
        """Check if U, V, W form a valid tensor decomposition."""
        rank = len(U[0])
        # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This should equal 1 if i,j,k form a valid multiplication entry, else 0
        # Actually, the tensor for matrix mult C = AB where A is n×m, B is m×p
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T has entries T[i,j,k] = 1 for the multiplication tensor
        # So we need: for all i,k: sum_j (sum_r U[i*m+j,r]*W[k*n+i,r]) * (sum_r V[j*p+k,r]*W[k*n+i,r]) ... 
        # Wait, that's not right. The tensor decomposition is:
        # T_{i,j,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
        # where T_{i,j,k} = 1 (the multiplication tensor has all entries 1 in the relevant positions)
        # Actually no. The matrix multiplication tensor M_{i,j,k} is defined such that
        # (AB)_{i,k} = sum_j M_{i,j,k} A_{i,j} B_{j,k}
        # So M_{i,j,k} = 1 for all valid i,j,k.
        # The tensor decomposition requires: M_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
        # So for each (i,j,k) in range(n)×range(m)×range(p), we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    def validate_fast(U, V, W):
        """Fast validation using numpy."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Reshape: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # For each r, we have outer products that contribute to the tensor
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i*m+j]
                    v_val = V_arr[j*p+k]
                    w_val = W_arr[k*n+i]
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Try to find decompositions of lower rank
    # Known lower bound: max(n*m, m*p, p*n) = max(6, 15, 10) = 15
    # Try ranks from 15 upward
    
    def try_random_decomposition(target_rank, max_attempts=5000):
        """Try random integer decompositions."""
        np.random.seed(42)
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Generate random integer matrices with entries in {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(n*m, target_rank))
            V = np.random.randint(-1, 2, size=(m*p, target_rank))
            W = np.random.randint(-1, 2, size=(p*n, target_rank))
            
            # Check if this decomposes the tensor
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_val = U[i*m+j]
                        v_val = V[j*p+k]
                        w_val = W[k*n+i]
                        s = np.dot(u_val, v_val * w_val)
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
        return None
    
    # Try structured approaches
    # The border rank / rank of matrix multiplication tensors has been studied
    # For 2×3 × 3×5, let's try various ranks
    
    # First, try rank 15 (the lower bound)
    for target_rank in [15, 16, 17, 18, 19, 20]:
        if time.time() > deadline:
            break
        
        result = try_random_decomposition(target_rank, max_attempts=2000)
        if result is not None:
            return result
    
    # If random search fails, try a more systematic approach
    # Use the fact that we can scale entries
    # Try with entries in {-2, -1, 0, 1, 2}
    def try_random_decomposition_scaled(target_rank, max_attempts=3000):
        """Try random integer decompositions with larger range."""
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            U = np.random.randint(-2, 3, size=(n*m, target_rank))
            V = np.random.randint(-2, 3, size=(m*p, target_rank))
            W = np.random.randint(-2, 3, size=(p*n, target_rank))
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_val = U[i*m+j]
                        v_val = V[j*p+k]
                        w_val = W[k*n+i]
                        s = np.dot(u_val, v_val * w_val)
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
        return None
    
    for target_rank in [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]:
        if time.time() > deadline:
            break
        
        result = try_random_decomposition_scaled(target_rank, max_attempts=1000)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return get_schoolbook()
# EVOLVE-BLOCK-END
