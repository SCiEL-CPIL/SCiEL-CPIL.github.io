"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Reconstruct tensor T_{i,j,k} from factors
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T_recon = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.dot(
                    U[i*m + j, :],
                    V[j*p + k, :] * W[k*n + i, :]
                )
    
    # Target tensor: T[i,j,k] = 1 for all i,j,k (since each (i,j,k) is independent)
    # Wait, that's not right. The matmul tensor is:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T_{i,j,k} = 1 (it's the identity in the j-index pairing)
    # Actually T[i,j,k] = 1 for all i,j,k because A[i,j]*B[j,k] contributes to C[i,k]
    # The tensor is T_{i,j,k} = 1
    
    T_target = np.ones((n, m, p))
    
    if not np.allclose(T_recon, T_target, atol=1e-10):
        return False
    
    # Check for exactness with rational/integer arithmetic
    # Convert to fractions for exact check
    from fractions import Fraction
    U_frac = [[Fraction(int(round(x * 10**6)) / 10**6 if abs(x * 10**6 - round(x * 10**6)) < 1e-9 else x) for x in row] for row in U]
    V_frac = [[Fraction(int(round(x * 10**6)) / 10**6 if abs(x * 10**6 - round(x * 10**6)) < 1e-9 else x) for x in row] for row in V]
    W_frac = [[Fraction(int(round(x * 10**6)) / 10**6 if abs(x * 10**6 - round(x * 10**6)) < 1e-9 else x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c] for c in range(rank))
                if val != 1:
                    return False
    return True

def _greedy_decomposition(n, m, p, max_rank, time_budget):
    """Try to find a decomposition with rank <= max_rank using greedy approach."""
    start_time = time.time()
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Try different ranks from lower bound to max_rank
    # Lower bound: max(n*p, m*p, n*m) at minimum, but actually the rank must be at least max(n*p, m*p, n*m) / something
    # A simple lower bound: the rank is at least max(n*p, m*p, n*m)
    # But for matmul, rank >= n*p is not necessarily true. Let's just try from 1 up.
    
    for target_rank in range(1, max_rank + 1):
        if time.time() - start_time > time_budget:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() - start_time > time_budget:
                break
            
            # Initialize with random values
            U = np.random.randint(-2, 3, size=(n*m, target_rank))
            V = np.random.randint(-2, 3, size=(m*p, target_rank))
            W = np.random.randint(-2, 3, size=(p*n, target_rank))
            
            # Greedy: iteratively adjust one factor at a time
            # For each (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is a system of n*m*p equations in the factors
            
            # Use a simple iterative approach: for each (i,j,k), if the current sum is wrong,
            # adjust one of the factors to fix it
            converged = False
            for iteration in range(100):
                if time.time() - start_time > time_budget:
                    break
                
                all_correct = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            current = 0
                            for c in range(target_rank):
                                current += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            if abs(current - 1) > 1e-10:
                                all_correct = False
                                break
                        if not all_correct:
                            break
                    if not all_correct:
                        break
                
                if all_correct:
                    converged = True
                    break
                
                # Try to fix by adjusting W for the first incorrect (i,j,k)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            current = 0
                            for c in range(target_rank):
                                current += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            if abs(current - 1) > 1e-10:
                                # Adjust W[k*n+i, :] to fix
                                # We need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                                # Let's set W[k*n+i, c] = (1 - sum_{c'!=c} ...) / (U[i*m+j,c]*V[j*p+k,c])
                                # This is tricky. Instead, let's use a different approach.
                                pass
                        break
                    break
                break
            
            if converged:
                return U, V, W, target_rank
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # First, get the schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find a lower rank decomposition
    # Known lower bound for rank of matmul: at least max(n*p, m*p, n*m)
    # For 2x3x7: max(14, 21, 6) = 21
    lower_bound = max(n*p, m*p, n*m)
    
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        result = _greedy_decomposition(n, m, p, target_rank, deadline - time.time() - 5)
        if result is not None:
            U, V, W, r = result
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = r
                break  # Found the minimum possible rank
    
    return best_result
# EVOLVE-BLOCK-END
