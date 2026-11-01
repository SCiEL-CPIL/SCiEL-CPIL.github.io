"""Rewritten program for 2x3x9 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition as dense arrays."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = U.shape[1]
    # Check: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c] should equal W[k*n+i,c] contribution
    # Actually, the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matrix mult, T[i,j,k] = 1 if we're computing (A*B)[i,k] using A[i,j]*B[j,k]
    # The tensor for mat mult has T[i,j,k] = 1 for all valid (i,j,k)
    # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                prod_sum = np.dot(u_val, v_val * w_val)
                if abs(prod_sum - 1.0) > 1e-9:
                    return False
    return True

def _to_fraction_list(arr):
    """Convert numpy array to list of Fractions."""
    result = []
    for row in arr:
        row_list = []
        for val in row:
            if abs(val - round(val)) < 1e-9:
                row_list.append(Fraction(int(round(val))))
            else:
                # Try to find a simple fraction
                frac = Fraction(val).limit_denominator(1000)
                row_list.append(frac)
        result.append(row_list)
    return result

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3x9 matrix multiplication."""
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    U0, V0, W0 = _build_schoolbook(n, m, p)
    best_rank = U0.shape[1]
    best_U, best_V, best_W = U0, V0, W0
    
    # Try to reduce rank by finding linear dependencies
    # The tensor T has entries T[i,j,k] = 1 for all (i,j,k) in [0,n)x[0,m)x[0,p)
    # We want to find the minimum rank r such that there exist U,V,W with
    # T[i,j,k] = sum_{c=0}^{r-1} U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Strategy: try random combinations and see if we can achieve lower rank
    # For small dimensions, we can try systematic approaches
    
    # Key insight: For matrix multiplication, the tensor rank is at most n*m*p
    # but can be lower. For 2x3x9, let's try to find if rank < 54 is possible.
    
    # Approach: Use the fact that for fixed i, the sub-tensor over (j,k) has rank at most m*p
    # We can try to share factors across different i values.
    
    # Let's try a structured approach:
    # Group by i (2 groups of 3x9 = 27 entries each)
    # For each i, we need U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] to sum to 1
    
    # Try rank = 45, 40, 36, etc.
    # For now, let's try to see if we can reduce by exploiting structure
    
    # Method: Try to find if some columns are linearly dependent in a way that allows merging
    
    # Actually, let's try a more direct approach:
    # The tensor T[i,j,k] = 1 for all (i,j,k)
    # This is a "full" tensor in the sense that all entries are 1
    # But the structure matters: it's the matmul tensor
    
    # For matmul, we know that rank >= max(n*p, m*p, n*m) = max(18, 27, 6) = 27
    # And rank <= n*m*p = 54
    
    # Let's try to find a decomposition with rank < 54
    
    # Simple idea: Can we combine some of the schoolbook terms?
    # Term (i,j,k) contributes U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1
    # If we have two terms (i1,j1,k1) and (i2,j2,k2) with i1=i2, j1=j2, k1!=k2
    # They differ only in W index. We can't easily combine them.
    
    # Let's try random search for lower rank
    np.random.seed(42)
    
    for target_rank in range(53, 26, -1):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Try to find a decomposition with this rank
        # Use random initialization and check
        
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts and time.time() - start_time < DEADLINE - 10:
            attempts += 1
            
            # Random decomposition attempt
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Check if this can represent the tensor
            # We need: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is a system of equations. Let's check if it's satisfied.
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        
        if best_rank == target_rank:
            break
    
    # Convert to list of Fractions
    if best_rank == 54:
        # Use schoolbook
        U_list = [[1 if c == i*m+j else 0 for c in range(54)] for i in range(n) for j in range(m)]
        V_list = [[1 if c == i*m*p + j*p+k else 0 for c in range(54)] for j in range(m) for k in range(p)]
        W_list = [[1 if c == i*m*p + j*p+k else 0 for c in range(54)] for k in range(p) for i in range(n)]
        
        # Fix indexing: column index = i*m*p + j*p + k
        U_list = [[0]*54 for _ in range(n*m)]
        V_list = [[0]*54 for _ in range(m*p)]
        W_list = [[0]*54 for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_list[i*m+j][col] = 1
                    V_list[j*p+k][col] = 1
                    W_list[k*n+i][col] = 1
                    col += 1
    else:
        U_list = _to_fraction_list(best_U)
        V_list = _to_fraction_list(best_V)
        W_list = _to_fraction_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
