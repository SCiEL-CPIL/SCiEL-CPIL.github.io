import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 145  # Leave 5s safety margin

def _check_time():
    return (time.time() - START_TIME) > TIME_LIMIT

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check that U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] sums to 1
                # and all other combinations sum to 0
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if val != 1:
                    return False
    # Check that for any (i,j,k) where the product should be 0, it is indeed 0
    # The tensor T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Actually, the decomposition must satisfy:
    # sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = delta_{i,j,k} for the appropriate indices
    # More precisely, for matrix multiplication C = A*B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all valid i,j,k, and the decomposition is:
    # T[i,j,k] = sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    # This must equal 1 for all i,j,k in range
    # Wait, that's not right. The tensor for matrix multiplication is:
    # T_{ijk} = 1 if we're computing C_{i,k} from A_{i,j} and B_{j,k}
    # Actually, the standard decomposition is:
    # sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] should equal 1 for all i,j,k
    # But that would mean every entry of the tensor is 1, which is not correct.
    
    # Let me reconsider. The matrix multiplication tensor T is defined as:
    # T[i, j, k] = 1 if and only if the product A[i,j] * B[j,k] contributes to C[i,k]
    # In the standard decomposition, we have:
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    # For this to equal sum_j A[i,j] * B[j,k], we need:
    # sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1 for all i,j,k
    # AND the sum over j gives the correct result.
    
    # Actually, I think the condition is simpler. The tensor T has entries T[i,j,k] = 1 
    # for all i in [0,n), j in [0,m), k in [0,p), and 0 otherwise. The decomposition 
    # must satisfy:
    # sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = T[i,j,k] = 1 for all valid i,j,k
    
    # But wait, that can't be right either because then the rank would be at least n*m*p.
    
    # Let me look at this more carefully. The matrix multiplication tensor is a 3-tensor 
    # T of dimensions n x m x p where T[i,j,k] = 1. The rank of this tensor is the 
    # minimum number of rank-1 tensors that sum to T.
    
    # A rank-1 tensor is U[i*m+j] * V[j*p+k] * W[k*n+i] for fixed i,j,k? No.
    
    # Actually, I think I'm overcomplicating this. Let me use the standard definition:
    # The tensor T is in R^{n*m} x R^{m*p} x R^{p*n} where
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    # and T[a, b, c] = 0 otherwise.
    
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # and T[a,b,c] = 0 for all other (a,b,c).
    
    # The decomposition is: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # where U is n*m x rank, V is m*p x rank, W is p*n x rank.
    
    # So we need: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    # = 1 if (a,b,c) = (i*m+j, j*p+k, k*n+i) for some i,j,k
    # = 0 otherwise
    
    # Let's verify this properly
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                expected = 0
                i, j = divmod(a, m)
                j2, k = divmod(b, p)
                k2, i2 = divmod(c, n)
                if j == j2 and i == i2 and k == k2:
                    expected = 1
                actual = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if actual != expected:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Create the schoolbook decomposition with rank n*m*p."""
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
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, target_rank, time_limit=10):
    """Try to reduce the rank by finding linearly dependent columns."""
    start = time.time()
    current_U, current_V, current_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    current_rank = len(current_U[0])
    
    while current_rank > target_rank and (time.time() - start) < time_limit:
        if _check_time():
            break
        # Try to find a column that can be removed
        found = False
        for r in range(current_rank):
            # Check if column r is linearly dependent on others
            # This is expensive, so we'll use a heuristic: try removing each column
            # and see if the decomposition still works
            test_U = [row[:] for row in current_U]
            test_V = [row[:] for row in current_V]
            test_W = [row[:] for row in current_W]
            
            # Remove column r
            for row in test_U:
                del row[r]
            for row in test_V:
                del row[r]
            for row in test_W:
                del row[r]
            
            if _verify_decomposition(test_U, test_V, test_W, n, m, p):
                current_U = test_U
                current_V = test_V
                current_W = test_W
                current_rank -= 1
                found = True
                break
        
        if not found:
            break
    
    return current_U, current_V, current_W, current_rank

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = len(U[0])
    
    # Try to reduce rank
    for target in range(best_rank - 1, 60, -1):
        if _check_time():
            break
        U_test, V_test, W_test, rank_test = _try_reduce_rank(U, V, W, n, m, p, target, time_limit=10)
        if rank_test < best_rank:
            best_rank = rank_test
            U, V, W = U_test, V_test, W_test
    
    return {"U": U, "V": V, "W": W}
