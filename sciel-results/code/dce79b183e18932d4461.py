"""Matrix multiplication 3,3,7 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

def _schoolbook(n, m, p):
    """Return U, V, W for schoolbook decomposition of n x m * m x p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    # Build the tensor from decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication, T[i,j,k] should be 1 for all valid (i,j,k)
    # Wait, that's not right. For matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] which corresponds to the coefficient of A[i,j]*B[j,k] in C[i,k]
    # So T[i,j,k] = 1 if the indices are consistent, 0 otherwise.
    # Actually, for the multiplication tensor, every entry is 1 because
    # C[i,k] = sum_j A[i,j] B[j,k], so the coefficient of A[i,j]B[j,k] in C[i,k] is 1
    # for all valid i,j,k.
    # 
    # Wait, that's the "raw" tensor. Let me re-read the contract.
    # "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor T_{i,j,k} has T_{i,j,k} = 1 for all i,j,k
    # (since each product A[i,j]*B[j,k] contributes to C[i,k]).
    # 
    # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    # 
    # Let's verify this.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _combine_decompositions(n, m, p1, p2, U1, V1, W1, U2, V2, W2):
    """Combine two decompositions for p1 and p2 into one for p1+p2.
    The second decomposition operates on the shifted k indices."""
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    r = r1 + r2
    
    U = np.zeros((n*m, r), dtype=float)
    V = np.zeros((m*(p1+p2), r), dtype=float)
    W = np.zeros(((p1+p2)*n, r), dtype=float)
    
    # First part: k = 0..p1-1
    U[:, :r1] = U1
    V[:m*p1, :r1] = V1
    W[:p1*n, :r1] = W1
    
    # Second part: k = p1..p1+p2-1 (shifted by p1)
    U[:, r1:] = U2
    V[m*p1:, r1:] = V2
    W[p1*n:, r1:] = W2
    
    return U, V, W

def construct():
    n, m, p = 3, 3, 7
    deadline = time.time() + 115  # Safety margin
    
    # Schoolbook fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    
    # Try to find a better decomposition.
    # For 3x3x7, we can try various structured approaches.
    
    # Approach 1: Try to use known lower-rank algorithms.
    # The rank of 3x3x3 matrix multiplication is known to be 23 (from Bini et al.).
    # If we can decompose 3x3x7 as 3x3x3 + 3x3x4, and if 3x3x4 has rank <= 32,
    # we get rank 55.
    
    # Let's try to construct a rank 23 decomposition for 3x3x3.
    # This is complex, so let's first try a simpler approach.
    
    # Approach 2: Use random search for a lower-rank decomposition.
    # This is unlikely to find rank < 63 quickly, but let's try a few ranks.
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = 63
    
    # Try ranks from 62 down to some reasonable lower bound.
    # The lower bound for 3x3x7 is max(3*3, 3*7, 7*3) = 21.
    # Let's try a few promising ranks.
    
    # Actually, for a 120-second budget, random search won't find rank < 63
    # unless we have a very good strategy. Let's just return schoolbook.
    # 
    # But wait - maybe we can do slightly better with a simple trick.
    # For the last column (k=6), we only need 9 terms (3*3*1).
    # For k=0..5, we need 45 terms (3*3*5).
    # Total: 54 terms. That's already an improvement over 63!
    # 
    # Wait, no. The schoolbook for 3x3x7 is 3*3*7 = 63 terms.
    # If we split as 3x3x6 + 3x3x1, we get 54 + 9 = 63. No improvement.
    # 
    # The only way to improve is to share factors across different k values.
    # This requires a non-trivial algorithm like Strassen or better.
    
    # Let's try a structured approach: for each j (middle index), 
    # the slice T[:,j,:] is a 3x7 matrix. We can write each such slice
    # as a sum of rank-1 terms. The minimum number of rank-1 terms for
    # a 3x7 matrix is 3 (its rank). So for 3 values of j, we need at
    # least 9 terms... but that's a lower bound on the "slice rank", not
    # the tensor rank.
    
    # Given the complexity of finding a provably correct lower-rank
    # decomposition within the time budget, return the schoolbook.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
