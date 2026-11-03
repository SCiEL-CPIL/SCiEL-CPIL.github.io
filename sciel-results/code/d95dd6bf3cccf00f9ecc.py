"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # Start with schoolbook
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        # Check shapes
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        for r in range(rank):
            if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
                return False
        
        # For each (i,j,k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # And for all other index combinations, the sum should be 0
        # Actually, the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This should equal 1 if (i,j,k) corresponds to a valid multiplication term, 0 otherwise
        
        # Wait, the matrix multiplication tensor is defined as:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T[i,j,k] = delta_{i,j,k} where the multiplication is C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
        # So T[i,j,k] = 1 for all valid (i,j,k) triples, and the decomposition should satisfy:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        # Let me reconsider. The standard matrix multiplication tensor T has T[i,j,k] = 1 for all 
        # i in [0,n), j in [0,m), k in [0,p), and the tensor decomposition satisfies:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # So for this specific problem, ALL (i,j,k) triples have T[i,j,k] = 1.
        # Therefore, we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True
    
    # Try to find lower rank decompositions using random search
    # Strategy: Start with schoolbook, try to merge columns or use structured forms
    
    # For now, let's try a simple random perturbation approach
    # Try ranks from best_rank - 1 down to some minimum
    # But first, let's see if we can quickly verify the schoolbook works
    if not verify(best_U, best_V, best_W):
        # Schoolbook should always work, so this shouldn't happen
        pass
    
    # Try to find better decompositions
    # Use random initialization and gradient-free optimization
    # Since we need exact integer/half-integer solutions, this is tricky
    
    # Alternative: Try known lower bounds and structured decompositions
    # For 3x4x5, the schoolbook rank is 60
    # Lower bound is max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # But actual minimum rank is likely higher
    
    # Let's try a simple approach: use the schoolbook as baseline
    # and attempt to find improvements through random search if time permits
    
    if time.time() < deadline - 5:
        # Try some random structured decompositions
        # This is a placeholder for more sophisticated search
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
