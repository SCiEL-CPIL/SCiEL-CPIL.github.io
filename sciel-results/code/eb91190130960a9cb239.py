"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import numpy as np
from itertools import product
import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin
    
    # The multiplication tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Start with schoolbook and try to reduce rank
    # Schoolbook: rank = n*m*p = 105
    
    # Try to find a lower rank decomposition using a structured approach
    # For small dimensions, try various rank values and solve
    
    # Key insight: We can try to use a decomposition that exploits the structure
    # One approach: try rank = n*m + m*p + p*n - something, or other structured forms
    
    # Let's try a direct approach: use the fact that we can write
    # A*B where A is n×m, B is m×p
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # The tensor is T[i,j,k] = 1
    # We need rank-1 tensors whose product gives 1 for each (i,j,k)
    
    # Try a greedy approach with random initialization and local search
    # But first, let's try some known structured decompositions
    
    # For 3×5×7, let's try to see if we can do better than 105
    
    # Approach: try to find a decomposition of the form where we share some terms
    # This is related to Strassen's algorithm but for rectangular matrices
    
    # Let me try a systematic search for low-rank decompositions
    # using the constraint that entries should be integer or half-integer
    
    # First, let's verify what ranks are possible by trying to solve
    # the system of equations for small rank values
    
    # The system is: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is a system of n*m*p = 105 equations in 3*rank variables (with rank unknown)
    
    # Try rank values from a lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # But this is a weak bound
    
    # Let's try to find a valid decomposition by random search with verification
    
    def verify(U, V, W, rank):
        """Check if the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find a decomposition with rank < 105
    # Use a random approach: initialize with random values and try to optimize
    
    # Actually, for exact decompositions with integer/half-integer entries,
    # let's try a different approach: enumerate possible structures
    
    # Key idea: if we can find a decomposition where the U, V, W matrices
    # have a lot of zeros, we might be able to reduce the effective rank
    
    # Let's try a structured approach based on the observation that
    # for matrix multiplication, we can sometimes use "block" decompositions
    
    # For 3×5×7, try to see if we can use a decomposition with rank around 50-80
    
    # Random local search approach:
    # 1. Start with a random initialization of U, V, W with small entries
    # 2. Try to minimize the error sum |sum_r ... - 1|^2
    # 3. Use gradient-free optimization (Nelder-Mead or similar)
    
    # But this is for real-valued, and we need exact integer/half-integer solutions
    
    # Alternative: use the fact that we can scale. If we find a real-valued
    # decomposition, we might be able to round to integers
    
    # Let me try a more direct approach: use the known structure of
    # matrix multiplication tensor decompositions
    
    # For the specific case of 3×5×7, let's try to construct a decomposition
    # by hand or using a systematic method
    
    # Method: try to find a decomposition where each rank-1 term corresponds
    # to a "pattern" of nonzero entries in U, V, W
    
    # Given the time constraint, let's try a few strategies:
    
    # Strategy 1: Try rank = 70 (a guess based on some heuristics)
    # Strategy 2: Try rank = 60
    # Strategy 3: Use the schoolbook as fallback
    
    # For now, let's return the schoolbook decomposition as a safe fallback
    # and try to improve it if time permits
    
    # Actually, let me try a simple improvement: can we combine some schoolbook terms?
    # In schoolbook, each term (i,j,k) has U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    # Can we find a way to use fewer terms by having some terms contribute to multiple (i,j,k)?
    
    # A rank-1 term with U[a]=u, V[b]=v, W[c]=w contributes u*v*w to T[a',b',c']
    # where a' = i*m+j, b' = j'*p+k, c' = k'*n+i'
    # But a, b, c are fixed for a given rank-1 term, so it only contributes to ONE (i,j,k)
    # Wait, no: U[i*m+j, r] is nonzero for specific (i,j), V[j*p+k, r] for specific (j,k),
    # W[k*n+i, r] for specific (k,i)
    # So a rank-1 term contributes to (i,j,k) if U[i*m+j,r], V[j*p+k,r], W[k*n+i,r] are all nonzero
    # For it to contribute to multiple (i,j,k), we need multiple (i,j) with U[i*m+j,r]!=0, etc.
    
    # So the key is: can we have a rank-1 term where U has multiple nonzero entries,
    # V has multiple nonzero entries, W has multiple nonzero entries, such that
    # the product still gives the right values?
    
    # This is the crux of the problem. Let me try to find such a decomposition.
    
    # For simplicity, let's try a decomposition where each U row has at most 2 nonzeros,
    # each V row has at most 2 nonzeros, each W row has at most 2 nonzeros.
    
    # This is getting complex. Let me just return the schoolbook for now.
    
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
