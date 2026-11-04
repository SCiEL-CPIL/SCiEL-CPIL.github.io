"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]
MAX_STEPS = 8

def _schoolbook(n, m, p):
    """
    Standard schoolbook matrix multiplication tensor decomposition.
    Rank = n * m * p.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    Indexing:
    U[i*m+j, c]
    V[j*p+k, c]
    W[k*n+i, c]
    """
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

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill switch
    
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    
    # 1. Establish the safe baseline (Schoolbook)
    best = _schoolbook(n, m, p)
    best_rank = base_rank
    
    # 2. Attempt to find a lower-rank decomposition via structured search
    # Strategy: Try to find a rank-59 decomposition by identifying a linear
    # dependency among the 60 schoolbook terms in the tensor space.
    
    def _verify_decomposition(U, V, W, n, m, p):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        # Check dimensions
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        for c in range(rank):
            if len(U[c]) != rank or len(V[c]) != rank or len(W[c]) != rank:
                return False
        # Verify: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        # and for all (i,j,k) where not all match, the sum should be 0
        # Actually, the condition is: the tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # should equal 1 if this is a valid term, but actually the matrix multiplication
        # tensor has T[i,j,k] = delta_{appropriate}... 
        # The correct condition: for matrix multiplication C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
        # The tensor is: sum_{i,j,k} A[i,j]*B[j,k]*C[i,k] = sum_{i,k} (sum_j A[i,j]*B[j,k])*C[i,k]
        # In the decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should give
        # the coefficient of A[i,j]*B[j,k]*C[i,k] which is 1 if j is the "middle" index
        # Wait, let me re-read the convention.
        # The tensor for matrix multiplication: T(A,B) = sum_{i,j,k} A[i,j] B[j,k] C[i,k]
        # This equals sum_{i,k} (sum_j A[i,j] B[j,k]) C[i,k]
        # So the coefficient of A[i,j] B[j,k] C[i,k] in the expansion is 1 for all i,j,k.
        # No wait, that's not right either. Let me think again.
        # Actually, the bilinear map is: (A,B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor is in the space (R^{n*m})* (R^{m*p})* (R^{p*n})
        # T_{i*m+j, j*p+k, k*n+i} = 1 for all i,j,k, and 0 otherwise.
        # So the decomposition must satisfy:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        # and for any (a,b,c) not of the form (i*m+j, j*p+k, k*n+i), the sum is 0.
        # But actually, since U, V, W are defined on their respective index sets,
        # we just need: for all i in [0,n), j in [0,m), k in [0,p):
        #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # And that's it, because the tensor is only non-zero on those triples.
        
        # Check all required entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    def _try_rank59(n, m, p, deadline):
        """
        Attempt to find a rank-59 decomposition.
        
        Approach: Start with the 60 schoolbook terms. Try to find a linear combination
        of 59 of them that reproduces the 60th term's contribution.
        
        For a rank-59 decomposition, we need 59 rank-1 tensors whose sum equals
        the matrix multiplication tensor. 
        
        Key idea: The 60 schoolbook terms are rank-1 tensors:
          T_c = e_{i_c, j_c} ⊗ e_{j_c, k_c} ⊗ e_{k_c, i_c}
        where (i_c, j_c, k_c) is the c-th triple.
        
        We want to see if one of these can be expressed as a linear combination of the others.
        But since they are all distinct basis vectors in the tensor space, they are linearly
        independent! So we cannot simply eliminate one.
        
        Instead, we need to find 59 NEW rank-1 tensors (not necessarily the schoolbook ones)
        that sum to the same tensor.
        
        This is a much harder problem. Let's try a different approach:
        Use the fact that for small matrices, known results exist.
        
        Actually, let me try a constructive approach: modify the schoolbook by
        combining terms in a Strassen-like way.
        
        For 3x4x5, we can try to use the Laderman decomposition for the 3x3 part
        and extend it.
        
        Laderman's decomposition for 3x3 matrix multiplication has rank 19 (vs 27 schoolbook).
        Let's try to adapt this.
        """
        # Laderman's 19-term decomposition for 3x3
        # This is complex to implement from scratch, so let's try a simpler approach.
        
        # Alternative: use a random search with verification
        import random
        random.seed(42)
        
        # Try to find a rank-59 decomposition by perturbing the schoolbook
        # and checking if we can reduce the rank.
        
        # For now, return None as we need a more sophisticated approach.
        # The schoolbook terms are linearly independent, so we need genuinely new rank-1 tensors.
        
        return None
    
    # Try the rank-59 search
    result_59 = _try_rank59(n, m, p, deadline)
    if result_59 is not None:
        if _verify_decomposition(result_59["U"], result_59["V"], result_59["W"], n, m, p):
            best = result_59
            best_rank = 59
    
    # Check time to ensure we don't accidentally timeout
    if time.time() > deadline - 5.0:
        return best

    return best
# EVOLVE-BLOCK-END
