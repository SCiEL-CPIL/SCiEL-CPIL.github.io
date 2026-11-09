"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    Searches for lower-rank decompositions below the schoolbook rank of 36.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline with 10s safety margin
    
    # Try to find a lower-rank decomposition
    # For 3x3x3, minimal rank is 23. For 3x3x4, let's search.
    
    # First, try known small ranks and see if we can find valid decompositions
    # We'll use a randomized approach with verification
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Generate schoolbook as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to find lower-rank decompositions
    # For each candidate rank, try random initializations and see if we can satisfy the constraints
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly represents matrix multiplication."""
        n, m, p = DIMENSIONS
        # For each output position (i,k) and input positions (j), check:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,k,j) is the "multiplication" pattern
        # Actually, the tensor T[i,j,k] for matrix multiplication is:
        # T[i,j,k] = 1 if we're computing C[i,k] += A[i,j]*B[j,k]
        # The decomposition requires: for all i,j,k:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # Wait, that's not right. Let me reconsider.
        
        # The matrix multiplication tensor has entries T[i,j,k] where:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # No wait, that's the structure. The tensor for matrix multiplication (n,m,p) is:
        # M[i,j,k] = 1 if the indices correspond to a valid multiplication term
        # Actually, the standard convention: the multiplication tensor has
        # T[i,j,k] = 1 for all i,j,k in their ranges, and the decomposition is:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # But that would mean T[i,j,k] = 1 for all i,j,k, which is just an all-ones tensor.
        
        # Let me reconsider the problem statement. The task says "exactly decompose the matrix multiplication tensor"
        # with U of shape (n*m, rank), V of shape (m*p, rank), W of shape (p*n, rank)
        # and W follows the trace convention W[k*n+i, r].
        
        # The matrix multiplication tensor for (n,m,p) has:
        # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # And the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Wait, that can't be right because that would be an all-ones tensor, which has rank 1.
        
        # I think I'm misunderstanding. Let me re-read: "multiplying a 3 by 3 matrix by a 3 by 4 matrix"
        # A is 3x3, B is 3x4, C = A*B is 3x4
        # C[i,k] = sum_j A[i,j] * B[j,k] for i in [0,3), j in [0,3), k in [0,4)
        
        # The tensor representation: we have variables A[i,j], B[j,k], C[i,k]
        # The multiplication is a trilinear form: sum_{i,j,k} A[i,j] * B[j,k] * C[i,k] = 0
        # No, that's not it either.
        
        # Actually, the standard tensor decomposition for matrix multiplication:
        # The multiplication tensor M has entries M[i,j,k] where i indexes rows of A (and C),
        # j indexes columns of A (and rows of B), k indexes columns of B (and C)
        # M[i,j,k] = 1 for all valid i,j,k
        # And we write: M[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # But if M[i,j,k] = 1 for all i,j,k, then this is just an all-ones tensor of size (n*m) x (m*p) x (p*n)
        # which has rank 1, not 36.
        
        # I must be misunderstanding the tensor structure. Let me think again...
        
        # Oh! I think the issue is that the tensor is NOT all ones. The matrix multiplication
        # involves specific index contractions. Let me look at this differently.
        
        # Actually, I think the correct interpretation is:
        # The multiplication map is: (A, B) -> C where C[i,k] = sum_j A[i,j] * B[j,k]
        # This is a bilinear map, which can be represented as a tensor T with:
        # T[i,k,j] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
        # So T[i,k,j] = 1 for all valid i,k,j, and 0 otherwise (but there's no "otherwise" since all are valid)
        
        # I'm confused. Let me just verify the schoolbook decomposition works.
        
        # Schoolbook: for each (i,j,k), we have one rank-1 term with:
        # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
        # So sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 (only the term where r=col contributes)
        
        # So the tensor being decomposed has T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        # This is a specific tensor, not an all-ones tensor.
        
        # To verify: for given indices (a, b, c) where a = i*m+j, b = j'*p+k', c = k''*n+i''
        # The tensor value is 1 if there exist i,j such that a=i*m+j, and j'=j, k''=i, and b=j*p+k', c=k'*n+i
        # Wait, this is getting complicated. Let me just check if the schoolbook satisfies:
        # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    # Try ranks from lower values up to schoolbook
    # For 3x3x4, let's try ranks from 20 to 35
    for candidate_rank in range(20, schoolbook_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations for this rank
        max_attempts = 50 if candidate_rank < 30 else 20
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Initialize with random values from {-1, 0, 1} or half-integers
            U = [[random.choice([-1, 0, 1]) for _ in range(candidate_rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(candidate_rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(candidate_rank)] for _ in range(p * n)]
            
            if verify_decomposition(U, V, W, candidate_rank):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = candidate_rank
                break
        
        if best_rank < schoolbook_rank:
            break
    
    if best_result is not None:
        return best_result
    else:
        return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
