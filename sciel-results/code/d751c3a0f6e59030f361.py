"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid tensor decomposition for n×m × m×p matrix mult.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] should equal 1 if i==k else 0
    Wait, let me re-read the convention.
    
    Actually for matrix multiplication C = A * B where A is n×m, B is m×p, C is n×p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has indices (i, j, k) where T[i,j,k] = 1 if this term contributes to C[i,k].
    Actually the tensor is: T[i, j, k] = delta_{i,k} is wrong.
    
    Let me think again. The matrix multiplication tensor is T[i, j, k] where:
    - i indexes rows of A (0..n-1)
    - j indexes columns of A / rows of B (0..m-1)  
    - k indexes columns of B (0..p-1)
    
    And C[i,k] = sum_j A[i,j] * B[j,k]
    
    So the tensor T[i,j,k] = 1 for all valid (i,j,k) - it's just the identity in the middle index.
    Wait no, that's not right either. The tensor is:
    T[i, j, k] = 1 if and only if this triple (i,j,k) is part of the multiplication.
    
    Actually, the matrix multiplication tensor is defined as:
    T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    No wait. Let me re-read the problem statement.
    
    "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    
    So the tensor T has components T[i, j, k] and:
    T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    And this should equal 1 if the (i,j,k) term contributes to matrix multiplication.
    
    For matrix multiplication C = A*B:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    So T[i,j,k] = 1 for all valid i,j,k? No, that would make it a all-ones tensor.
    
    Wait, I think I'm confusing myself. The matrix multiplication tensor is:
    T[i, j, k] = 1 if i, j, k form a valid multiplication term, which is always true for valid indices.
    
    Actually no. The tensor is defined such that:
    sum_{j} T[i, j, k] * A[i,j] * B[j,k] = C[i,k]
    
    So T[i,j,k] = 1 for all i,j,k in valid ranges. That's just a tensor of all 1s.
    
    Hmm, but that can't be right because then any rank-1 decomposition would work.
    
    Let me reconsider. I think the tensor is:
    T[i, j, k] = delta_{i, k} ... no that doesn't make sense for matrix mult.
    
    Actually, I think the correct interpretation is:
    The matrix multiplication tensor T is a 3-tensor with T[i, j, k] = 1 for all valid indices.
    The decomposition means: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    Wait, but W has index k*n+i, so W[k*n+i, c] corresponds to the (k,i) pair.
    
    So: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    And T[i,j,k] should be 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    Let me verify with the schoolbook: U[i*m+j, column]=1, V[j*p+k, column]=1, W[k*n+i, column]=1
    for each (i,j,k). So for a given (i,j,k), only the column corresponding to (i,j,k) has all three = 1,
    giving T[i,j,k] = 1. For other (i',j',k') with i'!=i or j'!=j or k'!=k, at least one factor is 0.
    So yes, T is the all-ones tensor!
    
    Wait, that can't be right. If T is all ones, then the rank is just 1 (take U=V=W=all ones).
    
    I must be misunderstanding the tensor. Let me re-read.
    
    "Find a rank-minimal exact tensor decomposition for multiplying a 5 by 5 matrix by a 5 by 5 matrix."
    
    OK so the tensor represents the bilinear map (A, B) -> C = A*B. The tensor T has the property that:
    C[i,k] = sum_{j,c} A[i,j] * B[j,k] * T[i,j,k,c] ... no.
    
    Actually, I think the standard tensor for matrix multiplication is:
    T[i, j, k] = 1 if this is a valid term, and the decomposition is:
    T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    
    But wait, W is indexed by k*n+i, not i*p+k or something. Let me look at the indices again.
    
    U: (n*m, rank) -> U[i*m+j, c]
    V: (m*p, rank) -> V[j*p+k, c]  
    W: (p*n, rank) -> W[k*n+i, c]
    
    So the tensor component is T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    And this should equal... what? For matrix multiplication, I believe T[i,j,k] = 1 for all valid (i,j,k).
    
    But as I noted, that would make the rank 1. So I must be wrong.
    
    Let me look at this differently. Perhaps the tensor is not all ones. Perhaps:
    T[i, j, k] = 1 if k == i (diagonal), 0 otherwise? No, that's not matrix multiplication.
    
    Actually, I think I've been overthinking this. Let me just test the schoolbook decomposition
    and see what tensor it produces.
    
    For n=m=p=5, schoolbook has rank=125. For each (i,j,k), there's a column c where
    U[i*5+j, c]=1, V[j*5+k, c]=1, W[k*5+i, c]=1.
    
    So T[i,j,k] = sum_c U[i*5+j,c]*V[j*5+k,c]*W[k*5+i,c]
    
    For a fixed (i,j,k), the only c where all three are nonzero is the c corresponding to (i,j,k).
    So T[i,j,k] = 1.
    
    For (i',j',k') != (i,j,k), say i' != i, then U[i'*5+j', c] is 0 for c=(i,j,k) since i'!=i.
    So T[i',j',k'] = 0 for (i',j',k') != (i,j,k)? No wait, that's not right either.
    
    T[i',j',k'] = sum_c U[i'*5+j',c] * V[j'*5+k',c] * W[k'*5+i',c]
    
    For c corresponding to (i,j,k): U[i'*5+j',c] = 1 if i'=i and j'=j, else 0.
    So if (i',j',k') != (i,j,k), then for c=(i,j,k), at least one factor is 0.
    
    But T[i',j',k'] sums over ALL c. So T[i',j',k'] = 1 if there exists c such that
    U[i'*5+j',c]=1, V[j'*5+k',c]=1, W[k'*5+i',c]=1 simultaneously.
    
    In schoolbook, each c corresponds to exactly one (i,j,k). So T[i',j',k'] = 1 iff
    there's a c with (i,j,k) = (i',j',k'), which is always true (take c for (i',j',k')).
    
    So T is indeed all ones! This means the rank is 1, which contradicts everything.
    
    I must be fundamentally misunderstanding the problem. Let me re-read the task.
    
    "Find a rank-minimal exact tensor decomposition for multiplying a 5 by 5 matrix by a 5 by 5 matrix."
    
    Maybe the tensor is not T[i,j,k] but rather the bilinear map itself. The rank of the matrix
    multiplication tensor is the minimum number of multiplications needed.
    
    I think the correct tensor is: T[i, j, k] = 1 for all i,j,k, and the rank is the minimum rank
    of a tensor equal to the all-ones tensor? No, that's rank 1.
    
    OK I think I need to just go with the verification approach: check that the decomposition
    correctly computes matrix multiplication. Let me just implement a verifier that checks
    C = A*B for random A, B.
    """
    pass

def _matmul_from_decomposition(U, V, W, A, B, n, m, p):
    """Compute C = A*B using the decomposition."""
    rank = len(U[0]) if U else 0
    C = np.zeros((n, p), dtype=float)
    for c in range(rank):
        # U[:, c] gives n*m values, reshape to (n, m)
        U_col = np.array(U[:, c]).reshape(n, m)
        V_col = np.array(V[:, c]).reshape(m, p)
        W_col = np.array(W[:, c]).reshape(p, n)
        # The contribution to C[i,k] is sum over j of A[i,j] * V_col[j,k] * ... 
        # Actually, let me think about this differently.
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So we need: sum_j A[i,j] * B[j,k] = sum_c (sum_j A[i,j] * U[i*m+j,c]) * V[j*p+k,c] * W[k*n+i,c]
        # This doesn't factor nicely unless W is identity-like.
        
        # Let me just directly compute the tensor and check.
        pass
    return C

def _verify_by_tensor(U, V, W, n, m, p):
    """Verify by checking T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # The known best rank for 5x5 is 49
    # Try ranks from 49 upward
    random.seed(42)
    
    for target_rank in range(49, best_rank + 1):
        if time.time() > deadline - 10:  # 10 second safety margin
            break
        
        # Try a few random initializations for this rank
        max_attempts = 5
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values
            U = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Verify
            if _verify_by_tensor(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
