"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Schoolbook rank
    rank_school = n * m * p  # 54
    
    # Try to find a lower rank decomposition
    # The tensor T has entries T[i, j, k] = 1 if the output element (i, k) depends on A[i,j] * B[j,k]
    # Actually, the matrix multiplication tensor is: sum over j of A[i,j] * B[j,k] = C[i,k]
    # So the tensor is T[i,j,k] = delta (it's 1 for each (i,j,k) triple in the sum)
    # The decomposition: C[i,k] = sum_r U[i,j... wait, need to be careful with indices
    
    # The interface: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] summed over r gives the coefficient
    # for A[i,j] * B[j,k] in the computation of C[i,k]
    
    # For exact matrix multiplication, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    
    # So the tensor we're decomposing has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait, that's not right either. Let me re-read.
    
    # The multiplication tensor: C[i,k] = sum_j A[i,j] B[j,k]
    # This is a trilinear form. The tensor T has indices (i, j, k) and T[i,j,k] = 1
    # because each term A[i,j]*B[j,k] contributes to C[i,k] with coefficient 1.
    
    # The decomposition requires: for all i, j, k:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a rank-r decomposition of the tensor T where T[i,j,k] = 1.
    
    # The trivial decomposition has rank n*m*p with U[i*m+j, i*m*p+j*p+k] = 1, etc.
    
    # To find lower rank, we need to find r < n*m*p such that there exist
    # U, V, W with the above property.
    
    # Known results: For 2x2x2, rank 7 < 8 (Strassen)
    # For 3x3x3, the rank is between 19 and 23.
    # For 3x3x6, we need to figure out the rank.
    
    # Let's try to construct a decomposition using Strassen for the 2x2x2 blocks.
    
    # Approach: Use the fact that 3 = 2 + 1 and 6 = 3 + 3
    # Or 6 = 2 + 2 + 2
    
    # Let me try a direct construction. I'll use the following strategy:
    # Decompose the problem into independent sub-problems and use Strassen where possible.
    
    # For now, let's try to find a decomposition with rank < 54.
    # I'll use a greedy/optimization approach.
    
    best_rank = rank_school
    best_U = None
    best_V = None
    best_W = None
    
    # First, let's create the schoolbook decomposition as fallback
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    # Now try to find a lower rank decomposition
    # Strategy: Try ranks from a lower bound up to 54
    # Lower bound: The tensor T[i,j,k]=1 for all i,j,k has a certain rank.
    # For the all-ones tensor of size n x m x p, the rank is min(n, m, p) if we allow real entries?
    # No, that's not right. The all-ones tensor has rank 1: U[i,j] = 1, V[j,k] = 1, W[k,i] = 1
    # But that doesn't satisfy our constraint because we need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # If we set U[i*m+j, 0] = 1 for all i,j, V[j*p+k, 0] = 1 for all j,k, W[k*n+i, 0] = 1 for all k,i
    # then the sum is 1*1*1 = 1. So rank 1 works?!
    
    # Wait, that can't be right. Let me re-examine.
    # If U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1, then
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] = 1*1*1 = 1
    # So rank 1 works!
    
    # But that seems too good. Let me verify with the actual matrix multiplication.
    # The decomposition means: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k]) * W[k*n+i, r]
    # No wait, that's not how it works. The tensor decomposition is for the bilinear form.
    
    # Let me re-read the task: "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor for (n,m,p) is T in R^(n*m) x R^(m*p) x R^(p*n)
    # defined by T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise?
    # Or is it T[i*m+j, j*p+k, k*n+i] = 1?
    
    # Actually, I think the tensor is defined such that the matrix multiplication is:
    # C[i,k] = sum_{j} A[i,j] * B[j,k]
    # This can be written as: C = sum_r (U_r * A) * (B * V_r) * W_r or something similar.
    
    # Let me think about it differently. The standard tensor decomposition for matrix multiplication:
    # We want: C[i,k] = sum_r U[i, j, r] * A[i,j] * V[j, k, r] * B[j,k] * W[k, i, r]
    # No, that's not standard either.
    
    # The correct formulation: The matrix multiplication is a bilinear map (A, B) -> C.
    # A tensor decomposition of rank r means: C = sum_r (U_r^T * A) * (B * V_r) * W_r
    # where U_r is n x m, V_r is m x p, W_r is p x n... no.
    
    # Let me look at the interface again:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The condition is: for the multiplication to be exact, we need
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
    
    # This is because the coefficient of A[i,j]*B[j,k] in the computation of C[i,k] must be 1,
    # and the coefficient in any other C[i',k'] must be 0.
    
    # Wait, that's the key insight I missed! The sum must be 1 when the output index (k*n+i) matches
    # the (i,k) pair, and 0 otherwise.
    
    # More precisely: The tensor T has entries T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a, b, c] = 1 if i=i', j=j', k=k', and 0 otherwise.
    # So T[i*m+j, j*p+k, k*n+i] = 1, and T[a,b,c] = 0 otherwise.
    
    # The decomposition requires: sum_r U[a, r] * V[b, r] * W[c, r] = T[a, b, c]
    
    # So for a = i*m+j, b = j*p+k, c = k*n+i: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # For other combinations: sum_r U[a, r] * V[b, r] * W[c, r] = 0
    
    # This is the correct formulation. The all-ones approach doesn't work because it would give
    # nonzero values for wrong output indices.
    
    # So we're back to needing a proper decomposition.
    
    # Let me try to construct one using Strassen.
    # For 2x2x2: rank 7. The Strassen decomposition.
    # For 3x3x3: Can be built from 2x2x2 blocks.
    # For 3x3x6: Can be built from 3x3x3 blocks or directly.
    
    # Let me try a different approach: use random initialization and optimize.
    
    # Actually, let me just return the schoolbook decomposition for now, since finding
    # a lower rank decomposition in 120 seconds is challenging.
    
    # But wait, let me try one more thing: the rank of the 3x3x3 multiplication tensor.
    # If I can find a rank-23 decomposition for 3x3x3, then for 3x3x6 I might use
    # two 3x3x3 decompositions (for the two halves of the 6 columns), giving rank 46.
    # Or I can share some structure.
    
    # For now, let me return the schoolbook decomposition.
    U = U_school.tolist()
    V = V_school.tolist()
    W = W_school.tolist()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
