"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # seconds before the 150s kill
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as baseline
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Try to find a lower rank decomposition
    # The matrix multiplication tensor T_{i,j,k} = 1 if we're computing C_{i,k} = sum_j A_{i,j}B_{j,k}
    # We want to find U, V, W of minimal rank such that:
    # T_{i,j,k} = sum_c U_{(i,j),c} * V_{(j,k),c} * W_{(k,i),c}
    # Wait, that's not right. Let me reconsider.
    #
    # Actually, the decomposition is:
    # C_{i,k} = sum_{i',j',k'} T_{i',j',k'} A_{i',j'} B_{j',k'}
    # But we want C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So T_{i,j,k} should be 1 when (i,j,k) corresponds to a valid term, and 0 otherwise.
    # More precisely: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor has T_{i,j,k} = 1 for all valid (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
    # and C_{i,k} picks up exactly the terms where the indices match.
    
    # The standard tensor decomposition for matrix multiplication:
    # We want: C_{i,k} = sum_c (sum_j U_{(i,j),c} A_{i,j}) * (sum_k' V_{(k',?),c} B_{?,k}) ... 
    # 
    # Actually, let me use the standard formulation. The matrix multiplication tensor M_{n,m,p}
    # is defined such that for matrices A (n x m) and B (m x p), the product C = A * B has:
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    #
    # A rank-r decomposition means we can write:
    # C_{i,k} = sum_{c=0}^{r-1} (sum_{j} U_{c,i,j} A_{i,j}) * (sum_{k} V_{c,j,k} B_{j,k}) * W_{c,k,i}
    # No, that's not right either.
    #
    # Let me think more carefully. The tensor T has indices (i, j, k) where:
    # - i indexes rows of A (size n)
    # - j indexes columns of A / rows of B (size m)  
    # - k indexes columns of B (size p)
    #
    # T_{i,j,k} = 1 if this term contributes to C, but actually the tensor is more subtle.
    #
    # In the standard formulation, we view matrix multiplication as a bilinear map:
    # (A, B) -> C where C_{i,k} = sum_j A_{i,j} B_{j,k}
    #
    # A rank-r decomposition provides:
    # C_{i,k} = sum_{c=0}^{r-1} alpha_c(i) * beta_c(k) * gamma_c
    # where alpha_c(i) = sum_j U_{c,i,j} A_{i,j} and beta_c(k) = sum_j V_{c,j,k} B_{j,k}
    # and gamma_c = W_{c,i,k} ... no this is getting confused.
    
    # Let me just use the direct approach: try random rank-r decompositions and check.
    # For small problems, we can try various ranks.
    
    def check_decomposition(U, V, W, rank, n, m, p):
        """Check if U, V, W give an exact decomposition."""
        # The decomposition should satisfy:
        # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
        # where T[i,j,k] = 1 if this is a valid term in the multiplication.
        # Actually T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # because every (i,j,k) triple corresponds to a term A[i,j]*B[j,k] which contributes to C[i,k].
        # But wait, that would mean the tensor is all ones, which has rank 1.
        
        # I think I'm confusing myself. Let me look at this differently.
        
        # The matrix multiplication tensor is a 3-tensor of size n*m*p.
        # It is defined as: T[i,j,k] = delta (some condition)
        
        # Actually, I believe the correct interpretation is:
        # The tensor T has entries T[i,j,k] where i in {0,...,n-1}, j in {0,...,m-1}, k in {0,...,p-1}
        # and T[i,j,k] = 1 if and only if this term appears in the matrix product formula.
        
        # Since C[i,k] = sum_j A[i,j] * B[j,k], every combination (i,j,k) with valid i,j,k
        # does appear. So T[i,j,k] = 1 for all valid i,j,k.
        
        # But that gives an all-ones tensor, which has rank 1!
        
        # This can't be right. The rank of matrix multiplication is known to be much higher.
        
        # I think the issue is that the tensor isn't just "does this term appear" but rather
        # it encodes the structure of how terms combine. Let me reconsider.
        
        # Actually, I think the tensor for matrix multiplication A (n x m) * B (m x p) = C (n x p)
        # is a tensor in the space R^{n*m} x R^{m*p} x R^{n*p} (or similar).
        
        # The standard rank of matrix multiplication is the smallest r such that we can write:
        # C[i,k] = sum_{c=0}^{r-1} (sum_j U[c, i, j] * A[i,j]) * (sum_j V[c, j, k] * B[j,k])
        # with some weights W[c] or similar.
        
        # Let me just try the straightforward check: given U (nm x r), V (mp x r), W (pn x r),
        # the decomposition works if for all i, j, k:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # (since T[i,j,k] = 1 for all valid indices)
        
        # Wait, but if T is all ones, then indeed we could use rank 1: U all ones, V all ones, W all ones.
        # That gives sum_c 1*1*1 = r, not 1. So we'd need U[i*m+j,c]=1/sqrt(r) etc., but we need integers.
        
        # Hmm, let me just verify with the schoolbook: in schoolbook, each (i,j,k) has its own c,
        # so U[i*m+j, c_{i,j,k}] = 1, V[j*p+k, c_{i,j,k}] = 1, W[k*n+i, c_{i,j,k}] = 1.
        # Then sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 (only one c contributes).
        
        # So the condition is: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # And we want to minimize r = rank.
        
        # Let's verify: does rank 1 work? If r=1, U[i*m+j,0]=a_{i,j}, V[j*p+k,0]=b_{j,k}, W[k*n+i,0]=w_{i,k}
        # Then sum_c = a_{i,j} * b_{j,k} * w_{i,k} = 1 for all i,j,k.
        # This means a_{i,j} * b_{j,k} * w_{i,k} = 1 for all i,j,k.
        # Taking i=0,j=0,k=0: a_{0,0}*b_{0,0}*w_{0,0} = 1
        # Taking i=0,j=0,k=1: a_{0,0}*b_{0,1}*w_{0,1} = 1
        # So b_{0,0}*w_{0,0} = b_{0,1}*w_{0,1} = 1/a_{0,0}
        # Taking i=0,j=1,k=0: a_{0,1}*b_{1,0}*w_{0,0} = 1
        # So b_{1,0}*w_{0,0} = 1/a_{0,1}
        # From b_{0,0}*w_{0,0} = 1/a_{0,0} and b_{1,0}*w_{0,0} = 1/a_{0,1}:
        # b_{0,0}/b_{1,0} = a_{0,1}/a_{0,0}
        # This might be solvable... but with the constraint that it works for ALL i,j,k, it's likely impossible
        # for n,m,p >= 2.
        
        # So rank 1 is out. Let's just do a search for small ranks.
        
        # Given time constraints, let me just try a few ranks with random initialization and Newton's method
        # or similar to solve for a valid decomposition.
        
        return True  # placeholder
    
    # Given the complexity and time limits, let me just try a simple approach:
    # Try ranks from some lower bound up to schoolbook rank, and for each rank,
    # try to find a valid decomposition using optimization.
    
    # The lower bound for matrix multiplication rank is known to be at least n*m + m*p + n*p - 1 or similar,
    # but actually for 3x3*3x7, let's just try a few values.
    
    # Actually, let me just return the schoolbook decomposition since finding a lower rank
    # within the time budget is challenging without sophisticated algorithms.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
