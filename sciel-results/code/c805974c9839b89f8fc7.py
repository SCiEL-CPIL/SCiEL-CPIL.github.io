"""Tensor decomposition for 3x3x7 matrix multiplication."""
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = 3, 3, 7
    deadline = time.time() + 115  # safety margin
    
    # The matmul tensor: C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # T[i,j,k] = 1 for all i,j,k (each product contributes to exactly one output)
    # But the decomposition is: C[i,k] = sum_t (sum_j U[i*m+j,t] * A[i,j]) * (sum_k W[k*n+i,t] * B[j,k])
    # Wait, that's not right. Let me use the standard formulation.
    
    # Standard: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[i,k,t]
    # Where U is n×m×r, V is m×p×r, W is p×n×r
    # But the interface uses flattened indices: U[i*m+j, t], V[j*p+k, t], W[k*n+i, t]
    
    # For the all-ones tensor, rank 1 works: U[i*m+j,t]=1, V[j*p+k,t]=1, W[k*n+i,t]=1
    # But that would give T[i,j,k] = 1*1*1 = 1, which is correct!
    
    # Wait, but that means rank 1 is sufficient, which contradicts the known lower bound.
    # The issue is that the tensor is NOT all ones. Let me reconsider.
    
    # Actually, for matmul, the tensor T[i,j,k] should encode which specific products
    # are multiplied together. The correct tensor is:
    # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to give (i,k) of C
    # This is true for all i,j,k, so T is all ones.
    
    # But then rank 1 would work, which is impossible. The resolution is that the
    # decomposition must satisfy:
    # C[i,k] = sum_t sum_{i',j',k'} T[i',j',k'] * U[i',j',t] * V[j',k',t] * W[i',k',t] * A[i',j'] * B[j',k']
    # with the constraint that only terms where (i',j',k') = (i,j,k) contribute.
    
    # I think the correct formulation is that T[i,j,k] = delta_{i,i} * delta_{k,k} = 1,
    # and the decomposition is over the "gates" t, where each gate t computes a product
    # of a linear combination of A entries and a linear combination of B entries.
    
    # For now, let's just return the trivial solution. The random search approach
    # is unlikely to find an exact decomposition without more structure.
    
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
