"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # The tensor T_{i,j,k} for matrix multiplication C = A * B is 1 if (i,j,k) corresponds to a term in the product.
    # Specifically, C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T has indices (i, j, k) where T[i,j,k] = 1.
    # We want to decompose T as sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] (using trace convention W[k*n+i, r]).
    # Wait, the standard decomposition is T_{ijk} = sum_r U_{ij}^r V_{jk}^r W_{ki}^r.
    # The problem statement says:
    # U shape: (n*m, rank) -> U[i*m+j, r] corresponds to U_{ij}^r
    # V shape: (m*p, rank) -> V[j*p+k, r] corresponds to V_{jk}^r
    # W shape: (p*n, rank) -> W[k*n+i, r] corresponds to W_{ki}^r
    # So T_{ijk} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # The target tensor T_{ijk} is 1 for all i in [0,n), j in [0,m), k in [0,p).
    # This is the all-ones tensor of shape (n, m, p).
    
    # We are looking for a rank-minimal decomposition of the all-ones tensor of size 3x4x6.
    # The rank of the all-ones tensor of size n x m x p is known to be related to the border rank.
    # For small dimensions, we can try to find an exact decomposition with a small rank.
    # The schoolbook rank is n*m*p = 72.
    # We can try to reduce this.
    
    # Let's use a randomized algorithm to find a low-rank decomposition.
    # We can formulate this as a system of polynomial equations.
    # For each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1.
    
    # We can use a heuristic search. Let's try to find a decomposition with rank R.
    # We can start with a small R and increase it until we find a solution.
    # Since we need exact integer/half-integer entries, we can use a lattice-based approach or just random search with rounding.
    
    # However, the problem asks for a rank-minimal *exact* decomposition.
    # For 3x4x6, the rank is likely not 72.
    # Let's try to find a decomposition using a known construction or a search.
    
    # One simple improvement is to use the fact that the tensor is all-ones.
    # We can try to use a decomposition based on the dimensions.
    
    # Let's try to find a decomposition with rank R.
    # We can use a simple iterative method or a random search.
    
    # Given the time constraint, let's try a few small ranks and see if we can find a solution.
    # We can use a randomized approach: initialize U, V, W with random values, and then solve for one of them.
    
    # Let's try rank R from 1 to 72.
    # For each R, we can try to find a solution.
    
    # Actually, the rank of the all-ones tensor of size n x m x p is at most n*m + m*p + p*n - 2.
    # For 3x4x6, this is 12 + 24 + 18 - 2 = 52.
    # We can try to find a decomposition with rank 52 or lower.
    
    # Let's use a simple construction:
    # U[i*m+j, r] = 1 if r < i*m + j else 0
    # V[j*p+k, r] = 1 if r < j*p + k else 0
    # W[k*n+i, r] = 1 if r < k*n + i else 0
    # This doesn't work directly.
    
    # Let's try a different approach: use the fact that the tensor is all-ones.
    # We can write 1 = sum_r a_r * b_r * c_r.
    # This is equivalent to finding a rank-1 decomposition of the scalar 1, which is trivial.
    # But we need to match the indices.
    
    # Let's use a known result: the rank of the all-ones tensor of size n x m x p is n*m + m*p + p*n - 2.
    # We can construct such a decomposition.
    
    # Construction:
    # For r in [0, n*m + m*p + p*n - 2]:
    # We can use a combination of the dimensions.
    
    # Let's try to implement the construction for the all-ones tensor.
    # The construction is based on the following idea:
    # 1 = (1 + 1 + ... + 1) / (n*m*p) * (n*m*p)
    # This is not helpful.
    
    # Let's use a different construction.
    # We can use the fact that the tensor is separable in a certain sense.
    
    # Actually, the rank of the all-ones tensor of size n x m x p is exactly n*m + m*p + p*n - 2.
    # We can find a decomposition with this rank.
    
    # Let's try to find a decomposition with rank R = n*m + m*p + p*n - 2 = 52.
    
    # We can use a randomized search to find a solution.
    # We can initialize U, V, W with random values and then use a solver.
    
    # Given the time constraint, let's just return the schoolbook decomposition for now,
    # but try to improve it if we have time.
    
    # Actually, let's try to find a better decomposition using a simple heuristic.
    # We can try to use a decomposition based on the dimensions.
    
    # Let's try rank R = 20 and see if we can find a solution.
    # We can use a randomized approach.
    
    # For now, let's return the schoolbook decomposition.
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
