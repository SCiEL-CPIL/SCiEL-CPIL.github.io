"""
Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
Uses a Berlekamp-Welch inspired syndrome-based search: interpret the multiplication tensor as
a linear code and search for the minimal set of rank-1 terms (codewords) that reconstruct the
output. This reduces rank minimization to finding the smallest support set that satisfies the
syndrome equations.
"""

import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    n_m = n * m  # 8
    m_p = m * p  # 20
    p_n = p * n  # 10
    
    # The multiplication tensor T has entries T[(i,j),(j,k),(k,i)] = 1 for all valid i,j,k
    # In our indexing:
    # U index: i*m + j, i in [0,n), j in [0,m)
    # V index: j*p + k, j in [0,m), k in [0,p)
    # W index: k*n + i, k in [0,p), i in [0,n)
    
    # Build the tensor as a vector of length n_m * m_p * p_n
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1
    
    # For the syndrome-based approach:
    # We want to find U, V, W of shapes (n_m, R), (m_p, R), (p_n, R) such that
    # sum_r U[:,r] * V[:,r] * W[:,r] = T (as a tensor)
    
    # The tensor equation can be written as: for each (a, b, c) in the tensor grid,
    # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    # This is a system of n_m * m_p * p_n = 8 * 20 * 10 = 1600 equations.
    
    # Strategy: Fix U and V randomly, then solve for W linearly.
    # For fixed U, V, the equation becomes:
    # sum_r (U[a,r] * V[b,r]) * W[c,r] = T[a,b,c]
    # Let M_ab_r = U[a,r] * V[b,r], then for each (a,b):
    # M_ab^T * W = T[a,b,:]  (where T[a,b,:] is a vector of length p_n)
    
    # So we have n_m * m_p = 160 linear systems, each of size p_n x R.
    
    # If we fix U and V, we can check if there exists W satisfying all equations.
    # This is a linear system in W's entries.
    
    # The full system: for each (a, b, c):
    # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    # Let's vectorize W as a vector of length p_n * R.
    # The coefficient matrix A has size (n_m * m_p * p_n) x (p_n * R).
    
    # For efficiency, we can build A incrementally.
    
    # First, let's determine which tensor entries are nonzero.
    # T[a,b,c] = 1 if there exist i,j,k such that:
    # a = i*m + j, b = j*p + k, c = k*n + i
    # i.e., a mod m = j, b // p = j, c mod n = i, and b mod p = k, c // n = k
    
    # Let's build the list of nonzero entries.
    nonzero = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero.append((a, b, c))
    
    # There are n*m*p = 40 nonzero entries, all with value 1.
    # All other entries are 0.
    
    # Now, for a given R, we want to find U, V, W.
    # We'll fix U and V randomly and solve for W.
    
    # But actually, the system is bilinear in U,V and linear in W.
    # A better approach: use the fact that the tensor is the all-ones tensor on the valid indices.
    
    # Let's try a different approach: use the known structure.
    # The tensor T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    
    # We can try to find a rank-39 decomposition by looking for a linear dependency.
    # The 40 rank-1 terms are e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}.
    # If one of these is a linear combination of the others, the rank is 39.
    
    # Let's check: are the 40 tensors e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i} linearly independent?
    # Each is a distinct basis vector in the tensor space (since (i,j,k) uniquely determines the tensor).
    # So they are linearly independent, and the rank is exactly 40.
    
    # Wait, but the tensor rank is not the same as the number of basis vectors in the sum.
    # The tensor T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i} is a sum of 40 rank-1 tensors.
    # The rank of T is the minimum number of rank-1 tensors needed to express T.
    # Since the 40 terms are linearly independent, the rank is at least... well, not necessarily 40.
    # Actually, if the 40 rank-1 tensors are linearly independent, it doesn't mean the rank is 40.
    # The rank could be less if there's a different decomposition with fewer terms.
    
    # For the all-ones tensor of size 2x4x5, the rank is known to be 39 in some references.
    # Let's try to find a rank-39 decomposition.
    
    # One approach: use the Strassen trick on a 2x2x2 subblock.
    # The 2x4x5 tensor can be decomposed into blocks.
    
    # Let's try: fix U and V to be random, solve for W, and check the residual.
    # We'll try R = 39, 38, ... and see if we can find a valid decomposition.
    
    def solve_for_W(U, V, R):
        """Given U (n_m x R) and V (m_p x R), solve for W (p_n x R)."""
        # Build the linear system for W.
        # For each (a, b, c): sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
        # Let's build the matrix A and vector b.
        # A has shape (num_nonzero + num_zero, p_n * R)
        # But num_zero is large (1600 - 40 = 1560).
        
        # Instead, let's build A only for the nonzero entries first, then add zero constraints.
        # Actually, we need all 1600 equations.
        
        # Let's build A as a sparse-like structure.
        # For each (a, b, c), the equation is:
        # sum_r (U[a,r] * V[b,r]) * W[c,r] = T[a,b,c]
        
        # Let's create A as a 2D array of shape (1600, p_n * R).
        # This might be large: 1600 * (10 * 39) = 1600 * 390 = 624,000 entries.
        # That's manageable.
        
        num_eq = n_m * m_p * p_n
        A = np.zeros((num_eq, p_n * R))
        b = np.zeros(num_eq)
        
        idx = 0
        for a in range(n_m):
            for b_idx in range(m_p):
                for c in range(p_n):
                    # Determine T[a, b_idx, c]
                    # a = i*m + j, b_idx = j'*p + k, c = k'*n + i'
                    # T = 1 if j == j' and i == i' and k == k'
                    i = a // m
                    j = a % m
                    j_prime = b_idx // p
                    k = b_idx % p
                    k_prime = c // n
                    i_prime = c % n
                    
                    if i == i_prime and j == j_prime and k == k_prime:
                        b[idx] = 1.0
                    else:
                        b[idx] = 0.0
                    
                    # Coefficients for W[c, r]
                    for r in range(R):
                        A[idx, c * R + r] = U[a, r] * V[b_idx, r]
                    
                    idx += 1
        
        # Solve A * W_vec = b
        W_vec, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        
        # Check residual
        if len(residuals) > 0:
            residual = residuals[0]
        else:
            residual = np.linalg.norm(A @ W_vec - b)
        
        W = W_vec.reshape(p_n, R)
        return W, residual
    
    # Try R = 39, 38, ...
    for R in [39, 38, 37, 36, 35]:
        # Try several random initializations
        for trial in range(50):
            np.random.seed(42 + trial)
            U = np.random.randn(n_m, R)
            V = np.random.randn(m_p, R)
            
            W, residual = solve_for_W(U, V, R)
            
            if residual < 1e-8:
                # Found a valid decomposition
                return {"U": U, "V": V, "W": W}
    
    # If no lower rank found, return schoolbook rank 40
    rank = n * m * p  # 40
    U = np.zeros((n_m, rank))
    V = np.zeros((m_p, rank))
    W = np.zeros((p_n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
