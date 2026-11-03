"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]
import time
import numpy as np

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify exact decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
    r = U.shape[1]
    # Build the tensor equation efficiently
    # For each (i,j,k): sum over c of U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We can do this by iterating over c and accumulating
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Reshape for efficiency
    # For each c, we need:
    #   U_c reshaped to (n, m), V_c reshaped to (m, p), W_c reshaped to (p, n)
    #   The contribution to T[i,j,k] from term c is:
    #   U_c[i,j] * V_c[j,k] * W_c[k,i]
    
    # We can compute this as: for each c, compute (U_c * V_c) which is n x p,
    # but that's not quite right because we also multiply by W_c[k,i].
    
    # Actually, let's just do a direct check for small sizes.
    # n*m*p = 60 entries, r is at most 60, so 60*60 = 3600 operations.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_constructed_decompositions(n, m, p, deadline):
    """Try several structured decompositions that may achieve rank < n*m*p."""
    
    # Strategy 1: Exploit the structure of T[i,j,k] = 1.
    # Key insight: T[i,j,k] = 1 for ALL (i,j,k).
    # This means for fixed j, the 2D slice T[:,j,:] is a 2x6 all-ones matrix.
    # The all-ones matrix J_{2,6} can be written as u * v^T where u=(1,1)^T, v=(1,1,1,1,1,1)^T.
    # But in our tensor decomposition, the factors are coupled across all three indices.
    
    # Let's think about it differently.
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Consider the case where we use terms that are "separable" in a useful way.
    # 
    # Idea: Use terms where U, V, W are products of 1D sequences.
    # Let U[i*m+j, c] = a_c[i] * b_c[j]
    # Let V[j*p+k, c] = b_c[j] * d_c[k]  (sharing b_c[j])
    # Let W[k*n+i, c] = d_c[k] * a_c[i]  (sharing d_c[k] and a_c[i])
    # Then the product is a_c[i]^2 * b_c[j]^2 * d_c[k]^2
    # Sum: sum_c a_c[i]^2 * b_c[j]^2 * d_c[k]^2 = 1
    # This is a rank-r decomposition of the all-ones tensor using "separable" terms.
    # The all-ones tensor of shape (2,5,6) has tensor rank... 
    # Actually, the all-ones tensor 1_{2,5,6} can be written as u ⊗ v ⊗ w where
    # u=(1,1), v=(1,1,1,1,1), w=(1,1,1,1,1,1), which is rank 1!
    # But wait, our decomposition is different from a standard tensor decomposition.
    # In a standard tensor decomposition, T[i,j,k] = sum_c u_c[i] * v_c[j] * w_c[k].
    # Here, T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c].
    # The indices are coupled: U depends on (i,j), V on (j,k), W on (k,i).
    # This is NOT a standard tensor decomposition. It's a "matrix multiplication tensor"
    # decomposition with the specific index coupling.
    
    # So the all-ones tensor in this coupled sense is harder.
    
    # Let's try a different approach.
    # For the standard matrix multiplication tensor, the rank is known to be 
    # at most n*m*p (schoolbook) and at least max(n*m, m*p, p*n).
    # For n=2, m=5, p=6: lower bound = max(10, 30, 12) = 30.
    
    # Actually, the matrix multiplication tensor [n,m,p] has a well-studied rank.
    # The standard matrix multiplication tensor M_{n,m,p} has entries:
    # M_{n,m,p}[(i,j),(j,k),(k,i)] = 1 for all valid indices.
    # This is exactly our tensor! T[i,j,k] = 1 corresponds to M_{n,m,p}.
    # The rank of M_{2,5,6} is what we're looking for.
    
    # Known results:
    # - M_{1,1,1} has rank 1
    # - M_{2,2,2} has rank 7 (Strassen)
    # - For M_{n,m,p}, the rank is at most n*m*p and at least (n*m + m*p + p*n - 2)
    #   (this is a lower bound from algebraic geometry)
    # For n=2, m=5, p=6: lower bound = 10 + 30 + 12 - 2 = 50.
    
    # Hmm, that's close to 60. Let me check this lower bound more carefully.
    # The lower bound from "slice rank" or "partition rank" might be different.
    
    # Actually, for the matrix multiplication tensor, a well-known lower bound is:
    # rank(M_{n,m,p}) >= (n*m*p) / (something)
    # But more precisely, using the "flattening" or "slice" argument:
    # The tensor can be seen as a bilinear map, and the rank is related to
    # the rank of associated matrices.
    
    # For now, let's try to construct a decomposition with rank around 30-50.
    
    # Strategy: Use the fact that for n=2, we can potentially share structure
    # between the two "i" values.
    
    # Let's try a specific construction:
    # For each j in {0,...,m-1} and each k in {0,...,p-1}, we need to cover
    # both i=0 and i=1.
    # 
    # Term for (j, k, i=0):
    #   U[0*m+j, c] = 1, U[1*m+j', c] = 0 for all j'
    #   V[j*p+k, c] = 1, V[j',p+k', c] = 0 for all other (j',k')
    #   W[k*n+0, c] = 1, W[k',n+i', c] = 0 for all other (k',i')
    # 
    # This is the schoolbook term. Rank = n*m*p = 60.
    
    # To reduce rank, we need to find linear combinations of terms that
    # cancel or combine to reduce the total number of terms.
    
    # One approach: For fixed (j,k), can we use ONE term to cover both i=0 and i=1?
    # We need: U[0*m+j,c] * V[j*p+k,c] * W[k*n+0,c] = 1
    #          U[1*m+j,c] * V[j*p+k,c] * W[k*n+1,c] = 1
    # If we set V[j*p+k,c] = 1 (and 0 elsewhere), then we need:
    #   U[0*m+j,c] * W[k*n+0,c] = 1
    #   U[1*m+j,c] * W[k*n+1,c] = 1
    # We can set U[0*m+j,c] = 1, W[k*n+0,c] = 1, U[1*m+j,c] = 1, W[k*n+1,c] = 1.
    # But then for other (i',j',k') combinations, we need to make sure the
    # contributions are 0.
    # 
    # For a fixed c with the above setting, the contribution to T[i',j',k'] is:
    #   U[i'*m+j',c] * V[j'*p+k',c] * W[k'*n+i',c]
    # This is nonzero only if j'=j AND k'=k (from V) AND i' is either 0 or 1 (from U,W).
    # Specifically:
    #   If j'=j and k'=k: contribution = U[i'*m+j,c] * 1 * W[k*n+i',c]
    #     For i'=0: U[0*m+j,c]*W[k*n+0,c] = 1*1 = 1
    #     For i'=1: U[1*m+j,c]*W[k*n+1,c] = 1*1 = 1
    #   If j'!=j or k'!=k: contribution = 0
    # 
    # So this single term c covers BOTH i=0 and i=1 for the fixed (j,k)!
    # The total rank would be m*p = 30!
    
    # Let's verify this more carefully.
    # For each (j,k) pair, create one term c with:
    #   U[i*m+j, c] = 1 for i in {0,1} (i.e., U[0*m+j,c]=1 and U[1*m+j,c]=1)
    #   U[i*m+j', c] = 0 for j' != j
    #   V[j*p+k, c] = 1
    #   V[j'*p+k', c] = 0 for (j',k') != (j,k)
    #   W[k*n+i, c] = 1 for i in {0,1} (i.e., W[k*n+0,c]=1 and W[k*n+1,c]=1)
    #   W[k'*n+i', c] = 0 for (k',i') != (k,0) and (k',i') != (k,1)
    
    # Now check: for any (i,j,k), the sum over all c is:
    #   Only the term c corresponding to (j,k) contributes.
    #   Contribution = U[i*m+j, c_{j,k}] * V[j*p+k, c_{j,k}] * W[k*n+i, c_{j,k}]
    #               = 1 * 1 * 1 = 1.
    # 
    # For any (i,j',k') with (j',k') != (j,k):
    #   The term c_{j',k'} has V[j*p+k', c] = 0 (since j' != j or k' != k),
    #   so its contribution is 0.
    # 
    # This works! The rank is m*p = 30.
    
    # Wait, but I need to double-check the W index.
    # W has shape (p*n, rank), and W[k*n+i, c].
    # For the term c_{j,k}, we set W[k*n+0, c] = 1 and W[k*n+1, c] = 1.
    # All other entries of W[:, c] are 0.
    
    # Let's construct this and verify.
    
    r = m * p  # 30
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    c = 0
    for j in range(m):
        for k in range(p):
            # Set U: for i in range(n), U[i*m+j, c] = 1
            for i in range(n):
                U[i * m + j, c] = 1.0
            # Set V: V[j*p+k, c] = 1
            V[j * p + k, c] = 1.0
            # Set W: for i in range(n), W[k*n+i, c] = 1
            for i in range(n):
                W[k * n + i, c] = 1.0
            c += 1
    
    if _verify(U, V, W, n, m, p):
        return U, V, W, r
    
    # If the above didn't work (shouldn't happen), try other constructions.
    
    # Strategy 2: Try rank = n*p = 12
    # For each (i,k), create one term:
    #   U[i*m+j, c] for all j... this doesn't work as simply.
    
    # Let's try: for each (i,k), term c with:
    #   U[i*m+j, c] = 1 for all j (so U[i*m:, c] = ones(m))
    #   V[j*p+k, c] = 1 for all j (so V[:,k] entries... wait, V[j*p+k,c] for all j)
    #   W[k*n+i, c] = 1
    # 
    # Check: for (i',j',k'), contribution from term c_{i,k} is:
    #   U[i'*m+j', c_{i,k}] * V[j'*p+k', c_{i,k}] * W[k'*n+i', c_{i,k}]
    #   = (1 if i'=i else 0) * (1 if k'=k else 0) * (1 if i'=i and k'=k else 0)
    #   Wait, W[k'*n+i', c] is 1 only if k'=k and i'=i.
    #   So the product is 1 if i'=i and k'=k, else 0.
    #   But V[j'*p+k', c] = 1 for all j' when k'=k.
    #   So for (i',j',k') with i'=i and k'=k: contribution = 1 * 1 * 1 = 1.
    #   For other (i',j',k'): contribution = 0.
    # 
    # Sum over all c: for (i',j',k'), only the term c_{i',k'} contributes, giving 1.
    # This also works! Rank = n*p = 12.
    
    r2 = n * p  # 12
    U2 = np.zeros((n * m, r2))
    V2 = np.zeros((m * p, r2))
    W2 = np.zeros((p * n, r2))
    
    c = 0
    for i in range(n):
        for k in range(p):
            # Set U: U[i*m+j, c] = 1 for all j
            for j in range(m):
                U2[i * m + j, c] = 1.0
            # Set V: V[j*p+k, c] = 1 for all j
            for j in range(m):
                V2[j * p + k, c] = 1.0
            # Set W: W[k*n+i, c] = 1
            W2[k * n + i, c] = 1.0
            c += 1
    
    if _verify(U2, V2, W2, n, m, p):
        return U2, V2, W2, r2
    
    # Strategy 3: Try rank = n*m = 10
    # For each (i,j), term c with:
    #   U[i*m+j, c] = 1
    #   V[j*p+k, c] = 1 for all k
    #   W[k*n+i, c] = 1 for all k
    # 
    # Check: for (i',j',k'), contribution from c_{i,j} is:
    #   U[i'*m+j', c] * V[j'*p+k', c] * W[k'*n+i', c]
    #   = (1 if i'=i and j'=j else 0) * (1 if j'=j else 0) * (1 if i'=i else 0)
    #   = 1 if i'=i and j'=j, else 0.
    # 
    # Sum: for (i',j',k'), only c_{i',j'} contributes, giving 1.
    # Rank = n*m = 10.
    
    r3 = n * m  # 10
    U3 = np.zeros((n * m, r3))
    V3 = np.zeros((m * p, r3))
    W3 = np.zeros((p * n, r3))
    
    c = 0
    for i in range(n):
        for j in range(m):
            U3[i * m + j, c] = 1.0
            for k in range(p):
                V3[j * p + k, c] = 1.0
            for k in range(p):
                W3[k * n + i, c] = 1.0
            c += 1
    
    if _verify(U3, V3, W3, n, m, p):
        return U3, V3, W3, r3
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Try constructed decompositions
    result = _try_constructed_decompositions(n, m, p, deadline)
    if result is not None:
        U, V, W, r = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
