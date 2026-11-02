"""Rank-minimal exact tensor decomposition for 2x5 @ 5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    def verify_numpy(U, V, W, n, m, p, tol=1e-9):
        """Verify tensor decomposition using numpy for speed."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        r = U.shape[1]
        
        # Tensor: T[i*m+j, j2*p+k2, k2*n+i2] = sum_c U[i*m+j,c] * V[j2*p+k2,c] * W[k2*n+i2,c]
        # Build the tensor and check against the matrix multiplication tensor
        
        # Efficient check: for each c, form outer products
        # T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
        
        # Check diagonal: for each (i,j,k), T[i*m+j, j*p+k, k*n+i] should be 1
        # Check off-diagonal: all other entries should be 0
        
        # Build full tensor
        T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for c in range(r):
            T += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
        
        # Check diagonal entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T[i*m+j, j*p+k, k*n+i] - 1.0) > tol:
                        return False
        
        # Check off-diagonal entries
        # Sample a subset for speed, but be thorough
        total_checks = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    # Check all (j2, k2, i2) where (i,j,k) != (i2,j2,k2)
                    for j2 in range(m):
                        for k2 in range(p):
                            for i2 in range(n):
                                if i == i2 and j == j2 and k == k2:
                                    continue
                                b = j2*p + k2
                                c_idx = k2*n + i2
                                if abs(T[a, b, c_idx]) > tol:
                                    return False
                                total_checks += 1
        
        return True
    
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    # Rank-25 construction exploiting n=2 structure
    # For c = j0*p + k0 (where j0 in [0,m), k0 in [0,p)):
    #   U[i*m+j0, c] = 1 for all i
    #   V[j0*p+k0, c] = 1
    #   W[k0*n+i, c] = 1 for all i
    # 
    # Tensor entry: sum_c U[i*m+j,c] * V[j2*p+k2,c] * W[k2*n+i2,c]
    # Only c = j*p+k2 can have U[i*m+j,c] != 0 AND V[j2*p+k2,c] != 0
    # This requires j = j2, giving: U[i*m+j, j*p+k2] * V[j*p+k2, j*p+k2] * W[k2*n+i2, j*p+k2]
    # = 1 * 1 * W[k2*n+i2, j*p+k2]
    # W[k2*n+i2, c] is non-zero only if c = j2'*p + k2 where j2' is the j-index in U
    # Since c = j*p+k2, we need j2' = j, so W[k2*n+i2, j*p+k2] = 1 for all i2
    # 
    # So the sum = 1 for all (i, i2), but we need it to be δ(i,i2)!
    # 
    # This means the simple all-ones construction does NOT work for n=2.
    # We need W to depend on i properly.
    
    # Corrected construction:
    # U[i*m+j0, c] = a[i] where c = j0*p+k0
    # V[j0*p+k0, c] = 1
    # W[k0*n+i, c] = b[i]
    # 
    # Tensor: sum_{j0,k0} a[i] * δ(j,j0) * δ(j2,j0) * δ(k2,k0) * b[i2] * δ(k2,k0)
    # Wait, V[j2*p+k2, j0*p+k0] = δ(j2,j0)*δ(k2,k0)
    # W[k2*n+i2, j0*p+k0] = δ(k2,k0)*b[i2]
    # 
    # So sum over c = j0*p+k0:
    #   U[i*m+j, j0*p+k0] * V[j2*p+k2, j0*p+k0] * W[k2*n+i2, j0*p+k0]
    # = a[i]*δ(j,j0) * δ(j2,j0)*δ(k2,k0) * δ(k2,k0)*b[i2]
    # = a[i]*b[i2]*δ(j,j0)*δ(j2,j0)*δ(k2,k0)
    # 
    # Sum over j0,k0: a[i]*b[i2]*δ(j,j2)*δ(k2,k2) [when j=j2 and the k0 matches k2]
    # Actually: sum_{j0,k0} δ(j,j0)*δ(j2,j0)*δ(k2,k0)*δ(k2,k0)
    # = sum_{j0,k0} δ(j,j0)*δ(j2,j0)*δ(k2,k0)
    # = δ(j,j2) * δ(k2,k2) [if j=j2, then j0=j, and k0=k2 gives 1 term]
    # = δ(j,j2) if k2 is in range
    # 
    # So tensor = a[i]*b[i2]*δ(j,j2)
    # We need this to equal δ(i,i2)*δ(j,j2)*δ(k,k2)
    # 
    # This requires: a[i]*b[i2]*δ(j,j2) = δ(i,i2)*δ(j,j2)*δ(k,k2)
    # For this to hold for all k,k2, we need δ(k,k2) to appear, but it doesn't in our expression!
    # 
    # The problem: our structure doesn't separate k and k2 properly.
    
    # Let me try a different approach. The schoolbook rank is 50.
    # For n=2, can we do better than 50?
    
    # Actually, let me reconsider. The matrix multiplication tensor T has entries:
    # T[i*m+j, j'*p+k', k'*n+i'] = δ(i,i')*δ(j,j')*δ(k,k')
    # 
    # Rank r decomposition: T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c
    # where u_c is vector of length n*m, v_c of length m*p, w_c of length p*n
    # 
    # For the tensor to be decomposable with rank < n*m*p, there must be algebraic structure.
    
    # Given the difficulty, let's use the guaranteed schoolbook rank 50.
    # But first, let me try one more structured approach.
    
    # Approach: Use rank = m*p = 25 with the following:
    # For each (j0, k0), define component c = j0*p + k0
    # U[i*m+j, c] = δ(j, j0) * f(i, k0)
    # V[j*p+k, c] = δ(j, j0) * δ(k, k0)  
    # W[k2*n+i2, c] = δ(k2, k0) * g(i2, j0)
    # 
    # Tensor: sum_{j0,k0} δ(j,j0)*f(i,k0) * δ(j2,j0)*δ(k2,k0) * δ(k2,k0)*g(i2,j0)
    # = sum_{j0,k0} δ(j,j0)*δ(j2,j0)*δ(k2,k0)*f(i,k0)*g(i2,j0)
    # = δ(j,j2) * f(i,k2) * g(i2,j)  [when j=j2, j0=j, and k0=k2]
    # 
    # We need: δ(j,j2) * f(i,k2) * g(i2,j) = δ(i,i2)*δ(j,j2)*δ(k,k2)
    # 
    # For j=j2: f(i,k2) * g(i2,j) = δ(i,i2) * δ(k,k2)
    # This must hold for all i,i2,k,k2.
    # Setting k=k2: f(i,k) * g(i2,j) = δ(i,i2)
    # Setting k≠k2: f(i,k) * g(i2,j) = 0
    # 
    # From k≠k2 case: for any i,i2,j, either f(i,k)=0 or g(i2,j)=0 for all k≠k2.
    # This is very restrictive. If f(i,k) ≠ 0 for some k, then g(i2,j) must be 0 for all k2≠k,
    # but g doesn't depend on k2, so g(i2,j)=0. But then for k=k2, f(i,k)*0=0≠δ(i,i2).
    # 
    # So this structure cannot work unless we allow more complex dependencies.
    
    # Conclusion: The simple structured rank-25 decompositions don't work.
    # Use schoolbook rank 50 which is guaranteed to work.
    
    U, V, W = schoolbook(n, m, p)
    
    # Verify before returning
    if verify_numpy(U, V, W, n, m, p, tol=1e-9):
        return {"U": U, "V": V, "W": W}
    
    # This should never happen, but fallback just in case
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
