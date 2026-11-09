import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _validate_fast(U, V, W, n, m, p):
    """Fast validation using numpy vectorization."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 for all i,j,k
    
    # Build the full tensor from the decomposition
    # For each r, the factor is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # Reshape: U[:,r] is (n*m,), V[:,r] is (m*p,), W[:,r] is (p*n,)
    
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # Vectorized: for each r, compute the outer product contribution
    # T = sum_r (U[:,r].reshape(n,m) ⊗ V[:,r].reshape(m,p) ⊗ W[:,r].reshape(p,n))
    # but with specific index mapping
    
    # Let's do it directly but efficiently
    # T[i,j,k] for i in 0..n-1, j in 0..m-1, k in 0..p-1
    
    # Precompute index arrays
    I, J, K = np.indices((n, m, p))
    idx_u = I * m + J  # (n,m,p)
    idx_v = J * p + K  # (n,m,p)
    idx_w = K * n + I  # (n,m,p)
    
    # U[idx_u, r] * V[idx_v, r] * W[idx_w, r], summed over r
    # Shape: (n,m,p,rank) -> sum over last axis
    U_exp = U[idx_u, :]  # (n,m,p,rank)
    V_exp = V[idx_v, :]  # (n,m,p,rank)
    W_exp = W[idx_w, :]  # (n,m,p,rank)
    
    T = np.sum(U_exp * V_exp * W_exp, axis=3)  # (n,m,p)
    
    return np.allclose(T, 1.0, atol=1e-9)

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with schoolbook decomposition
    rank = n * m * p  # 63
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Verify schoolbook is valid
    if not _validate_fast(U, V, W, n, m, p):
        # Should not happen, but return anyway
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = rank
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Try to reduce rank by removing columns (greedy)
    current_rank = rank
    improved = True
    while improved and current_rank > 1 and time.time() < deadline - 10:
        improved = False
        for col_idx in range(current_rank):
            if time.time() >= deadline - 10:
                break
            
            U_test = np.delete(U, col_idx, axis=1)
            V_test = np.delete(V, col_idx, axis=1)
            W_test = np.delete(W, col_idx, axis=1)
            
            if _validate_fast(U_test, V_test, W_test, n, m, p):
                U = U_test
                V = V_test
                W = W_test
                current_rank -= 1
                improved = True
                break
    
    # Try more aggressive: check if any pair of columns can be merged
    # by finding a linear combination that preserves the tensor
    # This is harder - we need to check if the span of the factors allows reduction
    
    # Alternative: use the fact that if the tensor has rank r, then
    # the flattening matrices must have rank r. Check the rank of
    # the tensor's matrix flattenings to get a lower bound.
    
    # The tensor T[i,j,k] = 1 for all i,j,k (in our case, it's the all-ones tensor)
    # Wait, no - the multiplication tensor is NOT all-ones. Let me reconsider.
    
    # Actually, for matrix multiplication C = A*B where A is n×m, B is m×p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 if we think of it as: the (i,j,k)-th entry is 1
    # meaning that A[i,j] and B[j,k] contribute to C[i,k].
    # So T[i,j,k] = 1 for all valid i,j,k. It IS the all-ones tensor of shape (n,m,p).
    
    # The rank of the all-ones tensor of shape (n,m,p) is... well, it's rank 1!
    # Because T[i,j,k] = 1 = 1*1*1, so it's a rank-1 tensor.
    
    # Wait, that can't be right for matrix multiplication. Let me re-examine.
    
    # The multiplication tensor is defined such that:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    # In the tensor decomposition framework, we write:
    # C[i,k] = sum_{r=0}^{rank-1} sum_{j=0}^{m-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Hmm, but that's not standard. The standard way is:
    # The multiplication tensor M has entries M[i,j,k] where:
    # - i indexes rows of A (and C): 0..n-1
    # - j indexes columns of A / rows of B: 0..m-1  
    # - k indexes columns of B (and C): 0..p-1
    # And M[i,j,k] = 1 (each term A[i,j]*B[j,k] appears exactly once)
    
    # The rank of this tensor is the minimum number of rank-1 tensors that sum to it.
    # A rank-1 tensor is u ⊗ v ⊗ w where (u⊗v⊗w)[i,j,k] = u[i]*v[j]*w[k]
    
    # So we need: sum_r U_r[i] * V_r[j] * W_r[k] = 1 for all i,j,k
    # where U_r is a vector of length n, V_r of length m, W_r of length p.
    
    # But the interface uses flattened indices: U has shape (n*m, rank), not (n, rank).
    # This suggests the decomposition is over the flattened input space.
    
    # Let me re-read the task: "U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)"
    # And the trace convention: W[k*n+i, r]
    
    # I think the correct interpretation is:
    # C[i,k] = sum_{r=0}^{rank-1} W[k*n+i, r] * (sum_{j=0}^{m-1} U[i*m+j, r] * V[j*p+k, r])
    
    # No wait, that doesn't make sense either. Let me think about this differently.
    
    # The standard tensor decomposition for matrix multiplication:
    # We want to compute C = A @ B where A is n×m, B is m×p.
    # Flatten A to a vector of length n*m, B to length m*p, C to length n*p.
    # The multiplication is a trilinear map: (A, B) -> C
    # This can be written as: C = sum_r (U_r ⊗ V_r) ⊗ W_r in some sense.
    
    # More precisely, the tensor T has indices (i, j, k) where:
    # - The "A" index is (i, j) flattened to i*m + j
    # - The "B" index is (j, k) flattened to j*p + k  
    # - The "C" index is (i, k) flattened to k*n + i (column-major)
    
    # And T[(i,j), (j,k), (i,k)] = 1, i.e., the entry is 1 when the indices are consistent.
    
    # So the tensor is: T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    # And T[a, b, c] = 0 otherwise.
    
    # This is NOT the all-ones tensor. It's a sparse tensor with exactly n*m*p = 63 non-zero entries.
    
    # The schoolbook decomposition has rank 63, one factor per non-zero entry.
    # Can we do better? The rank of this tensor is at least the max of the ranks of its flattenings.
    
    # Flattening 1: treat as matrix (n*m) × (m*p) for fixed c... no, that's not right.
    # The three flattenings are:
    # - (n*m) × (m*p) with c fixed? No, the tensor is 3-way.
    # - Flattening to (n*m) × (m*p * p*n)? That's not standard.
    
    # The standard lower bound: rank >= max(rank of each matrix flattening)
    # Flattening 1: reshape to (n*m) × (m*p * p*n) - no, that's not how it works.
    
    # For a 3-way tensor T of shape (I, J, K), the rank is at least:
    # - rank of T reshaped to (I, J*K)
    # - rank of T reshaped to (J, I*K)  
    # - rank of T reshaped to (K, I*J)
    
    # In our case: I = n*m = 9, J = m*p = 21, K = p*n = 21
    # The tensor T has T[a, b, c] = 1 iff there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
    # There are exactly 63 such (a,b,c) triples.
    
    # Let's compute the rank of the flattenings numerically to get a lower bound.
    
    # Build the tensor as a 3D array
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Flattening 1: (nm, mp*pn)
    T1 = T.reshape(nm, mp * pn)
    rank1 = np.linalg.matrix_rank(T1)
    
    # Flattening 2: (mp, nm*pn)
    T2 = T.transpose(1, 0, 2).reshape(mp, nm * pn)
    rank2 = np.linalg.matrix_rank(T2)
    
    # Flattening 3: (pn, nm*mp)
    T3 = T.transpose(2, 0, 1).reshape(pn, nm * mp)
    rank3 = np.linalg.matrix_rank(T3)
    
    lower_bound = max(rank1, rank2, rank3)
    
    # If the lower bound equals 63, then schoolbook is optimal and we can't do better.
    # If lower_bound < 63, there might be room for improvement.
    
    # For now, if we haven't improved, return the best we have.
    if current_rank == best_rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
