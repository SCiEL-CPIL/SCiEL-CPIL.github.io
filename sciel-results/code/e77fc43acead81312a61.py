"""
Rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.

The multiplication tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1 for all
valid (i,j,k). Since all entries are 1, we can exploit algebraic identities
to find rank reductions below the schoolbook rank of 63.

Key insight: The all-ones structure allows using roots of unity or other
algebraic identities to combine multiple (i,j,k) terms into fewer rank-1 terms.
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 7]
START_TIME = time.time()
TIME_LIMIT = 130  # safety margin before 150s kill


def _elapsed():
    return time.time() - START_TIME


def _deadline_exceeded():
    return _elapsed() > TIME_LIMIT


def _schoolbook_decomposition(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
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
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p):
    """Check if U, V, W form a valid decomposition of the multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                decomp_val = float(np.sum(
                    U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :]
                ))
                if abs(decomp_val - 1.0) > 1e-9:
                    return False
    
    # Check that non-multiplication entries are zero
    for r in range(rank):
        u_col = U_arr[:, r]
        v_col = V_arr[:, r]
        w_col = W_arr[:, r]
        nz_u = np.where(u_col != 0)[0]
        nz_v = np.where(v_col != 0)[0]
        nz_w = np.where(w_col != 0)[0]
        for a in nz_u:
            for b in nz_v:
                for c in nz_w:
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if not (j == j2 and k == k2 and i == i2):
                        if abs(u_col[a] * v_col[b] * w_col[c]) > 1e-9:
                            return False
    return True


def _roots_of_unity_decomposition(n, m, p):
    """
    Use the identity: 1 = (1/n) * sum_{i=0}^{n-1} (1 + omega^i + omega^(2i))
    where omega = exp(2*pi*i/n) is a primitive n-th root of unity.
    
    For n=3: 1 = (1/3) * (1 + 1 + 1) = 1, but more usefully:
    We can write the all-ones tensor as a sum of rank-1 terms using
    structured patterns.
    
    Actually, for the all-ones tensor T[i,j,k]=1, we can use:
    T = (1/3) * sum_{a=0}^{2} u_a ⊗ v ⊗ w_a
    where u_a[i] = omega^(a*i), w_a[k] = omega^(-a*k), v[j] = 1
    
    But this gives complex entries. For real/rational entries, we need
    a different approach.
    
    Alternative: Use the fact that for 3x3x7, we can group by j and
    use rank-3 decompositions for each (j,k) slice.
    """
    n, m, p = 3, 3, 7
    
    # For each (j,k), we need to represent the vector [1,1,1] (over i)
    # as a sum of rank-1 terms. The simplest is rank 3 (schoolbook).
    # But we can potentially share structure across different (j,k) pairs.
    
    # Try a structured decomposition where we use the same U and W patterns
    # for all (j,k), with V providing the "selection"
    
    # Actually, let's try a different approach: use the identity
    # 1 = (1/3) * (1+1+1) but with structured cancellations
    
    # For now, let's try to find a rank-21 decomposition by using
    # rank-3 for each (j,k) pair but with shared structure
    
    # Better approach: Use the fact that T[i,j,k] = 1 for all i,j,k
    # This means T = u ⊗ v ⊗ w where u[i]=1, v[j]=1, w[k]=1
    # But that's only if we allow the "full" tensor, which has rank 1!
    # Wait, no - the tensor is indexed as T[i*m+j, j*p+k, k*n+i]
    # so the indices are coupled.
    
    # Let me reconsider. The tensor is:
    # T[a, b, c] = 1 if a=i*m+j, b=j'*p+k, c=k'*n+i' with j=j', k=k', i=i'
    #             = 0 otherwise
    
    # So for fixed (i,j,k), T[i*m+j, j*p+k, k*n+i] = 1
    # And T is zero elsewhere.
    
    # This is NOT an all-ones tensor in the (a,b,c) space!
    # It's a sparse tensor with exactly n*m*p = 63 non-zero entries.
    
    return None


def _try_rank_reduction(n, m, p, target_rank):
    """
    Try to find a rank-target_rank decomposition using algebraic methods.
    
    For the 3x3x7 case, we exploit the structure where the tensor has
    a block structure over j (the middle index).
    """
    if _deadline_exceeded():
        return None
    
    # Strategy: For each j, the sub-tensor T_j has entries T_j[i,k] = 1
    # at positions (i*m+j, j*p+k, k*n+i). This is a 3x7 matrix of 1s.
    
    # We can try to decompose each T_j with rank r_j, then combine.
    # If we can find a rank-3 decomposition for each 3x7 "slice", we get
    # total rank 3*3 = 9... but that's not quite right because the
    # decompositions must be consistent.
    
    # Let's try a direct approach: assume a decomposition of the form:
    # U[i*m+j, c] = u_j[i, c]  (depends on j and i)
    # V[j*p+k, c] = v_j[k, c]  (depends on j and k)  
    # W[k*n+i, c] = w_k[i, c]  (depends on k and i)
    
    # Then: sum_c u_j[i,c] * v_j[k,c] * w_k[i,c] = 1 for all i,k
    
    # This is a system of equations. For fixed j, we need:
    # sum_c u_j[i,c] * v_j[k,c] * w_k[i,c] = 1 for all i in [0,3), k in [0,7)
    
    # This is 21 equations per j, with 3*r_j unknowns for u_j, 7*r_j for v_j,
    # and shared w_k across all j.
    
    # Let's try r_j = 3 for each j (total rank 9)
    
    # Actually, let's try a simpler approach: use known low-rank identities
    # for small matrix multiplication and extend them.
    
    # For 3x3x3, Strassen achieves rank 7. For 3x3x7, we might be able
    # to achieve rank around 50-60.
    
    # Let's try a randomized search with verification
    import random
    
    for attempt in range(200):
        if _deadline_exceeded():
            return None
        
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        # Generate sparse random patterns
        for c in range(target_rank):
            # Choose support for each matrix
            nz_u = random.sample(range(n * m), random.randint(2, 5))
            nz_v = random.sample(range(m * p), random.randint(2, 5))
            nz_w = random.sample(range(p * n), random.randint(2, 5))
            
            for idx in nz_u:
                U[idx, c] = random.choice([-2, -1, 1, 2])
            for idx in nz_v:
                V[idx, c] = random.choice([-2, -1, 1, 2])
            for idx in nz_w:
                W[idx, c] = random.choice([-2, -1, 1, 2])
        
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None


# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' as lists of lists.
        U: (n*m, rank) = (9, rank)
        V: (m*p, rank) = (21, rank)
        W: (p*n, rank) = (21, rank)
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 63
    
    # Start with the guaranteed-valid schoolbook decomposition
    best = _schoolbook_decomposition(n, m, p)
    best_rank = schoolbook_rank
    
    # Try to find a lower-rank decomposition
    # Try ranks from 55 down to 40
    for target_rank in range(55, 39, -1):
        if _deadline_exceeded():
            break
        
        improved = _try_rank_reduction(n, m, p, target_rank)
        if improved is not None:
            best = improved
            best_rank = target_rank
            break
    
    return best
# EVOLVE-BLOCK-END
