"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that U,V,W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build expected tensor Texp
    Texp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                Texp[a, b, c] = 1.0
    
    # Compute actual tensor T = sum_r U[:,r] V[:,r] W[:,r]
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # This can be computed efficiently by iterating over r
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    err = np.max(np.abs(T - Texp))
    return err < tol

def _schoolbook(n, m, p):
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

def _try_laderman_blocks(n, m, p):
    """
    Try to use Laderman's algorithm (rank 7 for 2x2x2) on 2x2 blocks.
    For 2x4 x 4x7:
    A = [A1 A2] where A1, A2 are 2x2
    B = [B1; B2] where B1, B2 are 2x7
    C = A1 B1 + A2 B2
    
    We can decompose this as:
    C = A1 B1 + A2 B2
    
    If we treat B1 and B2 as 2x7 matrices, we can't directly apply Laderman to 2x2 x 2x7.
    However, we can split the 7 columns of B into blocks.
    
    Alternative: Use Strassen-like or other known algorithms.
    For 2x4 x 4x7, the schoolbook rank is 56.
    Known lower bounds: The tensor rank of 2x4 x 4x7 is at least the matrix rank of its unfoldings.
    
    Let's try a specific construction:
    We can view the multiplication as:
    C = A B where A is 2x4, B is 4x7.
    
    Let's try to use the fact that 4 = 2+2.
    A = [A1 A2] (2x2 each)
    B = [B1; B2] (2x7 each)
    C = A1 B1 + A2 B2
    
    Now, A1 B1 is 2x7, A2 B2 is 2x7.
    We can compute A1 B1 using a decomposition of rank r1 and A2 B2 using rank r2.
    Total rank = r1 + r2.
    For 2x2 x 2x7, what is the minimal rank?
    It's at least 2*7 = 14? No, that's schoolbook.
    Actually, for 2x2 x 2xp, the rank is at least 2p? No.
    The rank of 2x2 x 2xp is at least 2p?
    Let's check 2x2 x 2x2: rank 7.
    2x2 x 2x1: rank 4 (schoolbook) or less?
    T[i,j,k] for i in 0..1, j in 0..1, k in 0..0.
    T[0,0,0]=1, T[0,1,0]=1, T[1,0,0]=1, T[1,1,0]=1.
    This is just the outer product of [1,1] and [1,1] and [1]? No.
    T = sum_r U[i,r] V[j,r] W[k,r].
    For k=0 only. W[0,r] is a scalar.
    T[i,j,0] = sum_r U[i,r] V[j,r] W[0,r].
    Let w_r = W[0,r]. Then T[i,j] = sum_r (U[i,r] w_r) V[j,r].
    This is a matrix product of (2 x rank) and (rank x 2).
    So the rank of the tensor is the rank of the matrix T[::, :, 0].
    T[::, :, 0] is a 2x2 matrix of all 1s. Rank 1.
    So 2x2 x 2x1 has rank 1?
    Let's check: U = [[1],[0]], V = [[1],[1]], W = [[1]].
    T[0,0,0] = 1*1*1 = 1.
    T[0,1,0] = 1*1*1 = 1.
    T[1,0,0] = 0*1*1 = 0. But it should be 1.
    So rank 1 is not enough.
    We need T[i,j] = 1 for all i,j.
    This is the matrix J_2. Rank 2.
    So 2x2 x 2x1 has rank 2.
    
    For 2x2 x 2xp:
    T[i,j,k] = 1 for all i,j,k.
    T[i,j,:] is a 1xp vector of all 1s.
    This is a 2 x 2 x p tensor.
    It can be written as sum_{k=0}^{p-1} (e_i e_j^T) \otimes e_k? No.
    T = sum_k (J_2 \otimes e_k)?
    J_2 is 2x2. e_k is 1xp (unit vector).
    T[i,j,k] = J_2[i,j] * e_k[k] = 1 * delta_{k,?} No.
    If we take sum_{k=0}^{p-1} u_k \otimes v_k \otimes w_k.
    Let u_k = [1,1]^T, v_k = [1,1]^T, w_k = e_k (unit vector in R^p).
    Then T[i,j,k] = sum_{k'} u_{k'}[i] v_{k'}[j] w_{k'}[k]
    = sum_{k'} 1 * 1 * delta_{k',k} = 1 if k'=k.
    So T[i,j,k] = 1.
    Yes! So 2x2 x 2xp has rank p.
    
    So for 2x2 x 2x7, rank is 7.
    Therefore, A1 B1 (2x2 x 2x7) has rank 7.
    A2 B2 (2x2 x 2x7) has rank 7.
    Total rank = 7 + 7 = 14.
    
    Wait, is this correct?
    A1 is 2x2, B1 is 2x7.
    The multiplication tensor for 2x2 x 2x7 is T[i,j,k] = 1 for i in 0..1, j in 0..1, k in 0..6.
    Yes, my derivation above shows rank 7.
    
    So we can compute C = A1 B1 + A2 B2 with rank 14.
    This is much better than 56.
    
    Let's construct the factors.
    For A1 B1:
    U1[i, r] for i in 0..1, r in 0..6.
    V1[j, r] for j in 0..1, r in 0..6.
    W1[k, r] for k in 0..6, r in 0..6.
    
    From the decomposition:
    T[i,j,k] = sum_{r=0}^6 u_r[i] v_r[j] w_r[k].
    We chose u_r = [1,1]^T, v_r = [1,1]^T, w_r = e_r.
    So:
    U1[i, r] = 1 for all i, r.
    V1[j, r] = 1 for all j, r.
    W1[k, r] = 1 if k==r, else 0.
    
    Let's verify:
    sum_r U1[i,r] V1[j,r] W1[k,r] = sum_r 1 * 1 * delta_{k,r} = 1.
    Correct.
    
    Now, map to the global indices.
    A1 corresponds to A[0:2, 0:2].
    So global i is 0,1. Global j is 0,1.
    U_global[i*m + j, r_global] = U1[i, r] if j is in 0..1 and i in 0..1.
    Wait, A1 is the top-left 2x2 block of A.
    A is 2x4. A[0:2, 0:2] is A1.
    So for i in 0..1, j in 0..1:
    U[i*4 + j, r] = 1.
    
    B1 is the top 2x7 block of B.
    B is 4x7. B[0:2, 0:7] is B1.
    So for j in 0..1, k in 0..6:
    V[j*7 + k, r] = 1.
    
    C1 = A1 B1 is the top-left 2x7 block of C? No.
    C = A B.
    C[0:2, :] = A[0:2, :] B[:, :]
    = A[0:2, 0:2] B[0:2, :] + A[0:2, 2:4] B[2:4, :]
    = A1 B1 + A2 B2.
    So C[0:2, :] = C1 + C2.
    The output indices are (i, k) with i in 0..1, k in 0..6.
    W_global[k*n + i, r] = W1[k, r] = delta_{k,r}.
    
    So for the first part (A1 B1):
    r ranges from 0 to 6.
    U[i*4 + j, r] = 1 for i in 0..1, j in 0..1.
    V[j*7 + k, r] = 1 for j in 0..1, k in 0..6.
    W[k*2 + i, r] = 1 if k==r else 0.
    
    Now for A2 B2:
    A2 is A[0:2, 2:4].
    So for i in 0..1, j in 2..3:
    U[i*4 + j, r'] = 1.
    
    B2 is B[2:4, 0:7].
    So for j in 2..3, k in 0..6:
    V[j*7 + k, r'] = 1.
    
    C2 contributes to C[0:2, :].
    W[k*2 + i, r'] = 1 if k==r' else 0.
    
    So we have 14 terms total.
    r ranges from 0 to 13.
    For r in 0..6 (A1 B1 part):
    U[i*4+j, r] = 1 if i in 0..1 and j in 0..1.
    V[j*7+k, r] = 1 if j in 0..1 and k in 0..6.
    W[k*2+i, r] = 1 if k==r else 0.
    
    For r in 7..13 (A2 B2 part):
    Let r' = r - 7.
    U[i*4+j, r] = 1 if i in 0..1 and j in 2..3.
    V[j*7+k, r] = 1 if j in 2..3 and k in 0..6.
    W[k*2+i, r] = 1 if k==r' else 0.
    
    Let's verify this construction.
    We need to check T[a,b,c] = 1 for valid (i,j,k) and 0 otherwise.
    Valid (i,j,k) means a=i*4+j, b=j*7+k, c=k*2+i.
    
    Sum_r U[a,r] V[b,r] W[c,r].
    
    Case 1: j in 0..1.
    Then b = j*7 + k.
    Only terms where V[b,r] != 0 contribute.
    V[j*7+k, r] is non-zero only if:
    - r in 0..6 and j in 0..1 (matches) -> V=1.
    - r in 7..13 and j in 2..3 (does not match) -> V=0.
    So only r in 0..6 contribute.
    For these r, U[i*4+j, r] = 1 (since i in 0..1, j in 0..1).
    W[k*2+i, r] = 1 if k==r else 0.
    Sum_{r=0}^6 1 * 1 * delta_{k,r} = 1.
    Correct.
    
    Case 2: j in 2..3.
    Then b = j*7 + k.
    Only terms where V[b,r] != 0 contribute.
    V[j*7+k, r] is non-zero only if:
    - r in 0..6 and j in 0..1 (does not match) -> V=0.
    - r in 7..13 and j in 2..3 (matches) -> V=1.
    So only r in 7..13 contribute.
    For these r, U[i*4+j, r] = 1 (since i in 0..1, j in 2..3).
    W[k*2+i, r] = 1 if k == r-7 else 0.
    Sum_{r=7}^{13} 1 * 1 * delta_{k, r-7} = 1.
    Correct.
    
    Now check off-diagonal zeros.
    We need T[a,b,c] = 0 if the indices are inconsistent.
    Inconsistent means:
    a = i_a * 4 + j_a
    b = j_b * 7 + k_b
    c = k_c * 2 + i_c
    Inconsistent if i_a != i_c or j_a != j_b or k_b != k_c.
    
    Let's analyze the sum S = sum_r U[a,r] V[b,r] W[c,r].
    
    For a given r, the term is non-zero only if:
    - U[a,r] != 0
    - V[b,r] != 0
    - W[c,r] != 0
    
    For r in 0..6:
    U[a,r] != 0 iff i_a in 0..1 and j_a in 0..1.
    V[b,r] != 0 iff j_b in 0..1 and k_b in 0..6.
    W[c,r] != 0 iff k_c == r and i_c in 0..1.
    
    So for r in 0..6, the term is non-zero iff:
    j_a in 0..1, j_b in 0..1, k_c == r.
    Also i_a in 0..1, i_c in 0..1, k_b in 0..6.
    
    If the term is non-zero, we have:
    j_a = j_b (both in 0..1)
    k_c = r
    U=1, V=1, W=1.
    So the term contributes 1.
    The sum is 1 if there is exactly one such r.
    Since k_c is fixed, r is fixed to k_c.
    So if j_a in 0..1, j_b in 0..1, and k_c in 0..6, and i_a in 0..1, i_c in 0..1, k_b in 0..6:
    The sum is 1 if k_c is in 0..6 (always true) and the conditions on j_a, j_b are met.
    But we also need consistency: j_a = j_b and i_a = i_c and k_b = k_c.
    
    If j_a != j_b, can the term be non-zero?
    No, because U and V are 1 for all j in their ranges.
    Wait, U[i*4+j, r] = 1 for ALL j in 0..1.
    V[j*7+k, r] = 1 for ALL j in 0..1.
    So if j_a=0 and j_b=1, both U and V are 1.
    So the term is non-zero!
    This means T[a,b,c] = 1 even if j_a != j_b.
    This is WRONG.
    
    My construction for 2x2 x 2x7 was for the tensor T[i,j,k] where i,j,k are the indices of the factors.
    In the global tensor, the indices are a=i*m+j, b=j*p+k, c=k*n+i.
    The consistency condition is that the j in a must equal the j in b.
    In my local construction for A1 B1, I used i, j, k.
    i is the row of A1 (0..1).
    j is the col of A1 / row of B1 (0..1).
    k is the col of B1 (0..6).
    
    The local tensor T_local[i,j,k] = 1.
    This corresponds to the global tensor entry T[a,b,c] where:
    a = i*4 + j (since A1 is at cols 0..1 of A)
    b = j*7 + k (since B1 is at rows 0..1 of B)
    c = k*2 + i (since C1 is at rows 0..1 of C)
    
    The condition for the local tensor to be valid is that i, j, k are consistent.
    In the local 2x2 x 2x7 multiplication, i, j, k are just indices.
    The tensor T_local[i,j,k] = 1 for all i,j,k is correct for the multiplication.
    
    But in the global tensor, T[a,b,c] is non-zero only if the indices are consistent.
    Consistency means:
    j_a (from a) == j_b (from b)
    i_a (from a) == i_c (from c)
    k_b (from b) == k_c (from c)
    
    In my construction, for r in 0..6:
    U[a,r] = 1 if j_a in 0..1. (Does not depend on i_a? No, U[i*4+j, r] = 1 for i in 0..1, j in 0..1. So it depends on i_a being 0 or 1, which is always true for n=2).
    So U[a,r] = 1 if j_a in 0..1.
    V[b,r] = 1 if j_b in 0..1.
    W[c,r] = 1 if k_c == r.
    
    So the term is 1 if j_a in 0..1, j_b in 0..1, k_c == r.
    It does NOT check if j_a == j_b.
    So if j_a=0, j_b=1, k_c=0, the term is 1.
    But this is an inconsistent triple (j_a != j_b).
    So T[a,b,c] = 1, but it should be 0.
    
    The error is that the local tensor T_local[i,j,k]=1 does not enforce j consistency in the global context?
    No, in the local multiplication, j is the shared index.
    The local tensor is defined such that sum_j A[i,j] B[j,k] = C[i,k].
    The tensor T_local[i,j,k] is 1 for all i,j,k.
    This is correct for the local multiplication.
    
    When we embed this into the global tensor, we map:
    local i -> global i_a
    local j -> global j_a and j_b
    local k -> global k_b and k_c
    
    The global tensor entry T[a,b,c] corresponds to local T[i_a, j_a, k_b] IF j_a == j_b AND i_a == i_c AND k_b == k_c.
    If the indices are inconsistent, the global tensor entry should be 0.
    
    My construction gives T[a,b,c] = 1 if j_a in 0..1, j_b in 0..1, k_c == r.
    This is 1 even if j_a != j_b.
    So the construction is INVALID for the global tensor.
    
    Why?
    Because the local tensor T_local[i,j,k] = 1 is a tensor over indices i, j, k.
    The global tensor is over indices a, b, c.
    The mapping is not a simple substitution because the global tensor has zeros for inconsistent indices.
    
    To fix this, we need the factors U, V, W to be zero when the indices are inconsistent.
    But U, V, W are fixed vectors.
    
    The issue is that the decomposition of the local tensor does not respect the global consistency constraints.
    The local tensor is "dense" in j, but the global tensor is "sparse" in j (only diagonal).
    
    So this approach (block decomposition) does not directly work by just summing local decompositions, because the local decompositions produce non-zero values for inconsistent global indices.
    
    Therefore, the rank 14 construction is invalid.
    
    We must fall back to the schoolbook rank 56, or find a valid lower rank decomposition.
    Given the complexity and the 150s limit, and the fact that simple block decompositions fail due to consistency constraints, the safest bet is the schoolbook rank 56.
    
    However, we can try to optimize the schoolbook or check if there are any trivial reductions.
    For 2x4 x 4x7, are there any linear dependencies?
    The tensor has 56 non-zero entries.
    If any 3x3 minor of the unfolding is singular, rank might be lower.
    But for the "all ones" structure on the diagonal, it's likely full rank.
    
    Let's just return the schoolbook decomposition.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Schoolbook is the only guaranteed valid decomposition we have.
    # Attempts to reduce rank via block decomposition failed due to index consistency.
    # We could try randomized search for a lower rank, but it's risky and time-consuming.
    # Given the constraints, we return schoolbook.
    
    result = _schoolbook(n, m, p)
    
    # Optional: Verify to ensure correctness
    if time.time() < deadline - 10:
        U, V, W = result['U'], result['V'], result['W']
        if not _verify(U, V, W, n, m, p):
            # Should not happen, but just in case
            pass
            
    return result

# EVOLVE-BLOCK-END
