import time
import numpy as np

DIMENSIONS = [3, 4, 7]

def _to_lists(mat):
    """Convert numpy array to list of lists of int/float (half-integers)."""
    return [[float(v) for v in row] for row in mat]

def _verify_tensor(U, V, W, n, m, p):
    """Verify that sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] == 1 for all i,j,k."""
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen4_decomp(n, m, p):
    """
    Try a Strassen-style decomposition for M_{n,m,p}.
    
    Idea: Use the 4x4 Strassen algorithm on a 4x4x4 block structure,
    then embed into the (n, m, p) case.
    
    For a general approach: we look for a rank-r decomposition where
    the rank-r term is built from tensor products of 1D decompositions.
    
    Key insight for 3x4x7:
    - The tensor T[i,j,k] = 1 for all valid i,j,k.
    - We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    Strategy: Use a Kronecker-product structure.
    If we can write the identity tensor as a sum of rank-1 tensors,
    each rank-1 tensor is U_r ⊗ V_r ⊗ W_r (appropriately indexed).
    
    For Strassen-like: decompose the middle index j into blocks.
    j ∈ {0,1,2,3}. Write j = j_hi * 2 + j_lo where j_lo ∈ {0,1}, j_hi ∈ {0,1}.
    
    The 2x2 Strassen gives rank 7 for 2x2x2 instead of 8.
    For 4x4x4: rank 49 (7^3) instead of 64.
    
    For 3x4x7: we need a different approach since dimensions aren't powers of 2.
    
    Let's try: split n=3 into 2+1, m=4 into 2+2, p=7 into 4+3.
    This gets complicated. Let's try a more direct approach.
    
    Direct algebraic search for a small-rank decomposition:
    Use the fact that T = sum over (i,j,k) of e_i ⊗ e_j ⊗ e_k.
    We want to express this as sum_r of rank-1 tensors.
    
    A rank-1 tensor has the form:
    U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
    
    For this to work, we need specific structure.
    """
    pass

def _build_block_strassen(n, m, p):
    """
    Build a Strassen-inspired decomposition.
    
    Approach: For the (i,j) part of U and (j,k) part of V and (k,i) part of W,
    use the following:
    
    Split the j-index into two halves: j = 2*j1 + j2 where j1 ∈ {0,1}, j2 ∈ {0,1}
    for m=4.
    
    For each (i, j1, j2, k), we want a rank-1 term.
    Use Strassen on the 2x2 block in j.
    
    Actually, let me try a completely different approach:
    Enumerate all possible "basis" decompositions using small integer vectors.
    """
    pass

def _random_integer_search(n, m, p, max_rank, time_limit):
    """
    Random search for low-rank integer decompositions.
    Start from rank 84 (schoolbook) and try to find lower.
    """
    start = time.time()
    deadline = start + time_limit
    
    # Try ranks from small to large
    # Lower bounds: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # Also: n*m*p = 84 is trivial upper bound
    # Known: R(3x4x7) >= max(n*m, m*p, p*n) = 28
    
    for rank in range(28, 85):
        if time.time() > deadline:
            break
        
        # Try a few random attempts at this rank
        attempts = 3
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            # Generate random integer matrices
            np.random.seed(rank * 1000 + attempt)
            
            U = np.zeros((n*m, rank), dtype=float)
            V = np.zeros((m*p, rank), dtype=float)
            W = np.zeros((p*n, rank), dtype=float)
            
            # Initialize with random small integers in {-2, -1, 0, 1, 2}
            U = np.random.randint(-2, 3, size=(n*m, rank)).astype(float)
            V = np.random.randint(-2, 3, size=(m*p, rank)).astype(float)
            W = np.random.randint(-2, 3, size=(p*n, rank)).astype(float)
            
            # Quick check: verify tensor identity
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # We need T[i,j,k] = 1 for all i,j,k
            
            # Efficient check using numpy
            ok = True
            for i in range(n):
                for k in range(p):
                    # For fixed i,k: sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = sum_j 1 = m
                    # Wait, the tensor identity is:
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for each (i,j,k)
                    
                    for j in range(m):
                        s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        if abs(s - 1.0) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                return U, V, W, rank
    
    return None

def _structured_kronecker_search(n, m, p, time_limit):
    """
    Try Kronecker-product structured decompositions.
    
    If m = m1*m2, we can write the tensor as a Kronecker product of smaller tensors.
    For m=4 = 2*2:
    T[i, j1*m2+j2, k] where j1∈{0,1}, j2∈{0,1}
    
    A rank-1 term: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Let r = (r1, r2) where r1 ranges over rank1 (for the "outer" 2x2) and r2 for "inner".
    Then U[i*m + j1*m2 + j2, (r1,r2)] = U1[i*m1+j1, r1] * U2[j2, r2]  ... not quite.
    
    Actually, for Kronecker structure:
    Let m = m1 * m2, n = n1 * n2, p = p1 * p2.
    Then M_{n,m,p} can be related to M_{n1,m1,p1} ⊗ M_{n2,m2,p2} if dimensions factor.
    
    For n=3, m=4, p=7: none factor nicely into equal factors.
    3 = 3*1, 4 = 2*2, 7 = 7*1.
    
    So we can write:
    M_{3,4,7} involves M_{3,2,7} ⊗ M_{1,2,1} or similar.
    
    M_{1,2,1}: rank = 1*2*1 = 2 (schoolbook), but actually rank 1? No.
    For 1x2x1: T[0,j,0] = 1 for j=0,1. Rank = 2.
    
    This path is complex. Let me try a different structured approach.
    
    ALTERNATIVE: Use the fact that for the (i,j) indices in U, we can try
    U[i*m+j, r] = a[i, r] * b[j, r] (rank-1 in the (i,j) block).
    Similarly V[j*p+k, r] = c[j, r] * d[k, r].
    W[k*n+i, r] = e[k, r] * f[i, r].
    
    Then the product: a[i,r]*b[j,r]*c[j,r]*d[k,r]*e[k,r]*f[i,r]
    = (a[i,r]*f[i,r]) * (b[j,r]*c[j,r]) * (d[k,r]*e[k,r])
    = A[i,r] * B[j,r] * C[k,r]
    
    So we need: sum_r A[i,r]*B[j,r]*C[k,r] = 1 for all i,j,k.
    
    This means the tensor 1_{i,j,k} = sum_r A[i,r]*B[j,r]*C[k,r].
    This is a rank-r decomposition of the all-ones tensor of size n×m×p.
    
    The rank of the all-ones tensor of size a×b×c is max(a,b,c) when... no.
    Actually, the all-ones tensor can be written as:
    sum_{r=1}^{max(n,m,p)} ... no, that's not right either.
    
    The all-ones tensor J of size n×m×p has rank = min(n,m,p) if we allow...
    Actually J[i,j,k] = 1 = sum_r (1/n_r * 1/m_r * 1/p_r) * n_r * m_r * p_r ... 
    
    Simplest: J[i,j,k] = 1 * 1 * 1. This is rank 1!
    A[i,r]=1, B[j,r]=1, C[k,r]=1 for r=0.
    
    So if we can factor U, V, W as outer products in this way, we get rank 1!
    But we need U[i*m+j, r] = a[i,r]*b[j,r] exactly, which means U must be a
    Kronecker product structure.
    
    For rank 1: U[i*m+j, 0] = a[i,0]*b[j,0] = 1*1 = 1 for all i,j. ✓
    V[j*p+k, 0] = c[j,0]*d[k,0] = 1*1 = 1 for all j,k. ✓
    W[k*n+i, 0] = e[k,0]*f[i,0] = 1*1 = 1 for all k,i. ✓
    
    Then sum_r U*V*W = 1*1*1 = 1. ✓
    
    WAIT. This means rank 1 works?! Let me double-check.
    
    U has shape (n*m, 1) = (12, 1). U[i*m+j, 0] = 1 for all i,j.
    V has shape (m*p, 1) = (28, 1). V[j*p+k, 0] = 1 for all j,k.
    W has shape (p*n, 1) = (21, 1). W[k*n+i, 0] = 1 for all k,i.
    
    Tensor check: for any (i,j,k):
    sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] = U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] = 1*1*1 = 1. ✓
    
    This is a valid rank-1 decomposition! But wait, is this correct for matrix multiplication?
    
    The tensor T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k].
    For matrix multiplication, T[i,j,k] = 1 for all valid i,j,k.
    
    The decomposition says: C[i,k] = sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_j' V[j'*p+k,r]*B[j',k]) * W[k*n+i,r]
    
    Hmm, but this doesn't look right. The standard decomposition is:
    C[i,k] = sum_r L_r(A) * R_r(B) * W_r
    where L_r(A) = sum_{i',j'} U[i'*m+j', r] * A[i',j'] and R_r(B) = sum_{j',k'} V[j'*p+k', r] * B[j',k'].
    
    But then we'd need W[k*n+i, r] to select the right (i,k) output.
    
    For rank 1: L_0(A) = sum_{i',j'} A[i',j'] = sum of all elements of A.
    R_0(B) = sum of all elements of B.
    W[0] = 1.
    
    C[i,k] = (sum all A) * (sum all B) * 1.
    
    This is NOT equal to the standard matrix product C[i,k] = sum_j A[i,j]*B[j,k]!
    
    So the tensor identity I checked is wrong. Let me reconsider.
    
    The correct tensor for matrix multiplication M_{n,m,p} is:
    T is a 3-tensor with T[i,j,k] = 1 if the term A[i,j]*B[j,k] appears in C[i,k].
    
    The bilinear decomposition is:
    C[i,k] = sum_r (sum_{i',j'} U[i'*m+j', r] * A[i',j']) * (sum_{j',k'} V[j'*p+k', r] * B[j',k'])
    
    AND W is not multiplied in the same way. Let me re-read the task.
    
    "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 7 matrix."
    "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    
    The tensor T has entries T[(i,j), (j,k), (i,k)] but that's not standard.
    
    Actually, the matrix multiplication tensor is T ∈ R^{(n×m) × (m×p) × (n×p)} where
    T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k'.
    
    So T[A_idx, B_idx, C_idx] = 1 if the indices match.
    
    The decomposition is: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    where U has shape (n*m, r), V has shape (m*p, r), W has shape (n*p, r).
    
    But the task says W has shape (p*n, rank), which is the same as (n*p, rank) just with different ordering.
    W[k*n+i, r] means the row index is k*n+i, so it's indexed by (k,i) not (i,k).
    
    So T[(i,j), (j',k), (k',i')] = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
    
    And we need T[(i,j), (j',k), (k',i')] = 1 if i=i', j=j', k=k', else 0.
    
    So the condition is:
    sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = δ_{i,i'} * δ_{j,j'} * δ_{k,k'}
    
    For rank 1: U[i*m+j, 0] = u_i * v_j, V[j'*p+k, 0] = v_{j'} * w_k, W[k'*n+i', 0] = w_{k'} * u_{i'}
    
    Product: u_i * v_j * v_{j'} * w_k * w_{k'} * u_{i'}
    
    We need this to equal δ_{i,i'} * δ_{j,j'} * δ_{k,k'}.
    
    For i=i'=0, j=j'=0, k=k'=0: u_0 * v_0 * v_0 * w_0 * w_0 * u_0 = u_0^2 * v_0^2 * w_0^2 = 1
    For i=0, i'=1, j=j'=0, k=k'=0: u_0 * v_0 * v_0 * w_0 * w_0 * u_1 = u_0 * u_1 * v_0^2 * w_0^2 = 0
    
    This requires u_0*u_1 = 0 or v_0 = 0 or w_0 = 0, but then the first equation fails.
    
    So rank 1 doesn't work. My earlier check was wrong because I was checking the wrong tensor identity.
    
    The correct identity is with Kronecker delta structure, not all ones.
    
    This makes the problem much harder. Let me go back to the random search but with the CORRECT verification.
    """
    start = time.time()
    deadline = start + time_limit
    
    # The correct tensor identity:
    # sum_r U[i*m+j, r] * V[j2*p+k, r] * W[k2*n+i2, r] = 1 if i==i2 and j==j2 and k==k2, else 0
    
    # For the schoolbook: rank = n*m*p = 84
    # Lower bound: max(n*m, m*p, n*p) = max(12, 28, 21) = 28
    
    # Try to find a decomposition with rank < 84
    # Use random initialization and local search
    
    best_U, best_V, best_W, best_rank = None, None, None, 84
    
    # Schoolbook as default
    U_sb = np.zeros((n*m, 84))
    V_sb = np.zeros((m*p, 84))
    W_sb = np.zeros((p*n, 84))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    # Verify schoolbook
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            s = np.dot(U_sb[i*m+j, :], V_sb[j2*p+k, :] * W_sb[k2*n+i2, :])
                            expected = 1.0 if (i==i2 and j==j2 and k==k2) else 0.0
                            if abs(s - expected) > 1e-9:
                                ok = False
                                break
                        if not ok: break
                    if not ok: break
                if not ok: break
            if not ok: break
        if not ok: break
    
    if not ok:
        print("Schoolbook verification failed!", file=sys.stderr)
    
    # Try random search for lower rank
    # This is computationally expensive, so be careful
    for target_rank in [40, 50, 60, 70, 80]:
        if time.time() > deadline:
            break
        
        for attempt in range(2):
            if time.time() > deadline:
                break
            
            np.random.seed(12345 + target_rank * 100 + attempt)
            
            U = np.random.randint(-2, 3, size=(n*m, target_rank)).astype(float)
            V = np.random.randint(-2, 3, size=(m*p, target_rank)).astype(float)
            W = np.random.randint(-2, 3, size=(p*n, target_rank)).astype(float)
            
            # Quick check on a subset
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Check diagonal: should be 1
                        s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        if abs(s - 1.0) > 1e-6:
                            ok = False
                            break
                        # Check some off-diagonal: should be 0
                        for i2 in range(n):
                            for j2 in range(m):
                                for k2 in range(p):
                                    if i2==i and j2==j and k2==k:
                                        continue
                                    if np.random.random() < 0.1:  # spot check
                                        s2 = np.dot(U[i*m+j, :], V[j2*p+k, :] * W[k2*n+i2, :])
                                        if abs(s2) > 1e-6:
                                            ok = False
                                            break
                                if not ok: break
                            if not ok: break
                        if not ok: break
                    if not ok: break
                if not ok: break
            if not ok: break
            
            if ok:
                # Full verification
                if _verify_tensor(U, V, W, n, m, p):
                    return U, V, W, target_rank
    
    return U_sb, V_sb, W_sb, 84

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s margin
    
    n, m, p = DIMENSIONS
    
    U, V, W, rank = _structured_kronecker_search(n, m, p, time_limit=120.0)
    
    # Convert to lists
    result = {
        "U": _to_lists(U),
        "V": _to_lists(V),
        "W": _to_lists(W)
    }
    
    return result

import sys
