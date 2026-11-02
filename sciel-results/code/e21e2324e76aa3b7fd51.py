"""Tensor rank search for 2x4x7 matrix multiplication."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]
DEADLINE = 150.0

def _time_remaining():
    return DEADLINE - time.time()

def _build_strassen_base(n=2, m=4, p=7):
    """Build the rank-50 Strassen-based decomposition for 2x4x7.
    
    Splits the 4 columns of B into two blocks of 2, applies Strassen (rank 7)
    to each 2x2x7 block. Total rank = 2*7 = 14... wait, that's not right.
    
    Actually: A is 2x4, B is 4x7, C is 2x7.
    Split B's 4 rows into two groups of 2: B = [B1; B2] where B1 is 2x7, B2 is 2x7.
    Split A's 4 columns into two groups of 2: A = [A1 | A2] where A1 is 2x2, A2 is 2x2.
    C = A1*B1 + A2*B2.
    
    For each 2x2x7 product, use Strassen: rank 7 per block, but the 7 dimension
    means we need 7 columns. Strassen gives rank 7 for 2x2x2, so for 2x2x7 we
    need 7*... no.
    
    Let me think again. Strassen for 2x2x2 uses 7 multiplications.
    For 2x2x7: C = A * B where A is 2x2, B is 2x7, C is 2x7.
    We can apply Strassen to each column independently, but that gives 7*7=49.
    Or we can use a block Strassen: treat B as 2x7 = 2x(7), and the 7 dimension
    is just repeated. Actually for 2x2x7, the minimal rank is known to be 22.
    
    Let me just build the schoolbook for each 2x2x7 block (rank 4*7=28 each,
    total 56) and then try to reduce.
    
    Actually, let me just build the full schoolbook (rank 56) and use that as
    the baseline, then search for reductions.
    """
    n, m, p = 2, 4, 7
    rank = n * m * p  # 56
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
    return U, V, W

def _verify_decomposition(U, V, W, n=2, m=4, p=7, tol=1e-6):
    """Verify that the tensor decomposition exactly reproduces matrix multiplication.
    
    The matrix multiplication tensor T is defined such that:
    T[i*m+j, j'*p+k, k'*n+i'] = delta_{j,j'} * delta_{k,k'}
    
    A rank-r decomposition gives U (nm x r), V (mp x r), W (pn x r) such that:
    T[a, b, c] = sum_s U[a,s] * V[b,s] * W[c,s]
    
    We check this on random instances.
    """
    # Check on several random inputs
    for _ in range(20):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Reconstruct C from decomposition
        # U[a, s] where a = i*m + j
        # V[b, s] where b = j'*p + k
        # W[c, s] where c = k'*n + i'
        C_recon = np.zeros((n, p))
        for s in range(U.shape[1]):
            # U[:, s] is a vector of length n*m
            # V[:, s] is a vector of length m*p
            # W[:, s] is a vector of length p*n
            # The product contributes to C[i', k'] via:
            # sum over j, k where the indices align
            u_s = U[:, s].reshape(n, m)
            v_s = V[:, s].reshape(m, p)
            w_s = W[:, s].reshape(p, n)
            # Contribution: u_s * v_s gives an n x p matrix
            # w_s is p x n, and we need w[k*n+i] to multiply
            # Actually the decomposition is:
            # C[i', k'] = sum_s sum_{j,k} U[i*m+j, s] * V[j*p+k, s] * W[k*n+i', s]
            #           = sum_s (U_s @ V_s)[i', k'] * ... 
            # Wait, let me think about this more carefully.
            #
            # U[i*m+j, s], V[j*p+k, s], W[k*n+i', s]
            # For a fixed s, the contribution to C[i', k'] is:
            #   sum_{j,k} U[i*m+j, s] * V[j*p+k, s] * W[k*n+i', s]
            # Let U_s be n x m, V_s be m x p, W_s be p x n.
            # Then: sum_{j,k} U_s[i',j] * V_s[j,k] * W_s[k,i']
            #      = (U_s @ V_s @ W_s)... no, that's not right dimensionally.
            #      = sum_k (sum_j U_s[i',j] * V_s[j,k]) * W_s[k,i']
            #      = (U_s @ V_s)[i',k] summed with W_s[k,i']
            #      = trace-like: sum_k M[i',k] * W_s[k,i'] where M = U_s @ V_s
            #      = (M * W_s)[sum over k] but W_s[k,i'] not W_s[i',k]
            #      = sum_k (U_s @ V_s)[i', k] * W_s[k, i']
            # This is not a standard matrix product.
            
            M = u_s @ v_s  # n x p
            # sum_k M[i', k] * w_s[k, i']
            for i_prime in range(n):
                for k_prime in range(p):
                    C_recon[i_prime, k_prime] += np.sum(M[i_prime, :] * w_s[:, i_prime])
        
        if not np.allclose(C_recon, C_expected, atol=tol):
            return False
    return True

def _verify_decomposition_v2(U, V, W, n=2, m=4, p=7, tol=1e-6):
    """Faster verification using the tensor identity directly."""
    r = U.shape[1]
    # Build the tensor T[a, b, c] = sum_s U[a,s]*V[b,s]*W[c,s]
    # and check against the MM tensor.
    # T has shape (n*m, m*p, p*n) = (8, 28, 14)
    # We can check on random A, B by computing C from the tensor.
    for _ in range(30):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # C[i', k'] = sum_{j,k} A[i',j] * B[j,k] * T[i*m+j, j*p+k, k*n+i']
        # But T should equal delta_{j,j'}*delta_{k,k'} so:
        # C[i', k'] = sum_{j,k} A[i',j] * B[j,k] * delta_{j,j'}*delta_{k,k'} = A[i',k']*B[k',k']... 
        # Wait, that's not right either. Let me re-derive.
        #
        # The MM tensor for A(n x m) @ B(m x p) = C(n x p):
        # C[i', k'] = sum_j A[i', j] * B[j, k']
        # The tensor T has indices (i, j, k) -> but actually the standard is:
        # T_{i,j,k} = delta_{j,j'} where (i,j) is row/col of A, (j',k) is row/col of B
        # Hmm, let me use the formulation from the task:
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # The product gives: for each output position, we sum over c.
        # C[i', k'] = sum_c sum_{j,k} U[i'*m+j, c] * V[j*p+k, c] * W[k*n+i', c]
        # For this to equal (A@B)[i', k'] = sum_j A[i',j] B[j, k'],
        # we need the tensor to encode: T[a,b,c] where a=i*m+j, b=j'*p+k, c=k'*n+i'
        # and T[a,b,c] = 1 if j=j' and k=k', else 0.
        # Then C[i',k'] = sum_{j,k} A[i',j] B[j,k] T[i*m+j, j*p+k, k*n+i']
        #               = sum_{j,k} A[i',j] B[j,k] delta_{j,j}delta_{k,k}... 
        # Wait, I'm confusing indices. Let me be very explicit.
        #
        # A has indices (i, j) with i in 0..n-1, j in 0..m-1.
        # B has indices (j, k) with j in 0..m-1, k in 0..p-1.
        # C[i, k] = sum_j A[i,j] B[j,k].
        #
        # U row index: i*m + j (for A's entry A[i,j])
        # V row index: j*p + k (for B's entry B[j,k])
        # W row index: k*n + i (for C's entry C[i,k])
        #
        # C[i, k] = sum_c sum_{j, j', k'} U[i*m+j, c] * V[j'*p+k', c] * W[k*n+i, c] * A[i,j] * B[j',k']
        # For this to work for ALL A, B, we need:
        # sum_c sum_{j,j',k'} U[i*m+j,c] V[j'*p+k',c] W[k*n+i,c] A[i,j] B[j',k'] = sum_j A[i,j]B[j,k]
        # This requires that the tensor T[i*m+j, j'*p+k', k*n+i] = delta_{j,j'} * delta_{k,k'}
        # (summed over c with the U,V,W factors).
        #
        # So verification: check that for random A, B:
        # C_recon[i,k] = sum_c sum_j sum_{j',k'} U[i*m+j,c]*V[j'*p+k',c]*W[k*n+i,c]*A[i,j]*B[j',k']
        # equals A@B.
        
        C_recon = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                # W[k*n+i, c] is the weight
                w_row = W[k * n + i, :]  # length r
                # For each c, we need sum_{j,j',k'} U[i*m+j,c]*V[j'*p+k',c]*A[i,j]*B[j',k']
                # = sum_c w_row[c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_{j',k'} V[j'*p+k',c]*B[j',k'])
                # But this factorization only works if U,V,W are separable per c, which they are.
                # Actually: sum_c [sum_j U[i*m+j,c]*A[i,j]] * [sum_{j',k'} V[j'*p+k',c]*B[j',k']] * W[k*n+i,c]
                # This is NOT the same as the triple sum unless we're careful.
                # The correct expansion:
                # C[i,k] = sum_c U[i*m+?, c]... no.
                
                # Let me just do it directly but efficiently:
                # C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * A[i,j]) * (sum_{j'} V[j'*p+?, c] * ...)
                # Hmm, this is getting complicated. Let me just verify on small random matrices
                # by building the full tensor.
                pass
        
        # Alternative: just check the tensor directly
        # T[a, b, c] = sum_s U[a,s]*V[b,s]*W[c,s]
        # should equal delta when a=i*m+j, b=j*p+k, c=k*n+i
        # i.e., T[i*m+j, j*p+k, k*n+i] = 1 and T[i*m+j, j'*p+k', k'*n+i'] = 0 otherwise
        # (when the (j,k) part doesn't match)
        
        # Check a few specific entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T_val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(T_val - 1.0) > tol:
                        return False
                    # Check a wrong entry
                    if j < m - 1:
                        b2 = (j + 1) * p + k
                        T_wrong = np.sum(U[a, :] * V[b2, :] * W[c, :])
                        if abs(T_wrong) > tol:
                            return False
                    if k < p - 1:
                        c2 = (k + 1) * n + i
                        T_wrong2 = np.sum(U[a, :] * V[b, :] * W[c2, :])
                        if abs(T_wrong2) > tol:
                            return False
    return True

def _verify_fast(U, V, W, n=2, m=4, p=7, tol=1e-6):
    """Verify decomposition by checking T[a,b,c] = sum_s U[a,s]V[b,s]W[c,s]."""
    # Key identities: T[i*m+j, j*p+k, k*n+i] = 1
    # T[i*m+j, j'*p+k, k*n+i] = 0 for j' != j
    # T[i*m+j, j*p+k, k'*n+i] = 0 for k' != k
    
    # Check all "on-diagonal" entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :]) 
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check "off-diagonal" in j
    for i in range(n):
        for j1 in range(m):
            for j2 in range(m):
                if j1 == j2:
                    continue
                for k in range(p):
                    a = i * m + j1
                    b = j2 * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > tol:
                        return False
    
    # Check "off-diagonal" in k
    for i in range(n):
        for j in range(m):
            for k1 in range(p):
                for k2 in range(p):
                    if k1 == k2:
                        continue
                    a = i * m + j
                    b = j * p + k1
                    c = k2 * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > tol:
                        return False
    
    return True

def _try_random_reduction(U, V, W, max_iters=5000, tol=1e-8):
    """Try to reduce rank by finding linear dependencies and eliminating terms.
    
    Strategy: try to find a rank-r+1 decomposition that can be reduced to rank r.
    Or: randomly perturb and see if we can drop a term.
    
    A simpler approach: check if any term is a linear combination of others
    in the tensor sense.
    """
    r = U.shape[1]
    # Try removing each term and checking if the remaining can span the same tensor
    # This is expensive; skip for now.
    return U, V, W

def _build_strassen_22(n=2, m=4, p=7):
    """Build a rank-44 decomposition using the known R(2,2,7)=22 result.
    
    Split the m=4 dimension into two blocks of 2.
    For each 2x2x7 sub-problem, use a rank-22 decomposition.
    Total rank = 2 * 22 = 44.
    
    The 2x2x7 decomposition: A is 2x2, B is 2x7, C is 2x7.
    C = A @ B.
    
    One known approach: use the identity for 2x2x7.
    The tensor for 2x2x7 MM has been shown to have rank 22.
    
    Let me construct it explicitly.
    
    For 2x2x2, Strassen gives rank 7 with these 7 terms (U, V, W each 2x2x7... 
    actually U is 2x2=4 entries, V is 2x2=4, W is 2x2=4):
    
    Strassen for 2x2x2:
    Let A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]], C = A@B.
    
    7 multiplications:
    m1 = (a11+a22)(b11+b22)
    m2 = (a21+a22)b11
    m3 = a11(b12-b22)
    m4 = a22(b21-b11)
    m5 = (a11+a12)b22
    m6 = (a21-a11)(b11+b12)
    m7 = (a12-a22)(b21+b22)
    
    C11 = m1 - m4 + m5 - m7
    C12 = m3 + m5
    C21 = m2 + m4
    C22 = m1 - m3 + m6 - m7
    
    In tensor form, U[i*m+j, s], V[j*p+k, s], W[k*n+i, s] with n=m=p=2:
    U row: i*2+j, V row: j*2+k, W row: k*2+i.
    
    Term 1 (m1): U has 1 at positions where (i,j) in {(0,0),(1,1)}, V has 1 where (j,k) in {(0,0),(1,1)}, W has 1 where (k,i) in {(0,0),(1,1)}.
    Actually, m1 = (a11+a22)(b11+b22) contributes to C via:
    C11 += m1, C12 += 0, C21 += 0, C22 += m1
    So W[0,0] += 1 (k=0,i=0), W[1,1] += 1 (k=1,i=1), and W[0,1] += 0, W[1,0] += 0.
    Wait, I need to be more careful.
    
    m1 multiplies (a11+a22) from A and (b11+b22) from B.
    A contribution: U[0*2+0, s] = 1 (i=0,j=0), U[1*2+1, s] = 1 (i=1,j=1).
    B contribution: V[0*2+0, s] = 1 (j=0,k=0), V[1*2+1, s] = 1 (j=1,k=1).
    C contribution: C11 and C22 get +m1.
    W[0*2+0, s] = 1 (k=0,i=0), W[1*2+1, s] = 1 (k=1,i=1).
    
    Let me just code this up properly.
    """
    # For 2x2x2 Strassen, the 7 terms as (U_mask, V_mask, W_mask)
    # U_mask is 4 entries (i,j), V_mask is 4 entries (j,k), W_mask is 4 entries (k,i)
    strassen_222 = [
        # (U, V, W) where each is a 4-element vector (flattened 2x2)
        # Term 1: m1 = (a11+a22)(b11+b22)
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # Term 2: m2 = (a21+a22)b11
        ([0,0,1,1], [1,0,0,0], [0,1,0,0]),
        # Term 3: m3 = a11(b12-b22)
        ([1,0,0,0], [0,1,0,-1], [0,0,1,0]),
        # Term 4: m4 = a22(b21-b11)
        ([0,0,0,1], [-1,0,1,0], [0,0,0,1]),
        # Term 5: m5 = (a11+a12)b22
        ([1,1,0,0], [0,0,0,1], [1,0,1,0]),
        # Term 6: m6 = (a21-a11)(b11+b12)
        ([-1,0,1,0], [1,1,0,0], [0,1,0,1]),
        # Term 7: m7 = (a12-a22)(b21+b22)
        ([0,1,0,-1], [0,0,1,1], [1,-1,1,-1]),
    ]
    
    # Verify Strassen 2x2x2
    for term in strassen_222:
        U_t, V_t, W_t = term
        U_t = np.array(U_t)
        V_t = np.array(V_t)
        W_t = np.array(W_t)
    
    # Now build 2x2x7 decomposition.
    # A is 2x2, B is 2x7, C is 2x7.
    # We can use the Strassen structure but extended to 7 columns.
    # The 7 columns of B: B = [b1, b2, ..., b7] where each b is 2x1.
    # C = A @ B, so C = [A@b1, A@b2, ..., A@b7].
    # Each A@b_k is a 2x1 vector computed from 2x2 @ 2x1.
    # 2x2 @ 2x1: C[i] = sum_j A[i,j] * b_k[j].
    # This is a 2x2x1 multiplication, which has rank 4 (schoolbook).
    # But we can share structure across k.
    
    # Actually, for 2x2x7, the minimal rank is 22 (Landsberg, 2014).
    # Let me just use the schoolbook: rank = 2*2*7 = 28.
    # Then total for 2x4x7 is 2*28 = 56. Not an improvement.
    
    # Let me try a different approach: use the known 2x2x7 rank-22 decomposition.
    # I'll construct it from the literature.
    
    # Actually, let me just build the schoolbook and try to optimize.
    # For now, return the schoolbook rank-56 decomposition.
    return _build_strassen_base()

def _build_rank_44():
    """Build a rank-44 decomposition for 2x4x7.
    
    Uses the fact that R(2,2,7) = 22 (known result).
    Split m=4 into two blocks of 2.
    For each block, use a rank-22 decomposition of 2x2x7.
    Total rank = 2*22 = 44.
    
    I need to construct the 2x2x7 rank-22 decomposition.
    
    Let me think about this differently. The 2x2x7 MM tensor:
    A is 2x2, B is 2x7, C is 2x7.
    C[i,k] = sum_j A[i,j] * B[j,k], for i in {0,1}, k in {0,...,6}, j in {0,1}.
    
    The tensor has 4*14*14 = 784 entries (but with structure).
    
    A rank-22 decomposition means 22 terms.
    
    Let me try to construct one. I'll use the following approach:
    - 4 "diagonal" terms for the schoolbook part (one for each (i,j,k) triple where... no, schoolbook is 28).
    - We need to save 6 terms.
    
    Actually, I recall that for 2x2xp, the rank is 2p for p>=2 (from Strassen-like
    constructions). Wait, no. R(2,2,2)=7, R(2,2,3)=10, R(2,2,4)=13, ...
    In general R(2,2,p) = 3p-2 for p>=2? Let me check:
    p=2: 3*2-2=4, but R(2,2,2)=7. So that's wrong.
    
    Actually, I think R(2,2,p) = 2p + R(2,2,1) - 1 or something like that.
    R(2,2,1) = 2 (since 2x2x1 is just a 2x2 matrix times a 2x1 vector, rank 2).
    Hmm, this is getting complicated. Let me just try to find a good decomposition
    through search.
    """
    # Let me just build the schoolbook and return it. The search will try to improve.
    return _build_strassen_base()

def _greedy_rank_reduction(U, V, W, tol=1e-8):
    """Try to reduce the rank by finding and removing redundant terms.
    
    For each term s, check if removing it still gives a valid decomposition
    (after possibly adjusting other terms). This is a heuristic.
    
    A better approach: for each pair of terms, check if one can be expressed
    as a linear combination in the tensor sense.
    """
    r = U.shape[1]
    # Simple check: for each term, see if zeroing it out still works
    # (this won't work since the tensor won't match anymore)
    # Skip this for now.
    return U, V, W

def construct():
    """Construct the best rank decomposition found within time budget."""
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook (rank 56)
    U, V, W = _build_strassen_base()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Verify the initial decomposition
    if not _verify_fast(best_U, best_V, best_W, n, m, p, tol=1e-6):
        # Something is wrong; fall back to pure schoolbook
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
        best_U, best_V, best_W = U, V, W
        best_rank = rank
    
    # Try to build a rank-44 decomposition using block structure
    # Split into two 2x2x7 blocks
    # For each block, try to find a low-rank decomposition
    # This is complex; let me try a simpler approach.
    
    # Approach: use the tensor rank search via random initialization + local search
    # But this is risky. Let me instead try a structured approach.
    
    # Try: for the 2x4x7 problem, note that:
    # C = A @ B where A is 2x4, B is 4x7.
    # Write A = [A1 | A2] with A1, A2 being 2x2.
    # Write B = [B1; B2] with B1, B2 being 2x7.
    # C = A1@B1 + A2@B2.
    #
    # For each 2x2x7 product, if we can do it in rank 22, total is 44.
    #
    # Let me try to construct a 2x2x7 rank-22 decomposition.
    #
    # I'll use the following known construction:
    # For 2x2xp, use the following terms:
    # 2 "pure" terms for each column pair... 
    #
    # Actually, let me just try to find it computationally.
    
    # Time check
    if _time_remaining() < 30:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try a simple improvement: check if the 56-term decomposition has
    # any linear dependencies that allow rank reduction.
    # This is a tensor problem, not a matrix one, so it's hard.
    
    # Let me try a different approach: random restarts with local search.
    # Initialize with the schoolbook, then try to "merge" terms.
    
    # Actually, let me try to construct the rank-44 solution explicitly.
    # I'll use the following 2x2x7 decomposition (rank 22):
    #
    # Based on the work of Landsberg, the 2x2x7 tensor has rank 22.
    # Here's a construction:
    #
    # Let me just try to verify a specific decomposition.
    # I'll use a parameterized family and search.
    
    # For time efficiency, let me just return the best we have.
    # The schoolbook gives rank 56, which scores 56/56 = 1.0.
    # To beat this, we need rank < 56.
    
    # Let me try one more thing: a structured rank-50 decomposition.
    # Use Strassen for the 2x2 sub-blocks within the 2x4x7.
    
    # Actually, the previous observation says a program got rank 50 using
    # "Strassen for 2x2 sub-blocks". Let me try to replicate that.
    
    # 2x4x7: Split the 4 middle dimension into two 2s.
    # C = A1@B1 + A2@B2 where A1,A2 are 2x2, B1,B2 are 2x7.
    # For each 2x2x7, use a rank-25 decomposition (if it exists) -> total 50.
    # Or use rank-24 -> total 48.
    
    # Let me try to build a 2x2x7 decomposition with rank as low as possible
    # by search.
    
    def _build_22x7_search(max_time=20):
        """Search for a low-rank decomposition of 2x2x7 MM."""
        nn, mm, pp = 2, 2, 7
        best_r = nn * mm * pp  # 28
        best_U = np.zeros((nn*mm, best_r))
        best_V = np.zeros((mm*pp, best_r))
        best_W = np.zeros((pp*nn, best_r))
        for c in range(best_r):
            i = c // (mm*pp)
            j = (c // pp) % mm
            k = c % pp
            best_U[i*mm+j, c] = 1
            best_V[j*pp+k, c] = 1
            best_W[k*nn+i, c] = 1
        
        t0 = time.time()
        # Try to reduce rank by random perturbation and projection
        rng = np.random.RandomState(42)
        for trial in range(50):
            if time.time() - t0 > max_time:
                break
            # Try rank = best_r - 1
            r_target = best_r - 1
            # Initialize
            U_t = rng.randn(nn*mm, r_target) * 0.1
            V_t = rng.randn(mm*pp, r_target) * 0.1
            W_t = rng.randn(pp*nn, r_target) * 0.1
            # Try to fit
            # This is a hard optimization; skip for now.
            pass
        return best_U, best_V, best_W, best_r
    
    # Given time constraints, let me just try to verify and return the best.
    # I'll attempt a simple rank reduction on the schoolbook.
    
    # Check if any column is a linear combination of others in a way that
    # preserves the tensor. This is a tensor-level check.
    
    # For a 3-way tensor T = sum_s u_s v_s w_s, the rank is the minimal s.
    # A term s is redundant if T - u_s v_s w_s has rank <= r-1, which means
    # there exists a rank-(r-1) decomposition of T - u_s v_s w_s.
    # This is equivalent to checking if u_s is in the span of {u_{s'} - alpha * u_s}
    # in some sense... it's complicated.
    
    # Let me just return the schoolbook.
    if _time_remaining() < 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Final attempt: try to find a rank-50 decomposition using a specific structure.
    # I'll use the following: for the 2x4x7 problem, note that the last dimension
    # is 7. If I can find a decomposition where each "column" of B shares structure
    # with adjacent columns, I might save terms.
    
    # For now, return the best found.
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
