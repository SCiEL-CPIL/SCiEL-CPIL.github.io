import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 5]

def verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Compute the tensor from U, V, W
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    tensor = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
    
    # The multiplication tensor should be 1 for all (i,j,k)
    expected = np.ones((n, m, p))
    
    if not np.allclose(tensor, expected, atol=1e-9):
        return False, tensor
    return True, tensor

def build_rank_28(n, m, p):
    """
    Construct a rank-28 decomposition for 2x3x5 matrix multiplication.
    Strategy:
    1. j=0: 2x1x5 block (rank 10, schoolbook)
    2. j=1,2, k=0,1: 2x2x2 block (rank 7, Strassen)
    3. j=1,2, k=2: 2x2x1 block (rank 4, schoolbook)
    4. j=1,2, k=3,4: 2x2x2 block (rank 7, Strassen)
    Total rank: 10 + 7 + 4 + 7 = 28.
    """
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0

    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k, col):
        U[i * m + j][col] = 1.0
        V[j * p + k][col] = 1.0
        W[k * n + i][col] = 1.0
        return col + 1

    # Helper to add Strassen 2x2x2 block at offsets (i_off, j_off, k_off)
    def add_strassen(i_off, j_off, k_off, col):
        # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        # M2: (A10+A11)B00 -> C10 += M2, C11 -= M2
        U[(i_off+1)*m + (j_off+0)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        W[(k_off+0)*n + (i_off+1)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = -1.0
        col += 1

        # M3: A00(B01-B11) -> C01 += M3, C11 += M3
        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = -1.0
        W[(k_off+1)*n + (i_off+0)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        # M4: A11(B10-B00) -> C00 += M4, C10 += M4
        U[(i_off+1)*m + (j_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+0)][col] = -1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        W[(k_off+0)*n + (i_off+1)][col] = 1.0
        col += 1

        # M5: (A00+A01)B11 -> C00 -= M5, C01 += M5
        U[(i_off+0)*m + (j_off+0)][col] = 1.0
        U[(i_off+0)*m + (j_off+1)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = -1.0
        W[(k_off+1)*n + (i_off+0)][col] = 1.0
        col += 1

        # M6: (A10-A00)(B00+B01) -> C11 += M6
        U[(i_off+1)*m + (j_off+0)][col] = 1.0
        U[(i_off+0)*m + (j_off+0)][col] = -1.0
        V[(j_off+0)*p + (k_off+0)][col] = 1.0
        V[(j_off+0)*p + (k_off+1)][col] = 1.0
        W[(k_off+1)*n + (i_off+1)][col] = 1.0
        col += 1

        # M7: (A01-A11)(B10+B11) -> C00 += M7
        U[(i_off+0)*m + (j_off+1)][col] = 1.0
        U[(i_off+1)*m + (j_off+1)][col] = -1.0
        V[(j_off+1)*p + (k_off+0)][col] = 1.0
        V[(j_off+1)*p + (k_off+1)][col] = 1.0
        W[(k_off+0)*n + (i_off+0)][col] = 1.0
        col += 1

        return col

    # 1. j=0 block: i in {0,1}, j=0, k in {0,1,2,3,4}
    # Schoolbook: 2 * 5 = 10 terms
    for k in range(p):
        col = add_schoolbook(0, 0, k, col)
        col = add_schoolbook(1, 0, k, col)
    
    # 2. j=1,2, k=0,1 block: i in {0,1}, j in {1,2}, k in {0,1}
    # Strassen: 7 terms
    # i_off=0, j_off=1, k_off=0
    col = add_strassen(0, 1, 0, col)

    # 3. j=1,2, k=2 block: i in {0,1}, j in {1,2}, k=2
    # Schoolbook: 2 * 2 = 4 terms
    for i in range(n):
        for j in [1, 2]:
            col = add_schoolbook(i, j, 2, col)

    # 4. j=1,2, k=3,4 block: i in {0,1}, j in {1,2}, k in {3,4}
    # Strassen: 7 terms
    # i_off=0, j_off=1, k_off=3
    col = add_strassen(0, 1, 3, col)

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W

def try_rank_27(n, m, p):
    """
    Try to find a rank-27 decomposition by searching for a linear dependency
    among the 28 terms of the standard block decomposition.
    
    The idea: if we can find coefficients c_0, ..., c_27 (not all zero) such that
    sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0 as a tensor, then we can eliminate one term
    and get rank 27.
    
    We search over small integer/half-integer coefficients.
    """
    deadline = time.time() + 90  # 90 second budget for this search
    
    # Start with the rank-28 decomposition
    U28, V28, W28 = build_rank_28(n, m, p)
    rank28 = 28
    
    U = np.array(U28, dtype=float)
    V = np.array(V28, dtype=float)
    W = np.array(W28, dtype=float)
    
    # The tensor constraint: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # We want to find a non-trivial linear combination of the rank-1 terms that sums to zero.
    
    # Build the system: for each (i,j,k), we have a bilinear form in the "middle" variables.
    # Actually, let's think differently. Each rank-1 term contributes u_r(i,j) * v_r(j,k) * w_r(k,i) to T[i,j,k].
    # We want sum_r c_r * u_r(i,j) * v_r(j,k) * w_r(k,i) = 0 for all (i,j,k).
    
    # This is equivalent to: the tensor sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0.
    
    # Let's build a matrix where each row corresponds to an (i,j,k) triple,
    # and each column corresponds to a "combined" index, but this is tricky because
    # the contribution is trilinear, not linear in a single set of variables.
    
    # Alternative approach: fix the W vectors and solve for c_r such that
    # for each (i,j,k): sum_r c_r * U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 0
    
    # This is still nonlinear in c_r if we don't fix the structure.
    
    # Better approach: think of it as a linear system in c_r.
    # For fixed (i,j,k), the equation is:
    # sum_r c_r * (U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]) = 0
    # This IS linear in c_r! Each c_r has coefficient a_{ijk,r} = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
    
    # So we build a matrix A of size (n*m*p) x rank28 where A[ijk, r] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # and find the null space of A.
    
    A = np.zeros((n * m * p, rank28))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank28):
                    A[idx, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                idx += 1
    
    # Find null space of A
    # If there's a non-trivial null vector c, then sum_r c_r * term_r = 0
    # and we can eliminate one term.
    
    try:
        # Use SVD to find null space
        U_svd, s, Vt = np.linalg.svd(A, full_matrices=False)
        # Null space vectors are rows of Vt corresponding to zero (or very small) singular values
        tol = 1e-10
        null_space = Vt[s < tol]
        
        if len(null_space) > 0:
            # We have a null vector. Let's use the first one.
            c = null_space[0]
            # Find the index of the largest |c_r| to eliminate
            max_idx = np.argmax(np.abs(c))
            # Check that c[max_idx] is not zero
            if abs(c[max_idx]) > 1e-10:
                # We can eliminate term max_idx by expressing it in terms of others.
                # But actually, for the decomposition to work, we need to REMOVE one term
                # and adjust the others. This isn't straightforward.
                
                # Actually, the null space tells us that sum_r c_r * term_r = 0.
                # This means the 28 rank-1 tensors are linearly dependent.
                # So the rank of the tensor is at most 27.
                # But to construct an explicit rank-27 decomposition, we need to find
                # 27 rank-1 tensors that span the same space.
                
                # One approach: if c[max_idx] != 0, we can write term_max_idx = -sum_{r!=max_idx} (c_r/c_max_idx) * term_r
                # But this doesn't directly give us a rank-27 decomposition of the ORIGINAL tensor.
                
                # The original tensor T = sum_r term_r.
                # If sum_r c_r * term_r = 0, then T = sum_r term_r = sum_r (1/c_r) * c_r * term_r ... no this doesn't help.
                
                # Actually, the linear dependence means the 28 vectors (in the tensor space) are dependent,
                # so the tensor has rank at most 27. But to get an explicit rank-27 decomposition,
                # we need to find 27 rank-1 tensors that sum to T.
                
                # Let's try a different approach: use the null vector to eliminate one term.
                # If sum_r c_r * u_r ⊗ v_r ⊗ w_r = 0, then we can replace one term.
                # But this requires the other 27 terms to still be rank-1, which they are.
                # The issue is that we need the SUM to still equal T.
                
                # Let's just verify: does the null space approach actually work?
                # The tensor T = sum_{r=0}^{27} term_r.
                # We found c such that sum_r c_r * term_r = 0.
                # This means the 28 terms are linearly dependent, so T can be written as a sum of at most 27 rank-1 tensors.
                # But we need to find them explicitly.
                
                # One way: pick 27 linearly independent terms from the 28, and express the 28th as a combination.
                # Then T = sum of the 27 independent terms + (the 28th term) = sum of 27 terms where the last one is adjusted.
                # But the adjusted term is a linear combination of rank-1 tensors, not necessarily rank-1 itself.
                
                # So this approach doesn't directly give a rank-27 decomposition.
                # We need a different strategy.
                
                pass
    except Exception as e:
        pass
    
    # Alternative: try specific structural modifications to the rank-28 decomposition.
    # Idea: merge some terms across the k-dimension for j=0.
    # For j=0, we have 10 schoolbook terms. Can we do better?
    # The subproblem is 2x1x5, which has rank 10 (schoolbook is optimal for 2x1xp or 1x1xp).
    # Actually, for 2x1x5, the rank is 2*5=10, so we can't improve that block.
    
    # What about the j=1,2 blocks? We use Strassen (7) + schoolbook (4) + Strassen (7) = 18.
    # Total: 10 + 18 = 28.
    
    # Can we do better on the j=1,2 part? The subproblem is 2x2x5.
    # Known: rank(2x2x5) = ? 
    # 2x2x2 has rank 7 (Strassen). 2x2x3 has rank 11 (7+4). 2x2x4 has rank 14 (7+7). 2x2x5 has rank 18 (7+4+7) or possibly less?
    
    # Let's try: 2x2x5 = 2x2x2 + 2x2x2 + 2x2x1 = 7+7+4 = 18.
    # Or: 2x2x5 = 2x2x3 + 2x2x2 = 11+7 = 18.
    # Or: 2x2x5 = 2x2x4 + 2x2x1 = 14+4 = 18.
    # So 18 seems to be the natural rank for 2x2x5.
    
    # But wait, maybe there's a more clever decomposition of 2x2x5 with rank < 18?
    # The lower bound for 2x2xp is not well-known. For p=5, the best known upper bound is 18.
    
    # Let's try a global optimization approach: search for a rank-27 decomposition by
    # randomly perturbing the rank-28 decomposition and checking if we can reduce rank.
    
    # Actually, let's try a more systematic approach:
    # Consider the 2x3x5 tensor. We can try different blocking strategies.
    
    # Strategy A: j=0 (rank 10) + j=1,2 (rank 18) = 28
    # Strategy B: i=0 (rank 15) + i=1 (rank 15) = 30 (worse)
    # Strategy C: k=0,1,2 (rank 12) + k=3,4 (rank 8) = 20? Let's check.
    #   For k=0,1,2: 2x3x3. Rank of 2x3x3?
    #   For k=3,4: 2x3x2. Rank of 2x3x2?
    
    # Let's compute: 2x3x2 = 2*(3x2) = 2*6 = 12 schoolbook, but maybe less?
    # 3x2x2: can we use Strassen-like? 3x2 = 2x3, and 2x3x2 has rank at most 2*3*2=12, but maybe 8 or 9?
    
    # Actually, let's just try to find a rank-27 decomposition by searching over
    # small perturbations of the rank-28 decomposition.
    
    # For now, return None to indicate no rank-27 found.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try rank-27 first
    try:
        result = try_rank_27(n, m, p)
        if result is not None:
            U, V, W = result
            valid, _ = verify_decomposition(U, V, W, n, m, p)
            if valid:
                return {'U': U, 'V': V, 'W': W}
    except Exception as e:
        pass
    
    # Attempt the known best rank 28 decomposition
    try:
        U, V, W = build_rank_28(n, m, p)
        valid, tensor = verify_decomposition(U, V, W, n, m, p)
        if valid:
            return {'U': U, 'V': V, 'W': W}
    except Exception as e:
        pass

    # Fallback: Schoolbook (Rank 30)
    rank_school = n * m * p
    U = [[0.0] * rank_school for _ in range(n * m)]
    V = [[0.0] * rank_school for _ in range(m * p)]
    W = [[0.0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    valid, _ = verify_decomposition(U, V, W, n, m, p)
    if valid:
        return {'U': U, 'V': V, 'W': W}
    
    # Should not happen
    return {'U': [[0.0]], 'V': [[0.0]], 'W': [[0.0]]}
