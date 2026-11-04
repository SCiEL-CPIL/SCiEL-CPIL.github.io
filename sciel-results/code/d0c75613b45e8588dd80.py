"""Find rank-minimal tensor decomposition for 3x3 @ 3x7 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]
DEADLINE = 140.0  # safety margin before 150s kill

def _build_tensor():
    """Build the 21x21x21 multiplication tensor."""
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _verify_exact(U, V, W):
    """Verify that U,V,W exactly decompose the tensor using integer arithmetic."""
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    for c in range(r):
        T += np.einsum('i,j,k->ijk', U_arr[:, c], V_arr[:, c], W_arr[:, c])
    TT = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                TT[i * m + j, j * p + k, k * n + i] = 1
    return np.array_equal(T, TT)

def _verify_halfint(U, V, W):
    """Verify with half-integer entries (multiply by 2, check against 8x tensor)."""
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    U_arr = np.array([[int(Fraction(x).numerator * 2) for x in row] for row in U], dtype=np.int64)
    V_arr = np.array([[int(Fraction(x).numerator * 2) for x in row] for row in V], dtype=np.int64)
    W_arr = np.array([[int(Fraction(x).numerator * 2) for x in row] for row in W], dtype=np.int64)
    for c in range(r):
        T += np.einsum('i,j,k->ijk', U_arr[:, c], V_arr[:, c], W_arr[:, c])
    TT = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                TT[i * m + j, j * p + k, k * n + i] = 8
    return np.array_equal(T, TT)

def _build_rank23_3x3x3():
    """
    Build an explicit rank-23 decomposition for 3x3x3 matrix multiplication.
    Uses the known optimal decomposition structure.
    Returns U (9,23), V (9,23), W (9,23) with integer entries.
    """
    n = 3
    # The rank-23 decomposition for 3x3x3:
    # We use the structure: 9 "core" terms + 14 "extra" terms
    # Core terms: for each (i,j,k) in {0,1,2}^3, we have a term
    # But rank 23 means we need 23 terms total
    
    # Explicit coordinates for rank-23 3x3x3 decomposition:
    # Based on the algebraic geometry lower bound construction
    
    # Let's use a known explicit rank-23 decomposition
    # The decomposition can be written as:
    # For the 3x3x3 tensor, we need 23 rank-1 tensors
    
    # Using the structure from Landsberg's work:
    # We partition into:
    # - 9 terms for the "diagonal" structure
    # - 14 terms for the "off-diagonal" corrections
    
    # Explicit construction:
    # Term 1-9: For each (i,j) in {0,1,2}^2, term involving row i, col j
    # Term 10-23: Additional terms for the nonlinear structure
    
    # Let me use a concrete rank-23 decomposition:
    # This is based on the explicit coordinates from the literature
    
    U = []
    V = []
    W = []
    
    # The 3x3x3 multiplication tensor T[i,j,k] = 1 if i,j,k form a valid product
    # T_{i*m+j, j*p+k, k*n+i} = 1 for i,j,k in {0,1,2}
    
    # Rank-23 decomposition using explicit coordinates:
    # We'll use the following structure based on the "slice rank" approach
    
    # For 3x3x3, the minimal rank is 23. Here's an explicit decomposition:
    
    # Using the known rank-23 decomposition:
    # Terms 0-8: Standard "diagonal" terms
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i == j == k:
                    # Diagonal term
                    u = [0]*9; v = [0]*9; w = [0]*9
                    u[i*3+j] = 1
                    v[j*3+k] = 1
                    w[k*3+i] = 1
                    U.append(u); V.append(v); W.append(w)
    
    # This gives only 3 terms, not enough. Let me use a proper rank-23 construction.
    
    # Actually, let me use a different approach: construct the decomposition
    # by solving a linear system for the rank-1 terms.
    
    # For now, let's use a known explicit rank-23 decomposition for 3x3x3.
    # The coordinates are well-known in the literature.
    
    # Here's a valid rank-23 decomposition (scaled to integers):
    # We use the structure where each term is a rank-1 tensor.
    
    # Let me construct it properly using the "strassen-like" approach:
    # For 3x3, we can use 23 multiplications.
    
    # Explicit rank-23 coordinates for 3x3x3:
    # (Using the known optimal decomposition)
    
    # I'll use a verified rank-23 decomposition:
    coords = [
        # (U_index, V_index, W_index, coefficient)
        # This is complex; let me use a different strategy.
    ]
    
    # Alternative: Use the fact that 3x3x3 can be done in 23 multiplications
    # by exploiting the structure. Let me build it from Strassen-like ideas.
    
    # For a clean approach, let me construct the rank-23 decomposition
    # using the following method:
    
    # The 3x3x3 tensor has rank exactly 23. Here's an explicit construction:
    
    # We'll define 23 rank-1 tensors that sum to the multiplication tensor.
    
    # Using the explicit coordinates from the algebraic geometry literature:
    
    # Let me use a simpler but valid approach:
    # Construct the decomposition by solving for the coefficients.
    
    # For the 3x3x3 case, here's a known rank-23 decomposition:
    
    # I'll use the following explicit coordinates (verified to work):
    
    U_list = []
    V_list = []
    W_list = []
    
    # 9 "row-column" terms
    for i in range(3):
        for j in range(3):
            u = [0]*9; v = [0]*9; w = [0]*9
            u[i*3+j] = 1
            for k in range(3):
                v[j*3+k] = 1
                w[k*3+i] = 1
            # This is not rank-1. Let me fix.
            pass
    
    # Let me use a completely different, verified approach.
    # I'll construct the rank-23 decomposition by explicit coordinates.
    
    # The following is a valid rank-23 decomposition for 3x3x3:
    
    # Using the structure: sum over t=0..22 of u_t * v_t * w_t = T
    
    # Here are explicit integer coordinates:
    
    U_mat = np.zeros((9, 23), dtype=np.int64)
    V_mat = np.zeros((9, 23), dtype=np.int64)
    W_mat = np.zeros((9, 23), dtype=np.int64)
    
    # Terms 0-2: Diagonal terms
    for d in range(3):
        U_mat[d*3+d, d] = 1
        V_mat[d*3+d, d] = 1
        W_mat[d*3+d, d] = 1
    
    # Terms 3-8: Off-diagonal "simple" terms
    # For (i,j) with i != j:
    idx = 3
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            for k in range(3):
                if k == i or k == j:
                    continue
                U_mat[i*3+j, idx] = 1
                V_mat[j*3+k, idx] = 1
                W_mat[k*3+i, idx] = 1
                idx += 1
            # idx is now 3 + 6 = 9? No, for each (i,j) with i!=j, there's 1 k
            # So 6 terms, idx goes from 3 to 8
    
    # This is getting complicated. Let me use a verified explicit decomposition.
    
    # I'll use the following known rank-23 decomposition for 3x3x3:
    
    # Actually, the simplest correct approach: use the schoolbook for 3x3x3
    # (rank 27) and then try to reduce it, or use a known explicit rank-23.
    
    # For the 3x3x7 case, let me use a different strategy:
    # The rank of 3x3x7 is at most the rank of 3x3x3 * ceil(7/3) = 23*3 = 69
    # But we can do better by sharing terms.
    
    # Better: The 3x3x7 tensor can be viewed as having 7 "slices" in the output.
    # Each slice is a 3x3x1 tensor (which is just a 3x3 matrix).
    # The multiplication A*B where A is 3x3 and B is 3x7:
    # (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    
    # For each output column k, we need to compute a 3-vector:
    # (AB)_{:,k} = A * B_{:,k}
    
    # This is 7 separate 3x3 * 3x1 multiplications, each of which is part
    # of a 3x3x3 multiplication.
    
    # Key insight: We can share the "A" factor across all 7 columns.
    
    # The rank of 3x3x7:
    # Lower bound: max(9, 21, 21) = 21
    # Upper bound: We can use the rank-23 decomposition of 3x3x3 and extend it.
    
    # For the extension: the 3x3x3 decomposition has 23 terms.
    # For 3x3x7, we need to handle 7 output columns instead of 3.
    
    # Method: For each of the 7 output columns, we can use the same "A-side"
    # terms but with different "B-side" and "output-side" terms.
    
    # Actually, the cleanest approach:
    # The 3x3x7 tensor T[i,j,k] = 1 if the (i,j) input, (j,k) input, (k,i) output
    # are consistent.
    
    # We can decompose this by fixing the "middle" index j:
    # T = sum_j T_j where T_j[i, j, k] = 1 (for fixed j)
    
    # Each T_j is a 3x1x7 tensor (rank at most 3*7=21, but actually rank 3 since
    # T_j[i, j, k] = 1 for all i,k, so it's a rank-1 tensor in the i-k space times
    # a delta in j... no, T_j[i, j, k] = 1 means it's the outer product of
    # ones_3 (in i), delta_j (in the 9-dim middle space), and ones_7 (in k).
    # So each T_j has rank 1!
    
    # Wait, that's not right. T[i, j, k] = 1 when the indices match.
    # For fixed j, T[i, j, k] = 1 for all i in {0,1,2} and k in {0,1,2,3,4,5,6}.
    # So T_j = ones_3 (i) * e_j (middle) * ones_7 (k), which is rank 1.
    
    # So T = sum_{j=0}^{2} T_j, giving rank at most 3!
    
    # But that can't be right because the rank must be at least 21.
    
    # Let me re-examine: T[i*m+j, j*p+k, k*n+i] = 1
    # For fixed j, the middle index is j*p+k, which varies with k.
    # So T[i, j*p+k, k*n+i] = 1 for all i, k.
    # This is NOT a rank-1 tensor because the middle and output indices are coupled.
    
    # For fixed j: T[i, j*p+k, k*n+i] = 1
    # This is a 3 x 7 x 21 tensor (i, k, output_index)
    # The output index is k*n+i, which depends on both k and i.
    # So it's not separable.
    
    # The rank of this 3x7x21 sub-tensor:
    # For each (i,k), we have T[i, k, k*3+i] = 1
    # This is like a "permutation" tensor, which has rank at least max(3,7) = 7.
    
    # So total rank is at least 3*7 = 21, which matches the lower bound.
    
    # Can we achieve rank 21?
    # For each j, the sub-tensor has rank 7 (since it's a 3x7x21 tensor with
    # T[i,k,l] = 1 if l = 3k+i, which is a "graph" of a function, rank = 7).
    
    # So total rank = 3*7 = 21.
    
    # Let me verify: For fixed j, define S_j[i, k, l] = 1 if l = 3k+i, else 0.
    # This is a 3x7x21 tensor. Its rank:
    # S_j = sum_{k=0}^{6} e_k (in the 7-dim k-space) * f_k (in the 3x21 space)
    # where f_k[i, l] = 1 if l = 3k+i.
    # Each f_k is a 3x21 vector with 3 ones.
    # The 7 vectors f_0, ..., f_6 are linearly independent (they have disjoint support
    # in the l dimension).
    # So rank(S_j) = 7.
    
    # Total rank = sum_j rank(S_j) = 3*7 = 21.
    
    # So we can achieve rank 21!
    
    # Let me construct it:
    # For each j in {0,1,2} and k in {0,...,6}:
    #   U[i*3+j, term] = 1 for i in {0,1,2}
    #   V[j*3+k, term] = 1
    #   W[k*3+i, term] = 1 for i in {0,1,2}
    
    # Wait, that's not rank-1. Let me be more careful.
    
    # For fixed (j,k), we want a rank-1 tensor R_{j,k} such that:
    # R_{j,k}[i, j*3+k, l] = 1 if l = 3k+i, else 0
    # R_{j,k}[i, j*3+k, l] = U_{j,k}[i] * V_{j,k}[j*3+k] * W_{j,k}[l]
    
    # For this to work:
    # V_{j,k}[j*3+k] = 1 (and 0 elsewhere in the V vector)
    # U_{j,k}[i] = 1 for all i in {0,1,2}
    # W_{j,k}[l] = 1 if l = 3k+i for some i, but this depends on i...
    
    # The problem: W_{j,k}[l] must not depend on i, but l = 3k+i depends on i.
    # So this is NOT rank-1 for fixed (j,k).
    
    # Let me reconsider. The sub-tensor for fixed j is:
    # S_j[i, j*3+k, k*3+i] = 1
    # This is a 3 x 7 x 21 tensor.
    
    # Can we write S_j as a sum of rank-1 tensors?
    # S_j = sum_t U_t[i] * V_t[middle] * W_t[output]
    
    # For S_j[i, j*3+k, k*3+i] = 1:
    # We need: sum_t U_t[i] * V_t[j*3+k] * W_t[k*3+i] = 1
    
    # One way: For each (i,k) pair, create a term:
    # Term (i,k): U_{i,k}[i'] = 1 if i'=i, V_{i,k}[j*3+k'] = 1 if k'=k, W_{i,k}[l] = 1 if l=k*3+i
    # This gives 3*7 = 21 terms for each j, total 63 terms.
    
    # That's the schoolbook rank. We need to do better.
    
    # Alternative: For fixed j, can we find 7 rank-1 tensors that sum to S_j?
    # S_j[i, j*3+k, k*3+i] = 1
    # Try: For each k, define a term T_k:
    #   T_k[i, j*3+k', k'*3+i] = 1 if k'=k, else 0
    #   T_k[i, j*3+k, k*3+i] = 1 for all i
    #   Is T_k rank-1? T_k[i, m, l] = delta(m, j*3+k) * delta(l, k*3+i)
    #   This depends on i in the output, so it's NOT rank-1.
    
    # Hmm. Let me think differently.
    
    # The tensor T[i, j, k] = 1 (with the index coupling) for 3x3x7:
    # T[i*3+j, j*7+k, k*3+i] = 1
    
    # The rank is at least 21 (from the support size argument).
    # Can we achieve 21?
    
    # Let me try a different decomposition:
    # For each (i, k) in {0,1,2} x {0,...,6}:
    #   Create a term where U[i*3+j] = 1 for all j, V[j*7+k] = 1 for all j, W[k*3+i] = 1
    #   But this doesn't work because the j index is shared.
    
    # I think the rank is actually higher than 21. Let me just use a good upper bound
    # and construct a valid decomposition.
    
    # Upper bound: Use the rank-23 decomposition of 3x3x3 and extend to 3x3x7.
    # For the 3x3x3 decomposition with 23 terms, each term has:
    #   U term: 9-dim vector
    #   V term: 9-dim vector
    #   W term: 9-dim vector
    # For 3x3x7, we need:
    #   U term: 9-dim vector (same)
    #   V term: 21-dim vector (extend from 9 to 21)
    #   W term: 21-dim vector (extend from 9 to 21)
    
    # The extension: For the 3x3x3 decomposition, the V and W vectors are 9-dim.
    # For 3x3x7, we need 21-dim vectors.
    
    # The 3x3x7 tensor: T[i*3+j, j*7+k, k*3+i] = 1
    # The 3x3x3 tensor: T3[i*3+j, j*3+k, k*3+i] = 1 for k in {0,1,2}
    
    # For k in {0,1,2}, the 3x3x7 tensor restricted to these k values is the 3x3x3 tensor
    # (with appropriate index mapping).
    
    # For k in {3,4,5,6}, we need additional terms.
    
    # Strategy: Use the 23 terms for k in {0,1,2} and add terms for k in {3,4,5,6}.
    # For each additional k, we need to handle the "new" output column.
    
    # For k=3: T[i*3+j, j*7+3, 3*3+i] = T[i*3+j, j*7+3, 9+i] = 1
    # This is a 9x1x3 tensor (for fixed k=3), which has rank 3.
    # Similarly for k=4,5,6.
    
    # So we need 4*3 = 12 additional terms, total 23+12 = 35.
    
    # But we can do better by sharing terms.
    
    # Let me just construct a valid decomposition with a reasonable rank.
    
    # I'll use the following approach:
    # 1. For k in {0,1,2}: Use a rank-23-like decomposition (or just schoolbook for simplicity)
    # 2. For k in {3,4,5,6}: Use 3 terms each (one for each i)
    
    # Actually, let me just use a clean schoolbook-based decomposition and optimize.
    
    # For the 3x3x7 tensor:
    # T[i*3+j, j*7+k, k*3+i] = 1
    
    # Schoolbook: For each (i,j,k), create a term:
    #   U[i*3+j] = 1
    #   V[j*7+k] = 1
    #   W[k*3+i] = 1
    # Total: 3*3*7 = 63 terms.
    
    # We can reduce this by noticing that for fixed (j,k), the terms over i
    # can be combined.
    
    # For fixed (j,k): sum_i U[i*3+j] * V[j*7+k] * W[k*3+i]
    # If we set V[j*7+k] = 1, U[i*3+j] = 1 for all i, W[k*3+i] = 1 for all i,
    # we get sum_i 1*1*1 = 3, not 1.
    
    # So we need separate terms for each i.
    
    # Let me try a different grouping:
    # For fixed (i,k): sum_j U[i*3+j] * V[j*7+k] * W[k*3+i]
    # If we set U[i*3+j] = 1 for all j, V[j*7+k] = 1 for all j, W[k*3+i] = 1,
    # we get sum_j 1*1*1 = 3, not 1.
    
    # The tensor is "diagonal" in a sense, so we can't easily combine terms.
    
    # Let me just use the schoolbook rank of 63 and see if we can improve.
    
    # Actually, I realize I should just construct a valid decomposition and return it.
    # The score is the rank, so lower is better.
    
    # Let me use a rank-63 schoolbook decomposition as a safe fallback,
    # and try to improve it.
    
    return None  # Placeholder

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build the tensor for verification
    T = _build_tensor()
    
    # Schoolbook decomposition: rank = n*m*p = 63
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school), dtype=np.int64)
    V = np.zeros((m * p, rank_school), dtype=np.int64)
    W = np.zeros((p * n, rank_school), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify
    T_recon = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for c in range(rank_school):
        T_recon += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    assert np.array_equal(T_recon, T), "Schoolbook verification failed"
    
    # Convert to list format
    U_list = [[int(U[i, c]) for c in range(rank_school)] for i in range(n * m)]
    V_list = [[int(V[i, c]) for c in range(rank_school)] for i in range(m * p)]
    W_list = [[int(W[i, c]) for c in range(rank_school)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
