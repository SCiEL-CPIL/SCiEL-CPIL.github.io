"""Rewritten program for 4x4x4 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1, else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomp(n, m, p, U, V, W, tol=1e-9):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] matches the tensor."""
    rank = len(U[0])
    T = _build_tensor(n, m, p)
    # Compute decomposition tensor
    # Shape: (n*m, m*p, p*n)
    # For each (a, b, c), compute sum_r U[a,r] * V[b,r] * W[c,r]
    # Efficient: use einsum
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    # T_decomp[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    T_decomp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    max_diff = np.max(np.abs(T_decomp - T))
    return max_diff < tol, max_diff

def _strassen_block_decomposition(n, m, p):
    """
    Build a rank-28 decomposition for 4x4x4 using Strassen at block level
    with schoolbook for 2x2 inner products.
    
    The idea:
    - Split 4x4 matrices into 2x2 blocks
    - Use Strassen's 7 multiplications at the block level
    - Each block multiplication (2x2 * 2x2) uses schoolbook (rank 4)
    - Total rank = 7 * 4 = 28
    """
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block indexing:
    # i = 2*bi + ai, k = 2*bk + ak, j = 2*bj + aj
    # where bi, ai, bk, ak, bj, aj in {0, 1}
    
    # Strassen terms for block-level 2x2 matrix multiplication:
    # C_00 = M1 + M4 - M5 + M7
    # C_01 = M3 + M5
    # C_10 = M2 + M4
    # C_11 = M1 - M2 + M3 + M6
    
    # where:
    # M1 = (A_00 + A_11)(B_00 + B_11)
    # M2 = (A_10 + A_11)B_00
    # M3 = A_00(B_01 - B_11)
    # M4 = A_11(B_10 - B_00)
    # M5 = (A_00 + A_01)B_11
    # M6 = (A_10 - A_00)(B_00 + B_01)
    # M7 = (A_01 - A_11)(B_10 + B_11)
    
    # Each M_t is a 2x2 matrix product, decomposed via schoolbook (rank 4).
    # For each schoolbook term (ai, ak, aj) in {0,1}^3:
    #   contributes to output block C_{bi,bj}[ai,aj]
    #   with A coefficient from A_{bi,bk}[ai,ak]
    #   and B coefficient from B_{bk,bj}[ak,aj]
    
    # Define Strassen factors:
    # For each term t in 1..7, define:
    #   a_coeff[t][bi][bk][ai][ak] = coefficient of A[2*bi+ai, 2*bk+ak]
    #   b_coeff[t][bk][bj][ak][aj] = coefficient of B[2*bk+ak, 2*bj+aj]
    #   c_coeff[t][bi][bj][ai][aj] = coefficient for C[2*bi+ai, 2*bj+aj]
    
    # M1 = (A_00 + A_11)(B_00 + B_11)
    # A factors: A_00 (bi=0,bk=0) and A_11 (bi=1,bk=1)
    # B factors: B_00 (bk=0,bj=0) and B_11 (bk=1,bj=1)
    # C output: C_00, C_11 (with signs)
    
    # Let's build this systematically.
    
    # Strassen term definitions:
    # term: (a_blocks, b_blocks, c_blocks)
    # where a_blocks is a dict mapping (bi,bk) -> coefficient
    #       b_blocks is a dict mapping (bk,bj) -> coefficient
    #       c_blocks is a dict mapping (bi,bj) -> coefficient
    
    strassen_terms = [
        # M1 = (A_00 + A_11)(B_00 + B_11)
        {
            'a': {(0,0): 1, (1,1): 1},
            'b': {(0,0): 1, (1,1): 1},
            'c': {(0,0): 1, (1,1): 1}
        },
        # M2 = (A_10 + A_11)B_00
        {
            'a': {(1,0): 1, (1,1): 1},
            'b': {(0,0): 1},
            'c': {(1,0): 1, (1,1): -1}
        },
        # M3 = A_00(B_01 - B_11)
        {
            'a': {(0,0): 1},
            'b': {(0,1): 1, (1,1): -1},
            'c': {(0,1): 1, (1,1): 1}
        },
        # M4 = A_11(B_10 - B_00)
        {
            'a': {(1,1): 1},
            'b': {(1,0): 1, (0,0): -1},
            'c': {(0,0): 1, (1,0): 1}
        },
        # M5 = (A_00 + A_01)B_11
        {
            'a': {(0,0): 1, (0,1): 1},
            'b': {(1,1): 1},
            'c': {(0,0): -1, (0,1): 1}
        },
        # M6 = (A_10 - A_00)(B_00 + B_01)
        {
            'a': {(1,0): 1, (0,0): -1},
            'b': {(0,0): 1, (0,1): 1},
            'c': {(1,1): 1}
        },
        # M7 = (A_01 - A_11)(B_10 + B_11)
        {
            'a': {(0,1): 1, (1,1): -1},
            'b': {(1,0): 1, (1,1): 1},
            'c': {(0,0): 1}
        },
    ]
    
    col = 0
    for t, term in enumerate(strassen_terms):
        a_blocks = term['a']
        b_blocks = term['b']
        c_blocks = term['c']
        
        # For each schoolbook term (ai, ak, aj) in {0,1}^3
        for ai in range(2):
            for ak in range(2):
                for aj in range(2):
                    # This schoolbook term corresponds to:
                    # A_{bi,bk}[ai, ak] * B_{bk,bj}[ak, aj] -> C_{bi,bj}[ai, aj]
                    
                    # For the A part: we need to combine all (bi,bk) in a_blocks
                    # that contribute to this (ai, ak) position
                    # But wait, the A coefficient depends on (bi,bk), not just (ai,ak)
                    # The schoolbook decomposition of a 2x2 product X*Y gives:
                    # (X*Y)[ai,aj] = sum_ak X[ai,ak] * Y[ak,aj]
                    # So each (ai, ak, aj) is one term.
                    
                    # For the Strassen term t, the "A matrix" is a sum of blocks:
                    # A_t = sum_{(bi,bk) in a_blocks} a_blocks[(bi,bk)] * A_{bi,bk}
                    # Similarly for B_t.
                    
                    # The product A_t * B_t is a 2x2 matrix (at the inner level).
                    # Its (ai,aj) entry is sum_{ak} A_t[ai,ak] * B_t[ak,aj]
                    # But A_t and B_t are themselves 2x2 matrices whose entries
                    # are sums of A_{bi,bk}[ai,ak] over the relevant blocks.
                    
                    # Actually, this is getting complicated. Let me think differently.
                    
                    # The Strassen algorithm computes C = A*B where A,B,C are 2x2 matrices
                    # of 2x2 submatrices. Each "multiplication" M_t = X_t * Y_t where
                    # X_t is a 2x2 matrix (linear combo of A blocks) and Y_t is a 2x2 matrix
                    # (linear combo of B blocks).
                    
                    # X_t[i,j] = sum_{(bi,bk) in support} coeff * A_{bi,bk}[i,j]
                    # Y_t[i,j] = sum_{(bk,bj) in support} coeff * B_{bk,bj}[i,j]
                    
                    # M_t[i,j] = sum_k X_t[i,k] * Y_t[k,j]
                    #           = sum_k sum_{(bi,bk) in a_supp} a_coeff * A_{bi,bk}[i,k]
                    #                     * sum_{(bk,bj) in b_supp} b_coeff * B_{bk,bj}[k,j]
                    
                    # And C_{bi,bj}[i,j] gets contribution c_coeff[t] * M_t[i,j]
                    
                    # So the full decomposition has terms indexed by (t, ai, ak, aj)
                    # where for each such term:
                    # U row: i = 2*bi + ai, k = 2*bk + ak  (but bi,bk vary over the support)
                    # This doesn't work cleanly because bi and bk are not fixed.
                    
                    # I think the correct approach is to expand everything out.
                    # Each term in the final decomposition is a scalar product
                    # A[2*bi+ai, 2*bk+ak] * B[2*bk+ak, 2*bj+aj] * C[2*bi+ai, 2*bj+aj]
                    # with some coefficient.
                    
                    # Let me just enumerate all possible (bi,bk,bj,ai,ak,aj) and compute
                    # the coefficient.
                    pass  # Skip this approach, do it below
    
    # Let's do it by direct enumeration of all 64 possible (i,k,j) terms
    # and compute the coefficient for each.
    
    # Actually, let me just compute the full decomposition by expanding.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t, term in enumerate(strassen_terms):
        a_blocks = term['a']
        b_blocks = term['b']
        c_blocks = term['c']
        
        for ai in range(2):
            for ak in range(2):
                for aj in range(2):
                    # For this inner (ai, ak, aj), the contribution to
                    # C_{bi,bj}[ai,aj] comes from:
                    # sum over (bi,bk) in a_blocks and (bk',bj) in b_blocks
                    # where the middle index matches: ak from A matches ak in B
                    # Wait, in the 2x2 product, the "k" index is ak.
                    # X_t[ai, ak] * Y_t[ak, aj]
                    
                    # X_t[ai, ak] = sum_{(bi,bk) in a_blocks} a_blocks[(bi,bk)] * A_{bi,bk}[ai, ak]
                    # Y_t[ak, aj] = sum_{(bk,bj) in b_blocks} b_blocks[(bk,bj)] * B_{bk,bj}[ak, aj]
                    
                    # M_t[ai, aj] = X_t[ai, ak] * Y_t[ak, aj]  (for fixed ak, but we sum over ak)
                    # Actually M_t[ai, aj] = sum_{ak} X_t[ai, ak] * Y_t[ak, aj]
                    
                    # So for fixed (ai, aj), M_t[ai, aj] = sum_{ak} ...
                    # And the contribution to C_{bi,bj}[ai,aj] is c_blocks[(bi,bj)] * M_t[ai,aj]
                    
                    # This means for each (t, ai, ak, aj), we have contributions from
                    # all (bi,bk) in a_blocks and (bk2,bj) in b_blocks.
                    
                    # The full term is:
                    # sum_{(bi,bk) in a_blocks} sum_{(bk2,bj) in b_blocks}
                    #   a_blocks[(bi,bk)] * b_blocks[(bk2,bj)] * c_blocks[(bi,bj)]
                    #   * A_{bi,bk}[ai,ak] * B_{bk2,bj}[ak,aj] * (C output at bi,bj)
                    
                    # But this only makes sense if bk == bk2 (the middle indices match).
                    # In the 2x2 product, the inner index is ak (0 or 1), and it's the same
                    # for both X and Y. So bk and bk2 are the block-level middle indices.
                    
                    # I think I'm overcomplicating this. Let me just directly construct
                    # the rank-28 decomposition by expanding the Strassen formula.
                    
                    pass
    
    # Direct construction approach:
    # For each Strassen term t and each schoolbook index (ai, ak, aj):
    # The A-side involves blocks A_{bi,bk} for (bi,bk) in a_blocks
    # The B-side involves blocks B_{bk,bj} for (bk,bj) in b_blocks
    # But the middle block index bk must be consistent.
    
    # Let me just hard-code the expansion.
    
    # Actually, the cleanest way: think of the 4x4 matrices as having indices
    # i = 2*bi + ai, k = 2*bk + ak, j = 2*bj + aj.
    
    # The Strassen algorithm at the block level says:
    # For each output block C_{bi,bj}, its entry [ai,aj] is:
    # sum_{t: (bi,bj) in c_blocks[t]} c_blocks[t][(bi,bj)] * M_t[ai,aj]
    
    # where M_t[ai,aj] = sum_{ak} X_t[ai,ak] * Y_t[ak,aj]
    # X_t[ai,ak] = sum_{(bi2,bk) in a_blocks[t]} a_blocks[t][(bi2,bk)] * A[2*bi2+ai, 2*bk+ak]
    # Y_t[ak,aj] = sum_{(bk,bj2) in b_blocks[t]} b_blocks[t][(bk,bj2)] * B[2*bk+ak, 2*bj2+aj]
    
    # So M_t[ai,aj] = sum_{ak} sum_{(bi2,bk) in a_blocks[t]} sum_{(bk2,bj2) in b_blocks[t]}
    #   a_blocks[t][(bi2,bk)] * b_blocks[t][(bk2,bj2)] * A[2*bi2+ai,2*bk+ak] * B[2*bk2+ak,2*bj2+aj]
    
    # And C[2*bi+ai, 2*bj+aj] = sum_{t} c_blocks[t].get((bi,bj),0) * M_t[ai,aj]
    
    # Now, the tensor decomposition term for a specific (i,k,j) = (2*bi+ai, 2*bk+ak, 2*bj+aj)
    # has coefficient 1 in the original tensor.
    
    # In the rank-28 decomposition, each column r corresponds to a specific
    # (t, ai, ak, aj) combination, and the U, V, W entries are determined by
    # the Strassen coefficients.
    
    # But wait, a single column r in the decomposition corresponds to a single
    # scalar product A[i,k] * B[k,j] * C[i,j]. The rank-28 decomposition has
    # 28 such products that combine to give the full tensor.
    
    # I think the issue is that in the Strassen decomposition, each "multiplication"
    # M_t is itself a 2x2 matrix product, which has 4 scalar products (schoolbook).
    # So each Strassen term t contributes 4 columns to the decomposition.
    
    # For column r = 4*t + 4*ak + ai*2 + aj (or some ordering):
    # U[2*bi+ai, 2*bk+ak] = a_blocks[t][(bi,bk)]  for the specific (bi,bk) ... 
    # No, this doesn't work because a single U entry can only have one value.
    
    # I think the correct interpretation is:
    # Each of the 28 columns corresponds to one (bi, bk, bj, ai, ak, aj) combination
    # with a coefficient. But there are 2*2*2*2*2*2 = 64 such combinations, and
    # we're claiming rank 28, so it's not a 1-to-1 mapping.
    
    # Let me reconsider. The rank-28 decomposition means:
    # T[i,k,j] = sum_{r=1}^{28} U[i,k,r] * V[k,j,r] * W[j,i,r]
    
    # where T[i,k,j] = 1 if the term A[i,k]*B[k,j] contributes to C[i,j] (which is
    # always the case for valid indices), so T[i,k,j] = 1 for all i,k,j.
    
    # Wait, that can't be right either. The tensor T is defined such that
    # sum_{i,k,j} T[i,k,j] A[i,k] B[k,j] = trace(C) or something.
    
    # Let me just verify: in the schoolbook, T[i*m+j, j*p+k, k*n+i] = 1 for all
    # (i,j,k), and the decomposition has rank n*m*p = 64 with U[i*m+j, r]=1,
    # V[j*p+k, r]=1, W[k*n+i, r]=1 for r = i*m*p + j*p + k.
    
    # So the tensor IS the all-ones tensor in the sense that every (i,j,k) triple
    # has T=1, and the schoolbook just uses 64 orthogonal rank-1 tensors.
    
    # For a lower-rank decomposition, we need to find 28 rank-1 tensors
    # (u_r, v_r, w_r) such that for every (i,j,k):
    # sum_{r=1}^{28} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a much harder problem! It's not just about Strassen blocks.
    
    # I think I've been misunderstanding the problem. Let me re-read.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4
    # matrix by a 4 by 4 matrix."
    
    # The matrix multiplication tensor M_{4,4,4} has rank at most 64 (schoolbook).
    # Strassen shows rank <= 28 for 4x4x4? Actually, Strassen gives rank 7 for
    # 2x2x2, and for 4x4x4 you can use 7*7=49 or 7*4=28 depending on the approach.
    
    # Actually, the known upper bound for 4x4x4 is 49 (Strassen squared) or 28
    # (Strassen + schoolbook). The lower bound is 27 (from asymptotic results).
    
    # So rank 28 should be achievable. Let me implement it properly.
    
    # The key insight: the matrix multiplication tensor M_{n,m,p} is the tensor
    # T where T[i,k,j] = delta_{i,j}... no.
    
    # Let me use the standard definition. The matrix multiplication tensor
    # M_{n,m,p} in V^* \otimes W^* \otimes Z^* has components:
    # M_{n,m,p} = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i^* \otimes e_j^* \otimes e_k^*
    # where the three factors correspond to A, B, C indices.
    
    # Actually, for C = AB, we have C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor is: M = sum_{i,j,k} A_{ij}^* \otimes B_{jk}^* \otimes C_{ik}^*
    # So M_{(i,j),(j,k),(i,k)} = 1 in the appropriate indexing.
    
    # In our indexing:
    # A has indices (i, j) where i in [0,n), j in [0,m)
    # B has indices (j, k) where j in [0,m), k in [0,p)
    # C has indices (i, k) where i in [0,n), k in [0,p)
    # 
    # U is indexed by (i, j) -> i*m + j
    # V is indexed by (j, k) -> j*p + k
    # W is indexed by (k, i) -> k*n + i  (trace convention)
    
    # The tensor component is: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
    
    # And we need: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # So for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is correct. Now, for the Strassen-based rank-28 decomposition,
    # I need to construct 28 rank-1 tensors that sum to the all-ones pattern.
    
    # Let me just use a numerical approach: set up the linear system and solve.
    
    # Actually, let me try a different strategy. Use the known Strassen construction.
    
    # For 2x2x2, Strassen gives rank 7. The 7 rank-1 tensors are well-known.
    # For 4x4x4, we can use the "Strassen over the reals" construction where
    # we view 4x4 matrices as 2x2 matrices of 2x2 blocks.
    
    # The Strassen algorithm for 2x2 block matrices gives 7 block multiplications,
    # each of which is a 2x2 * 2x2 = rank 4 (schoolbook) operation.
    # Total: 7 * 4 = 28 scalar multiplications.
    
    # Each scalar multiplication corresponds to one rank-1 tensor in the decomposition.
    
    # Let me construct this explicitly.
    
    # The 7 Strassen products for block matrices:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # where Aij, Bij, Cij are 2x2 blocks.
    
    # Each Mt = X_t * Y_t where X_t, Y_t are 2x2 matrices (linear combos of blocks).
    
    # For the schoolbook decomposition of a 2x2 product:
    # (X*Y)[r,c] = sum_k X[r,k] * Y[k,c]
    # This gives 4 scalar products.
    
    # So for each Strassen term t and each (r,k,c) in {0,1}^3, we have a scalar product.
    # Total: 7 * 4 = 28 scalar products.
    
    # Now, for each such scalar product, I need to determine the U, V, W vectors.
    
    # The scalar product is: X_t[r,k] * Y_t[k,c]
    # where X_t[r,k] is a linear combination of A block entries,
    # and Y_t[k,c] is a linear combination of B block entries.
    
    # But X_t[r,k] itself is a sum of A entries, so the "scalar product" is actually
    # a product of sums, which expands into multiple A*B products.
    
    # I think the correct way to view this is:
    # The Strassen algorithm computes C = AB using 7 intermediate matrix products.
    # Each intermediate product M_t = X_t * Y_t is computed using 4 scalar products
    # (schoolbook). So the total number of scalar A[i,k]*B[k,j] products is 28.
    
    # But these 28 products are NOT independent rank-1 tensors in the decomposition
    # sense. They are the 28 scalar multiplications performed by the algorithm.
    
    # In the tensor decomposition framework, each scalar multiplication
    # A[i,k] * B[k,j] corresponds to a rank-1 tensor with U[i*m+k] = 1,
    # V[k*p+j] = 1, W[j*n+i] = 1 (or with coefficients if there are linear combos).
    
    # For Strassen, the "A factor" for each multiplication is a linear combination
    # of A entries, and the "B factor" is a linear combination of B entries.
    # So the rank-1 tensor has U and V entries that are the coefficients of these
    # linear combinations.
    
    # Let me construct this.
    
    # For Strassen term t, the A-factor is:
    # X_t[r,k] = sum_{(bi,bk) in supp_a(t)} a_t[(bi,bk)] * A[2*bi+r, 2*bk+k]
    # 
    # And the B-factor is:
    # Y_t[k,c] = sum_{(bk,bj) in supp_b(t)} b_t[(bk,bj)] * B[2*bk+k, 2*bj+c]
    
    # The scalar product for (t, r, k, c) is:
    # X_t[r,k] * Y_t[k,c] = sum_{(bi,bk_a)} sum_{(bk_b,bj)} 
    #   a_t[(bi,bk_a)] * b_t[(bk_b,bj)] * A[2*bi+r, 2*bk_a+k] * B[2*bk_b+k, 2*bj+c]
    
    # For this to be a single rank-1 tensor, we need bk_a == bk_b (the middle index).
    # If not, we get multiple A*B products, which means this "scalar product"
    # actually involves multiple rank-1 tensors.
    
    # I think the correct interpretation is that each of the 28 "multiplications"
    # in the Strassen algorithm corresponds to a bilinear form, not a single
    # scalar product. The rank of the tensor is the minimum number of rank-1
    # tensors needed to express it.
    
    # Given the complexity, let me just use a numerical rank minimization approach.
    
    pass

def _rank_min_decomposition(n, m, p, max_rank=None, time_limit=100):
    """
    Try to find a low-rank decomposition using iterative approaches.
    """
    start_time = time.time()
    
    if max_rank is None:
        max_rank = n * m * p  # schoolbook
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Known: rank(4x4x4) is between 27 and 28
    lower_bound = 27
    upper_bound = min(max_rank, 64)
    
    best_result = None
    best_rank = upper_bound + 1
    
    for rank in range(lower_bound, upper_bound + 1):
        if time.time() - start_time > time_limit:
            break
        
        # Try to find a rank-r decomposition
        # Use the approach of solving the linear system
        # T[a,b,c] = sum_{r=1}^{rank} U[a,r] * V[b,r] * W[c,r]
        
        # This is a nonlinear system, so we use iterative methods.
        
        # Strategy: fix U and V, solve for W (or vice versa)
        # Use alternating minimization
        
        # Initialize randomly
        np.random.seed(42)
        
        for trial in range(50):
            if time.time() - start_time > time_limit - 10:
                break
            
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.zeros(p * n, rank)
            
            T = _build_tensor(n, m, p)
            
            # Alternating minimization
            for iteration in range(1000):
                if time.time() - start_time > time_limit - 5:
                    break
                
                # Solve for W: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # For fixed U,V: for each c, W[c,:] = (sum_{a,b} T[a,b,c]*U[a,:]*V[b,:]) / (sum_{a,b} U[a,:]*V[b,:]*W[c,:])
                # This is not quite right. Let me use the normal equations.
                
                # The residual is R = T - decomposed_tensor
                # We want to minimize ||R||^2 by adjusting W.
                
                # For each c, the contribution to the residual from W[c,:] is:
                # R_c = T[:,:,c] - sum_r U[:,r]*V[:,r]*W[c,r]
                # = T[:,:,c] - (U^T * V^T)[:,:,?] ... this is getting complicated.
                
                # Let me use a different approach: flatten and use linear algebra.
                
                # Actually, let me just try the known Strassen construction directly.
                
                break
            
            break  # Just try the direct construction
        
        # If we found a valid decomposition, return it
        if best_result is not None:
            return best_result
    
    # Fall back to schoolbook
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
    return {"U": U, "V": V, "W": W, "rank": rank}

def _strassen_rank28(n, m, p):
    """
    Direct construction of the rank-28 Strassen-based decomposition.
    """
    # This function constructs the rank-28 decomposition for 4x4x4
    # using Strassen at the block level with schoolbook for 2x2 products.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definition:
    # Each term t has:
    #   a_factors: list of (bi, bk, coeff) - A block coefficients
    #   b_factors: list of (bk, bj, coeff) - B block coefficients
    #   c_factors: list of (bi, bj, coeff) - C block coefficients
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For each term, the A-factor X_t is a 2x2 matrix where:
    # X_t[ai,ak] = sum over (bi,bk) in a_factors: coeff * A[2*bi+ai, 2*bk+ak]
    # Similarly for Y_t.
    
    # The schoolbook decomposition of X_t * Y_t gives 4 scalar products:
    # For (ai, ak, aj): X_t[ai,ak] * Y_t[ak,aj]
    
    # Each such product is:
    # (sum_{(bi,bk)} a_coeff * A[2*bi+ai, 2*bk+ak]) * (sum_{(bk,bj)} b_coeff * B[2*bk+ak, 2*bj+aj])
    
    # This is a bilinear form in A and B. In the tensor decomposition,
    # this contributes to the rank as a single rank-1 tensor ONLY IF
    # the A and B factors are single entries (not sums).
    
    # Since they ARE sums in general, each "scalar product" in the Strassen
    # algorithm actually corresponds to multiple rank-1 tensors in the
    # decomposition sense.
    
    # I believe the correct interpretation is that the rank-28 decomposition
    # uses 28 rank-1 tensors where each tensor corresponds to one of the
    # 28 scalar multiplications in the Strassen algorithm, and the U, V, W
    # entries encode the linear combinations.
    
    # Let me construct this by expanding.
    
    # For each (t, ai, ak, aj), the contribution is:
    # sum_{(bi,bk_a) in a_factors[t]} sum_{(bk_b,bj) in b_factors[t]}
    #   a_factors[t][(bi,bk_a)] * b_factors[t][(bk_b,bj)]
    #   * A[2*bi+ai, 2*bk_a+ak] * B[2*bk_b+ak, 2*bj+aj]
    #   * (output coefficient for C[2*bi+ai, 2*bj+aj])
    
    # But this has a constraint: for the middle index to match, we need
    # bk_a == bk_b (since both refer to the same "k" in the 2x2 product).
    
    # Actually, in the 2x2 product X*Y, the inner index is the same for
    # X's column and Y's row. So bk_a should equal bk_b. If they don't match
    # in the supports, those terms don't contribute.
    
    # Let me define the Strassen terms properly:
    
    strassen = [
        # (a_factors, b_factors, c_factors)
        # a_factors: { (bi,bk): coeff }
        # b_factors: { (bk,bj): coeff }
        # c_factors: { (bi,bj): coeff }
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),           # M2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),            # M3
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),            # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),            # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),            # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),            # M7
    ]
    
    col = 0
    for t, (a_fact, b_fact, c_fact) in enumerate(strassen):
        for ai in range(2):
            for ak in range(2):
                for aj in range(2):
                    # This is one of the 28 scalar products
                    # U entries: for each (bi, bk_a) in a_fact where bk_a == ak_block
                    # Actually, the U vector for this column should have:
                    # U[2*bi+ai, 2*bk+ak] = a_fact[(bi,bk)] for the specific (bi,bk)
                    
                    # But a single column can only have one U vector, which is
                    # a fixed assignment of values to all U[*,*] entries.
                    
                    # I think each column corresponds to a specific (bi, bk, bj, ai, ak, aj)
                    # with a coefficient, not to a Strassen term directly.
                    
                    # Let me just expand everything into individual (i,k,j) terms.
                    
                    pass
    
    # Direct expansion: for each (bi, bk, bj, ai, ak, aj), compute the coefficient
    # in the rank-28 decomposition.
    
    # The coefficient for A[2*bi+ai, 2*bk+ak] * B[2*bk+ak, 2*bj+aj] * C[2*bi+ai, 2*bj+aj]
    # is the sum over all Strassen terms t where (bi,bk) in a_fact[t] and (bk,bj) in b_fact[t]
    # of a_fact[t][(bi,bk)] * b_fact[t][(bk,bj)] * c_fact[t].get((bi,bj), 0)
    
    # Wait, but we also need to account for the (ai, ak, aj) indices.
    # In the 2x2 product, the (ai, ak, aj) indices are the inner indices.
    # The Strassen coefficients don't depend on (ai, ak, aj), only on the block indices.
    
    # So the coefficient for a specific (bi, bk, bj, ai, ak, aj) is:
    # sum_{t: (bi,bk) in a_fact[t] AND (bk,bj) in b_fact[t]}
    #   a_fact[t][(bi,bk)] * b_fact[t][(bk,bj)] * c_fact[t].get((bi,bj), 0)
    
    # And this coefficient should equal 1 for the decomposition to work,
    # but it will generally be some other value. The rank-28 decomposition
    # works because the sum of these rank-1 tensors gives the correct tensor.
    
    # Let me compute the coefficients and build the decomposition.
    
    # But wait, if I use ALL 64 (bi,bk,bj,ai,ak,aj) combinations, that's rank 64.
    # The rank-28 decomposition must use only 28 of them, or use linear combinations.
    
    # I think the key is that not all 64 combinations have nonzero coefficients,
    # and some coefficients are zero, reducing the effective rank.
    
    # Let me compute the coefficients for all 64 combinations.
    
    coeffs = {}
    for bi in range(2):
        for bk in range(2):
            for bj in range(2):
                for ai in range(2):
                    for ak in range(2):
                        for aj in range(2):
                            coeff = 0
                            for t, (a_fact, b_fact, c_fact) in enumerate(strassen):
                                if (bi, bk) in a_fact and (bk, bj) in b_fact:
                                    coeff += a_fact[(bi, bk)] * b_fact[(bk, bj)] * c_fact.get((bi, bj), 0)
                            coeffs[(bi, bk, bj, ai, ak, aj)] = coeff
    
    # Count nonzero coefficients
    nonzero = sum(1 for v in coeffs.values() if v != 0)
    print(f"Nonzero coefficients: {nonzero} out of 64")
    
    # Build the decomposition using only nonzero terms
    U = [[0] * 64 for _ in range(n * m)]
    V = [[0] * 64 for _ in range(m * p)]
    W = [[0] * 64 for _ in range(p * n)]
    
    col = 0
    for (bi, bk, bj, ai, ak, aj), coeff in coeffs.items():
        if coeff == 0:
            continue
        i = 2 * bi + ai
        k = 2 * bk + ak
        j = 2 * bj + aj
        U[i * m + k][col] = coeff
        V[k * p + j][col] = 1
        W[j * n + i][col] = 1
        col += 1
    
    # Trim to actual rank
    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    # Verify
    valid, max_diff = _verify_decomp(n, m, p, U, V, W)
    if valid:
        print(f"Rank {actual_rank} decomposition verified! Max diff: {max_diff}")
        return {"U": U, "V": V, "W": W, "rank": actual_rank}
    else:
        print(f"Rank {actual_rank} decomposition FAILED. Max diff: {max_diff}")
        # Fall back to schoolbook
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
        return {"U": U, "V": V, "W": W, "rank": rank}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try the Strassen rank-28 construction
    result = _strassen_rank28(n, m, p)
    
    elapsed = time.time() - start_time
    print(f"Construction took {elapsed:.2f}s, rank={result['rank']}")
    
    # Safety check
    if time.time() - start_time > 100:
        # Fall back to schoolbook
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
    
    return {"U": result["U"], "V": result["V"], "W": result["W"]}
# EVOLVE-BLOCK-END
