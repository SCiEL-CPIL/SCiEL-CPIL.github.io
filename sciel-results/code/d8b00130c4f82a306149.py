"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i,j,k] = 1 for all i,j,k (the identity tensor for matmul)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2():
    """Return the 7 rank-1 terms of Strassen's 2x2x2 algorithm.
    Each term is (u, v, w) where u is 2x2 flattened, v is 2x2 flattened, w is 2x2 flattened.
    Convention: u[a*2+b], v[b*2+c], w[c*2+a] for the tensor T[a,b,c].
    """
    # Strassen: C = A*B where A, B, C are 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # We need to express each M_l as a sum of rank-1 tensors in the (a,b,c) space.
    # Actually, each M_l is a single scalar product, contributing to specific C entries.
    # Let's build the U, V, W matrices for the 2x2x2 case directly.
    
    # For 2x2x2, the tensor T[a,b,c] = delta_{ac} (since C[a,c] = sum_b A[a,b]*B[b,c])
    # Wait, no. The multiplication tensor for 2x2x2 has T[a,b,c] = 1 if the (a,c) entry of C 
    # depends on A[a,b]*B[b,c]. So T[a,b,c] = 1 for all a,b,c in {0,1}.
    
    # Strassen gives 7 rank-1 decompositions. Let me list them as (u, v, w) vectors.
    # u is in R^4 (flattened 2x2 A), v is in R^4 (flattened 2x2 B), w is in R^4 (flattened 2x2 C).
    # The contribution is u ⊗ v ⊗ w, and we need sum_l u_l ⊗ v_l ⊗ w_l = T.
    
    # Let me use the standard Strassen decomposition.
    # M1: A00*A00 + A00*A11 + A11*A00 + A11*A11
    #   u1 = e00 + e11, v1 = e00 + e11, w1 = e00 + e11 (contributes to C00 and C11)
    #   Actually w1 should be such that w1[c*2+a] picks out the right C entries.
    #   M1 contributes to C00 (positive) and C11 (positive).
    #   So w1 = e_C00 + e_C11.
    
    # Let me be more careful. The tensor T[a,b,c] for 2x2 matrix multiplication is:
    # T[a,b,c] = 1 for all a,b,c in {0,1}.
    # A rank-1 term is u ⊗ v ⊗ w where u is indexed by (a,b), v by (b,c), w by (c,a).
    # The product contributes u[a,b]*v[b,c]*w[c,a] to T[a,b,c].
    
    # Strassen's 7 terms:
    terms = []
    # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
    terms.append((
        {0: 1, 3: 1},           # u: A00, A11
        {0: 1, 3: 1},           # v: B00, B11
        {0: 1, 3: 1}            # w: C00, C11
    ))
    # M2 = (A10+A11)B00 -> contributes to C10
    terms.append((
        {1: 1, 3: 1},           # u: A10, A11
        {0: 1},                 # v: B00
        {2: 1}                  # w: C10
    ))
    # M3 = A00(B01-B11) -> contributes to C01
    terms.append((
        {0: 1},                 # u: A00
        {1: 1, 3: -1},          # v: B01, -B11
        {1: 1}                  # w: C01
    ))
    # M4 = A11(B10-B00) -> contributes to C10 and C00
    terms.append((
        {3: 1},                 # u: A11
        {2: 1, 0: -1},          # v: B10, -B00
        {2: 1, 0: -1}           # w: C10, -C00
    ))
    # M5 = (A00+A10)B11 -> contributes to C01 and -C00
    terms.append((
        {0: 1, 1: 1},           # u: A00, A10
        {3: 1},                 # v: B11
        {1: 1, 0: -1}           # w: C01, -C00
    ))
    # M6 = (A10-A00)(B00+B10) -> contributes to C11
    terms.append((
        {1: 1, 0: -1},          # u: A10, -A00
        {0: 1, 2: 1},           # v: B00, B10
        {3: 1}                  # w: C11
    ))
    # M7 = (A01-A11)(B01+B11) -> contributes to C00
    terms.append((
        {2: 1, 3: -1},          # u: A01, -A11
        {1: 1, 3: 1},           # v: B01, B11
        {0: 1}                  # w: C00
    ))
    
    return terms

def _build_decomposition():
    """Build a decomposition using Strassen blocks where applicable."""
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Strategy: Partition the 3x4x5 multiplication into blocks.
    # We can use Strassen for 2x2x2 sub-blocks.
    # 
    # Approach: Use the schoolbook method but replace 2x2x2 sub-problems with Strassen.
    # 
    # Let's think of this as: C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We can partition rows of A into {0}, {1,2} and columns of B into {0,1}, {2,3}, {4}
    # But Strassen requires square matrices. Let's use a different approach.
    #
    # Actually, let's just use the schoolbook decomposition as a baseline and try to
    # find a better one using known results.
    #
    # Known: The minimum rank for 3x4x5 matrix multiplication is at most 56.
    # The schoolbook gives 3*4*5 = 60.
    #
    # Let me try a block approach:
    # Partition A (3x4) into rows {0}, {1,2}
    # Partition B (4x5) into columns {0,1,2,3} -> but we need 2x2 blocks for Strassen
    #
    # Alternative: Just use schoolbook for now and verify it works.
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

def _build_strassen_hybrid():
    """Build a hybrid decomposition using Strassen for 2x2x2 blocks."""
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Let's try to use Strassen for the (i=1,2) x (j=0,1) x (k=0,1) block,
    # and similarly for other 2x2x2 blocks.
    #
    # Actually, let's use a cleaner approach. We'll decompose the problem into
    # overlapping 2x2x2 Strassen blocks and 1x1x1 schoolbook terms.
    #
    # For a 2x2x2 sub-problem with row indices {i1,i2}, col indices {j1,j2}, 
    # output indices {k1,k2}, the Strassen decomposition gives 7 rank-1 terms.
    #
    # Let me enumerate all possible 2x2x2 sub-blocks and try to cover the full
    # 3x4x5 tensor with a combination of Strassen blocks and schoolbook terms.
    #
    # This is complex. Let me just use the schoolbook decomposition which is
    # guaranteed to work, and try to optimize if time permits.
    
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

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    # Start with schoolbook
    result = _build_decomposition()
    
    # Verify it works
    n, m, p = DIMENSIONS
    if _verify(result['U'], result['V'], result['W'], n, m, p):
        # Try to find a better decomposition
        if time.time() < deadline - 10:
            # Attempt Strassen-based improvement
            try:
                better = _build_strassen_hybrid()
                if _verify(better['U'], better['V'], better['W'], n, m, p):
                    rank_better = len(better['U'][0])
                    rank_current = len(result['U'][0])
                    if rank_better < rank_current:
                        result = better
            except Exception:
                pass
    
    return result
# EVOLVE-BLOCK-END
