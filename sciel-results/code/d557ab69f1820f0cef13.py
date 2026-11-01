"""Combined program for 3x3 x 3x4 matrix multiplication tensor decomposition."""
import time
import random
import sys
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n=3, m=3, p=4, tol=1e-9):
    """Verify rank-r decomposition: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if U_arr.shape != (n * m, rank) or V_arr.shape != (m * p, rank) or W_arr.shape != (p * n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n=3, m=3, p=4):
    """Schoolbook rank n*m*p decomposition."""
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

def _strassen_2x2x2():
    """
    Return rank-7 Strassen decomposition for 2x2 * 2x2.
    U rows indexed by i*2+j, V rows by j*2+k, W rows by k*2+i
    """
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # For each product Mc, we need U[i,j], V[j,k], W[k,i] coefficients
    # such that sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all (i,j,k)
    
    # Let's build the 7 terms:
    # Term 1: M1 = (A00+A11)(B00+B11)
    #   U: A00->1, A11->1 => U[0,0]=1, U[3,0]=1 (i*m+j: i=0,j=0 => 0; i=1,j=1 => 3)
    #   V: B00->1, B11->1 => V[0,0]=1, V[3,0]=1 (j*p+k: j=0,k=0 => 0; j=1,k=1 => 3)
    #   W: contributes to C00 (+1), C11 (+1) => W[0,0]=1 (k*n+i: k=0,i=0 => 0), W[3,0]=1 (k=1,i=1 => 3)
    
    # Term 2: M2 = (A10+A11)B00
    #   U: A10->1, A11->1 => U[2,1]=1, U[3,1]=1 (i=1,j=0 => 2; i=1,j=1 => 3)
    #   V: B00->1 => V[0,1]=1
    #   W: contributes to C10 (+1), C11 (-1) => W[2,1]=1 (k=0,i=1 => 2), W[3,1]=-1 (k=1,i=1 => 3)
    
    # Term 3: M3 = A00(B10-B11)
    #   U: A00->1 => U[0,2]=1
    #   V: B10->1, B11->-1 => V[2,2]=1, V[3,2]=-1 (j=1,k=0 => 2; j=1,k=1 => 3)
    #   W: contributes to C01 (+1), C11 (+1) => W[1,2]=1 (k=0,i=0 => 1), W[3,2]=1 (k=1,i=1 => 3)
    
    # Term 4: M4 = A11(B01-B00)
    #   U: A11->1 => U[3,3]=1
    #   V: B01->1, B00->-1 => V[1,3]=1, V[0,3]=-1 (j=0,k=1 => 1; j=0,k=0 => 0)
    #   W: contributes to C00 (+1), C10 (+1) => W[0,3]=1, W[2,3]=1
    
    # Term 5: M5 = (A00+A01)B11
    #   U: A00->1, A01->1 => U[0,4]=1, U[1,4]=1 (i=0,j=0 => 0; i=0,j=1 => 1)
    #   V: B11->1 => V[3,4]=1
    #   W: contributes to C00 (-1), C01 (+1) => W[0,4]=-1, W[1,4]=1
    
    # Term 6: M6 = (A10-A00)(B00+B10)
    #   U: A10->1, A00->-1 => U[2,5]=1, U[0,5]=-1
    #   V: B00->1, B10->1 => V[0,5]=1, V[2,5]=1
    #   W: contributes to C10 (+1) => W[2,5]=1
    
    # Term 7: M7 = (A01-A11)(B01+B11)
    #   U: A01->1, A11->-1 => U[1,6]=1, U[3,6]=-1
    #   V: B01->1, B11->1 => V[1,6]=1, V[3,6]=1
    #   W: contributes to C00 (-1), C11 (+1) => W[0,6]=-1, W[3,6]=1
    
    rank = 7
    U = [[0]*rank for _ in range(4)]  # 2*2 = 4
    V = [[0]*rank for _ in range(4)]  # 2*2 = 4
    W = [[0]*rank for _ in range(4)]  # 2*2 = 4
    
    # Term 0 (M1)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 1 (M2)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # Term 2 (M3)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # Term 3 (M4)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # Term 4 (M5)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Term 5 (M6)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[2][5] = 1
    
    # Term 6 (M7)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = 1
    
    return {"U": U, "V": V, "W": W}

def _combine_blocks(block1, block2, n1, m1, p1, n2, m2, p2, n_total, m_total, p_total):
    """
    Combine two block decompositions into a full decomposition.
    block1: for (n1 x m1) * (m1 x p1)
    block2: for (n2 x m2) * (m2 x p2)
    The full problem is (n_total x m_total) * (m_total x p_total)
    where n_total = n1+n2, m_total = m1+m2, p_total = p1+p2
    """
    r1 = len(block1['U'][0])
    r2 = len(block2['U'][0])
    rank = r1 + r2
    
    n, m, p = n_total, m_total, p_total
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Block 1 occupies rows 0..n1-1 in A, cols 0..p1-1 in B
    # U indices: i*m + j where i in [0,n1), j in [0,m1)
    # V indices: j*p + k where j in [0,m1), k in [0,p1)
    # W indices: k*n + i where k in [0,p1), i in [0,n1)
    
    for i in range(n1):
        for j in range(m1):
            for c in range(r1):
                U[i*m + j][c] = block1['U'][i*m1 + j][c]
    
    for j in range(m1):
        for k in range(p1):
            for c in range(r1):
                V[j*p + k][c] = block1['V'][j*p1 + k][c]
    
    for k in range(p1):
        for i in range(n1):
            for c in range(r1):
                W[k*n + i][c] = block1['W'][k*n1 + i][c]
    
    # Block 2: offset
    # A rows n1..n_total-1, B cols p1..p_total-1
    # U indices: i*m + j where i in [n1, n_total), j in [m1, m_total)
    # V indices: j*p + k where j in [m1, m_total), k in [p1, p_total)
    # W indices: k*n + i where k in [p1, p_total), i in [n1, n_total)
    
    for i in range(n2):
        for j in range(m2):
            for c in range(r2):
                U[(n1+i)*m + (m1+j)][r1+c] = block2['U'][i*m2 + j][c]
    
    for j in range(m2):
        for k in range(p2):
            for c in range(r2):
                V[(m1+j)*p + (p1+k)][r1+c] = block2['V'][j*p2 + k][c]
    
    for k in range(p2):
        for i in range(n2):
            for c in range(r2):
                W[(p1+k)*n + (n1+i)][r1+c] = block2['W'][k*n2 + i][c]
    
    return {"U": U, "V": V, "W": W}

def _cross_block_decomposition(n1, m1, p1, n2, m2, p2):
    """
    Create cross terms needed when combining blocks.
    For the full product, we also need:
    - A rows 0..n1-1 * B cols p1..p_total-1 -> C rows 0..n1-1
    - A rows n1..n_total-1 * B cols 0..p1-1 -> C rows n1..n_total-1
    
    These are handled by the schoolbook terms for the cross blocks.
    """
    # Cross block 1: (n1 x m1) * (m1 x p2) -> contributes to C[0:n1, p1:p_total]
    # Cross block 2: (n2 x m2) * (m2 x p1) -> contributes to C[n1:n_total, 0:p1]
    
    # But wait, in the standard block decomposition, if we have:
    # A = [[A11, A12], [A21, A22]] where A11 is n1xm1, A12 is n1xm2, A21 is n2xm1, A22 is n2xm2
    # B = [[B11, B12], [B21, B22]] where B11 is m1xp1, B12 is m1xp2, B21 is m2xp1, B22 is m2xp2
    # C = [[A11*B11 + A12*B21, A11*B12 + A12*B22], [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    # So we need 4 block multiplications, not 2!
    # This means we need decompositions for all 4 sub-blocks.
    
    pass

def _try_strassen_hybrid(n=3, m=3, p=4):
    """
    Try to use Strassen for a 2x2 sub-block and schoolbook for the rest.
    Split 3x3 x 3x4 as:
    - Top-left 2x2 x 2x2 block: use Strassen (rank 7)
    - Remaining terms: schoolbook
    
    Actually, let's think about this more carefully.
    We have A (3x3), B (3x4), C (3x4).
    
    Split A into:
    A = [[A_top (2x2), A_right (2x1)],
         [A_bottom (1x2), A_corner (1x1)]]
    
    B into:
    B = [[B_top (2x2), B_top_right (2x2)],
         [B_bottom (1x2), B_bottom_right (1x2)]]
    
    This is getting complex. Let's try a simpler approach.
    
    Use Strassen for the 2x2x2 sub-problem (top-left 2x2 of A, top-left 2x2 of B, 
    top-left 2x2 of C) and schoolbook for everything else.
    """
    n, m, p = 3, 3, 4
    
    # Strassen handles: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # This is a 2x2 x 2x2 problem with rank 7.
    
    # The remaining terms are:
    # 1. A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]  (2x2 x 2x2, rank 4 schoolbook)
    # 2. A[0:2,2:3] * B[0:2,0:2] -> C[0:2,0:2]  (2x1 x 2x2, rank 4 schoolbook)
    # 3. A[0:2,2:3] * B[0:2,2:4] -> C[0:2,2:4]  (2x1 x 2x2, rank 4 schoolbook)
    # 4. A[0:2,2:3] * B[2:3,0:2] -> C[0:2,0:2]  (2x1 x 1x2, rank 2 schoolbook)
    # 5. A[0:2,2:3] * B[2:3,2:4] -> C[0:2,2:4]  (2x1 x 1x2, rank 2 schoolbook)
    # 6. A[2:3,0:2] * B[0:2,0:2] -> C[2:3,0:2]  (1x2 x 2x2, rank 4 schoolbook)
    # 7. A[2:3,0:2] * B[0:2,2:4] -> C[2:3,2:4]  (1x2 x 2x2, rank 4 schoolbook)
    # 8. A[2:3,0:2] * B[2:3,0:2] -> C[2:3,0:2]  (1x2 x 1x2, rank 2 schoolbook)
    # 9. A[2:3,0:2] * B[2:3,2:4] -> C[2:3,2:4]  (1x2 x 1x2, rank 2 schoolbook)
    # 10. A[2:3,2:3] * B[0:2,0:2] -> C[2:3,0:2]  (1x1 x 2x2, rank 2 schoolbook)
    # 11. A[2:3,2:3] * B[0:2,2:4] -> C[2:3,2:4]  (1x1 x 2x2, rank 2 schoolbook)
    # 12. A[2:3,2:3] * B[2:3,0:2] -> C[2:3,0:2]  (1x1 x 1x2, rank 1 schoolbook)
    # 13. A[2:3,2:3] * B[2:3,2:4] -> C[2:3,2:4]  (1x1 x 1x2, rank 1 schoolbook)
    
    # Total schoolbook terms: 4+4+4+2+2+4+4+2+2+2+2+1+1 = 34
    # Plus Strassen: 7
    # Total: 41, which is worse than schoolbook (36).
    
    # This approach doesn't help. Let's try a different strategy.
    return None

def _try_direct_construction(n=3, m=3, p=4, target_rank=None):
    """
    Try to directly construct a lower-rank decomposition.
    For 3x3 x 3x3, the best known rank is 23 (Lenstra).
    For 3x3 x 3x4, we can potentially do better than 36.
    
    One approach: use the 3x3x3 rank-23 decomposition and extend it.
    But implementing Lenstra's decomposition is complex.
    
    Alternative: try randomized search with local optimization.
    """
    if target_rank is None:
        target_rank = 35
    
    # This is a placeholder for a more sophisticated search.
    # For now, return None.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify schoolbook
    if not _verify_decomposition(best['U'], best['V'], best['W'], n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try various approaches to find lower-rank decompositions
    approaches = [
        ('strassen_hybrid', lambda: _try_strassen_hybrid(n, m, p)),
        ('direct_35', lambda: _try_direct_construction(n, m, p, 35)),
        ('direct_34', lambda: _try_direct_construction(n, m, p, 34)),
        ('direct_33', lambda: _try_direct_construction(n, m, p, 33)),
    ]
    
    for name, func in approaches:
        if time.time() > deadline - 10:
            break
        try:
            result = func()
            if result is not None:
                r = len(result['U'][0])
                if r < best_rank and _verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
                    best = result
                    best_rank = r
        except Exception as e:
            pass
    
    return best

# EVOLVE-BLOCK-END
