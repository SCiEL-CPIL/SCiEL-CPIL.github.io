"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

DEADLINE = time.time() + 115.0

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly matches matrix multiplication."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = len(U[0]) if U else 0
    
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                computed = 0
                for c in range(rank):
                    computed += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(computed - 1) > 1e-10:
                    return False
    return True

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 by 2x2 multiplication.
    Each term is (U_2x2, V_2x2, W_2x2) where W is diagonal (only (0,0) and (1,1) entries).
    The decomposition: sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = delta_{i,k} for all i,j,k
    """
    # Strassen's 7 products for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For the tensor: T[i,j,k] = 1 for all valid (i,j,k) in 2x2x2
    # We need: sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1 for all i,j,k
    
    # W_c[k,i] is non-zero only when k==i (diagonal of W in the (k,i) indexing)
    # W_c[0,0] and W_c[1,1] are the coefficients for C[0,*] and C[1,*]
    
    # Term 0: M1 = (A11+A22)(B11+B22), contributes to C11 (+1) and C22 (+1)
    # U: [[1,0],[0,1]], V: [[1,0],[0,1]], W: [[1,0],[0,1]]
    # Term 1: M2 = (A21+A22)B11, contributes to C21 (+1) and C22 (-1)
    # U: [[0,0],[1,1]], V: [[1,0],[0,0]], W: [[0,0],[0,-1]]
    # Term 2: M3 = A11(B21-B22), contributes to C12 (+1) and C22 (+1)
    # U: [[1,0],[0,0]], V: [[0,1],[0,-1]], W: [[0,1],[0,0]]
    # Term 3: M4 = A22(B12-B11), contributes to C11 (+1) and C21 (+1)
    # U: [[0,0],[0,1]], V: [[-1,1],[0,0]], W: [[1,0],[0,0]]
    # Term 4: M5 = (A11+A12)B22, contributes to C11 (-1) and C12 (+1)
    # U: [[1,1],[0,0]], V: [[0,0],[0,1]], W: [[-1,0],[0,0]]
    # Term 5: M6 = (A21-A11)(B11+B12), contributes to C22 (+1)
    # U: [[-1,0],[1,0]], V: [[1,1],[0,0]], W: [[0,0],[0,1]]
    # Term 6: M7 = (A12-A22)(B21+B22), contributes to C11 (+1)
    # U: [[0,1],[0,-1]], V: [[0,1],[0,1]], W: [[1,0],[0,0]]
    
    terms = [
        # (U, V, W_diag) where W_diag = (w00, w11)
        ([[1,0],[0,1]], [[1,0],[0,1]], (1, 1)),    # M1
        ([[0,0],[1,1]], [[1,0],[0,0]], (0, -1)),   # M2
        ([[1,0],[0,0]], [[0,1],[0,-1]], (0, 0)),   # M3 -> C12: W[0,0]=1? No...
        ([[0,0],[0,1]], [[-1,1],[0,0]], (1, 0)),   # M4
        ([[1,1],[0,0]], [[0,0],[0,1]], (-1, 0)),   # M5
        ([[-1,0],[1,0]], [[1,1],[0,0]], (0, 1)),   # M6
        ([[0,1],[0,-1]], [[0,1],[0,1]], (1, 0)),   # M7
    ]
    return terms

def _build_rank10_decomposition(n, m, p):
    """
    Build a rank-10 decomposition for 3x3 * 3x4 matrix multiplication.
    
    Block structure:
    A (3x3): rows 0-1, col 0-1 = A11 (2x2); rows 0-1, col 2 = A12 (2x1)
              row 2, cols 0-1 = A21 (1x2); row 2, col 2 = A22 (1x1)
    B (3x4): rows 0-1, all cols = B11 (2x4); row 2, all cols = B12 (1x4)
    C (3x4): rows 0-1, all cols = C11 (2x4); row 2, all cols = C21 (1x4)
    
    C11 = A11 * B11 + A12 * B12  (2x2 * 2x4 + 2x1 * 1x4)
    C21 = A21 * B11 + A22 * B12  (1x2 * 2x4 + 1x1 * 1x4)
    
    For C11: use Strassen-like 7-term decomposition
    For C21: use 3 schoolbook terms
    """
    nm = n * m  # 9
    mp = m * p  # 12
    pn = p * n  # 12
    rank = 10
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    # Strassen terms for the 2x2x2 part (for C11 = A11*B11 + A12*B12)
    # The 2x2 block A11 has indices (i,j) where i in {0,1}, j in {0,1}
    # The 2x1 block A12 has indices (i,j) where i in {0,1}, j = 2
    # The 2x4 block B11 has indices (j,k) where j in {0,1}, k in {0,1,2,3}
    # The 1x4 block B12 has indices (j,k) where j = 2, k in {0,1,2,3}
    # The 2x4 block C11 has indices (k,i) where k in {0,1,2,3}, i in {0,1}
    # W index for C11: k*n + i = k*3 + i
    
    # Strassen's 7 products adapted for 2x2 * 2x4:
    # We treat B11 as a 2x4 matrix and B12 as a 1x4 matrix
    # The "2x2" Strassen applies to the A11 (2x2) and B11 (2x4) product
    # Plus the A12 (2x1) * B12 (1x4) product
    
    # Actually, let me think more carefully.
    # C11[i,k] = sum_{j=0}^{1} A11[i,j] * B11[j,k] + A12[i] * B12[k]
    #          = sum_{j=0}^{1} A11[i,j] * B11[j,k] + A12[i] * B12[k]
    
    # This is a (2x2) * (2x4) + (2x1) * (1x4) computation
    # We can view this as a 2x3 * 3x4 problem where:
    #   A' = [A11 | A12] (2x3)
    #   B' = [B11; B12] (3x4)
    #   C11 = A' * B'
    
    # So we need a decomposition of 2x3 * 3x4 multiplication!
    # Rank of 2x3 * 3x4: schoolbook is 24, but we can do better.
    # Actually, let me use a different approach.
    
    # Let me use the direct Strassen approach on the 2x2 part and handle
    # the cross terms separately.
    
    # For C11[i,k] = sum_{j=0}^{1} A11[i,j]*B11[j,k] + A12[i]*B12[k]:
    # Use Strassen for the 2x2 * 2x4 part (7 terms)
    # Plus 2 terms for A12[i]*B12[k] (one for each i in {0,1})
    # Total for C11: 9 terms
    
    # For C21[k] = sum_{j=0}^{1} A21[j]*B11[j,k] + A22*B12[k]:
    # Use 2 schoolbook terms for A21[j]*B11[j,k] (j=0,1)
    # Plus 1 term for A22*B12[k]
    # Total for C21: 3 terms
    
    # Grand total: 9 + 3 = 12 terms. Hmm, that's more than 10.
    
    # Let me reconsider. Can we do better?
    
    # Actually, the 2x2 * 2x4 part: this is equivalent to 2x2 * 2x2 done 4 times
    # (once for each column of the 4-column result). Strassen gives 7 multiplications
    # per 2x2*2x2, so 7*4 = 28? No, that's not right.
    
    # Strassen for 2x2 * 2x4: We can apply Strassen's algorithm where B is 2x4.
    # The 7 Strassen products are:
    # M1 = (A11+A22)(B11+B22) where B11,B22 are 1x4 vectors
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # where B11, B12, B21, B22 are the 1x4 row blocks of B (2x4).
    
    # Each M_c is a 1x4 vector (linear combination of A entries times B rows)
    # So each term c gives: U_c (2x2 matrix), V_c (2x4 matrix), and W_c contributes to C
    
    # For the tensor decomposition:
    # Term c: U_c[i,j] * V_c[j,k] * W_c[k,i] for i in {0,1}, j in {0,1}, k in {0,1,2,3}
    # W_c[k,i] is non-zero only for specific (k,i) pairs.
    
    # Let me define the 7 Strassen terms for 2x2 * 2x4:
    # B is 2x4, so B[j,k] for j in {0,1}, k in {0,1,2,3}
    # B11 = B[0,:] (row 0), B21 = B[1,:] (row 1) -- wait, I need to be careful with indexing
    
    # Let me use standard Strassen notation:
    # A = [[A11, A12], [A21, A22]] (2x2)
    # B = [[B11, B12], [B21, B22]] where each "element" is a 1x4 vector
    # So B11 = B[0,0:4], B12 = B[0,4:8]... no, B is 2x4 so:
    # B[0,0], B[0,1], B[0,2], B[0,3] are the entries of row 0
    # B[1,0], B[1,1], B[1,2], B[1,3] are the entries of row 1
    
    # For Strassen on 2x2 * 2x4, we treat B's columns as vectors:
    # Actually, Strassen works on the matrix structure. For 2x2 * 2x4:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We can group B's columns: B = [b0, b1, b2, b3] where bj is a 2x1 column
    # C = [c0, c1, c2, c3] where cj = A * bj (2x1 result)
    # Each cj = A * bj is a 2x2 * 2x1 multiplication = 2x2 * (2x1)
    
    # For 2x2 * 2x1: C = A * b where b is 2x1
    # C1 = A11*b1 + A12*b2
    # C2 = A21*b1 + A22*b2
    # This is a 2x2 * 2x1 problem, which has rank... 
    # Schoolbook: 4 terms. Can we do better?
    # Actually, 2x2 * 2x1 is equivalent to 2x2 * 2x2 with the second column of B being zero.
    # The rank of 2x2 * 2x1 tensor is 4 (it's the same as 2x2 * 2x2 restricted).
    
    # Hmm, this approach is getting complicated. Let me just use the schoolbook for the
    # cross terms and Strassen for the main block.
    
    # New plan:
    # C11 (2x4) = A11 (2x2) * B11 (2x4) + A12 (2x1) * B12 (1x4)
    # Use 7 Strassen terms for A11 * B11 (treating B11 as 2x4)
    # Use 2 terms for A12 * B12: term for i=0 (A12[0]*B12[k]) and i=1 (A12[1]*B12[k])
    # Total for C11: 9 terms
    
    # C21 (1x4) = A21 (1x2) * B11 (2x4) + A22 (1x1) * B12 (1x4)
    # Use 2 schoolbook terms for A21 * B11: j=0 and j=1
    # Use 1 term for A22 * B12
    # Total for C21: 3 terms
    
    # Grand total: 9 + 3 = 12 terms
    
    # Let me verify if 12 is better than 36. Yes! 12 < 36.
    
    # But can we do even better? Let me check if we can reduce the C11 part.
    # A11 (2x2) * B11 (2x4): The rank of this tensor is the same as 2x2 * 2x2 times 4
    # (since B11 has 4 columns, each column gives a 2x2 * 2x1 product).
    # Actually, the tensor for 2x2 * 2x4 has rank equal to 4 * rank(2x2 * 2x1).
    # rank(2x2 * 2x1) = 4 (schoolbook, and it's known to be minimal for this size).
    # So rank(2x2 * 2x4) = 16? That can't be right because Strassen gives 7 per 2x2*2x2.
    
    # I think I'm confusing myself. Let me just implement the 12-term decomposition
    # and verify it.
    
    # 7 Strassen terms for A11 (2x2) * B11 (2x4):
    # B11[j,k] for j in {0,1}, k in {0,1,2,3}
    # In the global indexing: B index = j*p + k = j*4 + k
    
    strassen_terms = [
        # (U_2x2, V_2x4_as_dict, W_indices)
        # Term 0: M1 = (A11+A22)(B11+B22)
        # U: [[1,0],[0,1]], V: row0 = B[0,:], row1 = B[1,:] -> V[j,k] = B[j,k] for j in {0,1}
        # W: C[0,:] += M1, C[1,:] += M1 -> W[k,0] += 1, W[k,1] += 1 for all k
        ([[1,0],[0,1]], {(0,0):1,(0,1):1,(0,2):1,(0,3):1,(1,0):1,(1,1):1,(1,2):1,(1,3):1},
         {0:1, 1:1, 2:1, 3:1, 3:1, 4:1, 5:1, 6:1}),  # W[k*3+i]: k=0,i=0 -> 0; k=0,i=1 -> 1; etc.
        # Term 1: M2 = (A21+A22)B11_row0
        # U: [[0,0],[1,1]], V: row0 = B[0,:], row1 = 0
        # W: C[1,:] += M2 -> W[k*3+1] += 1
        ([[0,0],[1,1]], {(0,0):1,(0,1):1,(0,2):1,(0,3):1},
         {1:1, 4:1, 7:1, 10:1}),
        # Term 2: M3 = A11(B21-B22) where B21=B[1,:], B22=0 (since B is 2x4, B22 doesn't exist in this form)
        # Wait, I need to reconsider. For 2x2 * 2x4, the Strassen decomposition is:
        # We split A into 4 scalars and B into 4 row-vectors (each 1x4).
        # A = [[A11, A12], [A21, A22]]
        # B = [B_row0; B_row1] where B_row0 is 1x4, B_row1 is 1x4
        # C = [A11*B_row0 + A12*B_row1; A21*B_row0 + A22*B_row1]
        # This is just schoolbook with 4 terms! Not Strassen.
        
        # I was confusing myself. Strassen applies to 2x2 * 2x2, not 2x2 * 2x4.
        # For 2x2 * 2x4, the natural decomposition is 4 terms (schoolbook).
        # To use Strassen, we'd need to split the 4 columns into groups.
        
        # Let me restart with a cleaner approach.
    ]
    
    # OK, let me use a completely different strategy.
    # For 3x3 * 3x4, let me use the following block decomposition:
    # 
    # A (3x3) = [A_top (2x3); A_bottom (1x3)]
    # B (3x4) = [B_left (3x2) | B_right (3x2)]
    # C (3x4) = [C_top_left (2x2) | C_top_right (2x2)]
    #           [C_bottom_left (1x2) | C_bottom_right (1x2)]
    #
    # C_top_left = A_top * B_left (2x3 * 3x2)
    # C_top_right = A_top * B_right (2x3 * 3x2)
    # C_bottom_left = A_bottom * B_left (1x3 * 3x2)
    # C_bottom_right = A_bottom * B_right (1x3 * 3x2)
    #
    # For each 2x3 * 3x2: rank is at most 12 (schoolbook), but can be lower.
    # For each 1x3 * 3x2: rank is at most 6 (schoolbook).
    #
    # Total schoolbook: 12*2 + 6*2 = 36. Same as before.
    
    # Let me try yet another approach: use Strassen on a 2x2 sub-block.
    # 
    # A (3x3): rows 0-1, cols 0-1 form a 2x2 block A11.
    # B (3x4): rows 0-1, cols 0-1 form a 2x2 block B11.
    # 
    # C[0:2, 0:2] gets contributions from A[0:2, 0:2] * B[0:2, 0:2] (the 2x2*2x2 part)
    # plus cross terms.
    
    # I think the cleanest approach is:
    # 1. Use Strassen (7 terms) for the 2x2 * 2x2 core (rows 0-1, cols 0-1 of A and B)
    # 2. Use schoolbook for all other terms
    # 
    # The 2x2*2x2 core handles: A[i,j]*B[j,k] for i,j,k in {0,1}
    # That's 8 schoolbook terms replaced by 7 Strassen terms.
    # Remaining schoolbook terms: 36 - 8 = 28
    # Total: 7 + 28 = 35. Only saves 1 term.
    
    # Better: use Strassen for multiple 2x2 blocks.
    # 
    # Let me think about this differently. The total number of (i,j,k) triples is 36.
    # Each Strassen term replaces 4 schoolbook terms with 1 term (saving 3).
    # If we can find 8 disjoint 2x2*2x2 sub-problems, we save 8*3 = 24 terms.
    # 36 - 24 = 12 terms. But we can't find 8 disjoint 2x2*2x2 sub-problems in 3x3*3x4.
    
    # Let me just implement the 12-term decomposition I described earlier:
    # 9 terms for C11 (2x4) = A11(2x2)*B11(2x4) + A12(2x1)*B12(1x4)
    # 3 terms for C21 (1x4) = A21(1x2)*B11(2x4) + A22(1x1)*B12(1x4)
    
    # For A11(2x2)*B11(2x4): 4 schoolbook terms
    # For A12(2x1)*B12(1x4): 2 schoolbook terms (one for each i in {0,1})
    # For A21(1x2)*B11(2x4): 2 schoolbook terms (one for each j in {0,1})
    # For A22(1x1)*B12(1x4): 1 schoolbook term
    # Total: 4 + 2 + 2 + 1 = 9 terms for the "extra" parts
    # Plus 7 Strassen terms for A11*B11? No, I already counted A11*B11 as 4 terms.
    
    # I'm going in circles. Let me just implement a simple improvement:
    # Use Strassen on the top-left 2x2 * 2x2 block, and schoolbook for the rest.
    
    # Strassen terms for A[0:2,0:2] * B[0:2,0:2] contributing to C[0:2,0:2]:
    # 7 terms replacing 8 schoolbook terms.
    # Remaining schoolbook: 36 - 8 = 28 terms.
    # Total: 7 + 28 = 35 terms.
    
    # That's an improvement from 36 to 35!
    
    # Let me implement this.
    
    U = [[0] * 35 for _ in range(nm)]
    V = [[0] * 35 for _ in range(mp)]
    W = [[0] * 35 for _ in range(pn)]
    
    # 7 Strassen terms for the 2x2*2x2 block (i,j in {0,1} for A, j,k in {0,1} for B, contributing to C[i,k] for i,k in {0,1})
    # Strassen:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # M3 = A00(B10-B11) -> C01 += M3, C11 += M3
    # M4 = A11(B01-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    
    # Term 0: M1
    U[0, 0] = 1; U[1, 0] = 1  # A00, A11 (indices: i*m+j = 0*3+0=0, 1*3+1=4)
    # Wait, U index is i*m+j. For i=0,j=0: 0. For i=1,j=1: 1*3+1=4.
    U = [[0] * 35 for _ in range(nm)]
    V = [[0] * 35 for _ in range(mp)]
    W = [[0] * 35 for _ in range(pn)]
    
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # Term 0: M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # U: A[0,0] and A[1,1] -> U[0,0]=1, U[4,0]=1
    # V: B[0,0] and B[1,1] -> V[0,0]=1, V[5,0]=1 (j*p+k: 0*4+0=0, 1*4+1=5)
    # W: C[0,0] and C[1,1] -> W[0,0]=1, W[4,0]=1 (k*n+i: 0*3+0=0, 1*3+1=4)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[5, 0] = 1
    W[0, 0] = 1; W[4, 0] = 1
    
    # Term 1: M2 = (A[1,0]+A[1,1]) * B[0,0]
    # U: A[1,0] and A[1,1] -> U[3,1]=1, U[4,1]=1
    # V: B[0,0] -> V[0,1]=1
    # W: C[1,0] += M2, C[1,1] -= M2 -> W[3,1]=1, W[4,1]=-1
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[3, 1] = 1; W[4, 1] = -1
    
    # Term 2: M3 = A[0,0] * (B[1,0]-B[1,1])
    # U: A[0,0] -> U[0,2]=1
    # V: B[1,0] and B[1,1] -> V[4,2]=1, V[5,2]=-1
    # W: C[0,1] += M3, C[1,1] += M3 -> W[1,2]=1, W[4,2]=1
    U[0, 2] = 1
    V[4, 2] = 1; V[5, 2] = -1
    W[1, 2] = 1; W[4, 2] = 1
    
    # Term 3: M4 = A[1,1] * (B[0,1]-B[0,0])
    # U: A[1,1] -> U[4,3]=1
    # V: B[0,1] and B[0,0] -> V[1,3]=1, V[0,3]=-1
    # W: C[0,0] += M4, C[1,0] += M4 -> W[0,3]=1, W[3,3]=1
    U[4, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[3, 3] = 1
    
    # Term 4: M5 = (A[0,0]+A[0,1]) * B[1,1]
    # U: A[0,0] and A[0,1] -> U[0,4]=1, U[1,4]=1
    # V: B[1,1] -> V[5,4]=1
    # W: C[0,0] -= M5, C[0,1] += M5 -> W[0,4]=-1, W[1,4]=1
    U[0, 4] = 1; U[1, 4] = 1
    V[5, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Term 5: M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # U: A[1,0] and A[0,0] -> U[3,5]=1, U[0,5]=-1
    # V: B[0,0] and B[0,1] -> V[0,5]=1, V[1,5]=1
    # W: C[1,1] += M6 -> W[4,5]=1
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[4, 5] = 1
    
    # Term 6: M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    # U: A[0,1] and A[1,1] -> U[1,6]=1, U[4,6]=-1
    # V: B[1,0] and B[1,1] -> V[4,6]=1, V[5,6]=1
    # W: C[0,0] += M7 -> W[0,6]=1
    U[1, 6] = 1; U[4, 6] = -1
    V[4, 6] = 1; V[5, 6] = 1
    W[0, 6] = 1
    
    # Now the remaining 28 schoolbook terms:
    # All (i,j,k) triples except those with i,j,k all in {0,1}
    # That is, exclude: (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
    
    col = 7
    for i in range(3):
        for j in range(3):
            for k in range(4):
                if i < 2 and j < 2 and k < 2:
                    continue  # Already handled by Strassen
                U[i * 3 + j, col] = 1
                V[j * 4 + k, col] = 1
                W[k * 3 + i, col] = 1
                col += 1
    
    # col should be 7 + 28 = 35
    assert col == 35, f"Expected 35 terms, got {col}"
    
    return U, V, W

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Try rank 35 (Strassen on 2x2 block + schoolbook for rest)
    U, V, W = _build_rank10_decomposition(n, m, p)
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    nm = n * m
    mp = m * p
    pn = p * n
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
