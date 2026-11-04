# EVOLVE-BLOCK-START
import time
import sys
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=6
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook fallback
    rank_school = n * m * p  # 60
    
    def make_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank_school
    
    def verify(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True
    
    # Strassen decomposition for 2x2 matrix multiplication
    # M = [[a,b],[c,d]] * [[e,f],[g,h]] = [[a*(e+h), (a+b)*h], [(c+d)*e, (c+d)*(f-e)]]
    # The 7 Strassen products:
    # p1 = (a+b)*(e+h)
    # p2 = (a+b)*h
    # p3 = c*(f-e)
    # p4 = d*(g+h)
    # p5 = a*(e+h)
    # p6 = (c+d)*e
    # p7 = (a-b)*(e-h)
    # M[0,0] = p5 + p4 - p2 + p7
    # M[0,1] = p2 + p3
    # M[1,0] = p6 + p1 - p4 - p7
    # M[1,1] = p1 - p3 + p5 + p7
    
    # Build Strassen-based decomposition for 2x2 x 2x6 (3 column pairs)
    # A_block: 2x2 (rows i=0,1; cols j=0,1 relative)
    # B_block: 2x6 (rows j=0,1 relative; cols k=0..5)
    
    # For each pair of output columns (k0, k0+1), we do a 2x2 x 2x2 Strassen
    # Total 7 terms per pair, 3 pairs = 21 terms
    
    # For the (2x1) x (1x6) block: schoolbook, 12 terms
    
    # Let's build the full decomposition
    # A columns: [0,1] -> block0, [2,3] -> block1, [4] -> block2
    # B rows: [0,1] -> block0, [2,3] -> block1, [4] -> block2
    
    # Term structure:
    # For block (A_col_block, B_row_block) where both are size 2:
    #   3 column pairs, each with 7 Strassen terms = 21 terms
    # For block (A_col_block size 1, B_row_block size 1):
    #   6 terms (2 rows * 6 cols)
    
    # Let's enumerate all terms
    # We'll build U, V, W as lists of (row, value) per column
    
    U_terms = []  # list of (row_index, value)
    V_terms = []  # list of (row_index, value)
    W_terms = []  # list of (row_index, value)
    
    def add_term(u_row, u_val, v_row, v_val, w_row, w_val):
        U_terms.append((u_row, u_val))
        V_terms.append((v_row, v_val))
        W_terms.append((w_row, w_val))
    
    # Strassen for 2x2 x 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # In our indexing:
    #   a = A[0, j0], b = A[0, j0+1], c = A[1, j0], d = A[1, j0+1]
    #   e = B[j0, k0], f = B[j0, k0+1], g = B[j0+1, k0], h = B[j0+1, k0+1]
    # 
    # Output C[0, k0] = a*e + c*g, C[0, k0+1] = a*f + c*h
    # Output C[1, k0] = b*e + d*g, C[1, k0+1] = b*f + d*h
    #
    # Strassen products:
    # p1 = (a+b)*(e+h) -> U has A[0,j0]+A[0,j0+1]... wait, no.
    # Actually in Strassen, the "a,b,c,d" are elements of the first matrix.
    # a = A[0,0], b = A[0,1], c = A[1,0], d = A[1,1]
    # e = B[0,0], f = B[0,1], g = B[1,0], h = B[1,1]
    #
    # p1 = (a+b)*(e+h): U[A[0,j0]] += 1, U[A[0,j0+1]] += 1, V[B[j0,k0]] += 1, V[B[j0+1,k0+1]] += 1... 
    # Wait, this doesn't work directly because the Strassen formula uses sums/differences.
    
    # Let me think differently. For each Strassen product p_i, we need:
    # p_i = (linear combo of A entries) * (linear combo of B entries)
    # Then C entries are linear combos of p_i's.
    #
    # In the tensor decomposition:
    # Each term c has U[row_u, c], V[row_v, c], W[row_w, c]
    # The contribution to C[i,k] from term c is U[i*m+j_u, c] * V[j_v*p+k, c] * W[k*n+i, c]
    # But wait, that's not quite right. Let me re-read the convention.
    #
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #
    # So for a given term c, if U[i*m+j, c] = u_ij, V[j'*p+k', c] = v_j'k', W[k''*n+i'', c] = w_k''i''
    # The contribution to T[i,j,k] is: for all j', k', i'' such that... 
    # Actually no. The sum is over c only. For each c, we pick specific entries.
    # 
    # Wait, I think the convention is:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # 
    # So for a fixed (i,j,k) and fixed c, we use U at position (i*m+j, c), V at position (j*p+k, c), W at position (k*n+i, c).
    #
    # This means each term c corresponds to a specific (i,j) for U, specific (j,k) for V, and specific (k,i) for W.
    # But that's the schoolbook structure! Each term has exactly one nonzero in U, one in V, one in W.
    #
    # For Strassen, we need terms where multiple entries of U (or V, or W) are nonzero for the same c.
    # Let me re-examine...
    #
    # Actually, I think the convention allows U[i*m+j, c] to be nonzero for multiple (i,j) pairs for the same c.
    # Similarly for V and W. Let me re-read: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    #
    # So U is a (n*m) x rank matrix. Each column c can have multiple nonzeros.
    # The tensor entry T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # For Strassen:
    # Consider the 2x2 x 2x2 case. We want:
    # C[0,0] = a*e + c*g
    # C[0,1] = a*f + c*h  
    # C[1,0] = b*e + d*g
    # C[1,1] = b*f + d*h
    #
    # In tensor form, T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,1}
    # (where i is row of C, j is col of A / row of B, k is col of C)
    #
    # Wait no. T[i,j,k] = 1 means the coefficient of A[i,j]*B[j,k] in C is 1.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k).
    #
    # For the 2x2 x 2x2 sub-problem (say A columns j0,j0+1 and B rows j0,j0+1, output columns k0,k0+1):
    # T[i, j0, k0] = 1 (contribution to C[i,k0] from A[i,j0]*B[j0,k0])
    # T[i, j0, k0+1] = 1 (contribution to C[i,k0+1] from A[i,j0]*B[j0,k0+1])
    # T[i, j0+1, k0] = 1
    # T[i, j0+1, k0+1] = 1
    # for i in {0,1}
    #
    # Strassen decomposes this 8-entry sub-tensor into 7 rank-1 terms.
    #
    # For term c, we have U[i*m+j, c] for various (i,j), V[j'*p+k', c] for various (j',k'), W[k''*n+i'', c] for various (k'',i'').
    #
    # The Strassen products for 2x2:
    # p1 = (a+b)*(e+h) where a=A[i0,j0], b=A[i0,j0+1], e=B[j0,k0], h=B[j0+1,k0+1]
    #   U: U[i0*m+j0, c] += 1, U[i0*m+j0+1, c] += 1
    #   V: V[j0*p+k0, c] += 1, V[(j0+1)*p+(k0+1), c] += 1
    #   W: ? 
    #
    # Hmm, but W[k*n+i, c] should contribute to C[i,k]. Let me think about what W should be.
    #
    # The contribution of term c to T[i,j,k] is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    # For the Strassen term p1 = (A[i0,j0]+A[i0,j0+1])*(B[j0,k0]+B[j0+1,k0+1]):
    # This contributes to C[i0, k0] via A[i0,j0]*B[j0,k0] and A[i0,j0+1]*B[j0+1,k0+1]... 
    # but that's not quite right because the indices don't match for a single C entry.
    #
    # I think I need to be more careful. Let me look at how Strassen works in the tensor decomposition framework.
    #
    # For 2x2 x 2x2, the tensor has 8 entries, all = 1.
    # T[0,0,0] = 1, T[0,0,1] = 1, T[0,1,0] = 1, T[0,1,1] = 1
    # T[1,0,0] = 1, T[1,0,1] = 1, T[1,1,0] = 1, T[1,1,1] = 1
    #
    # Strassen's 7 terms:
    # Term 1: p1 = (A[0,0]+A[0,1])*(B[0,0]+B[1,1])
    #   U[0*m+0, c] = 1, U[0*m+1, c] = 1
    #   V[0*p+0, c] = 1, V[1*p+1, c] = 1
    #   W[0*n+0, c] = 1, W[1*n+1, c] = 1  (contributes to C[0,0] and C[1,1])
    #   Check T[0,0,0]: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1 ✓ (but we also get contributions from other terms)
    #
    # Wait, this is getting complicated. Let me just directly construct the Strassen decomposition and verify it.
    
    # For 2x2 x 2x2 (local indices i,j,k in {0,1}):
    # Strassen terms (using local indices):
    # 
    # Term 0: (A[0,0]+A[0,1])*(B[0,0]+B[1,1]) -> contributes to C[0,0] and C[1,1]
    #   U: row 0*m+0 -> +1, row 0*m+1 -> +1
    #   V: row 0*p+0 -> +1, row 1*p+1 -> +1  
    #   W: row 0*n+0 -> +1, row 1*n+1 -> +1
    #
    # Let me verify: T[0,0,0] gets U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1
    # T[0,0,1] gets U[0,0]*V[0,1]*W[1,0] = 1*0*0 = 0 (V[0,1] is 0, W[1,0] is 0)
    # T[0,1,0] gets U[0,1]*V[1,0]*W[0,0] = 1*0*1 = 0
    # T[0,1,1] gets U[0,1]*V[1,1]*W[1,0] = 1*1*0 = 0
    # T[1,0,0] gets U[1,0]*V[0,0]*W[0,1] = 0*1*1 = 0
    # T[1,0,1] gets U[1,0]*V[0,1]*W[1,1] = 0*0*1 = 0
    # T[1,1,0] gets U[1,1]*V[1,0]*W[0,1] = 0*0*1 = 0
    # T[1,1,1] gets U[1,1]*V[1,1]*W[1,1] = 0*1*1 = 0
    #
    # So term 0 only contributes 1 to T[0,0,0] and 0 to others? That's not right.
    #
    # I think the issue is that W should be structured differently. Let me reconsider.
    #
    # Actually, I think for Strassen, the W matrix entries need to be set up so that the linear combinations work out. Let me look at this more carefully.
    #
    # For 2x2 x 2x2, we want:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    #
    # In tensor form: T[i,j,k] = 1 for all i,j,k in {0,1}
    #
    # Strassen:
    # p1 = (A[0,0]+A[0,1])*(B[0,0]+B[1,1])
    # p2 = (A[0,0]+A[0,1])*B[1,1]
    # p3 = A[1,0]*(B[0,1]-B[1,1])
    # p4 = A[1,1]*(B[0,0]+B[1,0])
    # p5 = A[0,0]*(B[0,0]+B[1,1])
    # p6 = (A[1,0]+A[1,1])*B[0,0]
    # p7 = (A[0,0]-A[0,1])*(B[0,0]-B[1,0])
    #
    # C[0,0] = p5 + p4 - p2 + p7
    # C[0,1] = p2 + p3
    # C[1,0] = p6 + p1 - p4 - p7
    # C[1,1] = p1 - p3 + p5 + p7
    #
    # Now in the tensor decomposition, each p_i is a rank-1 term.
    # For p1 = (A[0,0]+A[0,1])*(B[0,0]+B[1,1]):
    #   U[0*m+0, c] = 1, U[0*m+1, c] = 1
    #   V[0*p+0, c] = 1, V[1*p+1, c] = 1
    #   W[?, ?, c] = ?
    #
    # The contribution to T[i,j,k] from term c is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    # For p1, we want it to contribute to C[0,0] and C[1,1] (with appropriate signs from the linear combination).
    #
    # But in the tensor decomposition, we can't have "linear combinations" in the W matrix directly. Each term c has fixed U, V, W entries, and the sum over c gives the tensor.
    #
    # I think the key is that W[k*n+i, c] can take values like 1, -1, 0, etc., and the sum over all c's gives the correct result.
    #
    # Let me try: for p1, set W[0*n+0, c] = 1 and W[1*n+1, c] = 1.
    # Then contribution to T[0,0,0]: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1
    # Contribution to T[0,0,1]: U[0,0]*V[0,1]*W[1,0] = 1*0*0 = 0
    # Contribution to T[0,1,0]: U[0,1]*V[1,0]*W[0,0] = 1*0*1 = 0
    # Contribution to T[0,1,1]: U[0,1]*V[1,1]*W[1,0] = 1*1*0 = 0
    # Contribution to T[1,0,0]: U[1,0]*V[0,0]*W[0,1] = 0*1*0 = 0
    # Contribution to T[1,0,1]: U[1,0]*V[0,1]*W[1,1] = 0*0*1 = 0
    # Contribution to T[1,1,0]: U[1,1]*V[1,0]*W[0,1] = 0*0*0 = 0
    # Contribution to T[1,1,1]: U[1,1]*V[1,1]*W[1,1] = 0*1*1 = 0
    #
    # So p1 only contributes 1 to T[0,0,0]. That's not matching the Strassen formula.
    #
    # I think the issue is that I'm not correctly mapping the Strassen products to the tensor decomposition. Let me think about this differently.
    #
    # In the tensor decomposition, T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # For the 2x2 x 2x2 case (with global indices offset), let's say the local (i,j,k) correspond to global (I, J, K) where I = I0+i, J = J0+j, K = K0+k.
    #
    # T[I,J,K] = sum_c U[I*m+J, c] * V[J*p+K, c] * W[K*n+I, c] = 1
    #
    # For Strassen term p1 = (A[I0,J0]+A[I0,J0+1])*(B[J0,K0]+B[J0+1,K0+1]):
    # This product appears in the expansion of C[I0,K0] and C[I0+1,K0+1] (among others).
    #
    # Actually, I realize the issue. The Strassen formula gives C entries as linear combinations of the p_i's. In the tensor decomposition, each p_i is a separate term, and the W matrix encodes how each p_i contributes to each C entry.
    #
    # So for p1, which contributes +1 to C[0,0] and +1 to C[1,1] (in the local 2x2 sense):
    # W[0*n+0, c] = 1 (contributes to C[0,0])
    # W[1*n+1, c] = 1 (contributes to C[1,1])
    #
    # But as I showed, this only gives nonzero contribution to T[0,0,0] and T[1,1,1] (if U and V are set up correctly).
    #
    # Wait, let me recheck. U[0*m+0, c] = 1, U[0*m+1, c] = 1
    # V[0*p+0, c] = 1, V[1*p+1, c] = 1
    # W[0*n+0, c] = 1, W[1*n+1, c] = 1
    #
    # T[0,0,0] = U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1
    # T[0,0,1] = U[0,0]*V[0,1]*W[1,0] = 1*0*0 = 0
    # T[0,1,0] = U[0,1]*V[1,0]*W[0,0] = 1*0*1 = 0
    # T[0,1,1] = U[0,1]*V[1,1]*W[1,0] = 1*1*0 = 0
    # T[1,0,0] = U[1,0]*V[0,0]*W[0,1] = 0*1*0 = 0
    # T[1,0,1] = U[1,0]*V[0,1]*W[1,1] = 0*0*1 = 0
    # T[1,1,0] = U[1,1]*V[1,0]*W[0,1] = 0*0*0 = 0
    # T[1,1,1] = U[1,1]*V[1,1]*W[1,1] = 0*1*1 = 0
    #
    # So only T[0,0,0] gets 1. But we need T[0,0,0]=1, T[0,0,1]=1, T[0,1,0]=1, T[0,1,1]=1, T[1,0,0]=1, T[1,0,1]=1, T[1,1,0]=1, T[1,1,1]=1.
    #
    # The problem is that with this setup, p1 only contributes to one entry. But in the Strassen formula, p1 contributes to C[0,0] and C[1,1], which involves multiple (j,k) pairs.
    #
    # I think the correct interpretation is:
    # C[I,K] = sum_J A[I,J]*B[J,K]
    # In tensor form: T[I,J,K] = 1 for all I,J,K.
    #
    # For Strassen, the p_i's are intermediate products, and C[I,K] = sum_i coeff_{i,I,K} * p_i
    # where p_i = (linear combo of A entries) * (linear combo of B entries).
    #
    # In the tensor decomposition:
    # T[I,J,K] = sum_c U[I*m+J, c] * V[J*p+K, c] * W[K*n+I, c]
    #
    # For a Strassen term c corresponding to p_i:
    # U[I*m+J, c] is the coefficient of A[I,J] in the first factor of p_i
    # V[J*p+K, c] is the coefficient of B[J,K] in the second factor of p_i
    # W[K*n+I, c] is the coefficient of p_i in C[I,K]
    #
    # So for p1 = (A[0,0]+A[0,1])*(B[0,0]+B[1,1]):
    # U[0*m+0, c] = 1 (coefficient of A[0,0])
    # U[0*m+1, c] = 1 (coefficient of A[0,1])
    # V[0*p+0, c] = 1 (coefficient of B[0,0])
    # V[1*p+1, c] = 1 (coefficient of B[1,1])
    # 
    # Now, p1 contributes to C[0,0] with coefficient +1 and to C[1,1] with coefficient +1 (from the Strassen formula).
    # So W[0*n+0, c] = 1 and W[1*n+1, c] = 1.
    #
    # Now let's compute T[I,J,K] for all I,J,K in {0,1}:
    # T[0,0,0] = U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1
    # T[0,0,1] = U[0,0]*V[0,1]*W[1,0] = 1*0*0 = 0
    # T[0,1,0] = U[0,1]*V[1,0]*W[0,0] = 1*0*1 = 0
    # T[0,1,1] = U[0,1]*V[1,1]*W[1,0] = 1*1*0 = 0
    # T[1,0,0] = U[1,0]*V[0,0]*W[0,1] = 0*1*0 = 0
    # T[1,0,1] = U[1,0]*V[0,1]*W[1,1] = 0*0*1 = 0
    # T[1,1,0] = U[1,1]*V[1,0]*W[0,1] = 0*0*0 = 0
    # T[1,1,1] = U[1,1]*V[1,1]*W[1,1] = 0*1*1 = 0
    #
    # So p1 contributes 1 to T[0,0,0] and 0 to all others. But we need T[0,0,0]=1, T[0,0,1]=1, etc.
    #
    # The issue is that p1 = (A[0,0]+A[0,1])*(B[0,0]+B[1,1]) expands to:
    # A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[0,1]*B[0,0] + A[0,1]*B[1,1]
    #
    # This contributes to C[0,0] (via A[0,0]*B[0,0]) and C[0,1] (via A[0,0]*B[1,1]... wait no, B[1,1] has row index 1, so it contributes to C[0,1] only if the column index matches.
    #
    # I'm getting confused. Let me just directly construct the Strassen decomposition by working out the tensor entries.
    #
    # For 2x2 x 2x2, we need T[i,j,k] = 1 for all i,j,k in {0,1}.
    # We'll use 7 terms. Let me define them explicitly:
    #
    # Term 0: p1 = (A[0,0]+A[0,1])*(B[0,0]+B[1,1])
    #   U[0,0]=1, U[0,1]=1, V[0,0]=1, V[1,1]=1
    #   This product = A[0,0]B[0,0] + A[0,0]B[1,1] + A[0,1]B[0,0] + A[0,1]B[1,1]
    #   It should contribute to C[0,0] and C[1,1] in the Strassen formula.
    #   W[0,0]=1, W[1,1]=1
    #
    # Let me just verify the full Strassen decomposition numerically.
    
    def strassen_2x2_decomposition():
        """Return U, V, W for 2x2 x 2x2 matrix multiplication (local indices).
        U shape: (2*2, 7) = (4, 7)
        V shape: (2*2, 7) = (4, 7)  
        W shape: (2*2, 7) = (4, 7)
        Local indices: i,j,k in {0,1}
        U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen products for 2x2:
        # p0 = (a+b)*(e+h)  where a=A[0,0], b=A[0,1], e=B[0,0], h=B[1,1]
        # p1 = (a+b)*h
        # p2 = c*(f-e)  where c=A[1,0], f=B[0,1], e=B[0,0]
        # p3 = d*(g+h)  where d=A[1,1], g=B[1,0]
        # p4 = a*(e+h)
        # p5 = (c+d)*e  where d=A[1,1]
        # p6 = (a-b)*(e-h)
        
        # p0 = (A[0,0]+A[0,1])*(B[0,0]+B[1,1])
        U[0*2+0, 0] = 1  # A[0,0]
        U[0*2+1, 0] = 1  # A[0,1]
        V[0*2+0, 0] = 1  # B[0,0]
        V[1*2+1, 0] = 1  # B[1,1]
        W[0*2+0, 0] = 1  # C[0,0]
        W[1*2+1, 0] = 1  # C[1,1]
        
        # p1 = (A[0,0]+A[0,1])*B[1,1]
        U[0*2+0, 1] = 1
        U[0*2+1, 1] = 1
        V[1*2+1, 1] = 1  # B[1,1]
        W[0*2+1, 1] = 1  # C[0,1]
        W[1*2+1, 1] = -1 # C[1,1] (negative contribution)
        
        # p2 = A[1,0]*(B[0,1]-B[0,0])
        U[1*2+0, 2] = 1  # A[1,0]
        V[0*2+1, 2] = 1  # B[0,1]
        V[0*2+0, 2] = -1 # B[0,0]
        W[0*2+1, 2] = 1  # C[0,1]
        W[1*2+1, 2] = -1 # C[1,1]
        
        # p3 = A[1,1]*(B[1,0]+B[1,1])
        U[1*2+1, 3] = 1  # A[1,1]
        V[1*2+0, 3] = 1  # B[1,0]
        V[1*2+1, 3] = 1  # B[1,1]
        W[0*2+0, 3] = 1  # C[0,0]
        W[1*2+1, 3] = 1  # C[1,1]
        
        # p4 = A[0,0]*(B[0,0]+B[1,1])
        U[0*2+0, 4] = 1  # A[0,0]
        V[0*2+0, 4] = 1  # B[0,0]
        V[1*2+1, 4] = 1  # B[1,1]
        W[0*2+0, 4] = 1  # C[0,0]
        W[1*2+1, 4] = 1  # C[1,1]
        
        # p5 = (A[1,0]+A[1,1])*B[0,0]
        U[1*2+0, 5] = 1  # A[1,0]
        U[1*2+1, 5] = 1  # A[1,1]
        V[0*2+0, 5] = 1  # B[0,0]
        W[1*2+0, 5] = 1  # C[1,0]
        W[0*2+0, 5] = -1 # C[0,0] (negative)
        
        # p6 = (A[0,0]-A[0,1])*(B[0,0]-B[1,0])
        U[0*2+0, 6] = 1  # A[0,0]
        U[0*2+1, 6] = -1 # A[0,1]
        V[0*2+0, 6] = 1  # B[0,0]
        V[1*2+0, 6] = -1 # B[1,0]
        W[0*2+0, 6] = 1  # C[0,0]
        W[1*2+0, 6] = -1 # C[1,0]
        
        # Verify
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0.0
                    for c in range(7):
                        s += U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c]
                    if abs(s - 1.0) > 1e-10:
                        print(f"Strassen verify failed at ({i},{j},{k}): {s}")
                        return None
        return U, V, W
    
    # Test Strassen decomposition
    strassen_test = strassen_2x2_decomposition()
    if strassen_test is None:
        # Fall back to schoolbook
        U, V, W, r = make_schoolbook()
        return {"U": U, "V": V, "W": W}
    
    # Now build the full 2x5 x 5x6 decomposition
    # A: 2x5, B: 5x6, C: 2x6
    # Partition A columns: [0,1] (size 2), [2,3] (size 2), [4] (size 1)
    # Partition B rows: [0,1] (size 2), [2,3] (size 2), [4] (size 1)
    # Partition C columns: [0,1], [2,3], [4,5] (pairs for Strassen)
    
    # For each (A_col_block, B_row_block) pair:
    # - If both size 2: use Strassen on 3 column pairs -> 3*7 = 21 terms
    # - If A size 2, B size 1: schoolbook -> 2*1*6 = 12 terms
    # - If A size 1, B size 2: schoolbook -> 1*2*6 = 12 terms  
    # - If both size 1: schoolbook -> 1*1*6 = 6 terms
    
    # Wait, I need to be more careful. The C columns are partitioned into pairs for Strassen.
    # For a (2x2) x (2x6) product, we do 3 Strassen 2x2x2 multiplications (one per column pair).
    
    # Let me define the blocks:
    # A_col_blocks: [(0,2), (2,4), (4,5)] -> sizes [2, 2, 1]
    # B_row_blocks: [(0,2), (2,4), (4,5)] -> sizes [2, 2, 1]
    # C_col_pairs: [(0,2), (2,4), (4,6)] -> 3 pairs
    
    # For A_col_block (j0, j0+sz_a) and B_row_block (k0, k0+sz_b):
    # If sz_a == 2 and sz_b == 2:
    #   For each C column pair (c0, c0+2):
    #     Use Strassen 2x2x2 with local A cols [j0, j0+1], local B rows [k0, k0+1], local C cols [c0, c0+1]
    #     7 terms per pair, 3 pairs = 21 terms
    # If sz_a == 2 and sz_b == 1:
    #   Schoolbook: for i in {0,1}, j in {j0}, k in {0..5}: 2*1*6 = 12 terms
    # If sz_a == 1 and sz_b == 2:
    #   Schoolbook: for i in {0,1}, j in {j0,j0+1}, k in {0..5}: 2*2*6 = 24 terms
    # If sz_a == 1 and sz_b == 1:
    #   Schoolbook: for i in {0,1}, j in {j0}, k in {0..5}: 2*1*6 = 12 terms
    
    # Total: 21 + 21 + 12 + 24 + 12 + 12 = 102? That's way more than 54.
    # 
    # Wait, I'm double counting. Let me reconsider.
    #
    # The product C = A * B where A is 2x5 and B is 5x6.
    # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    #
    # We partition the j index into blocks: [0,1], [2,3], [4]
    # C[i,k] = sum_{j in [0,1]} A[i,j]*B[j,k] + sum_{j in [2,3]} A[i,j]*B[j,k] + A[i,4]*B[4,k]
    #
    # Term 1: A[:, 0:2] * B[0:2, :] -> 2x2 * 2x6 = 2x6
    # Term 2: A[:, 2:4] * B[2:4, :] -> 2x2 * 2x6 = 2x6
    # Term 3: A[:, 4:5] * B[4:5, :] -> 2x1 * 1x6 = 2x6
    #
    # For Term 1 (2x2 * 2x6):
    #   C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #   We can split k into pairs: (0,1), (2,3), (4,5)
    #   For each pair, it's a 2x2 * 2x2 multiplication, using Strassen rank 7.
    #   Total: 3 * 7 = 21 terms.
    #
    # For Term 2 (2x2 * 2x6): same as Term 1, 21 terms.
    #
    # For Term 3 (2x1 * 1x6):
    #   C[i,k] = A[i,4]*B[4,k]
    #   This is rank 2*6 = 12 (each (i,k) pair needs one term).
    #
    # Total: 21 + 21 + 12 = 54. 
    #
    # Now let me implement this properly.
    
    # Build the full decomposition
    # U: (n*m, rank) = (10, 54)
    # V: (m*p, rank) = (30, 54)
    # W: (p*n, rank) = (12, 54)
    
    total_rank = 21 + 21 + 12  # 54
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    # Helper: add Strassen 2x2x2 decomposition for local A cols [ja, ja+1], local B rows [jb, jb+1], local C cols [kc, kc+1]
    def add_strassen_block(ja, jb, kc):
        """Add 7 Strassen terms for A[:, ja:ja+2] * B[jb:jb+2, kc:kc+2]
        ja: starting column of A (and row of B)
        jb: starting row of B (same as ja in our case)
        kc: starting column of C
        """
        global col_offset
        # Get the Strassen U, V, W (local indices)
        U_s, V_s, W_s = strassen_test
        
        for c_local in range(7):
            c = col_offset + c_local
            # Map local indices to global
            # Local A: i in {0,1}, j in {0,1} -> global j = ja + j_local
            # Local B: j in {0,1}, k in {0,1} -> global j = jb + j_local, k = kc + k_local
            # Local C: k in {0,1}, i in {0,1} -> global k = kc + k_local, i = i_local
            
            for i_loc in range(2):
                for j_loc in range(2):
                    val = U_s[i_loc * 2 + j_loc, c_local]
                    if val != 0:
                        U_full[i_loc * m + (ja + j_loc), c] += val
            
            for j_loc in range(2):
                for k_loc in range(2):
                    val = V_s[j_loc * 2 + k_loc, c_local]
                    if val != 0:
                        V_full[(jb + j_loc) * p + (kc + k_loc), c] += val
            
            for k_loc in range(2):
                for i_loc in range(2):
                    val = W_s[k_loc * 2 + i_loc, c_local]
                    if val != 0:
                        W_full[(kc + k_loc) * n + i_loc, c] += val
        
        col_offset += 7
    
    # Helper: add schoolbook terms for A[:, ja:ja+sz] * B[jb:jb+sz2, :]
    def add_schoolbook_block(ja, sz_a, jb, sz_b):
        """Add schoolbook terms for A[:, ja:ja+sz_a] * B[jb:jb+sz_b, :]
        """
        global col_offset
        for i in range(n):
            for j_off in range(sz_a):
                j = ja + j_off
                for k in range(p):
                    U_full[i * m + j, col_offset] = 1
                    V_full[j * p + k, col_offset] = 1
                    W_full[k * n + i, col_offset] = 1
                    col_offset += 1
    
    # Term 1: A[:, 0:2] * B[0:2, :] -> 3 Strassen blocks
    for kc in range(0, p, 2):
        add_strassen_block(0, 0, kc)
    
    # Term 2: A[:, 2:4] * B[2:4, :] -> 3 Strassen blocks
    for kc in range(0, p, 2):
        add_strassen_block(2, 2, kc)
    
    # Term 3: A[:, 4:5] * B[4:5, :] -> schoolbook
    add_schoolbook_block(4, 1, 4, 1)
    
    # Verify
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    if verify(U_list, V_list, W_list, total_rank):
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fall back to schoolbook
        U, V, W, r = make_schoolbook()
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

