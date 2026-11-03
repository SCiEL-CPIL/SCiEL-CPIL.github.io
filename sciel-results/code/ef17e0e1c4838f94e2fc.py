"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Use Laderman's rank-7 decomposition for 2x2x2 matrix multiplication.
    # Laderman terms (P factors, Q factors, R factors) where indices are:
    # P: 0=P00, 1=P01, 2=P10, 3=P11
    # Q: 0=Q00, 1=Q01, 2=Q10, 3=Q11
    # R: 0=R00, 1=R01, 2=R10, 3=R11
    laderman_terms = [
        # T1
        ([(0,1)], [(0,1)], [(0,1)]),
        # T2
        ([(1,1)], [(3,1)], [(1,1)]),
        # T3
        ([(2,1)], [(2,1)], [(2,1)]),
        # T4
        ([(3,1)], [(1,1)], [(3,1)]),
        # T5
        ([(0,1),(1,1)], [(2,1),(3,1)], [(0,1),(3,1)]),
        # T6
        ([(2,1),(3,1)], [(0,1),(1,1)], [(2,1),(3,1)]),
        # T7
        ([(0,1),(2,1)], [(1,1),(3,1)], [(1,1),(2,1)]),
    ]
    
    # We need to decompose 2x4 * 4x6 -> 2x6
    # Strategy: Use 3 overlapping 2x2x2 blocks along the middle dimension.
    # Block k covers rows 2k..2k+1 of B and columns 2k..2k+1 of B (for k=0,1,2)
    # But we need to be careful about overlaps.
    
    # Alternative: Use 2 blocks for the 4-row middle, giving 2*3*7=42 (current).
    # To get rank 30, we need a smarter construction.
    
    # Let's try: split B's 4 rows into two groups of 2, and A's 4 columns into two groups of 2.
    # C[i,j] = sum_{k=0}^{3} A[i,k] * B[k,j]
    # = sum_{k=0}^{1} A[i,k] * B[k,j] + sum_{k=2}^{3} A[i,k] * B[k,j]
    # First sum: 2x2 * 2x6 -> can be done with 3 Laderman terms (one per 2x2 column block of B)
    # Second sum: 2x2 * 2x6 -> same, 3 Laderman terms
    # Total: 6 Laderman terms = 42. Same as before.
    
    # To reduce rank, we need to share some of the Laderman terms.
    # Idea: Use a rank-30 decomposition by exploiting that n=2 is small.
    
    # For n=2, the multiplication tensor T_{i,j,k} = delta_{i,k} has rank at most 2*m*p... 
    # but we can do better.
    
    # Let me try a direct rank-30 construction using the fact that we only need 2 rows of output.
    
    # Actually, let's use the known result: rank of 2x4x6 multiplication.
    # The trivial rank is 2*4*6/4 = 12... no, trivial is n*m*p = 48.
    # With Strassen on 2x2 blocks: 42.
    
    # Let me try to construct rank 30 directly.
    
    # C[i,j] = sum_k A[i,k] B[k,j] for i in {0,1}, j in {0,...,5}, k in {0,...,3}
    
    # We can write this as:
    # C[0,0] = A00*B00 + A01*B10 + A02*B20 + A03*B30
    # C[0,1] = A00*B01 + A01*B11 + A02*B21 + A03*B31
    # etc.
    
    # Group terms:
    # For each pair (i, j), we have 4 terms.
    
    # Let's use 30 rank-1 terms. Each term is U_col * V_col * W_col.
    
    # I'll use a systematic approach: for each of the 30 terms, assign coefficients.
    
    rank = 30
    U = np.zeros((n*m, rank))  # 8 x 30
    V = np.zeros((m*p, rank))  # 24 x 30
    W = np.zeros((p*n, rank))  # 12 x 30
    
    # U indices: i*4 + k for i in {0,1}, k in {0,1,2,3}
    # V indices: k*6 + j for k in {0,1,2,3}, j in {0,...,5}
    # W indices: j*2 + i for j in {0,...,5}, i in {0,1}
    
    # We'll use the Laderman decomposition in a clever way.
    # Split the 4 middle indices into 2 groups: {0,1} and {2,3}.
    # For group {0,1}: A[:,0:2] * B[0:2,:] -> 2x2 * 2x6
    # For group {2,3}: A[:,2:4] * B[2:4,:] -> 2x2 * 2x6
    
    # Each 2x2 * 2x6 can be decomposed using 3 Laderman terms (one per 2-column block of B).
    # But we can potentially share some terms between the two groups.
    
    # Let me just use 6 Laderman terms = 42, but try to find a rank-30 by optimization.
    
    # Actually, let me try a different approach: use rank 30 with explicit construction.
    
    # Term structure: We'll have 30 terms. For each term t, we define:
    # u_t: 8-vector, v_t: 24-vector, w_t: 12-vector
    # such that for all i,j: sum_t u_t[i*4+k] * v_t[k*6+j] * w_t[j*2+i] = C[i,j]
    # Wait, that's not right. The tensor is T[i,j,k] = delta_{i,k}... no.
    
    # The matrix multiplication tensor for A(2x4) * B(4x6) = C(2x6) is:
    # T[i, j, k] where i is row of A/C, j is col of B/C, k is middle index.
    # T[i,j,k] = 1 if we're computing C[i,j] from A[i,k]*B[k,j]
    
    # The decomposition requires:
    # sum_t U[i,k, t] * V[k,j, t] * W[i,j, t] = delta_{i,i} * delta_{k,k}... 
    # No. The standard form is:
    # C[i,j] = sum_k A[i,k] B[k,j] = sum_t (sum_k U[k,i,t] A[i,k]) * (sum_j V[j,k,t] B[k,j]) * W[i,j,t]
    
    # Wait, the interface says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # U index: i*m + k? or k*n + i? Let me re-read.
    # "U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "output factor follows the official trace convention W[k*n+i,r]"
    
    # So W[r, t] where r = k*n + i, meaning W is indexed by (j, i) with j in {0..p-1}, i in {0..n-1}.
    # W index = j*n + i.
    
    # For the decomposition to work:
    # C[i,j] = sum_t U[i*m + k, t] * V[k*p + j, t] * W[j*n + i, t] ... summed over k?
    
    # No, that's not right either. The standard tensor decomposition for matrix mult is:
    # T_{i,j,k} = sum_t U_{i,k}^{(t)} V_{k,j}^{(t)} W_{i,j}^{(t)}
    # where T_{i,j,k} = 1 if i==i and k==k (it's the identity in the middle).
    
    # Actually, the multiplication tensor is T[i, j, k] = delta_{i, i} * delta_{k, k} which doesn't make sense.
    
    # Let me think again. The tensor for multiplying an n×m matrix by an m×p matrix is:
    # T ∈ R^{n×p×m} where T[i, j, k] = 1 if we're using A[i,k] and B[k,j].
    # So T[i,j,k] = 1 for all valid i,j,k (it's just the structure of the multiplication).
    
    # The decomposition is: T[i,j,k] = sum_t U[i*m+k, t] * V[k*p+j, t] * W[j*n+i, t]
    # Wait, that would require summing over t, and the product U*V*W should give the "1" for each (i,j,k).
    
    # I think the correct interpretation is:
    # The bilinear form is: C[i,j] = sum_k A[i,k] B[k,j]
    # = sum_k sum_t U[i*m+k, t] * A[i,k] * V[k*p+j, t] * B[k,j] * W[j*n+i, t]
    # This must equal sum_k A[i,k] B[k,j] for all A, B.
    
    # So we need: sum_t U[i*m+k, t] * V[k*p+j, t] * W[j*n+i, t] = 1 for all i,j,k.
    
    # That's the constraint. Let me verify with the Strassen code above.
    
    # In the Strassen code, for a given term t (col_idx), and given i, k, j:
    # U[i*4+k, t] * V[k*6+j, t] * W[j*2+i, t] should sum to 1 over t.
    
    # Let me verify with a simple case. For source=0, c=0, t=0 (M1 of Strassen):
    # U factors: [(0,1), (3,1)] -> U[0*4+0, t]=1, U[0*4+3, t]=1 (for local indices 0 and 3)
    # Wait, local_idx 0 -> pi=0, pj=0 -> u_row = 0*4 + (0+0) = 0
    # local_idx 3 -> pi=1, pj=1 -> u_row = 1*4 + (1+0) = 5
    # So U[0, t] = 1, U[5, t] = 1
    
    # V factors: [(0,1), (3,1)] -> local 0: qi=0,qj=0 -> v_row=(0+0)*6+(0+0)=0
    # local 3: qi=1,qj=1 -> v_row=(1+0)*6+(0+1)=7
    # So V[0, t]=1, V[7, t]=1
    
    # W factors: [(0,1), (3,1)] -> local 0: ri=0,rj=0 -> w_row=(0+0)*2+0=0
    # local 3: ri=1,rj=1 -> w_row=(0+1)*2+1=3
    # So W[0, t]=1, W[3, t]=1
    
    # Now check: sum_t U[i*4+k,t] * V[k*6+j,t] * W[j*2+i,t]
    # For i=0, k=0, j=0: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1 (from term 0)
    # But we need this to sum to 1 over all t. Other terms might contribute 0 here.
    # This seems plausible.
    
    # OK so the constraint is: for all i in {0,1}, j in {0..5}, k in {0..3}:
    # sum_{t=0}^{rank-1} U[i*4+k, t] * V[k*6+j, t] * W[j*2+i, t] = 1
    
    # Let me try to construct rank 30.
    
    # I'll use a systematic construction with 30 terms.
    
    # Approach: Use 5 terms per (i, j) pair? No, that's 30 for 6 j values and... 
    # 2*6=12 pairs, 30/12 = 2.5 terms per pair. Not integer.
    
    # Let me try: 30 terms, each contributing to multiple (i,j,k) triples.
    
    # Actually, let me just try to find a rank-30 decomposition using the Laderman terms
    # with a different blocking strategy.
    
    # New idea: Use 2x2x2 Laderman for the (i, k) -> j part, but share across j blocks.
    
    # Let me try: For each i in {0,1}, and each k-pair in {(0,1), (2,3)}, 
    # we have a 2x2 block of A (rows i, cols 2k..2k+1) times B (rows 2k..2k+1, all 6 cols).
    # That's 2x2 * 2x6 = 2x6 output for each (i, k-pair).
    # 2 i values * 2 k-pairs = 4 blocks, each 2x2*2x6.
    # Each 2x2*2x6 can be done with rank 7*3=21? No, that's too much.
    
    # I think I need to just try rank 30 with a direct construction and verify.
    
    # Let me use 30 terms with the following pattern:
    # Terms 0-14: for i=0, using Laderman on 2x2 blocks
    # Terms 15-29: for i=1, using Laderman on 2x2 blocks
    
    # But that's still 42 if I use 7 per block.
    
    # I'll just try to output rank 30 with a valid construction by using
    # fewer terms and verifying.
    
    # For now, let me keep the rank 42 solution but try to reduce it.
    
    # Actually, let me try rank 24: 2 (i values) * 3 (k-pairs if we use overlapping) * 4 (some factor)
    
    # I'm going to try a different approach: use the fact that for n=2, 
    # we can use a rank-30 decomposition based on the 3x3x3 rank-23 decomposition
    # padded and projected.
    
    # For now, let me just return the rank 42 solution to ensure correctness,
    # and add a deadline guard.
    
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    strassen_terms = [
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, -1) ]),
        ([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1), (3, 1) ]),
        ([ (3, 1) ], [ (2, 1), (0, -1) ], [ (0, 1), (2, 1) ]),
        ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (0, -1), (1, 1) ]),
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ], [ (3, 1) ]),
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ], [ (0, 1) ]),
    ]
    
    col_idx = 0
    for source in range(2):
        for c in range(3):
            for t in range(7):
                u_factors, v_factors, w_factors = strassen_terms[t]
                
                for (local_idx, coeff) in u_factors:
                    pi = local_idx // 2
                    pj = local_idx % 2
                    u_row = pi * 4 + (pj + 2*source)
                    U[u_row, col_idx] += coeff
                
                for (local_idx, coeff) in v_factors:
                    qi = local_idx // 2
                    qj = local_idx % 2
                    v_row = (qi + 2*source) * 6 + (2*c + qj)
                    V[v_row, col_idx] += coeff
                
                for (local_idx, coeff) in w_factors:
                    ri = local_idx // 2
                    rj = local_idx % 2
                    w_row = (2*c + rj) * 2 + ri
                    W[w_row, col_idx] += coeff
                
                col_idx += 1
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
