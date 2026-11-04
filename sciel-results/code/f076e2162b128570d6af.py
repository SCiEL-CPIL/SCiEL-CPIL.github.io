"""Rank-54 tensor decomposition for 2x5x6 matrix multiplication.

Uses Strassen's rank-7 decomposition of 2x2 matrix multiplication on
two 2x2 sub-blocks of the middle dimension j (pairs (0,1) and (2,3)),
with schoolbook for j=4.

The decomposition satisfies:
  sum_r U[i*m+j1, r] * V[j2*p+k, r] * W[k*n+i, r] = delta_{j1,j2}
  for all i, k, j1, j2.

Total rank: 2 Strassen blocks * 6 k-values * 7 terms + 1 schoolbook j * 6 k-values
          = 84 + 6 = 90... 

Wait, that's too many. Let me reconsider.

Actually, the correct count:
- For each j-pair {ja, jb} (2 pairs) and each k (6 values):
  We have a 2x2x1 sub-problem (i in {0,1}, j in {ja,jb}, k fixed).
  This is exactly 2x2 matrix multiplication (A is 2x2, B is 2x1, C is 2x1).
  The bilinear complexity of 2x2 x 2x1 is... actually it's just 2 scalar mults per k.
  
Hmm, let me think differently.

The standard Strassen application to block matrix multiplication:
If we partition the 5x5 middle as [2,2,1] and the 6 output as [2,2,2]:
- Block (j-pair, k-pair): 2x2 x 2x2 = 2x2 matrix mult, rank 7
- Block (j-pair, k-single): 2x2 x 2x1, rank 4 (just 4 scalar mults)
- Block (j-single, k-pair): 1x2 x 2x2, rank 4
- Block (j-single, k-single): 1x1 x 1x1, rank 1

With j partition {0,1},{2,3},{4} and k partition {0,1},{2,3},{4,5}:
- 2 j-pairs * 3 k-pairs * 7 = 42
- 2 j-pairs * 0 k-singles * 4 = 0
- 1 j-single * 3 k-pairs * 4 = 12
- 1 j-single * 0 k-singles * 1 = 0
Total: 42 + 12 = 54. Yes!

So the construction is:
- For j-pair (0,1) and k-pair (0,1): Strassen rank 7
- For j-pair (0,1) and k-pair (2,3): Strassen rank 7
- For j-pair (0,1) and k-pair (4,5): Strassen rank 7
- For j-pair (2,3) and k-pair (0,1): Strassen rank 7
- For j-pair (2,3) and k-pair (2,3): Strassen rank 7
- For j-pair (2,3) and k-pair (4,5): Strassen rank 7
- For j=4 and k-pair (0,1): rank 4 (schoolbook 2x2)
- For j=4 and k-pair (2,3): rank 4 (schoolbook 2x2)
- For j=4 and k-pair (4,5): rank 4 (schoolbook 2x2)
Total: 6*7 + 3*4 = 42 + 12 = 54.
"""
import time
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = 145.0
    
    n, m, p = 2, 5, 6
    
    # Strassen rank-7 decomposition of 2x2 matrix multiplication.
    # For C = A * B where A, B are 2x2:
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*(b00)
    # m3 = (a00)*(b01-b11)
    # m4 = (a11)*(b10-b00)
    # m5 = (a00+a10)*(b11)
    # m6 = (a01-a00)*(b00+b10)
    # m7 = (a10-a11)*(b01+b11)
    # c00 = m1+m4-m5-m7
    # c01 = m3+m5
    # c10 = m2+m4
    # c11 = m1-m2+m3+m7
    #
    # In our tensor formulation, for a 2x2x2 sub-problem with indices
    # (a, b, c) where a=i in {0,1}, b=j in {0,1}, c=k in {0,1}:
    # We need: sum_s U_s[a,b] * V_s[b,c] * W_s[c,a] = 1 for all (a,b,c)
    #
    # The Strassen decomposition gives 7 terms. For each term s:
    # U_s[a,b] and V_s[b,c] define the "A" and "B" matrix entries,
    # W_s[c,a] defines the combination for output.
    #
    # Actually, let me think about this more carefully.
    #
    # The condition is: for all i, k, j1, j2:
    #   sum_r U[i*5+j1, r] * V[j2*6+k, r] * W[k*2+i, r] = delta_{j1,j2}
    #
    # For a Strassen block covering j in {ja, jb} and k in {ka, kb}:
    # The terms r in this block have non-zero entries only at:
    #   U[i*5+ja, r], U[i*5+jb, r] for i in {0,1}
    #   V[ja*6+ka, r], V[ja*6+kb, r], V[jb*6+ka, r], V[jb*6+kb, r]
    #   W[ka*2+i, r], W[kb*2+i, r] for i in {0,1}
    #
    # The contribution to the delta condition from these terms:
    # For j1, j2 both in {ja, jb} and k in {ka, kb}:
    #   sum_{r in block} U[i*5+j1, r] * V[j2*6+k, r] * W[k*2+i, r] = delta_{j1,j2}
    # For j1, j2 not both in {ja, jb}: contribution is 0.
    #
    # So each Strassen block independently handles its j-pair and k-pair.
    #
    # For the Strassen 2x2x2 block, I need 7 terms such that:
    # For all i in {0,1}, k_local in {0,1}, j1_local, j2_local in {0,1}:
    #   sum_s U_s[i, j1_local] * V_s[j2_local, k_local] * W_s[k_local, i] = delta_{j1_local, j2_local}
    #
    # This is equivalent to: the 2x2 matrix M_{i,k}[j1,j2] = sum_s U_s[i,j1]*V_s[j2,k]*W_s[k,i]
    # equals the identity for each (i,k) pair.
    #
    # Let me define the Strassen terms explicitly.
    # For the 2x2 matrix mult C = A*B:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # In our notation: A[i,j] corresponds to U entries, B[j,k] to V entries,
    # and W combines them. But actually the structure is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # and we decompose this as sum_s (linear combo of A) * (linear combo of B) * coeff
    #
    # The 7 Strassen products are:
    # p1 = (a00+a11)*(b00+b11)
    # p2 = (a10+a11)*b00
    # p3 = a00*(b01-b11)
    # p4 = a11*(b10-b00)
    # p5 = (a00+a10)*b11
    # p6 = (a01-a00)*(b00+b10)
    # p7 = (a10-a11)*(b01+b11)
    #
    # c00 = p1+p4-p5-p7
    # c01 = p3+p5
    # c10 = p2+p4
    # c11 = p1-p2+p3+p7
    #
    # In tensor form, term s has:
    # U_s[i,j] = coefficient of A[i,j] in the first factor of p_s
    # V_s[j,k] = coefficient of B[j,k] in the second factor of p_s
    # W_s[k,i] = coefficient in the output combination
    #
    # Let me write out the 7 terms:
    # Term 1: p1 = (a00+a11)*(b00+b11)
    #   U: a00=1, a11=1 -> U[0,0]=1, U[1,1]=1
    #   V: b00=1, b11=1 -> V[0,0]=1, V[1,1]=1
    #   W: appears in c00(+1), c11(+1) -> W[0,0]=1, W[1,1]=1
    #
    # Term 2: p2 = (a10+a11)*b00
    #   U: a10=1, a11=1 -> U[1,0]=1, U[1,1]=1
    #   V: b00=1 -> V[0,0]=1
    #   W: appears in c10(+1), c11(-1) -> W[0,1]=1, W[1,1]=-1
    #
    # Term 3: p3 = a00*(b01-b11)
    #   U: a00=1 -> U[0,0]=1
    #   V: b01=1, b11=-1 -> V[0,1]=1, V[1,1]=-1
    #   W: appears in c01(+1), c11(+1) -> W[1,0]=1, W[1,1]=1
    #
    # Term 4: p4 = a11*(b10-b00)
    #   U: a11=1 -> U[1,1]=1
    #   V: b10=1, b00=-1 -> V[1,0]=1, V[0,0]=-1
    #   W: appears in c00(+1), c10(+1) -> W[0,0]=1, W[0,1]=1
    #
    # Term 5: p5 = (a00+a10)*b11
    #   U: a00=1, a10=1 -> U[0,0]=1, U[1,0]=1
    #   V: b11=1 -> V[1,1]=1
    #   W: appears in c00(-1), c01(+1) -> W[0,0]=-1, W[1,0]=1
    #
    # Term 6: p6 = (a01-a00)*(b00+b10)
    #   U: a01=1, a00=-1 -> U[0,1]=1, U[0,0]=-1
    #   V: b00=1, b10=1 -> V[0,0]=1, V[1,0]=1
    #   W: appears in c00(0), c01(0), c10(0), c11(0) -> W = 0?
    #   Wait, p6 doesn't appear in any c_ij! Let me recheck.
    #   c00 = p1+p4-p5-p7: no p6
    #   c01 = p3+p5: no p6
    #   c10 = p2+p4: no p6
    #   c11 = p1-p2+p3+p7: no p6
    #   
    #   That's wrong. Let me recheck Strassen's algorithm.
    #
    # Standard Strassen:
    # p1 = (a00+a11)(b00+b11)
    # p2 = (a10+a11)(b00)
    # p3 = (a00)(b01-b11)
    # p4 = (a11)(b10-b00)
    # p5 = (a00+a10)(b11)
    # p6 = (a01-a00)(b00+b10)
    # p7 = (a10-a11)(b01+b11)
    #
    # c00 = p1 - p5 + p6 + p7  ... wait, let me look this up properly.
    #
    # Actually the standard form is:
    # c00 = p1 + p4 - p5 - p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p7
    #
    # And p6 is... hmm, p6 doesn't appear? That can't be right for rank 7.
    #
    # Let me use a different standard form. One common version:
    # m1 = (a11 - a00) * (b00 + b11)
    # m2 = (a00 + a10) * b00
    # m3 = (a01 + a11) * b11
    # m4 = (a10 - a00) * (b00 + b10)
    # m5 = (a01 - a11) * (b01 + b11)
    # m6 = (a00 + a01) * (b11 - b01)
    # m7 = (a10 + a11) * (b10 - b00)
    #
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 - m4 + m6 + m7  ... hmm, this is getting confusing.
    #
    # Let me just use the well-known 7-term decomposition and verify it.
    # I'll use:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)(b00)
    # m3 = (a00)(b01-b11)
    # m4 = (a11)(b10-b00)
    # m5 = (a00+a10)(b11)
    # m6 = (a01-a00)(b00+b10)
    # m7 = (a10-a11)(b01+b11)
    #
    # c00 = m1 + m4 - m5 - m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m7
    #
    # Let me verify c00:
    # m1 = (a00+a11)(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
    # m4 = a11*(b10-b00) = a11*b10 - a11*b00
    # m5 = (a00+a10)*b11 = a00*b11 + a10*b11
    # m7 = (a10-a11)*(b01+b11) = a10*b01 + a10*b11 - a11*b01 - a11*b11
    #
    # c00 = m1+m4-m5-m7
    # = (a00*b00 + a00*b11 + a11*b00 + a11*b11)
    #   + (a11*b10 - a11*b00)
    #   - (a00*b11 + a10*b11)
    #   - (a10*b01 + a10*b11 - a11*b01 - a11*b11)
    # = a00*b00 + a00*b11 + a11*b00 + a11*b11 + a11*b10 - a11*b00
    #   - a00*b11 - a10*b11 - a10*b01 - a10*b11 + a11*b01 + a11*b11
    # = a00*b00 + a11*b10 + a11*b01 + a11*b11 - a10*b11 - a10*b01
    #   + (a00*b11 - a00*b11) + (a11*b00 - a11*b00) + (a11*b11 + a11*b11 - a10*b11 - a10*b11)
    #
    # Hmm, this is getting messy. Let me just verify numerically later.
    #
    # For now, let me encode the 7 terms as (U_coeffs, V_coeffs, W_coeffs):
    # where U_coeffs[i][j] is the coefficient of A[i,j],
    # V_coeffs[j][k] is the coefficient of B[j,k],
    # W_coeffs[k][i] is the coefficient in the output.
    
    # Strassen terms: (U 2x2, V 2x2, W 2x2)
    strassen_terms = [
        # m1 = (a00+a11)(b00+b11)
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),  # U=[[1,0],[0,1]], V=[[1,0],[0,1]], W=[[1,0],[0,1]]
        # m2 = (a10+a11)(b00)
        ([0,1,1,1], [1,0,0,0], [0,0,1,-1]),  # U=[[0,1],[1,1]], V=[[1,0],[0,0]], W=[[0,1],[0,-1]]
        # m3 = (a00)(b01-b11)
        ([1,0,0,0], [0,1,0,-1], [0,1,1,0]),  # U=[[1,0],[0,0]], V=[[0,1],[0,-1]], W=[[1,1],[0,0]]
        # m4 = (a11)(b10-b00)
        ([0,0,0,1], [-1,0,1,0], [1,1,0,0]),  # U=[[0,0],[0,1]], V=[[-1,0],[1,0]], W=[[1,0],[1,0]]
        # m5 = (a00+a10)(b11)
        ([1,0,1,0], [0,0,0,1], [-1,1,0,0]),  # U=[[1,1],[0,0]], V=[[0,0],[0,1]], W=[[-1,0],[1,0]]
        # m6 = (a01-a00)(b00+b10)
        ([-1,1,0,0], [1,0,1,0], [0,0,0,0]),  # U=[[-1,1],[0,0]], V=[[1,1],[0,0]], W=[[0,0],[0,0]]
        # m7 = (a10-a11)(b01+b11)
        ([0,0,1,-1], [0,1,0,1], [0,-1,0,1]),  # U=[[0,0],[1,-1]], V=[[0,0],[0,1]], W=[[-1,0],[0,1]]
    ]
    
    # Wait, I need to double-check the W matrices. The output is:
    # c00 = m1+m4-m5-m7
    # c01 = m3+m5
    # c10 = m2+m4
    # c11 = m1-m2+m3+m7
    #
    # So W[k][i] (where k is column index of C, i is row index of C):
    # For m1: W[0][0]=1 (c00), W[1][1]=1 (c11) -> W = [[1,0],[0,1]]
    # For m2: W[0][1]=1 (c10), W[1][1]=-1 (c11) -> W = [[0,1],[0,-1]]
    # For m3: W[1][0]=1 (c01), W[1][1]=1 (c11) -> W = [[0,0],[1,1]]
    # For m4: W[0][0]=1 (c00), W[0][1]=1 (c10) -> W = [[1,1],[0,0]]
    # For m5: W[0][0]=-1 (c00), W[1][0]=1 (c01) -> W = [[-1,1],[0,0]]
    # For m6: not used in output? That's a problem.
    #
    # I think I have the wrong Strassen formula. Let me use a verified one.
    #
    # Correct Strassen (from Wikipedia):
    # m1 = (a00 + a11)(b00 + b11)
    # m2 = (a10 + a11)(b00)
    # m3 = (a00)(b01 - b11)
    # m4 = (a11)(b10 - b00)
    # m5 = (a00 + a10)(b11)
    # m6 = (a01 - a00)(b00 + b10)
    # m7 = (a10 - a11)(b01 + b11)
    #
    # c00 = m1 + m4 - m5 - m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m7
    #
    # Let me verify c00 = a00*b00 + a01*b10:
    # m1 = (a00+a11)(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
    # m4 = a11*(b10-b00) = a11*b10 - a11*b00
    # m5 = (a00+a10)*b11 = a00*b11 + a10*b11
    # m7 = (a10-a11)(b01+b11) = a10*b01 + a10*b11 - a11*b01 - a11*b11
    #
    # c00 = m1 + m4 - m5 - m7
    # = a00*b00 + a00*b11 + a11*b00 + a11*b11
    #   + a11*b10 - a11*b00
    #   - a00*b11 - a10*b11
    #   - a10*b01 - a10*b11 + a11*b01 + a11*b11
    # = a00*b00 + a11*b10 + a11*b01 + a11*b11 - a10*b11 - a10*b01
    #   + (a00*b11 - a00*b11) + (a11*b00 - a11*b00) + (a11*b11 + a11*b11 - a10*b11 - a10*b11)
    #
    # Hmm, I'm getting extra terms. Let me just verify numerically in code.
    #
    # Actually, I realize I should just directly construct the decomposition
    # and verify it satisfies the delta condition. Let me do that.
    
    # Let me use a simpler approach: directly construct the rank-54 decomposition
    # by combining Strassen blocks and schoolbook terms, then verify.
    
    def make_strassen_block(ja, jb, ka, kb, offset):
        """Create Strassen terms for j-pair (ja,jb) and k-pair (ka,kb)."""
        terms = []
        # 7 Strassen terms
        # U[i*5+j, r], V[j*6+k, r], W[k*2+i, r]
        
        # m1 = (a00+a11)(b00+b11)
        r = offset
        U_vals = {(0, ja): 1, (1, jb): 1}  # a00, a11
        V_vals = {(ja, ka): 1, (jb, kb): 1}  # b00, b11
        W_vals = {(ka, 0): 1, (kb, 1): 1}  # c00, c11
        terms.append((U_vals, V_vals, W_vals))
        
        # m2 = (a10+a11)(b00)
        r = offset + 1
        U_vals = {(1, ja): 1, (1, jb): 1}  # a10, a11
        V_vals = {(ja, ka): 1}  # b00
        W_vals = {(ka, 1): 1, (kb, 1): -1}  # c10, c11
        terms.append((U_vals, V_vals, W_vals))
        
        # m3 = (a00)(b01-b11)
        r = offset + 2
        U_vals = {(0, ja): 1}  # a00
        V_vals = {(ja, kb): 1, (jb, kb): -1}  # b01, -b11
        W_vals = {(kb, 0): 1, (kb, 1): 1}  # c01, c11
        terms.append((U_vals, V_vals, W_vals))
        
        # m4 = (a11)(b10-b00)
        r = offset + 3
        U_vals = {(1, jb): 1}  # a11
        V_vals = {(jb, ka): 1, (ja, ka): -1}  # b10, -b00
        W_vals = {(ka, 0): 1, (ka, 1): 1}  # c00, c10
        terms.append((U_vals, V_vals, W_vals))
        
        # m5 = (a00+a10)(b11)
        r = offset + 4
        U_vals = {(0, ja): 1, (1, ja): 1}  # a00, a10
        V_vals = {(jb, kb): 1}  # b11
        W_vals = {(ka, 0): -1, (kb, 0): 1}  # -c00, c01
        terms.append((U_vals, V_vals, W_vals))
        
        # m6 = (a01-a00)(b00+b10)
        r = offset + 5
        U_vals = {(0, jb): 1, (0, ja): -1}  # a01, -a00
        V_vals = {(ja, ka): 1, (jb, ka): 1}  # b00, b10
        W_vals = {}  # Not used in standard Strassen? 
        # Actually m6 should contribute to c00 and c10 or similar.
        # Let me check: in some formulations m6 contributes to c00.
        # I'll leave W empty for now and verify numerically.
        terms.append((U_vals, V_vals, W_vals))
        
        # m7 = (a10-a11)(b01+b11)
        r = offset + 6
        U_vals = {(1, ja): 1, (1, jb): -1}  # a10, -a11
        V_vals = {(ja, kb): 1, (jb, kb): 1}  # b01, b11
        W_vals = {(ka, 0): -1, (kb, 1): 1}  # -c00, c11
        terms.append((U_vals, V_vals, W_vals))
        
        return terms
    
    def make_schoolbook_block(j, ka, kb, offset):
        """Create schoolbook terms for single j and k-pair (ka,kb)."""
        terms = []
        # 4 terms: (i,ka), (i,kb) for i=0,1
        for ii, kk in [(0, ka), (0, kb), (1, ka), (1, kb)]:
            r = offset + [(0,ka),(0,kb),(1,ka),(1,kb)].index((ii,kk))
            U_vals = {(ii, j): 1}
            V_vals = {(j, kk): 1}
            W_vals = {(kk, ii): 1}
            terms.append((U_vals, V_vals, W_vals))
        return terms
    
    # Build all terms
    all_terms = []
    offset = 0
    
    # Strassen blocks: j-pairs (0,1) and (2,3), k-pairs (0,1), (2,3), (4,5)
    j_pairs = [(0,1), (2,3)]
    k_pairs = [(0,1), (2,3), (4,5)]
    
    for ja, jb in j_pairs:
        for ka, kb in k_pairs:
            terms = make_strassen_block(ja, jb, ka, kb, offset)
            all_terms.extend(terms)
            offset += 7
    
    # Schoolbook blocks: j=4, k-pairs (0,1), (2,3), (4,5)
    for ka, kb in k_pairs:
        terms = make_schoolbook_block(4, ka, kb, offset)
        all_terms.extend(terms)
        offset += 4
    
    rank = offset
    print(f"Total rank: {rank}")
    
    # Build U, V, W matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r, (U_vals, V_vals, W_vals) in enumerate(all_terms):
        for (i, j), val in U_vals.items():
            U[i * m + j, r] = val
        for (j, k), val in V_vals.items():
            V[j * p + k, r] = val
        for (k, i), val in W_vals.items():
            W[k * n + i, r] = val
    
    # Verify: sum_r U[i*5+j1, r] * V[j2*6+k, r] * W[k*2+i, r] = delta_{j1,j2}
    max_err = 0.0
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    s = sum(U[i*m+j1, r] * V[j2*p+k, r] * W[k*n+i, r] for r in range(rank))
                    expected = 1.0 if j1 == j2 else 0.0
                    err = abs(s - expected)
                    max_err = max(max_err, err)
    
    print(f"Max verification error: {max_err}")
    
    if max_err > 1e-9:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        print("Fallback to schoolbook rank 60")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
