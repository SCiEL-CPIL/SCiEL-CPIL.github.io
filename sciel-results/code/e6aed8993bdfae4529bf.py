"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _add_term(U, V, W, col, u_entry, v_entry, w_entry):
    """Add a single rank-1 term to the decomposition."""
    U[u_entry[0]][u_entry[1]] = u_entry[2]
    V[v_entry[0]][v_entry[1]] = v_entry[2]
    W[w_entry[0]][w_entry[1]] = w_entry[2]

def _strassen_2x2x2_terms(base_col, i0, j0, k0):
    """
    Generate Strassen terms for C[i0:i0+2, k0:k0+2] = A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
    Returns list of (U_idx, V_idx, W_idx, U_val, V_val, W_val) tuples.
    """
    terms = []
    # Strassen 7 products for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # Each product contributes to C entries via specific combinations
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M7
    # C11 = M1 - M2 + M3 + M6
    
    # We need to express this as rank-1 terms in the tensor decomposition
    # For the full tensor: U[i*m+j], V[j*p+k], W[k*n+i]
    
    # The 7 Strassen products create intermediate values, then we combine them.
    # In tensor decomposition, each "product" is one rank-1 term where:
    # U has the linear combination of A entries, V has the linear combination of B entries,
    # and W picks out the resulting C entry.
    
    # Actually, the standard way: each Strassen "multiply" Mx = (linear combo of A) * (linear combo of B)
    # contributes to multiple C entries. We need separate rank-1 terms for each C entry contribution.
    
    # M1 = (A00+A11)(B00+B11) contributes to C00 (+1), C11 (+1)
    # M2 = (A10+A11)B00 contributes to C10 (+1), C11 (-1)
    # M3 = A00(B10-B11) contributes to C01 (+1), C11 (+1)
    # M4 = A11(B01-B10) contributes to C00 (-1), C01 (+1)
    # M5 = (A00+A01)B11 contributes to C00 (+1), C11... wait let me recheck.
    
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M7
    # C11 = M1 - M2 + M3 + M6
    
    # For tensor decomposition, each (Mx, Cy) pair where Mx contributes to Cy with coefficient cx
    # becomes a rank-1 term: U = linear combo of A for Mx, V = linear combo of B for Mx, W = cx at position Cy
    
    # Let's define the A-combinations and B-combinations for each Mx:
    # A00 = A[i0,j0], A01 = A[i0,j0+1], A10 = A[i0+1,j0], A11 = A[i0+1,j0+1]
    # B00 = B[j0,k0], B01 = B[j0,k0+1], B10 = B[j0+1,k0], B11 = B[j0+1,k0+1]
    
    # M1: A_combo = {A00:1, A11:1}, B_combo = {B00:1, B11:1}
    # M2: A_combo = {A10:1, A11:1}, B_combo = {B00:1}
    # M3: A_combo = {A00:1}, B_combo = {B10:1, B11:-1}
    # M4: A_combo = {A11:1}, B_combo = {B01:1, B10:-1}
    # M5: A_combo = {A00:1, A01:1}, B_combo = {B11:1}
    # M6: A_combo = {A10:1, A00:-1}, B_combo = {B00:1, B10:1}
    # M7: A_combo = {A01:1, A11:-1}, B_combo = {B01:1, B11:1}
    
    # Contributions to C:
    # C00: M1(+1), M4(-1), M5(+1), M7(-1)
    # C01: M3(+1), M4(+1)
    # C10: M2(+1), M7(+1)
    # C11: M1(+1), M2(-1), M3(+1), M6(+1)
    
    # Each (Mx, Cy, coeff) is one rank-1 term.
    # Total terms: 4+2+2+4 = 12? No wait, that's not right.
    
    # Actually in tensor decomposition, we need the total number of rank-1 terms to equal the rank.
    # The Strassen algorithm uses 7 multiplications, but the "additions" are free in the tensor model?
    # No! In tensor decomposition, each rank-1 term is one multiplication. The additions are absorbed into the U and V matrices.
    
    # So each Mx is ONE rank-1 term (one multiplication), and its contribution to multiple C entries
    # means W has multiple non-zero entries for that term.
    
    # Wait, that's not how it works either. Let me think again.
    
    # In a rank-r tensor decomposition: T = sum_{t=1}^{r} u_t ⊗ v_t ⊗ w_t
    # where u_t is a vector in R^{n*m}, v_t in R^{m*p}, w_t in R^{p*n}
    
    # For matrix multiplication: T_{i,j,k} = delta_{j,k} (the standard tensor)
    # Actually T_{i,j,k} = 1 if the indices match appropriately.
    
    # The tensor for 2x3 x 3x9 multiplication: T[i,j,k] where i in 0..1, j in 0..2, k in 0..8
    # T[i,j,k] = 1 if... no, the tensor is defined by:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k) - no that's the schoolbook.
    
    # The tensor T has entries T[i,j,k] = 1 for all (i,j,k) in the domain, and the decomposition
    # T = sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t] means:
    # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all (i,j,k)
    
    # For Strassen on a 2x2x2 sub-problem, the 7 rank-1 terms work as follows:
    # Each term t has U_t, V_t, W_t vectors that are sparse.
    
    # Let me just directly construct the 7 Strassen terms for a 2x2x2 block.
    
    # For the 2x2x2 block at (i0,j0,k0):
    # Term 1 (M1): U has 1 at (i0,j0) and 1 at (i0+1,j0+1)
    #                V has 1 at (j0,k0) and 1 at (j0+1,k0+1)
    #                W has 1 at (k0,i0), 1 at (k0+1,i0+1), -1 at (k0,i0+1), -1 at (k0+1,i0)
    #                Wait, I need to be more careful.
    
    # Let me use the known result: Strassen 2x2x2 has rank 7. The 7 terms are:
    
    # I'll use a reference implementation. The 7 rank-1 tensors for Strassen 2x2x2:
    
    # Using standard indexing where A is 2x2, B is 2x2, C is 2x2:
    # T1: U=(1,0,0,1), V=(1,0,0,1), W=(1,0,0,-1)  -- for C[0,0] and C[1,1]
    #     Actually this is getting complicated. Let me just hardcode the known Strassen terms.
    
    # Known Strassen 2x2x2 rank-7 decomposition (from literature):
    # The 7 terms (u, v, w) where u in R^4 (A entries), v in R^4 (B entries), w in R^4 (C entries):
    
    # Let me use the formulation where:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # i = a*e + b*g, j = a*f + b*h, k = c*e + d*g, l = c*f + d*h
    
    # Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-c)
    # p5 = (a+b)h
    # p6 = (c-a)(e+g)
    # p7 = (b-d)(f+h)
    
    # i = p1 - p4 + p5 - p7
    # j = p3 + p4
    # k = p2 + p7
    # l = p1 - p2 + p3 + p6
    
    # Now, each p_m is a product of a linear combo of A and a linear combo of B.
    # In the tensor decomposition, each p_m corresponds to ONE rank-1 term where:
    # u_m = the linear combo coefficients for A
    # v_m = the linear combo coefficients for B
    # w_m = the coefficients for C (how p_m contributes to each C entry)
    
    # p1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)  [contributes +1 to i and l]
    # p2: u=(0,0,1,1), v=(1,0,0,0), w=(0,0,1,-1)  [contributes +1 to k, -1 to l]
    # p3: u=(1,0,0,0), v=(0,1,1,-1), w=(0,1,0,1)  [contributes +1 to j, +1 to l]
    # p4: u=(0,0,0,1), v=(0,-1,1,0), w=(-1,1,0,0)  [contributes -1 to i, +1 to j]
    # p5: u=(1,1,0,0), v=(0,0,0,1), w=(1,0,0,0)  [contributes +1 to i]
    # p6: u=(0,-1,1,0), v=(1,0,1,0), w=(0,0,0,1)  [contributes +1 to l]
    # p7: u=(0,1,0,-1), v=(0,1,0,1), w=(0,0,1,-1)  [contributes +1 to k, -1 to l]
    
    # Let me verify: 
    # i = p1 - p4 + p5 - p7 = 1 - (-1) + 1 - 0 = 3? That's wrong.
    # 
    # Wait, I think the w vectors should just indicate which C entry each p contributes to,
    # and the coefficient is the sign. But in tensor decomposition, w is a vector and the
    # product u⊗v⊗w gives a 4D tensor. The sum of all 7 should give the identity tensor.
    
    # Let me verify term by term for C[0,0] (entry i):
    # p1: u·a * v·b * w[0] = (a+d)(e+h) * 1
    # p4: u·a * v·b * w[0] = d*(g-c) * (-1) = -d(g-c)
    # p5: u·a * v·b * w[0] = (a+b)*h * 1
    # p7: u·a * v·b * w[0] = (b-d)(f+h) * 0 = 0
    # Sum: (a+d)(e+h) - d(g-c) + (a+b)h
    # = ae + ah + de + dh - dg + dc + ah + bh
    # = ae + 2ah + de + dh - dg + dc + bh
    # Should equal: ae + bg
    # This doesn't match. I'm making an error.
    
    # Let me look up the correct Strassen tensor decomposition.
    
    # Actually, I think the issue is that in the tensor decomposition, each rank-1 term
    # contributes to ALL C entries through the w vector, and the sum must equal the
    # multiplication tensor. Let me just use a verified implementation.
    
    # I'll use the approach from the inspiration that achieved rank 50, and try to
    # improve it. Let me first implement the rank-50 solution and verify, then try rank 48.
    
    pass

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] == 1 for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for t in range(rank):
                    total += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                if total != 1:
                    return False
    return True

def _construct_strassen_2x3x9():
    """Construct a Strassen-based decomposition for 2x3 x 3x9."""
    n, m, p = 2, 3, 9
    
    # We'll build the decomposition term by term.
    # Split: A is 2x3, B is 3x9
    # A_top = A[:, 0:2] (2x2), A_bot = A[:, 2:3] (2x1)
    # B_top = B[0:2, :] (2x9), B_bot = B[2:3, :] (1x9)
    # C = A_top @ B_top + A_bot @ B_bot
    
    # For A_top @ B_top (2x2 @ 2x9):
    # Split B_top into 4 blocks of 2x2 and 1 block of 2x1
    # B_top[:, 0:2], B_top[:, 2:4], B_top[:, 4:6], B_top[:, 6:8], B_top[:, 8:9]
    
    # For A_bot @ B_bot (2x1 @ 1x9): schoolbook, 18 terms
    
    # Let me define the Strassen 2x2x2 terms properly.
    # I'll use a verified set of 7 rank-1 tensors.
    
    # From the literature, the Strassen 2x2x2 tensor decomposition:
    # Using 0-indexed: A[i,j], B[j,k], C[i,k] for i,j,k in {0,1}
    # The 7 terms (u in R^4 indexed by (i,j), v in R^4 indexed by (j,k), w in R^4 indexed by (k,i)):
    
    # I'll use the formulation from "Tensor Decomposition" by Strassen:
    # Term 1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)
    # Term 2: u=(0,0,1,1), v=(1,0,0,0), w=(0,0,1,-1)  
    # Term 3: u=(1,0,0,0), v=(0,1,1,-1), w=(0,1,0,1)
    # Term 4: u=(0,0,0,1), v=(0,-1,1,0), w=(-1,1,0,0)
    # Term 5: u=(1,1,0,0), v=(0,0,0,1), w=(1,0,0,0)
    # Term 6: u=(0,-1,1,0), v=(1,0,1,0), w=(0,0,0,1)
    # Term 7: u=(0,1,0,-1), v=(0,1,0,1), w=(0,0,1,-1)
    
    # Let me verify term 1: 
    # (a+d)(e+h) contributes to C[0,0] with coeff 1 and C[1,1] with coeff 1
    # C[0,0] should be ae+bg. From term 1: (a+d)(e+h) = ae+ah+de+dh
    # That's already more than ae+bg, so other terms must cancel.
    
    # Let me just sum all 7 terms for C[0,0]:
    # T1: (a+d)(e+h)*1 = ae+ah+de+dh
    # T2: (c+d)*e*0 = 0
    # T3: a*(f-h)*0 = 0
    # T4: d*(g-c)*(-1) = -dg+dc
    # T5: (a+b)*h*1 = ah+bh
    # T6: (c-a)*(e+g)*0 = 0
    # T7: (b-d)*(f+h)*0 = 0
    # Sum: ae+ah+de+dh -dg+dc +ah+bh = ae + 2ah + de + dh - dg + dc + bh
    # Expected: ae + bg
    # This is clearly wrong. My Strassen terms are incorrect.
    
    # Let me use a different, verified approach. I'll directly compute the
    # Strassen decomposition by solving for it.
    
    # Actually, let me just use the known correct Strassen 7-multiplication scheme
    # and translate it to tensor decomposition terms.
    
    # The key insight: in tensor decomposition, we can't have "additions" as separate
    # operations. Each rank-1 term is a single multiplication. The Strassen algorithm
    # has 7 multiplications and some additions. The additions are "free" in the tensor
    # model because they're absorbed into the U, V, W vectors.
    
    # So the 7 Strassen multiplications correspond to 7 rank-1 terms.
    
    # Let me define them correctly:
    # p1 = (A00+A11) * (B00+B11)
    #   u = [1,0,0,1] (A00 and A11), v = [1,0,0,1] (B00 and B11)
    #   This product contributes to C00 and C11
    #   w = [1, 0, 0, 1] (C00 and C11, both +1)
    #   But wait, p1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    #   And C00 = A00B00 + A01B10, C11 = A10B01 + A11B11
    #   So p1 contributes A00B00 to C00, A11B11 to C11, and cross terms A00B11, A11B00
    #   which need to be cancelled by other terms.
    
    # This is getting complex. Let me just use a brute-force verification approach:
    # I'll construct the terms and verify they sum to the correct tensor.
    
    # Here's a cleaner way to think about it. The multiplication tensor T for 2x2x2 is:
    # T[i,j,k] = 1 for all (i,j,k) in {0,1}^3
    # Wait no. T[i,j,k] is the coefficient of A[i,j]*B[j,k] in C[i,k].
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 if the indices are consistent, which they always are for valid i,j,k.
    # Actually T is a 3-tensor with T[i,j,k] = 1 for all i,j,k in range.
    # No! T[i,j,k] = delta_{j,?} ... 
    
    # The matrix multiplication tensor is: T_{i,j,k} = 1 for all valid (i,j,k)
    # where i indexes rows of A/C, j indexes cols of A/rows of B, k indexes cols of B/C.
    # The decomposition T = sum u_t ⊗ v_t ⊗ w_t means:
    # sum_t u_t[i,j] * v_t[j,k] * w_t[k,i] = T[i,j,k] = 1
    
    # For 2x2x2, T[i,j,k] = 1 for all i,j,k in {0,1}.
    
    # Let me verify my Strassen terms sum to 1 for each (i,j,k):
    
    # For (i,j,k) = (0,0,0): T=1
    # Sum_t u_t[0,0]*v_t[0,0]*w_t[0,0]
    # T1: 1*1*1 = 1
    # T2: 0*1*0 = 0
    # T3: 1*0*0 = 0
    # T4: 0*0*(-1) = 0
    # T5: 1*0*1 = 0
    # T6: 0*1*0 = 0
    # T7: 0*0*0 = 0
    # Sum = 1 ✓
    
    # For (i,j,k) = (0,1,0): T=1
    # Sum_t u_t[0,1]*v_t[1,0]*w_t[0,0]
    # T1: 0*0*1 = 0
    # T2: 0*0*0 = 0
    # T3: 0*1*0 = 0
    # T4: 0*(-1)*(-1) = 0
    # T5: 1*0*1 = 0
    # T6: -1*0*0 = 0
    # T7: 1*1*0 = 0
    # Sum = 0 ✗ (should be 1)
    
    # So my terms are wrong. Let me look for the correct Strassen tensor decomposition.
    
    # I'll use a different strategy: just implement the schoolbook for now and
    # try to find a lower-rank decomposition through optimization.
    
    # Actually, let me just implement the rank-50 solution from the inspiration
    # and verify it works, then try to improve.
    
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try Strassen-based decomposition
    try:
        result = _construct_strassen_2x3x9()
        if result is not None:
            U, V, W = result
            rank = len(U[0])
            if _verify_decomposition(n, m, p, U, V, W):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
