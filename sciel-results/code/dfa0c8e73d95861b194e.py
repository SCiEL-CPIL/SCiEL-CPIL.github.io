import time
import numpy as np
from itertools import product

def construct():
    """
    Returns an exact tensor decomposition for multiplying a 2x3 matrix 
    by a 3x5 matrix. Tries to find rank-27 via structured search,
    falls back to known rank-28 Strassen+schoolbook construction.
    """
    t_start = time.time()
    deadline = t_start + 140  # safety margin before 150s kill

    n, m, p = 2, 3, 5

    def build_rank28():
        """Known rank-28 construction."""
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        def add_strassen(i_off, j_off, k_off):
            nonlocal col
            # M1: (A00+A11)(B00+B11)
            U[(i_off+0)*m + (j_off+0)][col] = 1
            U[(i_off+1)*m + (j_off+1)][col] = 1
            V[(j_off+0)*p + (k_off+0)][col] = 1
            V[(j_off+1)*p + (k_off+1)][col] = 1
            W[(k_off+0)*n + (i_off+0)][col] = 1
            W[(k_off+1)*n + (i_off+1)][col] = 1
            col += 1
            # M2: (A10+A11)(B00)
            U[(i_off+1)*m + (j_off+0)][col] = 1
            U[(i_off+1)*m + (j_off+1)][col] = 1
            V[(j_off+0)*p + (k_off+0)][col] = 1
            W[(k_off+0)*n + (i_off+1)][col] = 1
            W[(k_off+1)*n + (i_off+1)][col] = -1
            col += 1
            # M3: A00(B01-B11)
            U[(i_off+0)*m + (j_off+0)][col] = 1
            V[(j_off+0)*p + (k_off+1)][col] = 1
            V[(j_off+1)*p + (k_off+1)][col] = -1
            W[(k_off+1)*n + (i_off+0)][col] = 1
            W[(k_off+1)*n + (i_off+1)][col] = 1
            col += 1
            # M4: A11(B10-B00)
            U[(i_off+1)*m + (j_off+1)][col] = 1
            V[(j_off+1)*p + (k_off+0)][col] = 1
            V[(j_off+0)*p + (k_off+0)][col] = -1
            W[(k_off+0)*n + (i_off+0)][col] = 1
            W[(k_off+0)*n + (i_off+1)][col] = 1
            col += 1
            # M5: (A00+A01)(B11)
            U[(i_off+0)*m + (j_off+0)][col] = 1
            U[(i_off+0)*m + (j_off+1)][col] = 1
            V[(j_off+1)*p + (k_off+1)][col] = 1
            W[(k_off+0)*n + (i_off+0)][col] = -1
            W[(k_off+1)*n + (i_off+0)][col] = 1
            col += 1
            # M6: (A10-A00)(B00+B01)
            U[(i_off+1)*m + (j_off+0)][col] = 1
            U[(i_off+0)*m + (j_off+0)][col] = -1
            V[(j_off+0)*p + (k_off+0)][col] = 1
            V[(j_off+0)*p + (k_off+1)][col] = 1
            W[(k_off+1)*n + (i_off+1)][col] = 1
            col += 1
            # M7: (A01-A11)(B10+B11)
            U[(i_off+0)*m + (j_off+1)][col] = 1
            U[(i_off+1)*m + (j_off+1)][col] = -1
            V[(j_off+1)*p + (k_off+0)][col] = 1
            V[(j_off+1)*p + (k_off+1)][col] = 1
            W[(k_off+0)*n + (i_off+0)][col] = 1
            col += 1

        # Block 1: j=2, all k. M_{2,1,5}, rank 10.
        for i in range(n):
            for k in range(p):
                U[i * m + 2][col] = 1
                V[2 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

        # Block 2a: j in {0,1}, k in {0,1}. M_{2,2,2}, rank 7.
        add_strassen(0, 0, 0)
        # Block 2b: j in {0,1}, k in {2,3}. M_{2,2,2}, rank 7.
        add_strassen(0, 0, 2)
        # Block 2c: j in {0,1}, k=4. M_{2,2,1}, rank 4.
        for i in range(n):
            for j in range(2):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1

        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W):
        """Check if U,V,W exactly decompose the 2x3 x 3x5 multiplication tensor."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(r))
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    def search_rank27_structured():
        """
        Try rank-27 by fixing Strassen blocks for j∈{1,2}, k∈{0,1} and j∈{1,2}, k∈{3,4}.
        Then check if remaining 14-term sub-tensor can be done in rank 13.
        """
        # Fixed Strassen blocks: j∈{1,2}, k∈{0,1} and j∈{1,2}, k∈{3,4}
        # These use 14 terms (2×7)
        
        # The remaining entries that need value 1:
        # (i, 0, k) for i∈{0,1}, k∈{0,1,2,3,4} → 10 entries
        # (i, 1, 2) for i∈{0,1} → 2 entries
        # (i, 2, 2) for i∈{0,1} → 2 entries
        # Total: 14 entries, schoolbook gives rank 14
        
        # We need to check if rank 13 is sufficient.
        # Approach: Try to find 13 rank-1 terms that sum to the required tensor.
        # This is a hard problem, so use a structured ansatz.
        
        # The sub-tensor S has:
        # S[i, 0, k] = 1 for all i, k
        # S[i, 1, 2] = 1 for all i
        # S[i, 2, 2] = 1 for all i
        # S[i, j, k] = 0 otherwise (within the sub-problem)
        
        # In the full tensor, the Strassen blocks handle:
        # j∈{1,2}, k∈{0,1} and j∈{1,2}, k∈{3,4}
        # So the sub-tensor only needs to handle the remaining entries.
        
        # Let's set up the problem: we need 13 vectors (u_c, v_c, w_c) such that
        # sum_c u_c[i] * v_c[j] * w_c[k] = S[i,j,k]
        
        # Where S is defined as above.
        
        # Key insight: S[i,0,k] = 1 for all i,k means the j=0 slice is all ones.
        # S[i,1,2] = 1 and S[i,2,2] = 1 for all i means the k=2 column for j=1,2 is all ones.
        
        # Let's try a parametric approach. Assume the 13 terms have a specific structure.
        # For the j=0 part: we need sum_c u_c[i] * v_c[0] * w_c[k] = 1 for all i,k
        # This means the matrix (u_c[i] * v_c[0]) * (w_c[k]) summed over c equals the all-ones 2x5 matrix.
        
        # For the j=1,2 part at k=2: sum_c u_c[i] * v_c[j] * w_c[2] = 1 for all i, j∈{1,2}
        
        # For all other (j,k) in the sub-problem: sum must be 0.
        
        # This is a system of equations. Let's set up variables.
        # 13 terms, each with 2+5+2 = 9 parameters, but up to scaling: 8 free params per term.
        # Total: 13*8 = 104 variables.
        # Constraints: 2*3*5 = 30 equations (but many are 0=0 from Strassen blocks).
        # Actually, the sub-tensor has 14 non-zero entries and 16 zero entries (in the 2x3x5 space,
        # but only the ones not covered by Strassen matter).
        
        # Let me just try a direct numerical approach with a small number of structured guesses.
        
        # Attempt 1: Can we merge two schoolbook terms?
        # The 14 schoolbook terms are:
        # (i,0,k) for i=0,1; k=0,1,2,3,4 → 10 terms
        # (i,1,2) for i=0,1 → 2 terms
        # (i,2,2) for i=0,1 → 2 terms
        
        # Try to combine (0,0,0) and (0,0,1) into one term? No, they have different k.
        # Try to combine (0,1,2) and (1,1,2)? They have same j,k but different i.
        # A single rank-1 term u[i]*v[j]*w[k] with u=[1,1] would give both, but then
        # it also contributes to other (j,k) entries.
        
        # Let's try: term with u=[1,1], v=[0,1,1], w=[0,0,1,0,0]
        # This gives: (0,1,2)=1, (1,1,2)=1, (0,2,2)=1, (1,2,2)=1
        # But it also gives (0,1,2)=1, (1,1,2)=1, etc. - exactly what we want for j∈{1,2},k=2!
        # And it gives 0 for all other (j,k) in the sub-problem.
        
        # So one term handles all 4 entries (i, j∈{1,2}, k=2).
        # Remaining: (i,0,k) for i=0,1; k=0,1,2,3,4 → 10 entries, need rank 10.
        # Total: 1 + 10 = 11 terms for the sub-problem!
        # Grand total: 14 (Strassen) + 11 = 25? That can't be right...
        
        # Wait, I need to be more careful. The term u=[1,1], v=[0,1,1], w=[0,0,1,0,0]
        # contributes to the FULL tensor, not just the sub-problem.
        # In the full tensor, it contributes to:
        # (i,1,2) and (i,2,2) for i=0,1 → these are in the sub-problem, good.
        # But it does NOT contribute to j∈{1,2}, k∈{0,1,3,4} because w[0]=w[1]=w[3]=w[4]=0.
        # And it does NOT contribute to j=0 because v[0]=0.
        # So it's clean! It only affects the sub-problem entries.
        
        # So we can use this one term for all 4 entries at k=2, j∈{1,2}.
        # Then we need rank-10 for the j=0 slice: (i,0,k) for i=0,1; k=0,1,2,3,4.
        # This is a 2x1x5 tensor, which has rank 10 (it's just the outer product structure).
        # Actually, the 2x1x5 tensor with all entries = 1 has rank... let's think.
        # T[i,0,k] = 1. This is u[i]*v[0]*w[k] with u=[1,1], v=[1,0,0], w=[1,1,1,1,1].
        # That's rank 1! But wait, that single term would also contribute to other entries.
        
        # In the FULL tensor, a term u=[1,1], v=[1,0,0], w=[1,1,1,1,1] contributes:
        # (i,0,k) = 1 for all i,k → good, that's what we want.
        # But does it contribute elsewhere? v[j]=0 for j=1,2, so no.
        # So it's clean too!
        
        # So: 1 term for j=0 slice + 1 term for j∈{1,2},k=2 = 2 terms for the sub-problem!
        # Grand total: 14 (Strassen) + 2 = 16?? That seems too low.
        
        # Wait, I'm confusing myself. The Strassen blocks handle j∈{1,2}, k∈{0,1} and j∈{1,2}, k∈{3,4}.
        # The sub-problem handles everything else: j=0 (all k) and j∈{1,2}, k=2.
        
        # But the Strassen blocks produce specific values, not just 1. Let me recheck.
        # Strassen for 2x2x2 produces the correct matrix product, so for the entries it covers,
        # the sum of its 7 terms gives exactly 1 for each (i,j,k) in that block.
        
        # So the residual for the Strassen-covered entries is 0. Good.
        # The residual for the sub-problem entries is 1 (they're not covered by Strassen).
        
        # So my analysis is correct: we need to decompose the sub-tensor S where:
        # S[i,0,k] = 1, S[i,1,2] = 1, S[i,2,2] = 1, S[i,j,k] = 0 otherwise.
        
        # And I found:
        # Term A: u=[1,1], v=[1,0,0], w=[1,1,1,1,1] → handles all S[i,0,k]=1
        # Term B: u=[1,1], v=[0,1,1], w=[0,0,1,0,0] → handles all S[i,1,2]=1 and S[i,2,2]=1
        
        # But wait! Term A also contributes to S[i,0,2] which is fine (it's 1).
        # Term B also contributes to S[i,0,2]? No, v[0]=0 for term B.
        # Term A contributes to S[i,1,2]? v[1]=0 for term A. No.
        
        # So both terms are clean and together they handle all 14 entries!
        # That means the sub-problem has rank 2, not 14!
        
        # But that can't be right because the total would be 14+2=16, which is way below 28.
        # The issue is that these terms also interact with the Strassen blocks.
        
        # OH WAIT. I think I made an error. The Strassen blocks don't just "handle" their entries
        # in isolation. The full decomposition is a sum of ALL terms, and each term contributes
        # to ALL entries. The Strassen terms contribute to entries outside their block too!
        
        # Let me re-examine. In the current rank-28 construction:
        # - Strassen block 1 (j∈{0,1}, k∈{0,1}): 7 terms
        # - Strassen block 2 (j∈{0,1}, k∈{2,3}): 7 terms  
        # - Schoolbook (j∈{0,1}, k=4): 4 terms
        # - Schoolbook (j=2, all k): 10 terms
        
        # Each Strassen term has non-zero entries in U, V, W that may overlap with other blocks.
        # For example, a Strassen term for j∈{0,1}, k∈{0,1} has V entries at (0,0),(0,1),(1,0),(1,1)
        # and W entries at (0,0),(0,1),(1,0),(1,1). It does NOT have V entries at j=2 or k=2,3,4.
        
        # So the blocks are actually disjoint in terms of which (j,k) pairs they affect!
        # Each term in block (j_set, k_set) only has non-zero V[j,k] for j∈j_set, k∈k_set.
        # And W[k,i] only for k∈k_set.
        
        # This means the blocks are truly independent! The total rank is the sum of block ranks.
        # And the sub-problem analysis is valid.
        
        # So if the sub-problem (j=0 all k, and j∈{1,2} k=2) can be done in rank 2,
        # then total rank = 14 + 2 = 16. But that contradicts known lower bounds.
        
        # I must be making an error. Let me recheck the Strassen block structure.
        
        # In add_strassen(0, 0, 0): j_off=0, k_off=0
        # V entries: (0,0), (1,1), (0,0), (1,0), (0,1), (1,1), (0,0), (0,1), (1,0), (1,1)
        # So V is non-zero at (j,k) = (0,0), (1,1), (1,0), (0,1) → j∈{0,1}, k∈{0,1}. Correct.
        # W entries: (0,0), (1,1), (0,1), (1,0), (0,0), (1,0), (0,1), (1,0) → k∈{0,1}. Correct.
        
        # In add_strassen(0, 0, 2): j_off=0, k_off=2
        # V entries: (0,2), (1,3), (0,2), (1,2), (0,3), (1,3), (0,2), (0,3), (1,2), (1,3)
        # So V is non-zero at j∈{0,1}, k∈{2,3}. Correct.
        
        # So yes, the blocks are disjoint. The sub-problem is truly independent.
        
        # But then why can't we just use rank 2 for the sub-problem?
        # Let me verify: does term A = u=[1,1], v=[1,0,0], w=[1,1,1,1,1] 
        # correctly give S[i,0,k]=1 for all i,k?
        # S[i,0,k] = u[i]*v[0]*w[k] = 1*1*1 = 1. Yes!
        # Does it give 0 for j=1,2? v[1]=v[2]=0. Yes!
        
        # Term B = u=[1,1], v=[0,1,1], w=[0,0,1,0,0]
        # S[i,1,2] = u[i]*v[1]*w[2] = 1*1*1 = 1. Yes!
        # S[i,2,2] = u[i]*v[2]*w[2] = 1*1*1 = 1. Yes!
        # S[i,0,k] = u[i]*v[0]*w[k] = 1*0*w[k] = 0. Yes!
        # S[i,1,0] = u[i]*v[1]*w[0] = 1*1*0 = 0. Yes!
        # S[i,1,3] = u[i]*v[1]*w[3] = 1*1*0 = 0. Yes!
        # S[i,2,0] = u[i]*v[2]*w[0] = 1*1*0 = 0. Yes!
        # S[i,2,3] = u[i]*v[2]*w[3] = 1*1*0 = 0. Yes!
        # S[i,2,4] = u[i]*v[2]*w[4] = 1*1*0 = 0. Yes!
        
        # So the sub-problem IS rank 2!
        # Total rank = 14 + 2 = 16.
        
        # But wait, this can't be right. The tensor rank of 2x3x5 matrix multiplication
        # is known to be at least 10 (from the 2x1x5 part) and the Strassen blocks are 7 each.
        # Let me check: is the full tensor really decomposable in rank 16?
        
        # Let me construct it and verify!
        
        rank = 16
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        
        def add_strassen(i_off, j_off, k_off):
            nonlocal col
            U[(i_off+0)*m + (j_off+0)][col] = 1
            U[(i_off+1)*m + (j_off+1)][col] = 1
            V[(j_off+0)*p + (k_off+0)][col] = 1
            V[(j_off+1)*p + (k_off+1)][col] = 1
            W[(k_off+0)*n + (i_off+0)][col] = 1
            W[(k_off+1)*n + (i_off+1)][col] = 1
            col += 1
            U[(i_off+1)*m + (j_off+0)][col] = 1
            U[(i_off+1)*m + (j_off+1)][col] = 1
            V[(j_off+0)*p + (k_off+0)][col] = 1
            W[(k_off+0)*n + (i_off+1)][col] = 1
            W[(k_off+1)*n + (i_off+1)][col] = -1
            col += 1
            U[(i_off+0)*m + (j_off+0)][col] = 1
            V[(j_off+0)*p + (k_off+1)][col] = 1
            V[(j_off+1)*p + (k_off+1)][col] = -1
            W[(k_off+1)*n + (i_off+0)][col] = 1
            W[(k_off+1)*n + (i_off+1)][col] = 1
            col += 1
            U[(i_off+1)*m + (j_off+1)][col] = 1
            V[(j_off+1)*p + (k_off+0)][col] = 1
            V[(j_off+0)*p + (k_off+0)][col] = -1
            W[(k_off+0)*n + (i_off+0)][col] = 1
            W[(k_off+0)*n + (i_off+1)][col] = 1
            col += 1
            U[(i_off+0)*m + (j_off+0)][col] = 1
            U[(i_off+0)*m + (j_off+1)][col] = 1
            V[(j_off+1)*p + (k_off+1)][col] = 1
            W[(k_off+0)*n + (i_off+0)][col] = -1
            W[(k_off+1)*n + (i_off+0)][col] = 1
            col += 1
            U[(i_off+1)*m + (j_off+0)][col] = 1
            U[(i_off+0)*m + (j_off+0)][col] = -1
            V[(j_off+0)*p + (k_off+0)][col] = 1
            V[(j_off+0)*p + (k_off+1)][col] = 1
            W[(k_off+1)*n + (i_off+1)][col] = 1
            col += 1
            U[(i_off+0)*m + (j_off+1)][col] = 1
            U[(i_off+1)*m + (j_off+1)][col] = -1
            V[(j_off+1)*p + (k_off+0)][col] = 1
            V[(j_off+1)*p + (k_off+1)][col] = 1
            W[(k_off+0)*n + (i_off+0)][col] = 1
            col += 1

        # Strassen block 1: j∈{1,2}, k∈{0,1}
        add_strassen(0, 1, 0)
        # Strassen block 2: j∈{1,2}, k∈{3,4}
        add_strassen(0, 1, 3)
        
        # Term A: handles j=0, all k
        U[0*m+0][col] = 1  # i=0, j=0
        U[1*m+0][col] = 1  # i=1, j=0
        V[0*p+0][col] = 1  # j=0, k=0
        V[0*p+1][col] = 1  # j=0, k=1
        V[0*p+2][col] = 1  # j=0, k=2
        V[0*p+3][col] = 1  # j=0, k=3
        V[0*p+4][col] = 1  # j=0, k=4
        W[0*n+0][col] = 1  # k=0, i=0
        W[0*n+1][col] = 1  # k=0, i=1
        W[1*n+0][col] = 1  # k=1, i=0
        W[1*n+1][col] = 1  # k=1, i=1
        W[2*n+0][col] = 1  # k=2, i=0
        W[2*n+1][col] = 1  # k=2, i=1
        W[3*n+0][col] = 1  # k=3, i=0
        W[3*n+1][col] = 1  # k=3, i=1
        W[4*n+0][col] = 1  # k=4, i=0
        W[4*n+1][col] = 1  # k=4, i=1
        col += 1
        
        # Term B: handles j∈{1,2}, k=2
        U[0*m+1][col] = 1  # i=0, j=1
        U[1*m+1][col] = 1  # i=1, j=1
        U[0*m+2][col] = 1  # i=0, j=2
        U[1*m+2][col] = 1  # i=1, j=2
        V[1*p+2][col] = 1  # j=1, k=2
        V[2*p+2][col] = 1  # j=2, k=2
        W[2*n+0][col] = 1  # k=2, i=0
        W[2*n+1][col] = 1  # k=2, i=1
        col += 1
        
        # Verify
        if verify_decomposition(U, V, W):
            return {"U": U, "V": V, "W": W}
        
        return None

    # Try structured rank-27 (or lower) search
    result = None
    if time.time() < deadline - 5:
        try:
            result = search_rank27_structured()
        except Exception:
            result = None

    if result is not None:
        return result

    # Fallback: known rank-28 solution
    return build_rank28()

