import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = 3, 4, 4
    
    def _build_3x2x2_rank11():
        """Build the known rank-11 decomposition for 3x2x2 matmul."""
        U = np.zeros((6, 11))
        V = np.zeros((4, 11))
        W = np.zeros((6, 11))
        
        # Strassen part (cols 0-6) for rows 0,1 of A
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        # Extra part (cols 7-10) for Row 2 of A
        U[4, 7] = 1
        V[0, 7] = 1
        W[4, 7] = 1
        
        U[5, 8] = 1
        V[2, 8] = 1
        W[4, 8] = 1
        
        U[4, 9] = 1
        V[1, 9] = 1
        W[5, 9] = 1
        
        U[5, 10] = 1
        V[3, 10] = 1
        W[5, 10] = 1
        
        return U, V, W
    
    def _build_3x2x2_rank10():
        """Try to find rank-10 decomposition of 3x2x2 multiplication.
        
        Key insight: The 3x2x2 tensor can potentially be decomposed with rank 10
        by using Strassen for the 2x2x2 sub-tensor (rows 0,1) and 3 additional
        terms for row 2 that share structure with existing B-factors.
        
        Strategy: Use algebraic manipulation. The tensor T[a,b,c] for 3x2x2:
        - For a in {0,1,2,3,4,5}, b in {0,1,2,3}, c in {0,1,2,3,4,5}
        - a = a_row*2 + a_col, b = b_row*2 + b_col, c = c_col*3 + c_row
        - T[a,b,c] = 1 iff a_col == b_row, and c_row == a_row, c_col == b_col
        
        So the non-zero entries are:
        a_col=0 (a in {0,2,4}), b_row=0 (b in {0,1}): c = b_col*3 + a_row
          (0,0)->c=0, (0,1)->c=3, (2,0)->c=1, (2,1)->c=4, (4,0)->c=2, (4,1)->c=5
        a_col=1 (a in {1,3,5}), b_row=1 (b in {2,3}): c = b_col*3 + a_row
          (1,2)->c=0, (1,3)->c=3, (3,2)->c=1, (3,3)->c=4, (5,2)->c=2, (5,3)->c=5
        
        That's 12 non-zero entries.
        
        For rank 10, we need 10 products. The 2x2x2 sub-problem (a in {0,1,2,3}) 
        needs rank 7 (Strassen). The remaining entries involve a=4,5 and need 
        to be covered by 3 additional terms.
        
        The 6 entries for a=4,5 are:
        (4,0)->c=2: A[2,0]*B[0,0]
        (4,1)->c=5: A[2,0]*B[0,1]
        (5,2)->c=2: A[2,1]*B[1,0]
        (5,3)->c=5: A[2,1]*B[1,1]
        Plus c=0,1,3,4 entries for a=4,5 which are 0.
        
        So we need:
        M8, M9, M10 such that:
        - They contribute to c=2: A[2,0]*B[0,0] + A[2,1]*B[1,0]
        - They contribute to c=5: A[2,0]*B[0,1] + A[2,1]*B[1,1]
        - They contribute 0 to c=0,1,3,4
        - They contribute 0 to a=0,1,2,3 entries (or their contributions cancel)
        
        This is a linear system. Let's set up:
        For each new term r in {7,8,9}:
          U[4,r], U[5,r] are the A-factors for row 2
          V[0,r], V[1,r], V[2,r], V[3,r] are B-factors
          W[0,r]...W[5,r] are C-factors
        
        Constraints from c=0 (a_row=0, b_col=0):
          Sum over all terms: U[a,r]*V[b,r]*W[0,r] = 0 for a=4,5 with b=0,1 (a_col=0, b_row=0)
          Actually W[0,r] corresponds to c=0 which is c_col=0, c_row=0.
          The entry (a=4, b=0, c=0): a_col=0, b_row=0, so this IS a non-zero entry in the full tensor?
          Wait, c=0 means c_col=0, c_row=0. a=4 means a_row=2, a_col=0. b=0 means b_row=0, b_col=0.
          For the matmul: C[a_row, b_col] = sum_k A[a_row, k] * B[k, b_col]
          C[2, 0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
          So (a=4, b=0, c=0) should be 0 because c_row=0 != a_row=2.
          
          Let me recheck: c = c_col * 3 + c_row. c=0 -> c_col=0, c_row=0.
          The tensor entry T[a,b,c] = 1 iff a_col == b_row AND c_row == a_row AND c_col == b_col.
          
          (a=4, b=0, c=0): a_col=0, b_row=0 ✓, c_row=0, a_row=2 ✗. So T=0.
          (a=4, b=0, c=2): a_col=0, b_row=0 ✓, c_row=2, a_row=2 ✓, c_col=0, b_col=0 ✓. T=1.
          (a=4, b=1, c=5): a_col=0, b_row=0 ✓, c_row=2, a_row=2 ✓, c_col=1, b_col=1 ✓. T=1.
          (a=5, b=2, c=2): a_col=1, b_row=1 ✓, c_row=2, a_row=2 ✓, c_col=0, b_col=0 ✓. T=1.
          (a=5, b=3, c=5): a_col=1, b_row=1 ✓, c_row=2, a_row=2 ✓, c_col=1, b_col=1 ✓. T=1.
        
        So the 4 non-zero entries for row 2 are: (4,0,2), (4,1,5), (5,2,2), (5,3,5).
        
        For the 3 new terms, we need:
        Term r: U4_r * V0_r * W2_r + U5_r * V2_r * W2_r contributes to (a=4,b=0,c=2) and (a=5,b=2,c=2)
        Actually let me be more careful.
        
        The decomposition is: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        
        For the 3 new terms (r=7,8,9), the Strassen terms (r=0..6) already handle a in {0,1,2,3}.
        But they might also have non-zero entries at a=4,5 or c=2,5 that need to be zero.
        Let me check: In the Strassen part, U[4,r] and U[5,r] are 0 for r=0..6 (only U[0..3] are set).
        So the Strassen terms contribute 0 to all entries with a=4 or a=5. Good.
        
        Also need to check c=2 and c=5: W[2,r] and W[5,r] for r=0..6.
        W[2,1]=-1, W[2,3]=1, W[2,5]=1, W[5,r]=0 for all r=0..6 (only W[0,1,2,3,4] are set, plus W[5] is never set in Strassen).
        Wait, let me recheck the Strassen W values:
        W[0,0]=1, W[3,0]=1
        W[2,1]=1, W[3,1]=-1
        W[1,2]=1, W[3,2]=1
        W[0,3]=1, W[2,3]=1
        W[0,4]=-1, W[1,4]=1
        W[3,5]=1
        W[0,6]=1
        
        So W[2,1]=1, W[2,3]=1. These are non-zero. But since U[4,r]=U[5,r]=0 for r=0..6, the contribution to a=4,5 is 0 regardless of W values. Good.
        
        Now for the 3 new terms, we need:
        (a=4, b=0, c=2): sum_{r=7,8,9} U[4,r]*V[0,r]*W[2,r] = 1
        (a=4, b=1, c=5): sum_{r=7,8,9} U[4,r]*V[1,r]*W[5,r] = 1
        (a=5, b=2, c=2): sum_{r=7,8,9} U[5,r]*V[2,r]*W[2,r] = 1
        (a=5, b=3, c=5): sum_{r=7,8,9} U[5,r]*V[3,r]*W[5,r] = 1
        
        And all other entries must be 0:
        (a=4, b=0, c=0): sum U[4,r]*V[0,r]*W[0,r] = 0
        (a=4, b=0, c=1): sum U[4,r]*V[0,r]*W[1,r] = 0
        (a=4, b=0, c=3): sum U[4,r]*V[0,r]*W[3,r] = 0
        (a=4, b=0, c=4): sum U[4,r]*V[0,r]*W[4,r] = 0
        (a=4, b=1, c=0): sum U[4,r]*V[1,r]*W[0,r] = 0
        (a=4, b=1, c=1): sum U[4,r]*V[1,r]*W[1,r] = 0
        (a=4, b=1, c=2): sum U[4,r]*V[1,r]*W[2,r] = 0
        (a=4, b=1, c=3): sum U[4,r]*V[1,r]*W[3,r] = 0
        (a=4, b=1, c=4): sum U[4,r]*V[1,r]*W[4,r] = 0
        (a=5, b=2, c=0): sum U[5,r]*V[2,r]*W[0,r] = 0
        (a=5, b=2, c=1): sum U[5,r]*V[2,r]*W[1,r] = 0
        (a=5, b=2, c=3): sum U[5,r]*V[2,r]*W[3,r] = 0
        (a=5, b=2, c=4): sum U[5,r]*V[2,r]*W[4,r] = 0
        (a=5, b=2, c=5): sum U[5,r]*V[2,r]*W[5,r] = 0
        (a=5, b=3, c=0): sum U[5,r]*V[3,r]*W[0,r] = 0
        (a=5, b=3, c=1): sum U[5,r]*V[3,r]*W[1,r] = 0
        (a=5, b=3, c=2): sum U[5,r]*V[3,r]*W[2,r] = 0
        (a=5, b=3, c=3): sum U[5,r]*V[3,r]*W[3,r] = 0
        (a=5, b=3, c=4): sum U[5,r]*V[3,r]*W[4,r] = 0
        
        That's 4 equations = 1 and 16 equations = 0, total 20 constraints on 3*4*3 = 36 unknowns (U[4,7:10], U[5,7:10], V[0:4,7:10], W[0:6,7:10]).
        
        Actually it's 3 terms * (2 U + 4 V + 6 W) = 3*12 = 36 unknowns.
        20 constraints. So we have 16 degrees of freedom. Should be solvable.
        
        Let me try a structured approach:
        M7 = A[2,0] * B[0,0]  -> U[4,7]=1, V[0,7]=1, W[2,7]=1
        M8 = A[2,0] * B[0,1]  -> U[4,8]=1, V[1,8]=1, W[5,8]=1
        M9 = A[2,1] * (B[1,0] + B[1,1]) -> U[5,9]=1, V[2,9]=1, V[3,9]=1, W[2,9]=1, W[5,9]=1
        
        Check:
        (4,0,2): 1*1*1 + 0 + 0 = 1 ✓
        (4,1,5): 0 + 1*1*1 + 0 = 1 ✓
        (5,2,2): 0 + 0 + 1*1*1 = 1 ✓
        (5,3,5): 0 + 0 + 1*1*1 = 1 ✓
        
        Now check zero constraints:
        (4,0,0): 1*1*W[0,7] + 0 + 0 = W[0,7]. Need W[0,7]=0. ✓ (W[0,7] not set, defaults to 0)
        (4,0,1): W[1,7]=0 ✓
        (4,0,3): W[3,7]=0 ✓
        (4,0,4): W[4,7]=0 ✓
        (4,1,0): 0 + 1*1*W[0,8] + 0 = W[0,8]. Need 0. ✓
        (4,1,1): W[1,8]=0 ✓
        (4,1,2): W[2,8]=0 ✓
        (4,1,3): W[3,8]=0 ✓
        (4,1,4): W[4,8]=0 ✓
        (5,2,0): 0 + 0 + 1*1*W[0,9] = W[0,9]. Need 0. ✓
        (5,2,1): W[1,9]=0 ✓
        (5,2,3): W[3,9]=0 ✓
        (5,2,4): W[4,9]=0 ✓
        (5,2,5): 0 + 0 + 1*1*W[5,9] = W[5,9] = 1. But we need this to be 0! ✗
        
        Problem: (5,2,5) should be 0 but M9 gives 1*1*1=1.
        
        Let me fix: M9 = A[2,1]*B[1,0] only, and add M10 = A[2,1]*B[1,1].
        But that's 4 terms for row 2, giving rank 11.
        
        Alternative: M8 = A[2,0]*(B[0,0]+B[0,1]), M9 = A[2,1]*B[1,0], M10 = A[2,1]*B[1,1]
        (4,0,2): 1*1*1 = 1 ✓ (W[2,8]=1)
        (4,1,5): 1*1*1 = 1 ✓ (W[5,8]=1)
        (5,2,2): 1*1*1 = 1 ✓ (W[2,9]=1)
        (5,3,5): 1*1*1 = 1 ✓ (W[5,10]=1)
        
        Check zeros:
        (4,0,0): W[0,8]=0 ✓
        (4,0,1): W[1,8]=0 ✓
        (4,0,3): W[3,8]=0 ✓
        (4,0,4): W[4,8]=0 ✓
        (4,1,0): W[0,8]=0 ✓
        (4,1,1): W[1,8]=0 ✓
        (4,1,2): W[2,8]=1. But (4,1,2) should be 0! ✗
        
        Because M8 = A[2,0]*(B[0,0]+B[0,1]) has V[0,8]=1, V[1,8]=1, W[2,8]=1, W[5,8]=1.
        (4,1,2): U[4,8]*V[1,8]*W[2,8] = 1*1*1 = 1 ≠ 0.
        
        The issue is that a rank-1 term U[a,r]*V[b,r]*W[c,r] couples all three indices.
        
        Let me think differently. We need 3 bilinear forms in (A_row2, B) that produce the 4 required C values.
        
        C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
        C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
        
        With 3 products, we can compute 3 linear combinations. We need 2 outputs.
        So we have 1 degree of freedom.
        
        Let:
        M7 = alpha*A[2,0]*B[0,0] + beta*A[2,1]*B[1,0]  (but this must be rank 1!)
        
        A rank-1 term is of the form (linear in A) * (linear in B).
        So M7 = (a1*A[2,0] + a2*A[2,1]) * (b1*B[0,0] + b2*B[0,1] + b3*B[1,0] + b4*B[1,1])
        
        This expands to 8 terms. We need the cross terms to cancel or be zero.
        
        For M7 to only contribute to C[2,0] and C[2,1] (c=2 and c=5):
        W[2,7] and W[5,7] can be non-zero, others must be 0.
        
        The contribution to c=2 is: (a1*A[2,0]+a2*A[2,1])*(b1*B[0,0]+b2*B[0,1]+b3*B[1,0]+b4*B[1,1]) * W[2,7]
        We want this to equal some fraction of A[2,0]*B[0,0] + A[2,1]*B[1,0].
        
        For the product to only have A[2,0]*B[0,0] and A[2,1]*B[1,0] terms, we need:
        b2 = 0 (no B[0,1]), b4 = 0 (no B[1,1])
        And a1*b1 for A[2,0]*B[0,0], a2*b3 for A[2,1]*B[1,0]
        But we also get a1*b3*A[2,0]*B[1,0] and a2*b1*A[2,1]*B[0,0] which are unwanted.
        
        So we need a1*b3 = 0 and a2*b1 = 0.
        If a1≠0 and b3≠0, then a1*b3≠0. So either a1=0 or b3=0.
        If b3=0, then we can't get A[2,1]*B[1,0].
        If a1=0, then we can't get A[2,0]*B[0,0].
        
        So a single rank-1 term can't give exactly A[2,0]*B[0,0] + A[2,1]*B[1,0].
        We need at least 2 terms for C[2,0] alone (unless we use the Strassen terms to help).
        
        Wait! The Strassen terms might have non-zero W[2,r] or W[5,r] values that contribute to c=2 or c=5.
        But we established that U[4,r]=U[5,r]=0 for r=0..6, so they don't contribute to a=4,5 entries.
        
        Hmm, so we really do need the 3 new terms to handle all 4 entries for row 2.
        
        Let me try:
        M7 = (A[2,0]+A[2,1]) * B[0,0]  -> U[4,7]=1, U[5,7]=1, V[0,7]=1, W[2,7]=1
        M8 = A[2,0] * B[0,1]  -> U[4,8]=1, V[1,8]=1, W[5,8]=1
        M9 = A[2,1] * B[1,0]  -> U[5,9]=1, V[2,9]=1, W[2,9]=1
        
        Check:
        (4,0,2): 1*1*1 + 0 + 0 = 1 ✓
        (4,1,5): 0 + 1*1*1 + 0 = 1 ✓
        (5,2,2): 1*1*1 + 0 + 1*1*1 = 2 ✗ (should be 1)
        
        Let me adjust:
        M7 = (A[2,0]-A[2,1]) * B[0,0]  -> U[4,7]=1, U[5,7]=-1, V[0,7]=1, W[2,7]=1
        M8 = A[2,0] * B[0,1]  -> U[4,8]=1, V[1,8]=1, W[5,8]=1
        M9 = A[2,1] * B[1,0]  -> U[5,9]=1, V[2,9]=1, W[2,9]=1
        
        (4,0,2): 1*1*1 + 0 + 0 = 1 ✓
        (4,1,5): 0 + 1*1*1 + 0 = 1 ✓
        (5,2,2): (-1)*1*1 + 0 + 1*1*1 = 0 ✗ (should be 1)
        
        Try:
        M7 = A[2,0] * B[0,0]  -> U[4,7]=1, V[0,7]=1, W[2,7]=1
        M8 = A[2,0] * B[0,1]  -> U[4,8]=1, V[1,8]=1, W[5,8]=1
        M9 = (A[2,1]+A[2,0]) * B[1,0]  -> U[5,9]=1, U[4,9]=1, V[2,9]=1, W[2,9]=1
        
        (4,0,2): 1*1*1 + 0 + 1*1*1 = 2 ✗
        
        This is getting complicated. Let me just use 4 terms for row 2 (rank 11) and try to optimize.
        Actually, let me try a numerical approach with the full 3x2x2 tensor.
        
        """
        # Build the 3x2x2 tensor
        tensor = np.zeros((6, 4, 6))
        for a in range(6):
            a_row, a_col = divmod(a, 2)
            for b in range(4):
                b_row, b_col = divmod(b, 2)
                if a_col == b_row:
                    c = b_col * 3 + a_row
                    tensor[a, b, c] = 1.0
        
        # Try rank 10 with gradient-based optimization
        # Initialize with rank-11 solution, zero out one column, and optimize
        U11, V11, W11 = _build_3x2x2_rank11()
        
        # Flatten the tensor for vectorized operations
        # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # We can write this as: for each r, outer(U[:,r], V[:,r]) @ W[:,r] ... no
        # Actually: pred[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # = sum_r (U[:,r] * W[:,r]) outer with V[:,r]... 
        # Let's use einsum
        def _decompose_error(U, V, W, tensor):
            """Compute max absolute error."""
            pred = np.einsum('ar,br,cr->abc', U, V, W)
            return np.max(np.abs(pred - tensor))
        
        # Try several random initializations for rank 10
        np.random.seed(123)
        best_err = float('inf')
        best_U = best_V = best_W = None
        
        for trial in range(20):
            if time.time() > deadline - 40:
                break
            
            # Start from rank-11, remove one random column, and optimize
            U = U11[:, :10].copy()
            V = V11[:, :10].copy()
            W = W11[:, :10].copy()
            
            # Add small random perturbation to the last column
            U[:, 9] += np.random.randn(6) * 0.1
            V[:, 9] += np.random.randn(4) * 0.1
            W[:, 9] += np.random.randn(6) * 0.1
            
            # Gradient descent with vectorized operations
            lr = 0.005
            for it in range(3000):
                if time.time() > deadline - 30:
                    break
                
                # Compute prediction
                pred = np.einsum('ar,br,cr->abc', U, V, W)
                err = pred - tensor
                
                # Gradients (vectorized)
                # dU[a,r] = 2 * sum_{b,c} err[a,b,c] * V[b,r] * W[c,r]
                grad_U = 2.0 * np.einsum('abc,br,cr->ar', err, V, W)
                grad_V = 2.0 * np.einsum('abc,ar,cr->br', err, U, W)
                grad_W = 2.0 * np.einsum('abc,ar,br->cr', err, U, V)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                if it % 500 == 0:
                    cur_err = _decompose_error(U, V, W, tensor)
                    if cur_err < best_err:
                        best_err = cur_err
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                    if cur_err < 1e-8:
                        break
            
            cur_err = _decompose_error(U, V, W, tensor)
            if cur_err < best_err:
                best_err = cur_err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
        
        if best_err < 1e-6:
            # Try rounding to half-integers
            U_r = np.round(best_U * 2) / 2
            V_r = np.round(best_V * 2) / 2
            W_r = np.round(best_W * 2) / 2
            if _decompose_error(U_r, V_r, W_r, tensor) < 1e-6:
                return U_r, V_r, W_r
            else:
                return best_U, best_V, best_W
        
        return None
    
    def _try_remove_columns(U, V, W, tensor, max_time=10):
        """Try to remove columns from a decomposition by checking if they're redundant."""
        rank = U.shape[1]
        for col in range(rank):
            if time.time() > start_time + max_time:
                break
            # Try removing column col
            mask = np.ones(rank, dtype=bool)
            mask[col] = False
            U_sub = U[:, mask]
            V_sub = V[:, mask]
            W_sub = W[:, mask]
            if _decompose_error(U_sub, V_sub, W_sub, tensor) < 1e-8:
                return U_sub, V_sub, W_sub
        return U, V, W
    
    def _decompose_error(U, V, W, tensor):
        """Compute max absolute error."""
        pred = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(pred - tensor))
    
    # Build the 3x2x2 tensor for verification
    tensor_322 = np.zeros((6, 4, 6))
    for a in range(6):
        a_row, a_col = divmod(a, 2)
        for b in range(4):
            b_row, b_col = divmod(b, 2)
            if a_col == b_row:
                c = b_col * 3 + a_row
                tensor_322[a, b, c] = 1.0
    
    # Try rank 10
    U_loc = V_loc = W_loc = None
    rank = 11
    
    if time.time() < deadline - 50:
        result = _build_3x2x2_rank10()
        if result is not None:
            U_loc, V_loc, W_loc = result
            rank = 10
            # Try to reduce further
            U_loc, V_loc, W_loc = _try_remove_columns(U_loc, V_loc, W_loc, tensor_322, max_time=10)
            rank = U_loc.shape[1]
    
    if U_loc is None:
        # Fall back to rank 11
        U_loc, V_loc, W_loc = _build_3x2x2_rank11()
        rank = 11
        # Try to reduce
        U_loc, V_loc, W_loc = _try_remove_columns(U_loc, V_loc, W_loc, tensor_322, max_time=10)
        rank = U_loc.shape[1]
    
    total_rank = 4 * rank
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Mapping function
    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * rank
        
        for r in range(3):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_row = a_row_start + r
                global_a_col = a_col_start + c
                global_a_idx = global_a_row * m + global_a_col
                U[global_a_idx, col_offset:col_offset+rank] += U_loc[local_a_idx, :]
                
        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_row = b_row_start + r
                global_b_col = b_col_start + c
                global_b_idx = global_b_row * p + global_b_col
                V[global_b_idx, col_offset:col_offset+rank] += V_loc[local_b_idx, :]
                
        for r in range(3):
            for c in range(2):
                local_c_idx = r * 2 + c
                global_c_row = c_row_start + r
                global_c_col = c_col_start + c
                global_c_idx = global_c_col * n + global_c_row
                W[global_c_idx, col_offset:col_offset+rank] += W_loc[local_c_idx, :]

    add_block(0, 0, 0, 0, 0, 0, 0)
    add_block(1, 0, 2, 2, 0, 0, 0)
    add_block(2, 0, 0, 0, 2, 0, 2)
    add_block(3, 0, 2, 2, 2, 0, 2)
    
    # Verify decomposition is correct
    if time.time() < deadline:
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # Vectorized verification
        for a in range(n*m):
            a_row, a_col = divmod(a, m)
            for b in range(m*p):
                b_row, b_col = divmod(b, p)
                if a_col != b_row:
                    continue
                c_row = a_row
                c_col = b_col
                c = c_col * n + c_row
                expected = 1.0
                computed = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                if abs(computed - expected) > 1e-10:
                    # Fallback to trivial rank-48
                    U_fallback = np.zeros((n*m, n*m*p))
                    V_fallback = np.zeros((m*p, n*m*p))
                    W_fallback = np.zeros((p*n, n*m*p))
                    for a_idx in range(n*m):
                        a_r, a_c = divmod(a_idx, m)
                        for b_idx in range(m*p):
                            b_r, b_c = divmod(b_idx, p)
                            if a_c == b_r:
                                c_idx = b_c * n + a_r
                                rank_idx = a_idx * (m*p) + b_idx
                                U_fallback[a_idx, rank_idx] = 1
                                V_fallback[b_idx, rank_idx] = 1
                                W_fallback[c_idx, rank_idx] = 1
                    return {
                        "U": U_fallback.tolist(),
                        "V": V_fallback.tolist(),
                        "W": W_fallback.tolist()
                    }
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
