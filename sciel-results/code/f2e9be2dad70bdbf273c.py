"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill
    
    def verify_fast(U_mat, V_mat, W_mat, rank_val):
        """Fast vectorized verification using numpy."""
        U_np = np.array(U_mat, dtype=np.float64)
        V_np = np.array(V_mat, dtype=np.float64)
        W_np = np.array(W_mat, dtype=np.float64)
        
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.tile(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        a_idx = i_idx * m + j_idx
        b_idx = j_idx * p + k_idx
        c_idx = k_idx * n + i_idx
        
        vals = np.sum(U_np[a_idx] * V_np[b_idx] * W_np[c_idx], axis=1)
        
        return np.allclose(vals, 1.0, atol=1e-10)
    
    # Build rank-42 baseline (independent Strassen per 2x2 block)
    rank42 = 42
    U42 = [[0] * rank42 for _ in range(n * m)]
    V42 = [[0] * rank42 for _ in range(m * p)]
    W42 = [[0] * rank42 for _ in range(p * n)]
    
    U_forms = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    
    for u in range(6):
        for t in range(7):
            c = u * 7 + t
            for i in range(4):
                U42[i][c] = U_forms[t][i]
            if t == 0:
                V42[0*p + 2*u][c] += 1; V42[1*p + 2*u+1][c] += 1
            elif t == 1:
                V42[0*p + 2*u][c] += 1
            elif t == 2:
                V42[0*p + 2*u+1][c] += 1; V42[1*p + 2*u+1][c] -= 1
            elif t == 3:
                V42[1*p + 2*u][c] += 1; V42[0*p + 2*u][c] -= 1
            elif t == 4:
                V42[1*p + 2*u+1][c] += 1
            elif t == 5:
                V42[0*p + 2*u][c] += 1; V42[0*p + 2*u+1][c] += 1
            elif t == 6:
                V42[1*p + 2*u][c] += 1; V42[1*p + 2*u+1][c] += 1
            
            if t == 0:
                W42[(2*u)*n + 0][c] += 1; W42[(2*u+1)*n + 1][c] += 1
            elif t == 1:
                W42[(2*u)*n + 1][c] += 1; W42[(2*u+1)*n + 1][c] -= 1
            elif t == 2:
                W42[(2*u+1)*n + 0][c] += 1; W42[(2*u+1)*n + 1][c] += 1
            elif t == 3:
                W42[(2*u)*n + 0][c] += 1; W42[(2*u)*n + 1][c] += 1
            elif t == 4:
                W42[(2*u)*n + 0][c] -= 1; W42[(2*u+1)*n + 0][c] += 1
            elif t == 5:
                W42[(2*u+1)*n + 1][c] += 1
            elif t == 6:
                W42[(2*u)*n + 0][c] += 1
    
    best_result = {"U": U42, "V": V42, "W": W42}
    
    if not verify_fast(U42, V42, W42, rank42):
        raise ValueError("Baseline invalid")
    
    # Now attempt rank-39: three groups of 2x2 x 2x4, each with rank 13
    # Construct rank-13 decomposition for 2x2 x 2x4 via systematic search
    # The tensor for 2x2 x 2x4 has 2*2*4 = 16 non-zero entries
    # We need 13 rank-1 terms that sum to the tensor
    
    # Known result: rank of 2x2 x 2x4 multiplication tensor is 13
    # Let me construct it using the approach: take Strassen's 7 for first block,
    # Strassen's 7 for second block, then find 1 shared term that replaces 2 terms.
    
    # Alternative: use the explicit construction from literature.
    # For 2x2 x 2x4, the rank-13 decomposition can be built as follows:
    # Use 13 bilinear forms where the A-forms and B-forms are chosen to cover
    # all 16 entries of the output with 13 multiplications.
    
    # Let me try to construct it systematically.
    # A = [[a,b],[c,d]], B = [[e,f,g,h],[i,j,k,l]]
    # C = A*B = [[ae+bi, af+bj, ag+bk, ah+bl], [ce+di, cf+dj, cg+dk, ch+dl]]
    
    # 16 output entries, need 13 rank-1 terms.
    # Strategy: use Strassen-like forms but with shared structure.
    
    # Let me try a direct construction:
    # The 13 forms are:
    # t0: (a+d) * (e+j)  -> contributes to C[0,0] and C[1,1]
    # t1: (c+d) * (e)    -> contributes to C[1,0]
    # t2: (a) * (f)      -> contributes to C[0,1]
    # t3: (d) * (j)      -> contributes to C[1,1]
    # t4: (a+b) * (i)    -> contributes to C[0,0]
    # t5: (c-a) * (e)    -> contributes to C[1,0]
    # t6: (b-d) * (j)    -> contributes to C[0,1] and C[1,1]
    # t7: (a) * (g)      -> contributes to C[0,2]
    # t8: (d) * (k)      -> contributes to C[1,2]
    # t9: (a+b) * (h)    -> contributes to C[0,3]
    # t10: (c+d) * (l)   -> contributes to C[1,3]
    # t11: (a) * (k)     -> contributes to C[0,2]
    # t12: (d) * (g)     -> contributes to C[1,2]
    
    # This is getting complicated. Let me use a different approach:
    # Build the tensor and use a greedy/iterative rank reduction.
    
    # Actually, let me use a known explicit rank-13 decomposition.
    # From Coppersmith's work, for 2x2 x 2x4:
    
    r13 = 13
    # U: (4, 13) - linear forms in A = [[a00,a01],[a10,a11]]
    # V: (8, 13) - linear forms in B = [[b00,b01,b02,b03],[b10,b11,b12,b13]]
    # W: (8, 13) - linear forms in C = [[c00,c01,c02,c03],[c10,c11,c12,c13]]
    
    # Let me construct this by starting with 14 (7+7 for two blocks) and finding
    # a way to merge one pair.
    
    # Block 0 (columns 0,1): Strassen forms
    # Block 1 (columns 2,3): Strassen forms
    
    # The key insight: some A-forms can be shared between blocks.
    # In Strassen, the A-forms are: a00+a11, a10+a11, a00, a11, a00+a01, a10-a00, a01-a11
    # If we use the same A-forms for both blocks, we save on the A side.
    # But we still need separate B-forms for each block.
    
    # Total rank with shared A-forms: 7 A-forms * 2 blocks of B-forms = still 14
    # unless we can combine B-forms too.
    
    # Let me try a completely different structural approach:
    # Use the fact that 2x2 x 2x4 can be embedded in 2x2 x 2x2 x 2x2
    # and use the known rank-13 result.
    
    # I'll use a verified construction. Let me define it explicitly:
    
    # After careful analysis, here's a rank-13 decomposition:
    # Using the notation where A indices are (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices are (0,0)=0, (0,1)=1, (0,2)=2, (0,3)=3, (1,0)=4, (1,1)=5, (1,2)=6, (1,3)=7
    # C indices are same as B: (0,0)=0, ..., (1,3)=7
    
    U13 = np.zeros((4, r13))
    V13 = np.zeros((8, r13))
    W13 = np.zeros((8, r13))
    
    # Form 0: (a00+a11) * (b00+b11) -> C[0,0] += 1, C[1,1] += 1
    U13[0,0]=1; U13[3,0]=1
    V13[0,0]=1; V13[5,0]=1
    W13[0,0]=1; W13[5,0]=1
    
    # Form 1: (a10+a11) * b00 -> C[1,0] += 1
    U13[2,1]=1; U13[3,1]=1
    V13[0,1]=1
    W13[4,1]=1
    
    # Form 2: a00 * b01 -> C[0,1] += 1
    U13[0,2]=1
    V13[1,2]=1
    W13[1,2]=1
    
    # Form 3: a11 * b11 -> C[1,1] += 1
    U13[3,3]=1
    V13[5,3]=1
    W13[5,3]=1
    
    # Form 4: (a00+a01) * b10 -> C[0,0] += 1
    U13[0,4]=1; U13[1,4]=1
    V13[4,4]=1
    W13[0,4]=1
    
    # Form 5: (a10-a00) * b00 -> C[1,0] += 1
    U13[2,5]=1; U13[0,5]=-1
    V13[0,5]=1
    W13[4,5]=1
    
    # Form 6: (a01-a11) * b11 -> C[0,1] += 1, C[1,1] -= 1
    U13[1,6]=1; U13[3,6]=-1
    V13[5,6]=1
    W13[1,6]=1; W13[5,6]=-1
    
    # Now for columns 2,3:
    # Form 7: a00 * b02 -> C[0,2] += 1
    U13[0,7]=1
    V13[2,7]=1
    W13[2,7]=1
    
    # Form 8: a01 * b02 -> C[0,2] += 1
    U13[1,8]=1
    V13[2,8]=1
    W13[2,8]=1
    
    # Form 9: a10 * b12 -> C[1,2] += 1
    U13[2,9]=1
    V13[6,9]=1
    W13[6,9]=1
    
    # Form 10: a11 * b12 -> C[1,2] += 1
    U13[3,10]=1
    V13[6,10]=1
    W13[6,10]=1
    
    # Form 11: a00 * b03 -> C[0,3] += 1
    U13[0,11]=1
    V13[3,11]=1
    W13[3,11]=1
    
    # Form 12: a01 * b03 + a10 * b13 + a11 * b13 -> need 3 terms but only 1 left
    # This won't work. Let me reconsider.
    
    # The naive approach gives rank 16. We need to be smarter.
    # Let me use Strassen for each 2x2 block but share some A-forms.
    
    # Reset and try again with a proper construction.
    U13 = np.zeros((4, r13))
    V13 = np.zeros((8, r13))
    W13 = np.zeros((8, r13))
    
    # Use 7 Strassen forms for block 0 (cols 0,1) and 7 for block 1 (cols 2,3)
    # but merge one: the a00 and a11 forms are used in both blocks.
    # Actually in Strassen, each form is distinct. Let me just use 14 and see
    # if I can find a valid rank-13 by solving a system.
    
    # For now, let me try to verify if rank 39 is achievable by checking
    # if I can construct it. If not, fall back to 42.
    
    # Let me try a different known decomposition. The rank of M<2,2,4> is 13.
    # Here's an explicit one from the literature:
    
    # A-forms (4 entries: a00, a01, a10, a11):
    # B-forms (8 entries: b00..b03, b10..b13)
    # C-forms (8 entries: c00..c03, c10..c13)
    
    # I'll use the construction where we have:
    # 7 forms for the "Strassen core" covering 4x4=16 entries of a 2x2x2x2 tensor
    # plus 6 additional forms for the extra 2 columns.
    
    # Actually, let me just verify the rank-42 solution and return it.
    # The rank-39 construction is complex and I may get it wrong.
    
    if time.time() > deadline - 5:
        return best_result
    
    # Try to construct rank-39 by tiling a verified rank-13 block
    # Let me attempt a systematic construction using linear algebra
    
    # The multiplication tensor T has entries T[i,j,k] = 1 if i,j,k form a valid
    # multiplication triple. For n=2,m=2,p=12: T[i,j,k]=1 for all valid (i,j,k).
    # Actually T[i,j,k] = delta_{valid} where the tensor is:
    # T[i,j,k] = 1 if the (i,j,k) entry of the multiplication tensor is nonzero.
    
    # The tensor is: for each (i,j,k), T[i,j,k] = 1 (all entries are 1 in the
    # standard multiplication tensor for these dimensions? No...)
    
    # Actually the matrix multiplication tensor M_{n,m,p} has:
    # M[i,j,k] = 1 if and only if the entry (i,j) of A times entry (j,k) of B
    # contributes to entry (i,k) of C. In our case, since all indices are valid,
    # M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # Wait, that's not right either. The tensor M has:
    # M[i,j,k] = 1 for all valid (i,j,k) triples, and the decomposition is:
    # M[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # For our dimensions, there are n*m*p = 2*2*12 = 48 entries, all equal to 1.
    
    # So we need: for all (i,j,k) in {0,1}x{0,1}x{0,...,11}:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is what verify_fast checks. Good.
    
    # Let me try to find a rank-39 solution by modifying the rank-42 one.
    # In rank-42, each of the 6 column-pairs uses 7 independent Strassen forms.
    # If I can show that 2 adjacent column-pairs can be done with 13 forms,
    # I get 3*13 = 39.
    
    # Let me try to construct the rank-13 for 2x2 x 2x4 more carefully.
    # I'll set up the equations and solve.
    
    if time.time() > deadline - 10:
        return best_result
    
    # Try constructing rank-13 for 2x2 x 2x4 by solving a nonlinear system
    # This is hard. Let me use a known explicit construction instead.
    
    # From "The rank of the matrix multiplication tensor" - known result:
    # For 2x2 x 2x4, rank = 13.
    
    # Here's an explicit construction I'll verify:
    # Using the approach of taking Strassen's 7 forms for the first 2 columns,
    # and 6 additional forms for the next 2 columns, where 1 form is shared.
    
    # Let me define the 13 forms explicitly and verify:
    
    U13 = np.zeros((4, 13))
    V13 = np.zeros((8, 13))
    W13 = np.zeros((8, 13))
    
    # Forms 0-6: Strassen for columns 0,1
    # A-forms: [a00+a11, a10+a11, a00, a11, a00+a01, a10-a00, a01-a11]
    U13[0,0]=1; U13[3,0]=1   # a00+a11
    U13[2,1]=1; U13[3,1]=1   # a10+a11
    U13[0,2]=1                # a00
    U13[3,3]=1                # a11
    U13[0,4]=1; U13[1,4]=1   # a00+a01
    U13[2,5]=1; U13[0,5]=-1  # a10-a00
    U13[1,6]=1; U13[3,6]=-1  # a01-a11
    
    # B-forms for cols 0,1:
    V13[0,0]=1; V13[5,0]=1   # b00+b11
    V13[0,1]=1                # b00
    V13[1,2]=1; V13[5,2]=-1  # b01-b11
    V13[5,3]=1; V13[0,3]=-1  # b11-b00
    V13[5,4]=1                # b11
    V13[0,5]=1; V13[1,5]=1   # b00+b01
    V13[5,6]=1; V13[1,6]=1   # b11+b01
    
    # C-forms for cols 0,1:
    W13[0,0]=1; W13[5,0]=1   # c00+c11
    W13[1,1]=1; W13[5,1]=-1  # c01-c11
    W13[4,2]=1; W13[5,2]=1   # c10+c11
    W13[0,3]=1; W13[1,3]=1   # c00+c01
    W13[0,4]=-1; W13[4,4]=1  # -c00+c10
    W13[5,5]=1                # c11
    W13[0,6]=1                # c00
    
    # Forms 7-12: for columns 2,3
    # Use 6 additional forms
    # a00 * b02 -> c02
    U13[0,7]=1
    V13[2,7]=1
    W13[2,7]=1
    
    # a01 * b02 -> c02
    U13[1,8]=1
    V13[2,8]=1
    W13[2,8]=1
    
    # a10 * b12 -> c12
    U13[2,9]=1
    V13[6,9]=1
    W13[6,9]=1
    
    # a11 * b12 -> c12
    U13[3,10]=1
    V13[6,10]=1
    W13[6,10]=1
    
    # a00 * b03 -> c03
    U13[0,11]=1
    V13[3,11]=1
    W13[3,11]=1
    
    # Need one more form to cover: a01*b03, a10*b13, a11*b13
    # That's 3 entries with 1 form - impossible unless we combine.
    # This shows 13 might not work this way.
    
    # Let me verify this partial construction:
    # Check if forms 0-6 correctly handle cols 0,1:
    def verify_2x2x2x4(U, V, W, r):
        """Verify 2x2 x 2x4 decomposition."""
        nn, mm, pp = 2, 2, 4
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    val = sum(U[i*mm+j, rr] * V[j*pp+k, rr] * W[k*nn+i, rr] for rr in range(r))
                    if abs(val - 1.0) > 1e-10:
                        return False, (i,j,k), val
        return True, None, None
    
    # Verify just the first 7 forms for cols 0,1:
    ok, _, _ = verify_2x2x2x4(U13[:, :7], V13[:, :7], W13[:, :7], 7)
    # This should be False because forms 7-12 contribute to cols 2,3
    # and forms 0-6 should only contribute to cols 0,1
    
    # Let me check: do forms 0-6 have any V or W entries for cols 2,3?
    # V13[:, :7] has entries only in rows 0,1,5 (cols 0,1 of B)
    # W13[:, :7] has entries only in rows 0,1,4,5 (cols 0,1 of C)
    # So forms 0-6 only affect (i,j,k) with k in {0,1}. Good.
    
    # Now check if forms 0-6 correctly give 1 for all (i,j,k) with k in {0,1}:
    # This is exactly Strassen for 2x2 x 2x2, so it should work.
    
    # The issue is covering cols 2,3 with 6 forms.
    # Cols 2,3 have 8 entries: c02, c03, c12, c13 each with 2 contributing terms = 8 total
    # Wait: c02 = a00*b02 + a01*b02... no.
    # c02 = a00*b02 + a01*b12 (row 0 of A times col 2 of B)
    # c03 = a00*b03 + a01*b13
    # c12 = a10*b02 + a11*b12
    # c13 = a10*b03 + a11*b13
    
    # So 8 bilinear terms for cols 2,3. With 6 forms, we need to cover 8 terms
    # with 6 rank-1 terms. This requires some forms to cover multiple terms.
    
    # A rank-1 term U[i,j,r]*V[j,k,r]*W[k,i,r] contributes to exactly one (i,j,k)
    # entry. So 6 forms can cover at most 6 entries. But we need 8.
    # Unless... some forms have V or W entries that are sums.
    
    # Wait, I'm confusing myself. Each rank-1 term in the decomposition
    # contributes to ALL (i,j,k) entries simultaneously. The contribution to
    # entry (i,j,k) is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # A single rank-1 term with nonzero U[i0*m+j0, r], V[j0*p+k0, r], W[k0*n+i0, r]
    # contributes to entry (i0, j0, k0) with value U*V*W.
    # But it can ALSO contribute to other entries if U has multiple nonzero entries,
    # etc.
    
    # So a single rank-1 term CAN contribute to multiple (i,j,k) entries if
    # U has multiple nonzero entries (multiple (i,j) pairs), etc.
    
    # For example, if U[0,r]=1, U[1,r]=1, V[2,r]=1, W[2,r]=1:
    # Then it contributes to (0, j, k) where V[j*p+k,r] and W[k*n+0,r] are nonzero.
    
    # So the 6 forms for cols 2,3 CAN cover 8 entries if they're structured properly.
    
    # Let me try:
    # Form 7: (a00+a01) * b02 * 1 -> contributes to c02 with value (a00+a01)*b02
    #   But we need a00*b02 + a01*b12 for c02. This doesn't match.
    
    # I think I need to be more careful. Let me just return the rank-42 solution.
    # The rank-39 construction requires a verified explicit decomposition that I
    # don't have memorized correctly.
    
    return best_result
# EVOLVE-BLOCK-END
