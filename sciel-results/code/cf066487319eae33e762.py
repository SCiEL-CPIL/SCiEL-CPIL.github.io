"""
Attempt to find a rank-minimal tensor decomposition for 2x2x9 matrix multiplication
by exploring the "border rank" / degenerate limits of block-based decompositions.
Strategy:
- Start with the known rank-32 solution (4 Strassen blocks + 1 schoolbook).
- Attempt to "share" multiplications between blocks by perturbing the decomposition
  so that the same rank-1 term is used for multiple output coordinates, if possible.
  This is a heuristic search for a lower rank (e.g., rank 31, 30, ...).
- If a lower rank is found and verified, use it. Otherwise, return the rank-32 solution.
- Use a time budget to avoid exceeding the 120s limit.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify decomposition: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    U3 = np.array(U, dtype=float).reshape(n, m, -1)
    V3 = np.array(V, dtype=float).reshape(m, p, -1)
    W3 = np.array(W, dtype=float).reshape(p, n, -1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U3[i, j, :], V3[j, k, :] * W3[k, i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_rank32():
    """
    Build the known rank-32 decomposition: 4 Strassen blocks (cols 0-7) + 1 schoolbook (col 8).
    Returns (U, V, W) as lists of lists of floats.
    """
    n, m, p = 2, 2, 9
    rank = 32
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        # M1: (a00+a11)(b_{0,k0}+b_{1,k1})
        U[0][col] = 1.0; U[3][col] = 1.0
        V[0*p + k0][col] = 1.0; V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = 1.0; W[k1*n + 1][col] = 1.0
        col += 1
        # M2: (a10+a11) * b_{0,k0}
        U[2][col] = 1.0; U[3][col] = 1.0
        V[0*p + k0][col] = 1.0
        W[k0*n + 1][col] = 1.0; W[k1*n + 1][col] = -1.0
        col += 1
        # M3: a00 * (b_{0,k1} - b_{1,k1})
        U[0][col] = 1.0
        V[0*p + k1][col] = 1.0; V[1*p + k1][col] = -1.0
        W[k1*n + 0][col] = 1.0; W[k1*n + 1][col] = 1.0
        col += 1
        # M4: a11 * (b_{1,k0} - b_{0,k0})
        U[3][col] = 1.0
        V[1*p + k0][col] = 1.0; V[0*p + k0][col] = -1.0
        W[k0*n + 0][col] = 1.0; W[k0*n + 1][col] = 1.0
        col += 1
        # M5: (a00+a01) * b_{1,k1}
        U[0][col] = 1.0; U[1][col] = 1.0
        V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = -1.0; W[k1*n + 0][col] = 1.0
        col += 1
        # M6: (a10-a00) * (b_{0,k0}+b_{0,k1})
        U[2][col] = 1.0; U[0][col] = -1.0
        V[0*p + k0][col] = 1.0; V[0*p + k1][col] = 1.0
        W[k1*n + 1][col] = 1.0
        col += 1
        # M7: (a01-a11) * (b_{1,k0}+b_{1,k1})
        U[1][col] = 1.0; U[3][col] = -1.0
        V[1*p + k0][col] = 1.0; V[1*p + k1][col] = 1.0
        W[k0*n + 0][col] = 1.0
        col += 1

    def add_schoolbook(k):
        nonlocal col
        U[0][col] = 1.0
        V[0*p + k][col] = 1.0
        W[k*n + 0][col] = 1.0
        col += 1
        U[1][col] = 1.0
        V[1*p + k][col] = 1.0
        W[k*n + 0][col] = 1.0
        col += 1
        U[2][col] = 1.0
        V[0*p + k][col] = 1.0
        W[k*n + 1][col] = 1.0
        col += 1
        U[1][col] = 1.0  # Wait, the last term is a11 * b_{1,k} -> C_{k,1}
        # Correction: The 4th term is a11 (row 3) * b_{1,k} (row 1)
        U[1][col] = 0.0  # Reset if needed
        U[3][col] = 1.0
        V[1*p + k][col] = 1.0
        W[k*n + 1][col] = 1.0
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_schoolbook(8)
    
    return U, V, W

def _build_rank28():
    """
    Build a rank-28 decomposition using 3 groups of 3 columns.
    For each group of 3 columns (k0,k1,k2), use a 2x2x3 decomposition of rank ~9.33 avg.
    Actually, let's use a different strategy: 2 Strassen blocks (4 cols) + 1 block of 5 cols.
    For 5 cols, use schoolbook: 5*4=20. Total: 2*7+20=34. Worse.
    
    Let me try: use the structure that 9 = 3+3+3, and for each 3-col block,
    find a rank-9 decomposition (vs schoolbook 12). That gives 27.
    
    For 2x2x3, can we get rank 9? The tensor has 12 entries. 
    Let's try a specific construction.
    """
    n, m, p = 2, 2, 9
    rank = 27
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    
    def add_block_3cols(k0, k1, k2):
        """Add a rank-9 decomposition for 2x2x3 block."""
        nonlocal col
        # M1: a00 * b_{0,k0}
        U[0][col] = 1.0; V[0*p+k0][col] = 1.0; W[k0*n+0][col] = 1.0; col += 1
        # M2: a00 * b_{1,k0}  
        U[0][col] = 1.0; V[1*p+k0][col] = 1.0; W[k0*n+1][col] = 1.0; col += 1
        # M3: a10 * b_{0,k0}
        U[2][col] = 1.0; V[0*p+k0][col] = 1.0; W[k0*n+1][col] = 1.0; col += 1
        # M4: a11 * b_{1,k0}
        U[3][col] = 1.0; V[1*p+k0][col] = 1.0; W[k0*n+1][col] = 1.0; col += 1
        # M5: a01 * b_{0,k1}
        U[1][col] = 1.0; V[0*p+k1][col] = 1.0; W[k1*n+0][col] = 1.0; col += 1
        # M6: a01 * b_{1,k1}
        U[1][col] = 1.0; V[1*p+k1][col] = 1.0; W[k1*n+1][col] = 1.0; col += 1
        # M7: a11 * b_{0,k1}
        U[3][col] = 1.0; V[0*p+k1][col] = 1.0; W[k1*n+1][col] = 1.0; col += 1
        # M8: a10 * b_{1,k2}
        U[2][col] = 1.0; V[1*p+k2][col] = 1.0; W[k2*n+1][col] = 1.0; col += 1
        # M9: a00 * b_{0,k2} ... wait this doesn't cover all terms.
        # This is getting complicated. Let me just use schoolbook for each 3-col block: 3*4=12 per block, total 36.
        # That's not better.
        
        # Let me try a different approach: use Strassen for pairs and handle the 9th column separately.
        # 4 pairs (8 cols) with Strassen: 4*7=28, plus 1 col schoolbook: 4. Total 32.
        # That's the current solution.
        
        # Alternative: 3 groups of 3, each with a custom rank-9 decomposition.
        # For 2x2x3, the minimal rank is actually 12 (schoolbook) or possibly less.
        # Let me check if rank 10 is achievable.
        
        # For now, just do schoolbook for 3 cols: 12 terms
        for k in [k0, k1, k2]:
            for i in range(2):
                for j in range(2):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
    
    # This gives 3*12=36, which is worse. Let me abandon this approach.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Start with the known rank-32 solution
    U, V, W = _build_rank32()
    
    if not _verify(U, V, W, 2, 2, 9):
        # Fallback to full schoolbook rank 36
        rank_fb = 36
        U_fb = [[0.0] * rank_fb for _ in range(4)]
        V_fb = [[0.0] * rank_fb for _ in range(18)]
        W_fb = [[0.0] * rank_fb for _ in range(18)]
        c = 0
        for i in range(2):
            for j in range(2):
                for k in range(9):
                    U_fb[i*2+j][c] = 1.0
                    V_fb[j*9+k][c] = 1.0
                    W_fb[k*2+i][c] = 1.0
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    best_rank = 32
    best_U, best_V, best_W = U, V, W
    
    # Try to improve: use 3 Strassen blocks (6 cols) + 1 block of 3 cols with rank-9 decomposition.
    # 3*7 + 9 = 30. If we can find a rank-9 decomposition for 2x2x3, this gives rank 30.
    
    def _build_2x2x3_rank9():
        """
        Attempt to build a rank-9 decomposition for 2x2x3 matrix multiplication.
        If successful, returns (U, V, W) for n=2,m=2,p=3.
        """
        n2, m2, p2 = 2, 2, 3
        r = 9
        U2 = [[0.0]*r for _ in range(4)]
        V2 = [[0.0]*r for _ in range(6)]
        W2 = [[0.0]*r for _ in range(6)]
        
        # Use Strassen for first 2 columns (7 terms) + 1 term for 3rd column? No, that's 8.
        # Actually, for 2x2x3, let's try:
        # Columns 0,1: Strassen (7 terms)
        # Column 2: needs 4 terms for schoolbook, but maybe we can share.
        
        # Let me try a different split: use 3 terms for col 0, 3 for col 1, 3 for col 2.
        # Total 9.
        
        # Col 0: C[0,0]=a00*b00, C[0,1]=a01*b00, C[1,0]=a10*b00, C[1,1]=a11*b00
        # That's 4 terms just for col 0. Can't do 3.
        
        # So rank 9 for 2x2x3 is not trivially achievable with this structure.
        # Let me try rank 10: Strassen for 2 cols (7) + 3 terms for 3rd col.
        # But 3rd col needs 4 terms minimum if independent.
        
        # I think the minimal rank for 2x2x3 is 12 (schoolbook). Let me verify by checking
        # if any known lower-rank algorithm exists.
        
        # For now, return None to indicate this approach doesn't work.
        return None
    
    # Since 2x2x3 likely has rank 12, the 3-group-of-3 approach gives 36.
    # The 4-group-of-2 + 1-group-of-1 approach gives 32, which is better.
    
    # Try to find rank-31 by dropping a column and then doing a local optimization
    # to repair the residual. The idea: drop column c, compute the residual tensor R
    # (what's missing), then try to express R as a rank-1 tensor by adjusting one
    # of the remaining columns, or by finding that the residual is zero.
    
    if time.time() < deadline - 20:
        try:
            import numpy as np
            
            n, m, p = 2, 2, 9
            U0 = np.array(U, dtype=float)
            V0 = np.array(V, dtype=float)
            W0 = np.array(W, dtype=float)
            
            for drop_col in range(32):
                if time.time() > deadline - 20:
                    break
                
                U_t = np.delete(U0, drop_col, axis=1)
                V_t = np.delete(V0, drop_col, axis=1)
                W_t = np.delete(W0, drop_col, axis=1)
                
                # Compute the residual: what's missing from the tensor
                # Residual[i,j,k] = 1 - sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
                residual = np.ones((n, m, p))
                for r in range(31):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                residual[i,j,k] -= U_t[i*m+j, r] * V_t[j*p+k, r] * W_t[k*n+i, r]
                
                max_abs = np.max(np.abs(residual))
                if max_abs < 1e-9:
                    # The dropped column was redundant!
                    best_rank = 31
                    best_U = U_t.tolist()
                    best_V = V_t.tolist()
                    best_W = W_t.tolist()
                    return {"U": best_U, "V": best_V, "W": best_W}
                
                # Try to repair: can we adjust one existing column to absorb the residual?
                # This is hard, so skip for now.
        
        except Exception:
            pass
    
    # Try a different block structure: 3 Strassen blocks (21) + 1 block of 3 cols.
    # For 3 cols, can we do better than schoolbook (12)? Try rank 10 for 2x2x3.
    # Total would be 21 + 10 = 31.
    
    def _build_2x2x3_rank10():
        """
        Attempt a rank-10 decomposition for 2x2x3.
        Use Strassen for cols 0,1 (7 terms) + 3 terms for col 2.
        But col 2 needs 4 output entries. With 3 terms, we can cover at most 3 if
        each term contributes to one output. But a rank-1 term U_i*V_j*W_k contributes
        to exactly one output (i,j,k). So 3 terms cover 3 outputs, missing 1.
        
        Unless we use non-trivial coefficients where one term contributes to multiple
        outputs? No, each rank-1 term in the tensor decomposition contributes to
        exactly one (i,j,k) entry with coefficient U[i]*V[j]*W[k].
        
        Wait, that's not right. A rank-1 term is U[i,j,r]*V[j,k,r]*W[k,i,r] for all
        (i,j,k) simultaneously. So one rank-1 term contributes to ALL 12 entries,
        each with a different coefficient.
        
        So with 3 terms, we have 3*(4+6+6)=36 unknowns and 12 equations. Should be
        solvable. Let me try to construct it.
        
        For 2x2x3, let's use:
        - Strassen for cols 0,1: 7 terms
        - 3 additional terms for col 2
        
        The 7 Strassen terms for cols 0,1 handle all 8 outputs for cols 0,1.
        For col 2, we need 4 outputs: C[2,0], C[2,1] from A*B[:,2].
        
        With 3 more terms, we have 3 rank-1 tensors. Each has 4+6+6=16 degrees of freedom,
        but we only care about their contribution to col 2 (4 entries) and we need them
        to not disturb cols 0,1 (which should be zero).
        
        This is getting complex. Let me just try a numerical solve.
        """
        n2, m2, p2 = 2, 2, 3
        r = 10
        # Initialize: first 7 cols from Strassen for 2x2x2, last 3 cols zero
        U2 = np.zeros((4, r))
        V2 = np.zeros((6, r))
        W2 = np.zeros((6, r))
        
        # Strassen for 2x2x2 (cols 0,1 of B, which are k=0,1)
        # M1: (a00+a11)(b00+b11)
        U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[4,0]=1; W2[0,0]=1; W2[3,0]=1
        # M2: (a10+a11)b00
        U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[1,1]=1; W2[3,1]=-1
        # M3: a00(b01-b11)
        U2[0,2]=1; V2[1,2]=1; V2[5,2]=-1; W2[2,2]=1; W2[4,2]=1
        # M4: a11(b10-b00)
        U2[3,3]=1; V2[4,3]=1; V2[0,3]=-1; W2[0,3]=1; W2[2,3]=1
        # M5: (a00+a01)b11
        U2[0,4]=1; U2[1,4]=1; V2[5,4]=1; W2[0,4]=-1; W2[4,4]=1
        # M6: (a10-a00)(b00+b01)
        U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1; W2[4,5]=1
        # M7: (a01-a11)(b10+b11)
        U2[1,6]=1; U2[3,6]=-1; V2[4,6]=1; V2[5,6]=1; W2[0,6]=1
        
        # Now try to fill in cols 7,8,9 to handle k=2
        # We need: for k=2, C[2,i] = sum_j A[i,j]*B[j,2]
        # The first 7 terms contribute to k=2 via W2[2*n+i, r] for i=0,1
        # Check what they contribute:
        # Term 0: W2[2,0]=0, W2[3,0]=0 -> no contribution to k=2
        # Term 1: W2[2,1]=0, W2[3,1]=0 -> no contribution
        # Term 2: W2[2,2]=1, W2[3,2]=0 -> contributes to C[2,0] with a00*(b01-b11)*1
        # Wait, this is wrong. W2 is indexed by [k*n+i, r]. For k=2, indices are 4,5.
        # Let me recheck: W2 has shape (p*n, r) = (3*2, r) = (6, r)
        # W2[k*n+i, r] for k in [0,1,2], i in [0,1]
        # So k=2 -> indices 4,5.
        
        # Term 0: W2[4,0]=0, W2[5,0]=0
        # Term 1: W2[4,1]=0, W2[5,1]=0  
        # Term 2: W2[4,2]=1, W2[5,2]=0 -> contributes to C[2,0] = a00*(b01-b11)
        # Term 3: W2[4,3]=0, W2[5,3]=0
        # Term 4: W2[4,4]=1, W2[5,4]=0 -> contributes to C[2,0] = (a00+a01)*b11
        # Term 5: W2[4,5]=1, W2[5,5]=0 -> contributes to C[2,0] = (a10-a00)*(b00+b01)
        # Term 6: W2[4,6]=0, W2[5,6]=0
        
        # So current contribution to C[2,0] = a00*b01 - a00*b11 + a00*b11 + a01*b11 + a10*b00 + a10*b01 - a00*b00 - a00*b01
        # = a01*b11 + a10*b00 + a10*b01 - a00*b00
        # But we need C[2,0] = a00*b02 + a01*b12 (for k=2, B[j,2] where j=0,1)
        # This is completely wrong - the Strassen terms are for k=0,1, not k=2!
        
        # I made an error. The Strassen decomposition for 2x2x2 handles B columns 0 and 1,
        # producing C columns 0 and 1. For C column 2, we need separate terms.
        
        # So for 2x2x3, we need:
        # - 7 terms for C cols 0,1 (Strassen)
        # - 4 terms for C col 2 (schoolbook)
        # Total: 11 terms. That's rank 11, not 10.
        
        # Can we do better? Maybe use a different algorithm for 2x2x3.
        # The minimal rank for 2x2x3 is unknown, but likely 11 or 12.
        
        return None
    
    # Since rank-10 for 2x2x3 doesn't seem achievable, the 3-block approach gives 21+11=32.
    # Same as current.
    
    # Let me try: 2 Strassen blocks (14) + 1 block of 5 cols.
    # For 5 cols, schoolbook is 20. Total: 34. Worse.
    
    # Or: 1 Strassen block (7) + 1 block of 7 cols. 7+28=35. Worse.
    
    # The current 32 appears to be optimal for this block structure.
    # Return the rank-32 solution.
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
