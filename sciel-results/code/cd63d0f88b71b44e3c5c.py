import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit

    n, m, p = 4, 5, 5
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20

    # The matrix multiplication tensor T[i,j,k] = 1 for all (i,j,k) in [0,n) x [0,m) x [0,p)
    # We want a rank-r decomposition:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

    # STRATEGY:
    # Partition the 4x5x5 tensor into disjoint sub-blocks:
    # - One 3x3x3 sub-block: i in {0,1,2}, j in {0,1,2}, k in {0,1,2} -> use BCR rank-15
    # - One 2x2x2 sub-block: i in {2,3}, j in {3,4}, k in {3,4} -> use Strassen rank-7
    #   Wait, i={2,3} overlaps with i={0,1,2} at i=2. We need disjoint blocks.
    #
    # Let's choose disjoint partitions:
    # Option A:
    #   Block 1: i in {0,1,2}, j in {0,1,2}, k in {0,1,2} (3x3x3)
    #   Block 2: i in {2,3}, j in {3,4}, k in {3,4} -> i=2 overlaps. Not disjoint.
    #
    # Option B:
    #   Block 1: i in {0,1,2}, j in {0,1,2}, k in {0,1,2} (3x3x3, 27 terms standard)
    #   Block 2: i in {3}, j in {3,4}, k in {3,4} (1x2x2, 4 terms standard)
    #   Remaining: all other (i,j,k) not in Block 1 or Block 2
    #
    # Let's count what's left:
    # Total terms: 4*5*5 = 100
    # Block 1 covers: 3*3*3 = 27 terms
    # Block 2 covers: 1*2*2 = 4 terms
    # Remaining: 100 - 27 - 4 = 69 terms (standard rank 1 each)
    #
    # If we use BCR (rank 15) for Block 1 instead of 27, we save 12.
    # Total rank: 15 + 4 + 69 = 88
    #
    # Can we do better? Let's try:
    #   Block 1: i in {0,1,2}, j in {0,1,2}, k in {0,1,2} (3x3x3) -> BCR rank 15
    #   Block 2: i in {3}, j in {3,4}, k in {0,1} (1x2x2) -> standard rank 4
    #   Block 3: i in {2,3}, j in {3,4}, k in {2,3} -> overlaps with Block 1 at i=2,k=2,3
    #
    # This is getting complicated with overlaps. Let's use a cleaner approach:
    # Use BCR for the 3x3x3 core, and Strassen for 2x2x2 blocks that don't overlap.
    #
    # Clean partition:
    #   Core 3x3x3: i={0,1,2}, j={0,1,2}, k={0,1,2} -> BCR rank 15
    #   Strassen 2x2x2: i={2,3}, j={2,3}, k={2,3} -> overlaps with core at i=2,j=2,k=2
    #
    # Disjoint partition:
    #   Block A (3x3x3): i={0,1,2}, j={0,1,2}, k={0,1,2} -> 27 terms, use BCR rank 15
    #   Block B (2x2x2): i={2,3}, j={3,4}, k={3,4} -> 8 terms, use Strassen rank 7
    #     But i=2 is in both A and B. Not disjoint.
    #
    # Let me just use disjoint blocks:
    #   Block A (3x3x3): i={0,1,2}, j={0,1,2}, k={0,1,2}
    #   Block B (1x2x2): i={3}, j={3,4}, k={3,4}
    #   Everything else: standard
    #
    # Rank: 15 + 4 + (100-27-4) = 15 + 4 + 69 = 88
    #
    # Alternative: Use two 2x2x2 Strassen blocks that are disjoint from each other and from the 3x3x3.
    #   Block A (3x3x3): i={0,1,2}, j={0,1,2}, k={0,1,2}
    #   Block B (2x2x2): i={2,3}, j={3,4}, k={3,4} -> i=2,k=2,3 overlap with A
    #
    # Let's try:
    #   Block A (3x3x3): i={0,1,2}, j={0,1,2}, k={0,1,2}
    #   Block B (2x2x2): i={3}, j={3,4}, k={3,4} -> only 1x2x2, rank 4
    #
    # Or:
    #   Block A (2x2x2): i={0,1}, j={0,1}, k={0,1} -> Strassen rank 7
    #   Block B (2x2x2): i={2,3}, j={2,3}, k={2,3} -> Strassen rank 7
    #   Block C (1x1x1): i={0,1}, j={2,3,4}, k={2,3,4} -> 1*3*3=9 terms
    #   Block D (1x1x1): i={2,3}, j={0,1}, k={2,3,4} -> 2*1*3=6 terms
    #   Block E (1x1x1): i={2,3}, j={2,3}, k={0,1} -> 2*2*2=8 terms (but this overlaps with B at k=2,3)
    #
    # This is getting messy. Let me just go with the clean 3x3x3 + standard approach:
    #   3x3x3 core with BCR: rank 15 (saves 12)
    #   Rest: 73 standard terms
    #   Total: 88
    #
    # Actually, let me try a different clean partition:
    #   3x3x3: i={0,1,2}, j={0,1,2}, k={0,1,2} -> BCR rank 15
    #   2x2x2: i={2,3}, j={3,4}, k={3,4} -> but i=2 overlaps
    #
    # Final decision: Use 3x3x3 BCR + everything else standard.
    # Rank = 15 + 73 = 88
    #
    # But wait, can I also apply Strassen to a 2x2x2 block that is fully outside the 3x3x3?
    # The 3x3x3 uses i in {0,1,2}, j in {0,1,2}, k in {0,1,2}.
    # Outside: i=3, or j in {3,4}, or k in {3,4}
    # A 2x2x2 block fully outside would need 2 i-values from {3} (only 1 available), so no.
    # A 1x2x2 block: i={3}, j={3,4}, k={3,4} -> 4 terms, no Strassen benefit.
    #
    # So 88 is the target with this approach.
    #
    # Let me also consider: what if I use Strassen on two 2x2x2 blocks that are disjoint?
    #   Block 1: i={0,1}, j={0,1}, k={0,1} -> rank 7
    #   Block 2: i={2,3}, j={2,3}, k={2,3} -> rank 7
    #   These are disjoint!
    #   Remaining: 100 - 8 - 8 = 84 terms
    #   Total: 7 + 7 + 84 = 98. Worse than 88.
    #
    # What about one 3x3x3 (BCR rank 15) and one 2x2x2 (Strassen rank 7) that are disjoint?
    #   3x3x3: i={0,1,2}, j={0,1,2}, k={0,1,2}
    #   2x2x2: needs 2 i's from {3}, not possible.
    #   2x2x2: i={3}, not possible (need 2).
    #
    # So the best disjoint partition with BCR is just the 3x3x3 core.
    # Rank = 88.

    def build_bcr_3x3x3(U, V, W, col, i0, j0, k0):
        """
        Apply BCR rank-15 decomposition to a 3x3x3 sub-block starting at (i0, j0, k0).
        The sub-block has i in {i0, i0+1, i0+2}, j in {j0, j0+1, j0+2}, k in {k0, k0+1, k0+2}.
        Returns the next column index.
        
        The BCR decomposition for 3x3x3 matrix multiplication uses 15 scalar products.
        For local indices (a,b,c) in {0,1,2}^3, we need:
          sum_r U[a,b,r] * V[b,c,r] * W[c,a,r] = 1
        
        Construction:
        - Diagonal (a=a, b=b, c=c): positions (0,0,0),(1,1,1),(2,2,2)
          Use 3 terms with Lagrange interpolation on the diagonal.
        - Cyclic diag 1 (a+1=b, b+1=c, c+1=a mod 3): (0,1,2),(1,2,0),(2,0,1)
          Use 3 terms.
        - Cyclic diag 2 (a+2=b, b+2=c, c+2=a mod 3): (0,2,1),(1,0,2),(2,1,0)
          Use 3 terms.
        - Off-diagonal: 9 remaining positions need 6 more terms.
        """
        # Local indices
        # Position (a,b,c) in local coords maps to:
        # U[(i0+a)*m + (j0+b)], V[(j0+b)*p + (k0+c)], W[(k0+c)*n + (i0+a)]
        
        # Diagonal terms: for positions (0,0,0), (1,1,1), (2,2,2)
        # Term 1: U has 1 at (0,0),(1,1),(2,2); V has 1 at (0,0),(1,1),(2,2); W has 1 at (0,0),(1,1),(2,2)
        # This gives 1 for all three diagonal positions but also contributes to off-diagonal.
        # We need to be more careful.
        
        # Actually, the BCR construction is:
        # Let ω = exp(2πi/3). But we need real/rational coefficients.
        # 
        # A valid integer BCR decomposition:
        # For the main diagonal (i=j=k):
        #   M1: U=(1,1,1) on diag, V=(1,1,1) on diag, W=(1,1,1) on diag
        #   M2: U=(1,2,3) on diag, V=(1,2,3) on diag, W=(1,2,3) on diag  
        #   M3: U=(1,3,2) on diag, V=(1,3,2) on diag, W=(1,3,2) on diag
        # These three give: for position (t,t,t): 1*1*1 + 2*2*2 + 3*3*3... no that's wrong.
        # 
        # Let me think differently. We need:
        # For (0,0,0): sum U[0,0,r]*V[0,0,r]*W[0,0,r] = 1
        # For (1,1,1): sum U[1,1,r]*V[1,1,r]*W[1,1,r] = 1
        # For (2,2,2): sum U[2,2,r]*V[2,2,r]*W[2,2,r] = 1
        # And for off-diagonal (a,b,c) with not all equal: sum = 0
        #
        # The standard BCR approach uses:
        # 3 terms for each of the 3 "diagonals" (main, cyclic1, cyclic2) = 9 terms
        # 6 terms for the remaining 9 off-diagonal positions
        #
        # For the main diagonal, use:
        #   r=0: U[0,0]=1, U[1,1]=1, U[2,2]=1; V[0,0]=1, V[1,1]=1, V[2,2]=1; W[0,0]=1, W[1,1]=1, W[2,2]=1
        #   r=1: U[0,0]=1, U[1,1]=2, U[2,2]=3; V[0,0]=1, V[1,1]=2, V[2,2]=3; W[0,0]=1, W[1,1]=2, W[2,2]=3
        #   r=2: U[0,0]=1, U[1,1]=3, U[2,2]=2; V[0,0]=1, V[1,1]=3, V[2,2]=2; W[0,0]=1, W[1,1]=3, W[2,2]=2
        #
        # Check (0,0,0): 1*1*1 + 1*1*1 + 1*1*1 = 3 ≠ 1. Not right.
        #
        # I need to use Lagrange interpolation properly.
        # For value at t=0: L0(t) = (t-1)(t-2)/((0-1)(0-2)) = (t-1)(t-2)/2
        # For value at t=1: L1(t) = (t-0)(t-2)/((1-0)(1-2)) = -(t)(t-2)
        # For value at t=2: L2(t) = (t-0)(t-1)/((2-0)(2-1)) = t(t-1)/2
        #
        # So:
        #   r=0: U_diag = L0 = (1/2, -1, 1/2), V_diag = (1,1,1), W_diag = (1,1,1)
        #     Check (0,0,0): (1/2)*1*1 = 1/2
        #     Check (1,1,1): (-1)*1*1 = -1
        #     Check (2,2,2): (1/2)*1*1 = 1/2
        #
        #   r=1: U_diag = (1,1,1), V_diag = L0 = (1/2, -1, 1/2), W_diag = (1,1,1)
        #     Check (0,0,0): 1*(1/2)*1 = 1/2
        #     Check (1,1,1): 1*(-1)*1 = -1
        #     Check (2,2,2): 1*(1/2)*1 = 1/2
        #
        #   r=2: U_diag = (1,1,1), V_diag = (1,1,1), W_diag = L0 = (1/2, -1, 1/2)
        #     Check (0,0,0): 1*1*(1/2) = 1/2
        #     Check (1,1,1): 1*1*(-1) = -1
        #     Check (2,2,2): 1*1*(1/2) = 1/2
        #
        # Sum at (0,0,0): 1/2+1/2+1/2 = 3/2. Not 1.
        #
        # I need to scale. Let me use:
        #   r=0: U_diag = 2*L0 = (1, -2, 1), V_diag = (1,1,1), W_diag = (1,1,1)
        #   r=1: U_diag = (1,1,1), V_diag = 2*L0 = (1, -2, 1), W_diag = (1,1,1)
        #   r=2: U_diag = (1,1,1), V_diag = (1,1,1), W_diag = 2*L0 = (1, -2, 1)
        #
        # Sum at (0,0,0): 1*1*1 + 1*1*1 + 1*1*1 = 3. Scale by 1/3? But we need integers.
        #
        # Alternative: use a different approach. Let me just use the standard 27 terms
        # for now since getting BCR exactly right is complex. The Strassen approach already
        # gives rank 92. Let me focus on making that work correctly and efficiently.
        
        # Standard 27 terms for 3x3x3 (safe fallback)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[(i0+i)*m + (j0+j)][col] = 1
                    V[(j0+j)*p + (k0+k)][col] = 1
                    W[(k0+k)*n + (i0+i)][col] = 1
                    col += 1
        return col

    def build_strassen_2x2x2(U, V, W, col, i1, i2, j1, j2, k1, k2):
        """
        Apply Strassen rank-7 decomposition to a 2x2x2 sub-block.
        i values: i1, i2; j values: j1, j2; k values: k1, k2.
        """
        # M1 = (a11 + a22)(b11 + b22)
        U[i1*m+j1][col] += 1
        U[i2*m+j2][col] += 1
        V[j1*p+k1][col] += 1
        V[j2*p+k2][col] += 1
        W[k1*n+i1][col] += 1
        W[k2*n+i2][col] += 1
        col += 1

        # M2 = (a21 + a22)b11
        U[i2*m+j1][col] += 1
        U[i2*m+j2][col] += 1
        V[j1*p+k1][col] += 1
        W[k1*n+i2][col] += 1
        W[k2*n+i2][col] += -1
        col += 1

        # M3 = a11(b12 - b22)
        U[i1*m+j1][col] += 1
        V[j1*p+k2][col] += 1
        V[j2*p+k2][col] += -1
        W[k2*n+i1][col] += 1
        W[k2*n+i2][col] += 1
        col += 1

        # M4 = a22(b21 - b11)
        U[i2*m+j2][col] += 1
        V[j2*p+k1][col] += 1
        V[j1*p+k1][col] += -1
        W[k1*n+i1][col] += 1
        W[k1*n+i2][col] += 1
        col += 1

        # M5 = (a11 + a12)b22
        U[i1*m+j1][col] += 1
        U[i1*m+j2][col] += 1
        V[j2*p+k2][col] += 1
        W[k1*n+i1][col] += -1
        W[k2*n+i1][col] += 1
        col += 1

        # M6 = (a21 - a11)(b11 + b12)
        U[i2*m+j1][col] += 1
        U[i1*m+j1][col] += -1
        V[j1*p+k1][col] += 1
        V[j1*p+k2][col] += 1
        W[k2*n+i2][col] += 1
        col += 1

        # M7 = (a12 - a22)(b21 + b22)
        U[i1*m+j2][col] += 1
        U[i2*m+j2][col] += -1
        V[j2*p+k1][col] += 1
        V[j2*p+k2][col] += 1
        W[k1*n+i1][col] += 1
        col += 1

        return col

    # Build with Strassen on all 2x2x2 blocks (rank 92)
    def build_strassen_decomposition():
        rank = 92
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        blocks = []
        for i_base in [0, 2]:
            for j_base in [0, 2]:
                for k_base in [0, 2]:
                    blocks.append((
                        [i_base, i_base + 1],
                        [j_base, j_base + 1],
                        [k_base, k_base + 1]
                    ))
        
        skip_set = set()
        for I, J, K in blocks:
            for i in I:
                for j in J:
                    for k in K:
                        skip_set.add((i, j, k))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in skip_set:
                        continue
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        for I, J, K in blocks:
            i1, i2 = I
            j1, j2 = J
            k1, k2 = K
            
            # M1 = (a11 + a22)(b11 + b22)
            U[i1 * m + j1][col] += 1
            U[i2 * m + j2][col] += 1
            V[j1 * p + k1][col] += 1
            V[j2 * p + k2][col] += 1
            W[k1 * n + i1][col] += 1
            W[k2 * n + i2][col] += 1
            col += 1
            
            # M2 = (a21 + a22)b11
            U[i2 * m + j1][col] += 1
            U[i2 * m + j2][col] += 1
            V[j1 * p + k1][col] += 1
            W[k1 * n + i2][col] += 1
            W[k2 * n + i2][col] += -1
            col += 1
            
            # M3 = a11(b12 - b22)
            U[i1 * m + j1][col] += 1
            V[j1 * p + k2][col] += 1
            V[j2 * p + k2][col] += -1
            W[k2 * n + i1][col] += 1
            W[k2 * n + i2][col] += 1
            col += 1
            
            # M4 = a22(b21 - b11)
            U[i2 * m + j2][col] += 1
            V[j2 * p + k1][col] += 1
            V[j1 * p + k1][col] += -1
            W[k1 * n + i1][col] += 1
            W[k1 * n + i2][col] += 1
            col += 1
            
            # M5 = (a11 + a12)b22
            U[i1 * m + j1][col] += 1
            U[i1 * m + j2][col] += 1
            V[j2 * p + k2][col] += 1
            W[k1 * n + i1][col] += -1
            W[k2 * n + i1][col] += 1
            col += 1
            
            # M6 = (a21 - a11)(b11 + b12)
            U[i2 * m + j1][col] += 1
            U[i1 * m + j1][col] += -1
            V[j1 * p + k1][col] += 1
            V[j1 * p + k2][col] += 1
            W[k2 * n + i2][col] += 1
            col += 1
            
            # M7 = (a12 - a22)(b21 + b22)
            U[i1 * m + j2][col] += 1
            U[i2 * m + j2][col] += -1
            V[j2 * p + k1][col] += 1
            V[j2 * p + k2][col] += 1
            W[k1 * n + i1][col] += 1
            col += 1
        
        return U, V, W, rank
    
    U, V, W, rank = build_strassen_decomposition()
    
    # Verify the decomposition using NumPy for speed
    def verify(U, V, W, n, m, p, tol=1e-9):
        """Check if the decomposition is correct using vectorized operations."""
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        rank = U_np.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                    if abs(s - 1) > tol:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to standard decomposition
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to numpy for potential further optimization
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    current_rank = rank
    best_rank = current_rank
    best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
    
    # Try to find reducible terms using linear dependency search
    # Quick check: are there any all-zero columns?
    for r in range(current_rank):
        if all(U_np[i][r] == 0 for i in range(nm)) and \
           all(V_np[i][r] == 0 for i in range(mp)) and \
           all(W_np[i][r] == 0 for i in range(pn)):
            U_np = np.delete(U_np, r, axis=1)
            V_np = np.delete(V_np, r, axis=1)
            W_np = np.delete(W_np, r, axis=1)
            current_rank -= 1
            r -= 1
    
    if time.time() > deadline:
        U_final = [[int(round(U_np[i][r])) if abs(U_np[i][r] - round(U_np[i][r])) < 1e-9 else U_np[i][r] 
                    for r in range(current_rank)] for i in range(nm)]
        V_final = [[int(round(V_np[i][r])) if abs(V_np[i][r] - round(V_np[i][r])) < 1e-9 else V_np[i][r] 
                    for r in range(current_rank)] for i in range(mp)]
        W_final = [[int(round(W_np[i][r])) if abs(W_np[i][r] - round(W_np[i][r])) < 1e-9 else W_np[i][r] 
                    for r in range(current_rank)] for i in range(pn)]
        return {"U": U_final, "V": V_final, "W": W_final}
    
    # Convert back to lists with integer/half-integer entries
    U_final = []
    for i in range(nm):
        row = []
        for r in range(current_rank):
            val = U_np[i][r]
            if abs(val - round(val)) < 1e-9:
                row.append(int(round(val)))
            elif abs(val - round(val*2)/2) < 1e-9:
                row.append(round(val*2)/2)
            else:
                row.append(val)
        U_final.append(row)
    
    V_final = []
    for i in range(mp):
        row = []
        for r in range(current_rank):
            val = V_np[i][r]
            if abs(val - round(val)) < 1e-9:
                row.append(int(round(val)))
            elif abs(val - round(val*2)/2) < 1e-9:
                row.append(round(val*2)/2)
            else:
                row.append(val)
        V_final.append(row)
    
    W_final = []
    for i in range(pn):
        row = []
        for r in range(current_rank):
            val = W_np[i][r]
            if abs(val - round(val)) < 1e-9:
                row.append(int(round(val)))
            elif abs(val - round(val*2)/2) < 1e-9:
                row.append(round(val*2)/2)
            else:
                row.append(val)
        W_final.append(row)
    
    return {"U": U_final, "V": V_final, "W": W_final}
