import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100
    
    # We will build the decomposition
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We don't know the final rank in advance, so we'll use a list of tuples
    # Each tuple is (u_vec, v_vec, w_vec)
    # u_vec is length n*m, v_vec is length m*p, w_vec is length p*n
    
    decompositions = []
    
    def add_decomposition(u_coeffs, v_coeffs, w_coeffs):
        # u_coeffs: list of (index, coeff)
        # v_coeffs: list of (index, coeff)
        # w_coeffs: list of (index, coeff)
        u_vec = np.zeros(n*m)
        v_vec = np.zeros(m*p)
        w_vec = np.zeros(p*n)
        for idx, val in u_coeffs:
            u_vec[idx] = val
        for idx, val in v_coeffs:
            v_vec[idx] = val
        for idx, val in w_coeffs:
            w_vec[idx] = val
        decompositions.append((u_vec, v_vec, w_vec))
        
    # Helper to get index
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i

    # 1. Laderman for i in 0-2, j in 0-2, k in 0-2
    # Local indices 0,1,2 correspond to global 0,1,2
    # Laderman coefficients
    # This is complex to hardcode from memory. 
    # I will use a known set of 23 linear forms.
    # Or I can just use schoolbook for the 3x3x3 blocks if I can't remember Laderman exactly.
    # Schoolbook rank for 3x3x3 is 27.
    # Strassen for 2x2x2 is 7.
    
    # Let's stick to Strassen for 2x2x2 blocks to be safe, as I know the formulas.
    # If I use schoolbook for 3x3x3, rank is 27.
    # 27 + 27 + 35 = 89.
    # With Laderman: 23 + 23 + 35 = 81.
    
    # I will implement Laderman.
    # Reference: Laderman, 1976.
    # The 23 products are:
    # M1 = A11 B11
    # M2 = A12 B22
    # M3 = A21 B12
    # M4 = A22 B21
    # M5 = (A11 + A22) (B11 + B22)
    # ... this is getting long.
    
    # Let's try to find a simpler improvement.
    # What if I just use Strassen on as many 2x2x2 blocks as possible?
    # We found 2 blocks in the 3x3x3 area?
    # No, 3x3x3 doesn't decompose into 2x2x2 blocks easily.
    
    # Let's just use Strassen on the 2x2x2 blocks we identified.
    # Block 1: i=0,1; j=0,1; k=0,1.
    # Block 2: i=0,1; j=2,3; k=2,3.
    # Block 3: i=0,1; j=4,?; k=4,5. (No j left)
    
    # So we have 2 Strassen blocks.
    # Rank 90 - 2 = 88.
    
    # Let's try to fit more.
    # What if we use i=2 for the third block?
    # Block 3: i=2,?; j=4,?; k=4,5.
    # We need 2 i's. We have i=0,1 used in B1, B2.
    # Can we reuse i?
    # If we reuse i, the monomials are not distinct.
    # $A_{2,4} B_{4,4}$ is a distinct monomial.
    # $A_{0,4} B_{4,4}$ is a distinct monomial.
    # So we can have a block with i=0,2?
    # If we use i=0,2 for Block 3, we are reusing i=0 from Block 1.
    # Does this invalidate the rank reduction?
    # The rank reduction relies on the fact that the 8 monomials of the block can be replaced by 7 linear combos.
    # If the monomials are distinct from all other monomials in the sum, it's fine.
    # $A_{0,4} B_{4,4}$ is distinct from $A_{0,0} B_{0,0}$.
    # So yes, we can use i=0,2 for the third block.
    
    # Block 3: i=0,2; j=4,?; k=4,5.
    # We need 2 j's. We have j=4. We need one more.
    # j=0,1,2,3 are used in B1, B2.
    # Can we reuse j=0 for Block 3?
    # $A_{0,0} B_{0,4}$ is distinct from $A_{0,0} B_{0,0}$.
    # So yes.
    # Let's use j=4,0 for Block 3.
    
    # So:
    # B1: i=0,1; j=0,1; k=0,1.
    # B2: i=0,1; j=2,3; k=2,3.
    # B3: i=0,2; j=4,0; k=4,5.
    
    # Are all 24 monomials distinct?
    # B1: A_{0,0}B_{0,0}, A_{0,0}B_{0,1}, A_{0,1}B_{1,0}, A_{0,1}B_{1,1}, A_{1,0}B_{0,0}, A_{1,0}B_{0,1}, A_{1,1}B_{1,0}, A_{1,1}B_{1,1}.
    # B2: A_{0,2}B_{2,2}, A_{0,2}B_{2,3}, A_{0,3}B_{3,2}, A_{0,3}B_{3,3}, A_{1,2}B_{2,2}, A_{1,2}B_{2,3}, A_{1,3}B_{3,2}, A_{1,3}B_{3,3}.
    # B3: A_{0,4}B_{4,4}, A_{0,4}B_{4,5}, A_{0,0}B_{0,4}, A_{0,0}B_{0,5}, A_{2,4}B_{4,4}, A_{2,4}B_{4,5}, A_{2,0}B_{0,4}, A_{2,0}B_{0,5}.
    
    # All distinct?
    # B1 uses j=0,1. B2 uses j=2,3. B3 uses j=4,0.
    # B3 uses j=0. B1 uses j=0.
    # Monomials in B3 with j=0: $A_{0,0}B_{0,4}, A_{0,0}B_{0,5}, A_{2,0}B_{0,4}, A_{2,0}B_{0,5}$.
    # Monomials in B1 with j=0: $A_{0,0}B_{0,0}, A_{1,0}B_{0,0}$.
    # Distinct k's (4,5 vs 0,1). So yes, distinct.
    
    # So we have 3 Strassen blocks.
    # Rank reduction: 3.
    # New rank: 87.
    
    # Can we do better?
    # What if we use Laderman?
    # I will implement the 3 Strassen blocks.
    
    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        # Strassen 7 products
        # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
        # M2 = (A[i2,j1]+A[i2,j2]) * B[j1,k2]
        # M3 = A[i1,j1] * (B[j2,k1]-B[j1,k1])
        # M4 = A[i2,j2] * (B[j2,k2]-B[j1,k2])
        # M5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
        # M6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j1,k2])
        # M7 = (A[i1,j2]-A[i2,j2]) * (B[j2,k1]+B[j2,k2])
        
        # We need to add these 7 terms to the decomposition.
        # Each term is a product of a linear combo of A and a linear combo of B.
        # This corresponds to a rank-1 tensor in the (i,j,k) space?
        # No, it's a bilinear form.
        # The term $L_A \cdot L_B$ contributes to $C$.
        # $C_{ik} = \sum L_A L_B W_{ik}$.
        # The W part is determined by how the Strassen outputs combine.
        
        # Actually, it's easier to just generate the rank-1 tensors for the schoolbook 8 terms
        # and then replace them? No, that's not how it works.
        # We need to provide the U, V, W matrices for the 7 Strassen terms.
        
        # For each Strassen product M_r, we have:
        # U_r is the linear combo of A entries.
        # V_r is the linear combo of B entries.
        # W_r is the linear combo of C entries (output).
        
        # We need to know the W coefficients for Strassen.
        # C11 = M1 - M4 + M5 - M6 + M7
        # C12 = M3 + M4 + M5
        # C21 = M1 + M2 - M3 - M7
        # C22 = M1 - M2 + M4 + M6
        
        # So for each M_r, we add a term to U, V, and W.
        
        # M1:
        u = {idx_u(i1,j1): 1, idx_u(i2,j2): 1}
        v = {idx_v(j1,k1): 1, idx_v(j2,k2): 1}
        w = {idx_w(k1,i1): 1, idx_w(k1,i2): -1, idx_w(k2,i1): 1, idx_w(k2,i2): -1}
        # Wait, C11 is (i1,k1). C21 is (i2,k1). C12 is (i1,k2). C22 is (i2,k2).
        # M1 contributes +1 to C11, +1 to C21, -1 to C12? No.
        # C11 = M1 ...
        # C21 = M1 ...
        # C12 = -M1 ...? No, C12 = M3+M4+M5. M1 is not in C12.
        # C22 = M1 ...
        
        # Let's map local (0,0) to (i1,k1), (0,1) to (i1,k2), (1,0) to (i2,k1), (1,1) to (i2,k2).
        # C[0,0] = M1 - M4 + M5 - M6 + M7
        # C[0,1] = M3 + M4 + M5
        # C[1,0] = M1 + M2 - M3 - M7
        # C[1,1] = M1 - M2 + M4 + M6
        
        # M1:
        w1 = {idx_w(k1,i1): 1, idx_w(k2,i2): 1, idx_w(k1,i2): 1, idx_w(k2,i1): 0} # C11, C22, C21
        # Check: C11 (i1,k1) is 1. C22 (i2,k2) is 1. C21 (i2,k1) is 1.
        # C12 (i1,k2) is 0.
        
        # M2:
        u2 = {idx_u(i2,j1): 1, idx_u(i2,j2): 1}
        v2 = {idx_v(j1,k2): 1}
        w2 = {idx_w(k1,i2): 1, idx_w(k2,i2): -1} # C21, C22
        
        # M3:
        u3 = {idx_u(i1,j1): 1}
        v3 = {idx_v(j2,k1): 1, idx_v(j1,k1): -1}
        w3 = {idx_w(k2,i1): 1, idx_w(k1,i2): -1} # C12, C21
        
        # M4:
        u4 = {idx_u(i2,j2): 1}
        v4 = {idx_v(j2,k2): 1, idx_v(j1,k2): -1}
        w4 = {idx_w(k1,i1): -1, idx_w(k2,i1): 1, idx_w(k2,i2): 1} # C11, C12, C22
        
        # M5:
        u5 = {idx_u(i1,j1): 1, idx_u(i1,j2): 1}
        v5 = {idx_v(j2,k2): 1}
        w5 = {idx_w(k1,i1): 1, idx_w(k2,i1): 1} # C11, C12
        
        # M6:
        u6 = {idx_u(i2,j1): 1, idx_u(i1,j1): -1}
        v6 = {idx_v(j1,k1): 1, idx_v(j1,k2): 1}
        w6 = {idx_w(k1,i1): -1, idx_w(k2,i2): 1} # C11, C22
        
        # M7:
        u7 = {idx_u(i1,j2): 1, idx_u(i2,j2): -1}
        v7 = {idx_v(j2,k1): 1, idx_v(j2,k2): 1}
        w7 = {idx_w(k1,i1): 1, idx_w(k1,i2): -1} # C11, C21
        
        add_decomposition(list(u1.items()), list(v1.items()), list(w1.items()))
        add_decomposition(list(u2.items()), list(v2.items()), list(w2.items()))
        add_decomposition(list(u3.items()), list(v3.items()), list(w3.items()))
        add_decomposition(list(u4.items()), list(v4.items()), list(w4.items()))
        add_decomposition(list(u5.items()), list(v5.items()), list(w5.items()))
        add_decomposition(list(u6.items()), list(v6.items()), list(w6.items()))
        add_decomposition(list(u7.items()), list(v7.items()), list(w7.items()))

    # Add the 3 Strassen blocks
    add_strassen_block(0, 1, 0, 1, 0, 1)
    add_strassen_block(0, 1, 2, 3, 2, 3)
    add_strassen_block(0, 2, 4, 0, 4, 5)
    
    # Now we need to add the remaining schoolbook terms.
    # Total terms 90.
    # Strassen blocks cover 24 terms.
    # Remaining 66 terms.
    
    # We can just iterate through all 90 terms and add them if they are not in the Strassen blocks.
    # But we need to know which ones are in the Strassen blocks.
    # The Strassen blocks replace the 8 terms of the block.
    # So we should NOT add the schoolbook terms for the indices in the Strassen blocks.
    
    # Define the sets of (i,j,k) covered by Strassen
    covered = set()
    for i1, i2, j1, j2, k1, k2 in [(0, 1, 0, 1, 0, 1), (0, 1, 2, 3, 2, 3), (0, 2, 4, 0, 4, 5)]:
        for i in [i1, i2]:
            for j in [j1, j2]:
                for k in [k1, k2]:
                    covered.add((i, j, k))
                    
    # Add remaining schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    u = {idx_u(i, j): 1}
                    v = {idx_v(j, k): 1}
                    w = {idx_w(k, i): 1}
                    add_decomposition(list(u.items()), list(v.items()), list(w.items()))
                    
    # Convert to matrices
    rank = len(decompositions)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for r in range(rank):
        u, v, w = decompositions[r]
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
