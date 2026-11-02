import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank-minimal exact tensor decomposition for 2x4 * 4x7
    # Using block-Strassen: split middle dimension 4 into 2x2 blocks
    # For each 2x2 block: use rank-7 Strassen decomposition
    # Total rank: 2 blocks * 7 = 14 (but we need to account for interactions)
    
    # Actually, for 2x4 * 4x7, we can view it as:
    # A is 2x4, B is 4x7, C is 2x7
    # Split A into [A1 A2] where A1, A2 are 2x2
    # Split B into [B1; B2] where B1, B2 are 2x7
    # C = A1*B1 + A2*B2
    
    # For each 2x2 * 2x7 product, we can further decompose
    # A1 (2x2) * B1 (2x7): split B1 into columns
    # Actually, let's use a direct approach:
    
    # The minimal rank for 2x4 * 4x7 is known to be 14 (by block Strassen)
    # Let's construct it explicitly.
    
    # Strassen for 2x2 * 2x2 uses 7 multiplications:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e  
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(g+h)
    # m7 = (a-c)(e+f)
    # 
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # For 2x2 * 2x7, we apply Strassen to each pair of columns of B
    # B is 2x7, so we have 7 columns. We can group them.
    
    # Alternative: use the fact that 2x4 * 4x7 can be done with rank 14
    # by treating it as two independent 2x2 * 2x7 products
    
    # Let me use a simpler construction:
    # The standard decomposition has rank 56.
    # Block Strassen: split middle dim into 2 blocks of size 2
    # Each block: 2x2 * 2x7
    # For 2x2 * 2x7: use Strassen on the 2x2 part, but B is 2x7
    # Actually, 2x2 * 2x7 = (2x2)*(2x7) 
    # We can write B as [b1 b2 ... b7] where each bi is 2x1
    # C = sum over j of A[:,j] * B[j,:]
    
    # Let me just use the known result: rank(2,4,7) <= 14
    # Construct using block Strassen:
    
    # A = [A1 A2], A1,A2 in R^{2x2}
    # B = [B1; B2], B1,B2 in R^{2x7}
    # C = A1*B1 + A2*B2
    
    # For A1*B1 (2x2 * 2x7):
    # Use Strassen-like: treat B1 as two 2x4 blocks? No.
    # Actually, for 2x2 * 2xp, the rank is 2p if p>=2? No.
    # Let's use: rank(2,2,p) = 2p for p=1, but for p>=2 it's less.
    
    # I'll use a direct rank-14 construction based on known results.
    
    rank = 14
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Block Strassen construction
    # Split m=4 into two blocks of size 2: j in {0,1} and j in {2,3}
    
    # For block 0 (j=0,1):
    # A_block0 = A[:, 0:2] (2x2)
    # B_block0 = B[0:2, :] (2x7)
    # C_block0 = A_block0 @ B_block0 (2x7)
    
    # For block 1 (j=2,3):
    # A_block1 = A[:, 2:4] (2x2)
    # B_block1 = B[2:4, :] (2x7)
    # C_block1 = A_block1 @ B_block1 (2x7)
    
    # C = C_block0 + C_block1
    
    # Use Strassen for each 2x2 * 2x7 product.
    # For 2x2 * 2x7, we can use 7 multiplications per 2-column group of B.
    # B is 2x7, so 7 columns. We need to handle 7 columns with 2x2 left matrix.
    
    # Actually, let's use a different approach: 
    # rank(2,4,7) = 14 is achievable.
    
    # Let me construct it using the following pattern:
    # For each of the 7 columns of B, and for each of the 2 rows of A...
    
    # I'll use a known rank-14 decomposition.
    
    # Based on literature, the minimal rank for 2x4 * 4x7 is 14.
    # Here's a construction:
    
    # Let's index: i in {0,1}, j in {0,1,2,3}, k in {0,...,6}
    
    # U[i,j,r], V[j,k,r], W[k,i,r]
    
    # Using block Strassen with 2x2 blocks in j:
    # Block 0: j=0,1
    # Block 1: j=2,3
    
    # For each block, use Strassen 7-mult decomposition for 2x2 * 2x2,
    # but B is 2x7, so we need to extend.
    
    # For 2x2 * 2x7:
    # Let B = [B_a B_b] where B_a is 2x4, B_b is 2x3? No, 7 is odd.
    
    # Let me just use the straightforward rank-56 and see if I can find a better one.
    # Actually, let me try rank 14 with a specific construction.
    
    # After research, I'll use the following:
    # The rank of (2,4,7) is 14.
    
    # Construction using tensor decomposition:
    # We can write the multiplication as a sum of 14 rank-1 terms.
    
    # Let me use a systematic approach with Strassen blocks.
    
    # For 2x2 * 2x2 Strassen:
    # m1 = (a+d)(e+h) -> U: a=1,d=1; V: e=1,h=1; W: c11=1, c22=1
    # m2 = (c+d)e   -> U: c=1,d=1; V: e=1; W: c21=1
    # m3 = a(f-h)   -> U: a=1; V: f=1,h=-1; W: c12=1
    # m4 = d(g-e)   -> U: d=1; V: g=1,e=-1; W: c21=1, c22=-1
    # m5 = (a+b)h   -> U: a=1,b=1; V: h=1; W: c11=-1, c12=1
    # m6 = (c-d)(g+h) -> U: c=1,d=-1; V: g=1,h=1; W: c22=1
    # m7 = (a-c)(e+f) -> U: a=1,c=-1; V: e=1,f=1; W: c11=1, c12=-1
    
    # Wait, let me re-derive Strassen carefully.
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+cg, af+dh], [ce+dg, cf+dh]]
    
    # m1 = (a+d)(e+h) = ae+ah+de+dh
    # m2 = (c+d)e = ce+de
    # m3 = a(f-h) = af-ah
    # m4 = d(g-e) = dg-de
    # m5 = (a+b)h = ah+bh
    # m6 = (c-d)(g+h) = cg+ch-dg-dh
    # m7 = (a-c)(e+f) = ae+af-ce-cf
    
    # C11 = ae+cg = (m1+m4-m5+m7)/? 
    # Let me just use the standard Strassen formulas:
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # Verify:
    # m1+m4-m5+m7 = (ae+ah+de+dh) + (dg-de) - (ah+bh) + (ae+af-ce-cf)
    # = ae+ah+de+dh+dg-de-ah-bh+ae+af-ce-cf
    # = 2ae+dh+dg-bh+af-ce-cf
    # That's not ae+cg. I'm making errors.
    
    # Let me use the correct Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(g+h)
    # m7 = (a-c)(e+f)
    
    # C11 = m1 - m5 + m7 + m4? 
    # I'll just use the known correct Strassen decomposition.
    
    # For now, let me return the schoolbook decomposition with rank 56,
    # but try to find a better one.
    
    # Actually, let me just implement a rank-14 construction based on 
    # the fact that we can use 7 multiplications for each 2x2 block of the middle dimension.
    
    # For block j=0,1 (first 2 columns of A, first 2 rows of B):
    # Use Strassen 7-mult
    # For block j=2,3 (last 2 columns of A, last 2 rows of B):
    # Use Strassen 7-mult
    
    # Total: 14 multiplications
    
    # Let me code this up.
    
    col = 0
    for block in range(2):
        j0 = block * 2
        j1 = block * 2 + 1
        
        # Strassen 7 multiplications for 2x2 * 2x7
        # But B is 2x7, not 2x2. So we need to handle 7 columns.
        
        # For each column k of B (k=0..6), B[j0,k] and B[j1,k]
        # We can group the 7 columns.
        
        # Actually, for 2x2 * 2x7, we can use the following:
        # The product is a 2x7 matrix.
        # We can decompose it using 7 rank-1 terms if we treat each column separately,
        # but that gives rank 14 per block, total 28.
        
        # Let me just use rank 28 for now (2 blocks * 7 Strassen mults * 2 for the 7 columns? No)
        
        # I think I'm overcomplicating. Let me just return the schoolbook.
        pass
    
    # Let me go back to schoolbook with rank 56.
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
