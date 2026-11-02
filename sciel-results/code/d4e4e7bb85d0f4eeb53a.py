"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition.

Strategy: Use the schoolbook decomposition as baseline (rank 48), then attempt
to find a lower-rank exact decomposition using algebraic structure.

The 2x3x8 multiplication tensor can be viewed as a Kronecker-like product.
For 2x3x2 multiplication, we can potentially use a rank-5 or rank-6 decomposition.
Then combine with the 1x1x4 identity to get total rank 4*r where r is the
rank of the 2x3x2 component.

Known results:
- Rank(2x2x2) = 7 (Strassen)
- Rank(2x3x2) is between 5 and 8. We'll try to find a valid decomposition.

We'll implement a search that:
1. Starts with schoolbook (rank 48)
2. Tries rank 24 (4*6) using a 2x3x2 rank-6 decomposition
3. Tries rank 20 (4*5) using a 2x3x2 rank-5 decomposition
4. Verifies exactness with integer arithmetic
"""

import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # The multiplication tensor T has indices:
    # U: shape (n*m, rank) = (6, rank)
    # V: shape (m*p, rank) = (24, rank)  
    # W: shape (p*n, rank) = (16, rank)
    # T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
    # where i = a*m+b (a in 0..n-1, b in 0..m-1)
    #       j = b*p+c (b in 0..m-1, c in 0..p-1)
    #       k = c*n+a (c in 0..p-1, a in 0..n-1)
    # T[i,j,k] = 1 if the indices are consistent (same b and same c in the right way), else 0
    
    # Actually, for matrix multiplication (A B)_{i,k} = sum_b A_{i,b} B_{b,k}
    # The tensor T_{i,b,k} = 1 for all valid (i,b,k), else 0
    # In our indexing: i = a*m+b, j = b'*p+c, k = c'*n+a'
    # T is 1 when b = b' and c = c' and a = a' ... wait, let me re-derive.
    
    # Standard: (AB)_{ik} = sum_j A_{ij} B_{jk}
    # T_{i,j,k} = 1 for all (i,j,k) with i in 1..n, j in 1..m, k in 1..p
    # So T has exactly n*m*p = 48 ones.
    
    # U has rows indexed by (i,j) pairs: row = i*m + j for i in 0..n-1, j in 0..m-1
    # V has rows indexed by (j,k) pairs: row = j*p + k for j in 0..m-1, k in 0..p-1
    # W has rows indexed by (k,i) pairs: row = k*n + i for k in 0..p-1, i in 0..n-1
    
    # T[i,j,k] in the tensor = 1 always (for the multiplication tensor)
    # The reconstruction is: for each (i,k) output, (AB)_{ik} = sum_j A_{ij} B_{jk}
    # = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal sum_j A_{ij} B_{jk}
    
    # So we need: for all (i, j, k): 
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # And for "wrong" combinations... actually the tensor only has entries for valid triples.
    
    # Let me just verify with the schoolbook approach.
    
    def verify(U, V, W, rank):
        """Verify that the decomposition exactly reproduces matrix multiplication."""
        U_arr = np.array(U, dtype=object)
        V_arr = np.array(V, dtype=object)
        W_arr = np.array(W, dtype=object)
        
        # For each (i, j, k), check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    # Schoolbook: rank = n*m*p = 48
    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W
    
    # Try to get a lower rank using Kronecker structure.
    # The 2x3x8 multiplication can be seen as:
    # A is 2x3, B is 3x8.
    # We can partition B into four 3x2 blocks: B = [B1 | B2 | B3 | B4]
    # Then AB = [A*B1 | A*B2 | A*B3 | A*B4]
    # Each A*Bk is a 2x2 product.
    # If we had a rank-r decomposition for 2x2x2, total rank would be 4r.
    # Strassen: rank(2x2x2) = 7, so total rank = 28.
    
    # Strassen for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For our case, each "2x2 multiplication" uses the same A (2x3) but different 3x2 blocks of B.
    # Actually this doesn't directly apply because A is 2x3, not 2x2.
    
    # Let me think differently. 
    # A is 2x3, B is 3x8.
    # AB is 2x8.
    # We can write B = sum over 8 columns, each column is 3x1.
    # (AB)[:,k] = A * B[:,k]
    # So we need 8 matrix-vector products of 2x3 by 3x1.
    
    # Alternatively, partition A's rows: A = [row1; row2] where row1, row2 are 1x3.
    # (AB)_{1,k} = row1 * B[:,k] = sum_j row1[j] * B[j,k]
    # (AB)_{2,k} = row2 * B[:,k] = sum_j row2[j] * B[j,k]
    
    # This is essentially 2 linear combinations for each of 8 outputs.
    
    # Let me try a direct approach: assume rank r and see what's feasible.
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(6, 24, 16) = 24.
    # So rank >= 24.
    
    # Let's try rank 24.
    # For rank 24, we need to find U (6x24), V (24x24), W (16x24) such that
    # for all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is a system of 48 equations in 24*6 + 24*24 + 16*24 = 144 + 576 + 384 = 1104 variables.
    # With the constraint that entries are integers or half-integers.
    
    # One approach: use the fact that V can be chosen as a permutation/scaled identity-like matrix.
    # If V is diagonal with entries v_j for row j (j in 0..23), then:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # If V is 24x24 identity (or scaled), this simplifies.
    
    # Actually, let me try a structured approach.
    # Let's say rank = 24. Index the rank dimension by (b, k') where b in 0..2 (3 values for the middle index)
    # and k' in 0..7 (8 values). That's 3*8 = 24.
    
    # U[i*m+j, (b,k')] = u_{i,j,b,k'}
    # V[j*p+k, (b,k')] = v_{j,k,b,k'}
    # W[k*n+i, (b,k')] = w_{k,i,b,k'}
    
    # The condition is: for all (i,j,k):
    # sum_{b,k'} u_{i,j,b,k'} * v_{j,k,b,k'} * w_{k,i,b,k'} = 1
    
    # If we set v_{j,k,b,k'} = delta_{j,b} * delta_{k,k'} (i.e., V is a block diagonal with identity blocks),
    # then v_{j,k,b,k'} = 1 if j=b and k=k', else 0.
    # Then the sum becomes: u_{i,j,j,k} * 1 * w_{k,i,j,k} = u_{i,j,j,k} * w_{k,i,j,k} = 1
    # So we need u_{i,j,j,k} * w_{k,i,j,k} = 1 for all (i,j,k).
    # We can set u_{i,j,j,k} = 1 and w_{k,i,j,k} = 1, and all other entries 0.
    # But wait, this means U has non-zero entries only at position (i*m+j, j*p+k) in the rank space... 
    # but rank space is indexed by (b,k'), so U[i*m+j, (b,k')] is non-zero only when b=j, k'=k.
    # That's exactly the schoolbook! Rank 48, not 24.
    
    # The issue is that with rank 24, the rank index can't separate all (j,k) pairs uniquely
    # because there are 3*8=24 pairs but we need the product to work out.
    
    # Let me try a different structure. What if we use rank 24 with the rank index being (j, k) 
    # but allow V to mix them?
    
    # Actually, I realize that for the exact decomposition, we might not be able to go below 48
    # with simple structures. Let me try rank 32 or 24 with numerical search.
    
    # Given time constraints, let me try a few specific ranks with a simple numerical approach.
    
    def try_rank(r):
        """Try to find a decomposition of rank r using numpy least squares or similar."""
        if time.time() > deadline:
            return None
        
        # We need U (n*m x r), V (m*p x r), W (p*n x r)
        # such that for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        
        # This is a tensor decomposition problem. Let's use a greedy/iterative approach.
        # Initialize with random small integers and iterate.
        
        # For simplicity, try to fix V as a Hadamard-like or structured matrix.
        
        # Actually, let me try the simplest thing: rank = n*m*p = 48 (schoolbook) as fallback,
        # and try rank 24 with a specific structure.
        
        # For rank 24, let's try:
        # U[i*m+j, (b,k')] = 1 if b==j else 0, for all k'
        # Wait, that gives U row i*m+j has 8 ones (one for each k' when b=j).
        
        # Let me just return the schoolbook for now and try to improve.
        return None
    
    # Let's just use the schoolbook decomposition with rank 48.
    # But wait - can we do better?
    
    # The key insight: for 2x3x8, we can use the following:
    # Partition the 3 middle dimension into parts. 
    # Actually, 2x3 = 2x(1+2), so A = [A1 | A2] where A1 is 2x1, A2 is 2x2.
    # B = [B1; B2] where B1 is 1x8, B2 is 2x8.
    # AB = A1*B1 + A2*B2
    # A1*B1: 2x1 times 1x8 = 2x8, rank contribution = 2*1*8 = 16 (schoolbook) or can be done in rank 2*1*8=16.
    # Actually for 2x1x8, the multiplication is just scaling: (A1*B1)_{i,k} = A1[i]*B1[k].
    # This can be done with rank 1: U[i,0]=A1[i], V[0,0]=1, W[k,0]=B1[k]. But we need it to work for all inputs.
    # For the tensor: T_{i,j,k} where j is the only middle index. T_{i,0,k} = 1 for all i,k.
    # Rank of 2x1x8 tensor: we need sum_c U[i,c]*V[0,c]*W[k,c] = 1 for all i,k.
    # This is rank 1: U[i,0]=1, V[0,0]=1, W[k,0]=1. Yes!
    
    # A2*B2: 2x2 times 2x8. 
    # 2x2x8 multiplication. Rank of 2x2x8?
    # This is 2x2 times 2x8. We can partition B2 into four 2x2 blocks: B2 = [C1|C2|C3|C4].
    # A2*B2 = [A2*C1 | A2*C2 | A2*C3 | A2*C4].
    # Each A2*Ck is 2x2 times 2x2, which has rank 7 (Strassen).
    # Total: 4*7 = 28 for the 2x2x8 part.
    # Plus 1 for the 2x1x8 part.
    # Total: 29.
    
    # Wait, but we need to be careful about the tensor structure.
    
    # Let me construct this explicitly.
    
    # A is 2x3: A = [a00 a01 a02; a10 a11 a12]
    # B is 3x8: B = [b00 b01 ... b07; b10 b11 ... b17; b20 b21 ... b27]
    # AB is 2x8.
    
    # Split: A = [A_1 | A_2] where A_1 = [a00; a10] (2x1), A_2 = [a01 a02; a11 a12] (2x2)
    # B = [B_1; B_2] where B_1 = [b00 b01 ... b07] (1x8), B_2 = [b10 b11 ... b17; b20 b21 ... b27] (2x8)
    
    # (AB)_{ik} = a_i0*b_0k + a_i1*b_1k + a_i2*b_2k = (A_1 * B_1)_i + (A_2 * B_2)_ik
    
    # Part 1: A_1 * B_1 (2x1 times 1x8)
    # (A_1*B_1)_{ik} = a_i0 * b_0k
    # Tensor: T1_{i, j, k} where j is the single middle index (0). T1_{i,0,k} = 1.
    # Rank 1 decomposition: U1[i,0] = 1, V1[0,0] = 1, W1[k,0] = 1.
    # In our indexing: 
    #   A_1 corresponds to A's first column. In the full A, column index j=0.
    #   U row: i*m + j = i*3 + 0 = 3i for i in {0,1}. So U[0,0] and U[3,0].
    #   V row: j*p + k = 0*8 + k = k for k in 0..7. So V[0,0], V[1,0], ..., V[7,0].
    #   W row: k*n + i = k*2 + i for k in 0..7, i in {0,1}. So W[2k,0] and W[2k+1,0].
    
    # Part 2: A_2 * B_2 (2x2 times 2x8)
    # A_2 = [a01 a02; a11 a12], B_2 = [b10 ... b17; b20 ... b27]
    # (A_2*B_2)_{ik} = a_i1*b_1k + a_i2*b_2k
    # Tensor: T2_{i, j, k} where j in {1,2} (the middle indices within A_2/B_2), k in 0..7.
    # T2_{i,j,k} = 1 for i in {0,1}, j in {1,2}, k in 0..7.
    
    # Split B_2 into four 2x2 blocks: B_2 = [B_2^{(1)} | B_2^{(2)} | B_2^{(3)} | B_2^{(4)}]
    # where B_2^{(q)} is 2x2 for q=1,2,3,4.
    # B_2^{(1)} = [b10 b11; b20 b21], B_2^{(2)} = [b12 b13; b22 b23], etc.
    
    # (A_2*B_2^{(q)})_{ik} for i,k in {0,1} is a standard 2x2x2 multiplication.
    # Strassen decomposition for each: rank 7.
    
    # Total rank: 1 (for A_1*B_1) + 4*7 (for the four 2x2x2 blocks) = 1 + 28 = 29.
    
    # Let me construct this.
    
    # Strassen for 2x2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # c00 = M1+M4-M5+M7
    # c01 = M3+M5
    # c10 = M2+M4
    # c11 = M1-M2+M3+M6
    
    # For the full problem, let's build the decomposition.
    
    # This is getting complex. Let me just implement the rank 29 construction carefully.
    
    if time.time() > deadline:
        U, V, W = schoolbook()
        return {"U": U, "V": V, "W": W}
    
    # Let me implement the rank 29 construction.
    # 
    # Rank dimension: 1 (for part 1) + 4*7 (for the four Strassen blocks) = 29.
    # 
    # Part 1 (rank 1):
    #   U[3i, 0] = 1 for i in {0,1}
    #   V[k, 0] = 1 for k in 0..7
    #   W[2k+i, 0] = 1 for k in 0..7, i in {0,1}
    #
    # Part 2: For each block q in 0..3, we have a 2x2x2 Strassen.
    #   The middle indices for A_2 are j=1,2 (global j indices).
    #   The output indices for each block q are k in {4q, 4q+1, 4q+2, 4q+3}.
    #   Wait, B_2 has 2 rows and 8 columns. Splitting into 2x2 blocks:
    #   Block q uses columns 4q, 4q+1, 4q+2, 4q+3? No, 2x2 means 2 columns.
    #   B_2 is 2x8. Split into four 2x2 blocks: 
    #   Block 0: cols 0,1. Block 1: cols 2,3. Block 2: cols 4,5. Block 3: cols 6,7.
    #   
    #   For block q, the Strassen multiplies A_2 (2x2) by B_2_block_q (2x2).
    #   The A_2 indices: global i in {0,1}, global j in {1,2}.
    #   The B_2_block_q indices: global j in {1,2}, global k in {4q, 4q+1}.
    #   Wait, B_2's columns correspond to global B columns 1..7 (since B_1 is col 0).
    #   B_2 column 0 = global B column 1. B_2 column 1 = global B column 2. Etc.
    #   
    #   So for block q (q=0,1,2,3):
    #   B_2_block_q columns: global B columns (4q+1) and (4q+2).
    #   
    #   The output (A_2*B_2_block_q) has entries for i in {0,1} and k in {4q+1, 4q+2}.
    #   
    #   Strassen M factors:
    #   M1_q = (A2[0,0]+A2[1,1]) * (Bq[0,0]+Bq[1,1])
    #   ... etc.
    #   where A2[r, s] is the (r,s) entry of A_2, which is global A[r, s+1] (since A_2 starts at col 1).
    #   Bq[r, s] is the (r,s) entry of B_2_block_q, which is global B[s+4q+1, r]... 
    #   Wait, I'm getting confused with the indexing.
    
    # Let me be very explicit.
    # Global A: 2x3, A[i][j] for i in {0,1}, j in {0,1,2}.
    # Global B: 3x8, B[j][k] for j in {0,1,2}, k in {0,...,7}.
    # Global output C: 2x8, C[i][k] for i in {0,1}, k in {0,...,7}.
    # C[i][k] = sum_{j=0}^{2} A[i][j] * B[j][k].
    
    # Split j=0 as "part 1", j=1,2 as "part 2".
    # Part 1: C[i][k] += A[i][0] * B[0][k].
    #   U row: i*3+0 = 3i. V row: 0*8+k = k. W row: k*2+i.
    #   U[3i][col] = 1, V[k][col] = 1, W[2k+i][col] = 1 for one rank column col.
    
    # Part 2: C[i][k] += A[i][1]*B[1][k] + A[i][2]*B[2][k].
    #   For each pair of output columns (k1, k2) where k1=4q+1, k2=4q+2 (q=0,1,2,3):
    #   We compute the 2x2 matrix product of A_2 (2x2) and B_2_block (2x2).
    #   A_2 = [[A[0][1], A[0][2]], [A[1][1], A[1][2]]]
    #   B_2_block = [[B[1][k1], B[1][k2]], [B[2][k1], B[2][k2]]]
    #   
    #   Strassen computes this with 7 multiplications.
    #   
    #   For each Strassen factor M_q_t (q=block index, t=Strassen index 0..6):
    #   M_q_t = (linear combo of A_2 entries) * (linear combo of B_2_block entries)
    #   
    #   U contribution: for each A_2 entry A[i][j] (i in {0,1}, j in {1,2}),
    #   U[i*3+j][col] gets the coefficient from the A-side of M_q_t.
    #   
    #   V contribution: for each B_2_block entry B[j][k] (j in {1,2}, k in {k1,k2}),
    #   V[j*8+k][col] gets the coefficient from the B-side of M_q_t.
    #   
    #   W contribution: for each output C[i][k] (i in {0,1}, k in {k1,k2}),
    #   W[k*2+i][col] gets the coefficient from the output-side of M_q_t.
    
    # Let me code this up.
    
    rank = 29
    U = [[0]*rank for _ in range(6)]
    V = [[0]*rank for _ in range(24)]
    W = [[0]*rank for _ in range(16)]
    
    # Part 1: rank column 0
    for i in range(2):
        U[3*i][0] = 1
    for k in range(8):
        V[k][0] = 1
    for k in range(8):
        for i in range(2):
            W[2*k+i][0] = 1
    
    # Part 2: Strassen for four 2x2 blocks
    # Strassen coefficients:
    # M1 = (a00+a11)(b00+b11) -> A: (1,0,0,1), B: (1,0,0,1), C: (1,0,0,-1)
    # M2 = (a10+a11)b00 -> A: (0,0,1,1), B: (1,0,0,0), C: (0,0,1,0)
    # M3 = a00(b10-b11) -> A: (1,0,0,0), B: (0,0,1,-1), C: (0,1,0,0)
    # M4 = a11(b01-b00) -> A: (0,0,0,1), B: (0,1,-1,0), C: (1,0,0,0)
    # M5 = (a00+a01)b11 -> A: (1,1,0,0), B: (0,0,0,1), C: (-1,0,0,0)
    # M6 = (a10-a00)(b00+b01) -> A: (-1,0,1,0), B: (1,1,0,0), C: (0,0,0,1)
    # M7 = (a01-a11)(b10+b11) -> A: (0,1,0,-1), B: (0,0,1,1), C: (1,0,0,0)
    
    # A_2 indexing: (i, j) where i in {0,1}, j in {1,2} (global j).
    # In the 2x2 Strassen, a00=A[0][1], a01=A[0][2], a10=A[1][1], a11=A[1][2].
    # So A_2[r][s] where r,s in {0,1} maps to global A[r][s+1].
    # U row: i*3+j = r*3+(s+1).
    
    # B_2_block indexing: (j, k) where j in {1,2}, k in {k1, k2}.
    # b00=B[1][k1], b01=B[1][k2], b10=B[2][k1], b11=B[2][k2].
    # So B_2_block[r][s] maps to global B[r+1][k1+s] (if s=0 -> k1, s=1 -> k2).
    # V row: j*8+k = (r+1)*8 + (k1+s).
    
    # Output: C[i][k] where i in {0,1}, k in {k1, k2}.
    # c00=C[0][k1], c01=C[0][k2], c10=C[1][k1], c11=C[1][k2].
    # W row: k*2+i.
    # c00: k=k1, i=0 -> W[k1*2+0]
    # c01: k=k2, i=0 -> W[k2*2+0]
    # c10: k=k1, i=1 -> W[k1*2+1]
    # c11: k=k2, i=1 -> W[k2*2+1]
    
    # Strassen factor definitions: (A_coeffs for (a00,a01,a10,a11), B_coeffs for (b00,b01,b10,b11), C_coeffs for (c00,c01,c10,c11))
    strassen = [
        ((1,0,0,1), (1,0,0,1), (1,0,0,-1)),   # M1
        ((0,0,1,1), (1,0,0,0), (0,0,1,0)),     # M2
        ((1,0,0,0), (0,0,1,-1), (0,1,0,0)),    # M3
        ((0,0,0,1), (0,1,-1,0), (1,0,0,0)),    # M4
        ((1,1,0,0), (0,0,0,1), (-1,0,0,0)),    # M5
        ((-1,0,1,0), (1,1,0,0), (0,0,0,1)),    # M6
        ((0,1,0,-1), (0,0,1,1), (1,0,0,0)),    # M7
    ]
    
    col = 1
    for q in range(4):
        k1 = 2*q
        k2 = 2*q + 1
        for t in range(7):
            A_coeffs, B_coeffs, C_coeffs = strassen[t]
            # A_coeffs for (a00,a01,a10,a11) = (A[0][1], A[0][2], A[1][1], A[1][2])
            # a00 -> U row 0*3+1 = 1
            # a01 -> U row 0*3+2 = 2
            # a10 -> U row 1*3+1 = 4
            # a11 -> U row 1*3+2 = 5
            A_rows = [1, 2, 4, 5]
            for idx, coeff in enumerate(A_coeffs):
                if coeff != 0:
                    U[A_rows[idx]][col] += coeff
            
            # B_coeffs for (b00,b01,b10,b11) = (B[1][k1], B[1][k2], B[2][k1], B[2][k2])
            # b00 -> V row 1*8+k1 = 8+k1
            # b01 -> V row 1*8+k2 = 8+k2
            # b10 -> V row 2*8+k1 = 16+k1
            # b11 -> V row 2*8+k2 = 16+k2
            B_rows = [8+k1, 8+k2, 16+k1, 16+k2]
            for idx, coeff in enumerate(B_coeffs):
                if coeff != 0:
                    V[B_rows[idx]][col] += coeff
            
            # C_coeffs for (c00,c01,c10,c11) = (C[0][k1], C[0][k2], C[1][k1], C[1][k2])
            # c00 -> W row k1*2+0 = 2*k1
            # c01 -> W row k2*2+0 = 2*k2
            # c10 -> W row k1*2+1 = 2*k1+1
            # c11 -> W row k2*2+1 = 2*k2+1
            C_rows = [2*k1, 2*k2, 2*k1+1, 2*k2+1]
            for idx, coeff in enumerate(C_coeffs):
                if coeff != 0:
                    W[C_rows[idx]][col] += coeff
            
            col += 1
    
    # Verify
    if not verify(U, V, W, rank):
        # Fallback to schoolbook
        U, V, W = schoolbook()
        rank = 48
    else:
        rank = 29
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
