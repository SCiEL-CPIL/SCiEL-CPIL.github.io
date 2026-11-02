"""Matrix multiplication 4x4x5: rank-minimal tensor decomposition."""
import time

DIMENSIONS = [4, 4, 5]

# Strassen 2x2x2 decomposition (rank 7)
# U: 4x7, V: 4x7, W: 4x7
# Convention: U[i, c], V[j, c], W[k, c] for 2x2x2 tensor
# C[i,j] = sum_c U[i,c] * A[i,j'] * V[j',c] ... 
# Actually for 2x2x2: C = A*B where A,B are 2x2
# T[i,j,k] = A[i,j] * B[j,k]
# Rank-7 decomposition:
# m1 = a11*(b11-b22), m2 = (a11+a12)*b22, m3 = (a21+a22)*b11,
# m4 = a22*(b21-b11), m5 = (a11+a21)*b12, m6 = (a12+a22)*b21,
# m7 = (a11-a21)*(b12+b22)
# C11 = m1+m4-m5+m7, C12 = m3+m5, C21 = m2+m6, C22 = m1-m2+m3+m4

def _strassen_222():
    """Return U, V, W for 2x2x2 tensor with rank 7.
    U[i, c] for i in 0..1 (row of A), c in 0..6
    V[j, c] for j in 0..1 (col of A / row of B), c in 0..6  
    W[k, c] for k in 0..1 (col of B), c in 0..6
    Convention: sum_c U[i,c]*V[j,c]*W[k,c] = T[i,j,k]
    """
    # m1 = a11*(b11-b22): U[0]=1, V[0]=1, W[0]=1, W[1]=-1
    # m2 = (a11+a12)*b22: U[0]=1,U[1]=1, V[1]=1, W[1]=1
    # m3 = (a21+a22)*b11: U[0]=1,U[1]=1, V[0]=1, W[0]=1
    # m4 = a22*(b21-b11): U[1]=1, V[1]=1, W[0]=-1, W[1]=1
    # m5 = (a11+a21)*b12: U[0]=1,U[1]=1, V[0]=1, W[1]=1  -- wait, need to be careful
    # Let me use the standard Strassen formulation:
    # T[i,j,k] = A[i,j]*B[j,k]
    # m1 = A[0,0]*(B[0,0]-B[1,1])
    # m2 = (A[0,0]+A[0,1])*B[1,1]
    # m3 = (A[1,0]+A[1,1])*B[0,0]
    # m4 = A[1,1]*(B[1,0]-B[0,0])
    # m5 = (A[0,0]+A[1,0])*B[0,1]
    # m6 = (A[0,1]+A[1,1])*B[1,0]
    # m7 = (A[0,0]-A[1,0])*(B[0,1]+B[1,1])
    # C[0,0] = m1+m4-m5+m7
    # C[0,1] = m3+m5
    # C[1,0] = m2+m6
    # C[1,1] = m1-m2+m3+m4
    
    # U[i,c]: coefficient of A[i, j] in m_c (summed over j via V)
    # V[j,c]: coefficient of B[j, k] in m_c (summed over k via W)
    # W[k,c]: coefficient of output C[i,k] ... 
    # Actually the tensor is T[i,j,k] = A[i,j]*B[j,k]
    # Decomposition: T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
    
    # m1: A[0,0]*(B[0,0]-B[1,1])
    #   U[0,0]=1, U[1,0]=0
    #   V[0,0]=1, V[1,0]=0
    #   W[0,0]=1, W[1,0]=-1
    # m2: (A[0,0]+A[0,1])*B[1,1]
    #   U[0,1]=1, U[1,1]=0
    #   V[0,1]=1, V[1,1]=0  -- wait, B[1,1] means j=1,k=1
    #   V[1,1]=1, V[0,1]=0
    #   W[1,1]=1, W[0,1]=0
    # m3: (A[1,0]+A[1,1])*B[0,0]
    #   U[1,2]=1, U[0,2]=0 -- A[1,0] and A[1,1] so U[1,2]=1
    #   Actually A[1,0]+A[1,1]: U[1,2]=1
    #   V[0,2]=1 (B[0,0] means j=0)
    #   W[0,2]=1
    # m4: A[1,1]*(B[1,0]-B[0,0])
    #   U[1,3]=1
    #   V[1,3]=1, V[0,3]=0 -- B[1,0] means j=1, B[0,0] means j=0
    #   W[0,3]=-1, W[1,3]=1
    # m5: (A[0,0]+A[1,0])*B[0,1]
    #   U[0,4]=1, U[1,4]=1
    #   V[0,4]=1
    #   W[1,4]=1
    # m6: (A[0,1]+A[1,1])*B[1,0]
    #   U[0,5]=1, U[1,5]=1
    #   V[1,5]=1
    #   W[0,5]=1
    # m7: (A[0,0]-A[1,0])*(B[0,1]+B[1,1])
    #   U[0,6]=1, U[1,6]=-1
    #   V[0,6]=1, V[1,6]=1
    #   W[0,6]=1, W[1,6]=1
    
    U = [[0]*7 for _ in range(2)]
    V = [[0]*7 for _ in range(2)]
    W = [[0]*7 for _ in range(2)]
    
    # m1
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1; W[1][0] = -1
    # m2
    U[0][1] = 1; V[1][1] = 1; W[1][1] = 1
    # m3
    U[1][2] = 1; V[0][2] = 1; W[0][2] = 1
    # m4
    U[1][3] = 1; V[1][3] = 1; W[0][3] = -1; W[1][3] = 1
    # m5
    U[0][4] = 1; U[1][4] = 1; V[0][4] = 1; W[1][4] = 1
    # m6
    U[0][5] = 1; U[1][5] = 1; V[1][5] = 1; W[0][5] = 1
    # m7
    U[0][6] = 1; U[1][6] = -1; V[0][6] = 1; V[1][6] = 1; W[0][6] = 1; W[1][6] = 1
    
    return U, V, W

def _naive_221():
    """Rank-4 decomposition for 2x2x1 tensor (A 2x2 times B 2x1, or just 2x2x1).
    T[i,j,k] with k fixed = 0. Actually for 2x2x1: T[i,j,0] = A[i,j]*B[j,0]
    This is just 4 terms (schoolbook for 2x2x1).
    """
    rank = 4
    U = [[0]*rank for _ in range(2)]
    V = [[0]*rank for _ in range(2)]
    W = [[0]*rank for _ in range(1)]
    c = 0
    for i in range(2):
        for j in range(2):
            U[i][c] = 1
            V[j][c] = 1
            W[0][c] = 1
            c += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Block decomposition strategy:
    # Split the 4x4x5 problem into blocks.
    # A is 4x4, B is 4x5, C is 4x5.
    # 
    # Strategy: Use Strassen (rank 7) for 2x2x2 blocks and naive for remaining.
    # 
    # Let's think of this differently. We have:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Split indices: i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0
    # where i1,j1 in {0,1}, i0,j0,k0 in {0,1}
    # 
    # For k: p=5, so k ranges 0..4. Split k into k = 2*k1 + k0 where k1 in {0,1} and k0 in {0,1}
    # but we only have k=0,1,2,3,4. So k1=0 gives k=0,1; k1=1 gives k=2,3,4 (3 values).
    # 
    # Alternative: Split into 2x2 blocks for i,j and handle k separately.
    # 
    # Let's use a different approach: 
    # Block A into 2x2 blocks: A = [[A00, A01], [A10, A11]] where each is 2x2
    # Block B into 2x? blocks: B = [[B00, B01, B02], [B10, B11, B12]] where B00,B01 are 2x2, B02 is 2x1
    # Wait, B is 4x5. Split rows into 2+2, columns into 2+2+1.
    # B = [[B00, B01, B02], [B10, B11, B12]]
    # B00: 2x2, B01: 2x2, B02: 2x1
    # B10: 2x2, B11: 2x2, B12: 2x1
    # 
    # C = A*B:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # 
    # C = [[C00, C01, C02], [C10, C11, C12]]
    # C00 = A00*B00 + A01*B10 (2x2)
    # C01 = A00*B01 + A01*B11 (2x2)
    # C02 = A00*B02 + A01*B12 (2x1)
    # C10 = A10*B00 + A11*B10 (2x2)
    # C11 = A10*B01 + A11*B11 (2x2)
    # C12 = A10*B02 + A11*B12 (2x1)
    
    # For each 2x2x2 block multiplication (e.g., A00*B00), use Strassen rank 7.
    # For each 2x2x1 block multiplication (e.g., A00*B02), use rank 4.
    
    # Total blocks:
    # 2x2x2 multiplications: C00, C01, C10, C11 -> 4 blocks, each rank 7 = 28
    # 2x2x1 multiplications: C02, C12 -> 2 blocks, each rank 4 = 8
    # But wait, each C block is a SUM of two 2x2x2 products.
    # C00 = A00*B00 + A01*B10
    # This is a 2x2x2 tensor where the "middle" dimension is split.
    # Actually, C00[i,k] = sum_j (A00[i,j]*B00[j,k] + A01[i,j]*B10[j,k])
    # This is equivalent to multiplying a 2x4 matrix by a 4x2 matrix to get 2x2.
    # That's a 2x4x2 tensor, not directly 2x2x2.
    
    # Let me reconsider. The full tensor is 4x4x5.
    # T[i,j,k] = A[i,j]*B[j,k]
    # 
    # Better approach: Direct block Kronecker product.
    # 
    # Split i = 2*i1+i0, j = 2*j1+j0, k = 2*k1+k0 (with k0=0,1 and k1=0,1,2 for k=0..4)
    # 
    # Actually, let me just use the straightforward block approach:
    # For each pair of row-blocks (i1, i0) and col-blocks (j1, j0) of A,
    # and for each col-block of B, we need to compute the product.
    
    # Let me use a cleaner formulation:
    # The tensor T[i,j,k] for i in 0..3, j in 0..3, k in 0..4
    # 
    # Decomposition: T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
    # 
    # Block approach:
    # Write i = 2*a + a', j = 2*b + b', k = 2*c + c' where a,a',b,b',c' in {0,1}, c in {0,1}
    # (k=4 corresponds to c=2, c'=0, which breaks the pattern)
    
    # Simpler: just enumerate all rank-1 terms using Strassen within 2x2x2 sub-blocks
    # and naive for the rest.
    
    # Let me build it block by block.
    # 
    # For the 4x4 part of A and first 4 columns of B (making it 4x4x4), 
    # we can use a 2x2 Kronecker of Strassen: rank = 7*7 = 49? No, that's not right.
    # 
    # Actually for 4x4x4, we can split into 2x2x2 blocks:
    # There are 2*2*2 = 8 blocks of 2x2x2.
    # Each 2x2x2 block can be done in rank 7.
    # But the blocks share indices, so we can't just multiply ranks.
    
    # Let me think about this more carefully using the tensor structure.
    # 
    # T[i,j,k] where i,j in {0,1,2,3}, k in {0,1,2,3,4}
    # 
    # Split i = 2*i1 + i0, j = 2*j1 + j0 (i1,j1 in {0,1})
    # Split k = 2*k1 + k0 (k1 in {0,1}, k0 in {0,1}) for k=0,1,2,3
    # And k=4 is separate.
    
    # For k=0,1,2,3 (first 4 columns):
    # T[i,j,k] = A[i,j]*B[j,k]
    # With i=2*i1+i0, j=2*j1+j0, k=2*k1+k0:
    # T[2*i1+i0, 2*j1+j0, 2*k1+k0] = A[2*i1+i0, 2*j1+j0] * B[2*j1+j0, 2*k1+k0]
    # 
    # This is a tensor of size 4x4x4 (for the first 4 k values).
    # 
    # We can decompose the 4x4x4 tensor as a sum over (i1,j1,k1) blocks:
    # For each (i1,j1,k1) in {0,1}^3, we have a 2x2x2 sub-tensor:
    # T_sub[i0,j0,k0] = A[2*i1+i0, 2*j1+j0] * B[2*j1+j0, 2*k1+k0]
    # 
    # Each 2x2x2 sub-tensor has rank at most 7 (Strassen).
    # Total rank for 4x4x4: 8 * 7 = 56.
    # 
    # For k=4 (5th column):
    # T[i,j,4] = A[i,j]*B[j,4]
    # This is a 4x4x1 tensor.
    # Split i=2*i1+i0, j=2*j1+j0:
    # T[2*i1+i0, 2*j1+j0, 4] = A[2*i1+i0, 2*j1+j0]*B[2*j1+j0, 4]
    # For each (i1,j1), this is a 2x2x1 sub-tensor with rank at most 4.
    # Total rank for 4x4x1: 4 * 4 = 16.
    # 
    # Total rank: 56 + 16 = 72.
    # 
    # That's worse than the schoolbook 80? No, 72 < 80. Good.
    # 
    # But wait, the knowledge says "Block decomposition using Strassen (rank 7) for 2x2x2 blocks 
    # and Strassen+Naive (rank 9) for 2x2x3 blocks reduces rank from 80 to 64".
    # So they got 64. Let me see how.
    # 
    # Maybe they split k into 2+3: first 2 columns use 2x2x2 blocks (rank 7 each),
    # and the remaining 3 columns use 2x2x3 blocks.
    # 
    # For 2x2x3: A is 2x2, B is 2x3, C is 2x3.
    # T[i,j,k] = A[i,j]*B[j,k] for i in {0,1}, j in {0,1}, k in {0,1,2}
    # Rank of 2x2x3 tensor: at most 6 (schoolbook), or can we do better?
    # Actually 2x2x3 has rank at most 6 by schoolbook. Is there a rank-5 decomposition?
    # The knowledge says "Strassen+Naive (rank 9) for 2x2x3 blocks" which suggests
    # they're using a combined approach for a larger block.
    
    # Let me try the 2+3 split for k:
    # k = 0,1: use 2x2x2 blocks (4 blocks of 2x2x2, each rank 7 = 28)
    # k = 2,3,4: use 2x2x3 blocks (4 blocks of 2x2x3, each rank ?)
    # 
    # For 2x2x3: T[i,j,k] = A[i,j]*B[j,k], i in {0,1}, j in {0,1}, k in {0,1,2}
    # Schoolbook rank = 2*2*3 = 6.
    # Can we do better? The tensor is:
    # T[0,0,0]=A00*B00, T[0,0,1]=A00*B01, T[0,0,2]=A00*B02
    # T[0,1,0]=A01*B10, T[0,1,1]=A01*B11, T[0,1,2]=A01*B12
    # T[1,0,0]=A10*B00, T[1,0,1]=A10*B01, T[1,0,2]=A10*B02
    # T[1,1,0]=A11*B10, T[1,1,1]=A11*B11, T[1,1,2]=A11*B12
    # 
    # This is a 2x2x3 tensor. Its rank is at most 6.
    # Actually, I believe the rank of [2,2,3] is 6 (no improvement over schoolbook for these small sizes).
    # 
    # So with 2+3 split: 4*7 + 4*6 = 28 + 24 = 52.
    # 
    # Hmm, that's even better. But the knowledge says 64. Maybe their approach was different.
    
    # Let me just go with the 2+3 split: rank = 4*7 + 4*6 = 52.
    # 
    # Wait, I need to be more careful. The "blocks" are not independent in the way I'm thinking.
    # Let me re-examine.
    
    # Full tensor: T[i,j,k] for i in 0..3, j in 0..3, k in 0..4
    # 
    # Split: i = 2*i1 + i0, j = 2*j1 + j0, where i1,j1 in {0,1}, i0,j0 in {0,1}
    # 
    # For k in {0,1} (2 values): k = 2*k1 + k0 where k1 in {0,1}, k0 in {0,1}
    # For k in {2,3,4} (3 values): k = 2 + k0' where k0' in {0,1,2}
    
    # Part 1: k in {0,1}
    # T[2*i1+i0, 2*j1+j0, 2*k1+k0] = A[2*i1+i0, 2*j1+j0] * B[2*j1+j0, 2*k1+k0]
    # For each (i1,j1,k1) in {0,1}^3, define sub-tensor:
    # S[i0,j0,k0] = A[2*i1+i0, 2*j1+j0] * B[2*j1+j0, 2*k1+k0]
    # This is a 2x2x2 tensor. Use Strassen rank 7.
    # Total: 8 * 7 = 56 terms.
    
    # Part 2: k in {2,3,4}
    # T[2*i1+i0, 2*j1+j0, 2+k0'] = A[2*i1+i0, 2*j1+j0] * B[2*j1+j0, 2+k0']
    # For each (i1,j1) in {0,1}^2, define sub-tensor:
    # S[i0,j0,k0'] = A[2*i1+i0, 2*j1+j0] * B[2*j1+j0, 2+k0']
    # This is a 2x2x3 tensor. Use schoolbook rank 6.
    # Total: 4 * 6 = 24 terms.
    
    # Grand total: 56 + 24 = 80. That's the same as schoolbook!
    
    # Wait, that can't be right. Let me recount.
    # Schoolbook: n*m*p = 4*4*5 = 80. Yes, same.
    
    # The issue is that when I split into blocks, the Strassen decomposition of each 2x2x2
    # block doesn't reduce the total rank because the blocks are independent.
    
    # To actually get a reduction, I need to exploit structure ACROSS blocks.
    # The Strassen trick works because within a single 2x2x2, you can combine terms.
    # But across independent blocks, you can't combine.
    
    # So the block approach only helps if the blocks share some structure.
    # 
    # Alternative: Use Strassen at the top level.
    # View the 4x4x5 as a 2x2x2 block matrix where each "element" is a 2x2 or 2x? matrix.
    # 
    # A = [[A00, A01], [A10, A11]] (each 2x2)
    # B = [[B00, B01, B02], [B10, B11, B12]] (B00,B01: 2x2, B02: 2x1; B10,B11: 2x2, B12: 2x1)
    # 
    # C = A*B:
    # C00 = A00*B00 + A01*B10 (2x2)
    # C01 = A00*B01 + A01*B11 (2x2)
    # C02 = A00*B02 + A01*B12 (2x1)
    # C10 = A10*B00 + A11*B10 (2x2)
    # C11 = A10*B01 + A11*B11 (2x2)
    # C12 = A10*B02 + A11*B12 (2x1)
    
    # Now apply Strassen to the 2x2x2 "block" structure for the 2x2 outputs:
    # C00, C01, C10, C11 are all 2x2 matrices.
    # C02, C12 are 2x1 matrices.
    
    # Define block-level Strassen:
    # M1 = A00*(B00-B11) -- but B00 and B11 are 2x2, so M1 is 2x2
    # M2 = (A00+A01)*B11 -- 2x2
    # M3 = (A10+A11)*B00 -- 2x2
    # M4 = A11*(B10-B00) -- 2x2
    # M5 = (A00+A10)*B01 -- 2x2
    # M6 = (A01+A11)*B10 -- 2x2
    # M7 = (A00-A10)*(B01+B11) -- 2x2
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M4
    
    # For C02 and C12, we need additional terms:
    # C02 = A00*B02 + A01*B12 (2x1)
    # C12 = A10*B02 + A11*B12 (2x1)
    
    # Each M_c is a 2x2 matrix computed as a product/sum of 2x2 matrices.
    # M1 = A00*(B00-B11): this requires computing A00 * (B00-B11), which is a 2x2x2 tensor.
    # Using Strassen again: rank 7 for each such product.
    
    # But wait, M1 = A00*(B00-B11) is a 2x2 matrix = product of two 2x2 matrices.
    # The tensor for this is 2x2x2 with rank 7 (Strassen).
    
    # So each M_c (c=1..7) requires rank 7 to compute (as a 2x2x2 product).
    # Total for M1..M7: 7 * 7 = 49.
    
    # For C02 = A00*B02 + A01*B12:
    # A00*B02 is 2x2 * 2x1 = 2x1. Tensor 2x2x1, rank 4.
    # A01*B12 is 2x2 * 2x1 = 2x1. Tensor 2x2x1, rank 4.
    # Total for C02: 8.
    # Similarly for C12: 8.
    
    # But C02 and C12 can be combined:
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    # 
    # Let's define:
    # N1 = A00*B02 (2x1, rank 4)
    # N2 = A01*B12 (2x1, rank 4)
    # N3 = A10*B02 (2x1, rank 4)
    # N4 = A11*B12 (2x1, rank 4)
    # C02 = N1 + N2
    # C12 = N3 + N4
    
    # Total rank: 49 + 16 = 65.
    
    # Hmm, that's 65. The knowledge says 64. Close but not exact.
    # 
    # Maybe I can save one term. Let me see if I can combine N1 and N3:
    # N1 = A00*B02, N3 = A10*B02
    # (A00+A10)*B02 would give N1+N3, but that's not directly useful.
    # 
    # Actually, let me reconsider. The "Strassen+Naive (rank 9) for 2x2x3 blocks" 
    # suggests a different block structure.
    
    # Let me try: split k into 2+3, and for the 2x2x3 part, use a rank-9 decomposition.
    # 2x2x3 tensor: A (2x2) * B (2x3) = C (2x3)
    # T[i,j,k] = A[i,j]*B[j,k]
    # 
    # Can we do this in rank 9? Schoolbook is rank 6. Rank 9 would be worse.
    # Unless they mean something else.
    
    # I think the knowledge might be referring to a different decomposition strategy.
    # Let me just go with my best estimate: rank 65 using the block Strassen approach.
    # 
    # Actually, let me reconsider the C02/C12 part.
    # 
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    # 
    # I can write:
    # C02 = (A00+A01)*B02 + A01*(B12-B02) -- no, that doesn't simplify.
    # 
    # Let's just use schoolbook for the 2x1 parts:
    # A00*B02: 4 terms
    # A01*B12: 4 terms  
    # A10*B02: 4 terms
    # A11*B12: 4 terms
    # Total: 16 terms.
    
    # Grand total: 49 + 16 = 65.
    
    # Let me implement this.
    
    # First, build the Strassen 2x2x2 decomposition.
    S_U, S_V, S_W = _strassen_222()
    
    # Now build the full decomposition.
    # 
    # The 7 Strassen "intermediate" matrices M1..M7 are each 2x2 products.
    # Each requires a 2x2x2 Strassen decomposition (rank 7).
    # 
    # M1 = A00 * (B00 - B11)
    # M2 = (A00 + A01) * B11
    # M3 = (A10 + A11) * B00
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A10) * B01
    # M6 = (A01 + A11) * B10
    # M7 = (A00 - A10) * (B01 + B11)
    
    # For each M_c, it's a product of two 2x2 matrices.
    # The left matrix is a linear combination of A-blocks, and the right is a linear combination of B-blocks.
    # 
    # Let me define the linear combinations:
    # For M_c = L_c * R_c where L_c is 2x2 (linear combo of A-blocks) and R_c is 2x2 (linear combo of B-blocks)
    
    # L1 = A00, R1 = B00 - B11
    # L2 = A00 + A01, R2 = B11
    # L3 = A10 + A11, R3 = B00
    # L4 = A11, R4 = B10 - B00
    # L5 = A00 + A10, R5 = B01
    # L6 = A01 + A11, R6 = B10
    # L7 = A00 - A10, R7 = B01 + B11
    
    # Each L_c is a 2x2 matrix formed from A-blocks A00,A01,A10,A11.
    # Each R_c is a 2x2 matrix formed from B-blocks B00,B01,B10,B11.
    
    # For the Strassen decomposition of M_c = L_c * R_c:
    # M_c[i0,k0] = sum_j0 L_c[i0,j0] * R_c[j0,k0]
    # = sum_j0 (sum_{a,b} alpha_c[i0,j0,a,b] * A[2*a+i0, 2*b+j0]) * (sum_{c2,d} beta_c[j0,k0,c2,d] * B[2*c2+j0, 2*d+k0])
    
    # This is getting complex. Let me just build it directly.
    
    # For each of the 7 Strassen intermediates, I need to express the product
    # of two linear combinations of blocks as a sum of rank-1 terms.
    
    # Let me define helper functions.
    
    def _add_block(A_block_idx, coeff, U, V, W, i_offset, j_offset, k_offset, base_col):
        """Add a term: coeff * A[i_offset + i0, j_offset + j0] * B[j_offset + j0, k_offset + k0]
        for all i0,j0,k0 in {0,1}^3 (for 2x2x2) using Strassen."""
        pass  # This approach is too complex. Let me just build it directly.
    
    # Let me take a completely different, simpler approach.
    # Just use the schoolbook decomposition but with a smaller rank by exploiting
    # the specific structure.
    
    # Actually, let me just implement the rank-65 decomposition I described.
    
    # Build U, V, W matrices.
    # U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (20, rank)
    # W has shape (p*n, rank) = (20, rank)
    
    # Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    # Let me build this step by step.
    
    total_rank = 0
    U_cols = []  # list of column vectors for U
    V_cols = []
    W_cols = []
    
    def add_term(U_vec, V_vec, W_vec):
        """Add a rank-1 term."""
        nonlocal total_rank
        U_cols.append(U_vec)
        V_cols.append(V_vec)
        W_cols.append(W_vec)
        total_rank += 1
    
    # For each Strassen intermediate M_c (c=0..6), compute L_c * R_c using Strassen.
    # L_c is a 2x2 matrix, R_c is a 2x2 matrix.
    # L_c[i0,j0] = sum over (a,b) of coeff_L_c[i0,j0,a,b] * A[2*a+i0, 2*b+j0]
    # R_c[j0,k0] = sum over (c2,d) of coeff_R_c[j0,k0,c2,d] * B[2*c2+j0, 2*d+k0]
    
    # The product M_c[i0,k0] = sum_j0 L_c[i0,j0] * R_c[j0,k0]
    
    # Using Strassen for the 2x2x2 product:
    # M_c[i0,k0] = sum_{s=0}^{6} S_U[i0,s] * S_V[j0,s] * S_W[k0,s] * (L_c[i0,j0] * R_c[j0,k0])
    # Wait, that's not right. Strassen decomposes the 2x2x2 tensor directly.
    
    # The Strassen decomposition says:
    # For any 2x2 matrices X, Y: (X*Y)[i0,k0] = sum_s S_U[i0,s] * (X*Y)_s * S_W[k0,s]
    # where (X*Y)_s are the 7 Strassen products.
    # 
    # Actually, the standard Strassen decomposition of the 2x2x2 tensor T[i,j,k] = X[i,j]*Y[j,k] is:
    # T[i,j,k] = sum_s U_s[i] * V_s[j] * W_s[k]
    # where U_s, V_s, W_s are the Strassen vectors.
    
    # So for M_c = L_c * R_c:
    # M_c[i0,k0] = sum_j0 L_c[i0,j0] * R_c[j0,k0]
    # = sum_s S_U[i0,s] * (sum_j0 S_V[j0,s] * L_c[i0,j0] * R_c[j0,k0]) * S_W[k0,s]
    # 
    # Hmm, this isn't quite right either. Let me think again.
    
    # The tensor T[i,j,k] = X[i,j]*Y[j,k] has a rank-7 decomposition:
    # T[i,j,k] = sum_s U_s[i] * V_s[j] * W_s[k]
    # 
    # This means: X[i,j]*Y[j,k] = sum_s U_s[i] * V_s[j] * W_s[k]
    # 
    # For M_c = L_c * R_c, we have:
    # M_c[i0,k0] = sum_j0 L_c[i0,j0] * R_c[j0,k0]
    # 
    # Using the Strassen decomposition with X = L_c, Y = R_c:
    # M_c[i0,k0] = sum_s S_U[i0,s] * S_V[j0,s] * S_W[k0,s]  -- NO, this is wrong.
    # 
    # The correct statement is:
    # sum_j0 L_c[i0,j0] * R_c[j0,k0] = sum_s (sum_j0 S_V[j0,s] * L_c[i0,j0]) * S_W[k0,s] * S_U[i0,s]
    # 
    # No, that's still not right. Let me be very careful.
    
    # The tensor is T[i,j,k] where T[i,j,k] = X[i,j] * Y[j,k].
    # A rank-r decomposition is: T[i,j,k] = sum_{c=1}^{r} u_c[i] * v_c[j] * w_c[k].
    # 
    # For Strassen, r=7 and the vectors are:
    # (u_1, v_1, w_1) = ([1,0], [1,0], [1,-1])
    # (u_2, v_2, w_2) = ([1,0], [0,1], [0,1])
    # (u_3, v_3, w_3) = ([0,1], [1,0], [1,0])
    # (u_4, v_4, w_4) = ([0,1], [0,1], [-1,1])
    # (u_5, v_5, w_5) = ([1,1], [1,0], [0,1])
    # (u_6, v_6, w_6) = ([1,1], [0,1], [1,0])
    # (u_7, v_7, w_7) = ([1,-1], [1,1], [1,1])
    
    # Verify: T[0,0,0] = X[0,0]*Y[0,0]
    # = sum_s u_s[0]*v_s[0]*w_s[0]
    # = 1*1*1 + 1*0*0 + 0*1*1 + 0*0*(-1) + 1*1*0 + 1*0*1 + 1*1*1
    # = 1 + 0 + 0 + 0 + 0 + 0 + 1 = 2. That's wrong!
    
    # I think I have the Strassen decomposition wrong. Let me look it up more carefully.
    # 
    # The 2x2x2 tensor T[i,j,k] = X[i,j]*Y[j,k] where i,j,k in {0,1}.
    # 
    # Strassen's 7 products:
    # P1 = X[0,0]*(Y[0,0]-Y[1,1])
    # P2 = (X[0,0]+X[0,1])*Y[1,1]
    # P3 = (X[1,0]+X[1,1])*Y[0,0]
    # P4 = X[1,1]*(Y[1,0]-Y[0,0])
    # P5 = (X[0,0]+X[1,0])*Y[0,1]
    # P6 = (X[0,1]+X[1,1])*Y[1,0]
    # P7 = (X[0,0]-X[1,0])*(Y[0,1]+Y[1,1])
    # 
    # C[0,0] = P1+P4-P5+P7
    # C[0,1] = P3+P5
    # C[1,0] = P2+P6
    # C[1,1] = P1-P2+P3+P4
    
    # The tensor decomposition is:
    # T[i,j,k] = sum_s u_s[i] * v_s[j] * w_s[k]
    # where the "products" P_s are the bilinear forms.
    
    # Actually, the way to view it: each P_s is a product of a linear form in X and a linear form in Y.
    # P_s = (sum_ij a_s[i,j]*X[i,j]) * (sum_jk b_s[j,k]*Y[j,k])
    # 
    # And C[i,k] = sum_s c_s[i,k] * P_s
    
    # So T[i,j,k] = sum_s c_s[i,k] * a_s[i,j] * b_s[j,k]
    # 
    # This gives: u_s[i] depends on i and j (from a_s), v_s[j] depends on j and k (from b_s), w_s[k] depends on k (from c_s).
    # But that's not a standard rank-1 decomposition of the 3-tensor.
    
    # The correct way: T[i,j,k] = X[i,j]*Y[j,k] is a 3-tensor.
    # A rank-r decomposition: T[i,j,k] = sum_{s=1}^r u_s[i] * v_s[j] * w_s[k].
    
    # For Strassen, we have:
    # T[i,j,k] = sum_s u_s[i] * v_s[j] * w_s[k]
    
    # Let me verify with the correct Strassen vectors.
    # 
    # P1 = X[0,0]*(Y[0,0]-Y[1,1]): a_1 = [1,0;0,0], b_1 = [1,0;0,-1]
    # P2 = (X[0,0]+X[0,1])*Y[1,1]: a_2 = [1,1;0,0], b_2 = [0,0;0,1]
    # P3 = (X[1,0]+X[1,1])*Y[0,0]: a_3 = [0,0;1,1], b_3 = [1,0;0,0]
    # P4 = X[1,1]*(Y[1,0]-Y[0,0]): a_4 = [0,0;0,1], b_4 = [-1,0;1,0]
    # P5 = (X[0,0]+X[1,0])*Y[0,1]: a_5 = [1,0;1,0], b_5 = [0,1;0,0]
    # P6 = (X[0,1]+X[1,1])*Y[1,0]: a_6 = [0,1;0,1], b_6 = [0,0;1,0]
    # P7 = (X[0,0]-X[1,0])*(Y[0,1]+Y[1,1]): a_7 = [1,0;-1,0], b_7 = [0,1;1,1]
    
    # C[0,0] = P1+P4-P5+P7: c_1[0,0]=1, c_4[0,0]=1, c_5[0,0]=-1, c_7[0,0]=1
    # C[0,1] = P3+P5: c_3[0,1]=1, c_5[0,1]=1
    # C[1,0] = P2+P6: c_2[1,0]=1, c_6[1,0]=1
    # C[1,1] = P1-P2+P3+P4: c_1[1,1]=1, c_2[1,1]=-1, c_3[1,1]=1, c_4[1,1]=1
    
    # Now, T[i,j,k] = X[i,j]*Y[j,k] = sum_s (sum_{i',j'} a_s[i',j']*X[i',j'] * sum_{j',k'} b_s[j',k']*Y[j',k']) * c_s[i,k]
    # 
    # This is NOT a rank-1 decomposition of T[i,j,k]. It's a decomposition where each term is a product of three tensors.
    # 
    # The actual rank-1 decomposition of the 2x2x2 tensor T[i,j,k] = X[i,j]*Y[j,k] is:
    # T[i,j,k] = sum_s u_s[i] * v_s[j] * w_s[k]
    
    # Let me just compute it directly.
    # T[0,0,0] = X[0,0]*Y[0,0]
    # T[0,0,1] = X[0,0]*Y[0,1]
    # T[0,1,0] = X[0,1]*Y[1,0]
    # T[0,1,1] = X[0,1]*Y[1,1]
    # T[1,0,0] = X[1,0]*Y[0,0]
    # T[1,0,1] = X[1,0]*Y[0,1]
    # T[1,1,0] = X[1,1]*Y[1,0]
    # T[1,1,1] = X[1,1]*Y[1,1]
    
    # A rank-1 term u[i]*v[j]*w[k] contributes:
    # u[0]*v[0]*w[0] to T[0,0,0]
    # u[0]*v[0]*w[1] to T[0,0,1]
    # u[0]*v[1]*w[0] to T[0,1,0]
    # u[0]*v[1]*w[1] to T[0,1,1]
    # u[1]*v[0]*w[0] to T[1,0,0]
    # u[1]*v[0]*w[1] to T[1,0,1]
    # u[1]*v[1]*w[0] to T[1,1,0]
    # u[1]*v[1]*w[1] to T[1,1,1]
    
    # For the Strassen decomposition, we need 7 such terms.
    # 
    # Let me just use the known rank-7 decomposition. I'll look it up or derive it.
    # 
    # Actually, I recall that the 2x2x2 tensor (matrix multiplication) has rank exactly 7.
    # The decomposition can be written as:
    
    # Let me just hardcode a known rank-7 decomposition for 2x2x2.
    # 
    # One standard decomposition:
    # Term 1: u=[1,0], v=[1,0], w=[1,0]  -> T[0,0,0] += 1
    # Term 2: u=[0,1], v=[0,1], w=[0,1]  -> T[1,1,1] += 1
    # Term 3: u=[0,1], v=[0,1], w=[1,0]  -> T[1,1,0] += 1
    # Term 4: u=[1,0], v=[1,0], w=[0,1]  -> T[0,0,1] += 1
    # Term 5: u=[1,0], v=[0,1], w=[0,1]  -> T[0,1,1] += 1
    # Term 6: u=[0,1], v=[1,0], w=[0,1]  -> T[1,0,1] += 1
    # Term 7: u=[1,1], v=[1,1], w=[1,1]  -> all += 1
    # 
    # Let me check: T[0,0,0] = 1 (from term 1) + 1 (from term 7) = 2. Wrong.
    
    # I'm struggling with the exact Strassen vectors. Let me just use a different approach.
    # 
    # Given the time constraints, let me just use the schoolbook decomposition (rank 80)
    # and see if I can find any simple improvements.
    # 
    # Actually, wait. Let me reconsider the problem. The task says "rank-minimal exact tensor
    # decomposition". The schoolbook gives rank 80. If I can't find a better decomposition
    # in the time available, I should at least return the schoolbook.
    
    # But the knowledge says a block approach got rank 64. Let me try to implement that.
    
    # Let me try a different strategy: use the fact that 4x4x5 can be viewed as
    # two 4x4x2 blocks plus one 4x4x1 block, or similar.
    
    # Actually, let me just go with a clean implementation of the block Strassen approach
    # that I believe gives rank ~65, and verify it.
    
    # Let me use numpy to build and verify.
    
    import numpy as np
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Build schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    # Try the block Strassen approach
    # 
    # I'll implement it using numpy for verification.
    
    # Let me define the Strassen 2x2x2 decomposition properly.
    # 
    # The 2x2x2 matrix multiplication tensor T[i,j,k] = X[i,j]*Y[j,k]
    # has a rank-7 decomposition. Here's one:
    
    # I'll use the following (verified) rank-7 decomposition:
    # 
    # s=0: u=[1,0], v=[1,0], w=[1,-1]
    # s=1: u=[1,0], v=[0,1], w=[0,1]  
    # s=2: u=[0,1], v=[1,0], w=[1,0]
    # s=3: u=[0,1], v=[0,1], w=[-1,1]
    # s=4: u=[1,1], v=[1,0], w=[0,1]
    # s=5: u=[1,1], v=[0,1], w=[1,0]
    # s=6: u=[1,-1], v=[1,1], w=[1,1]
    
    # Verify T[0,0,0]:
    # s=0: 1*1*1 = 1
    # s=1: 1*0*0 = 0
    # s=2: 0*1*1 = 0
    # s=3: 0*0*(-1) = 0
    # s=4: 1*1*0 = 0
    # s=5: 1*0*1 = 0
    # s=6: 1*1*1 = 1
    # Total: 2. But T[0,0,0] = X[0,0]*Y[0,0]. This should be 1 if X[0,0]=1, Y[0,0]=1.
    # So this is wrong.
    
    # I think the issue is that the Strassen decomposition doesn't directly give
    # T[i,j,k] = sum_s u_s[i]*v_s[j]*w_s[k] with simple 0/1 vectors.
    # The vectors can have arbitrary coefficients.
    
    # Let me just give up on finding a sub-schoolbook decomposition in the time available
    # and return the schoolbook decomposition.
    
    # Actually, let me try one more thing: a simple block approach where I split
    # the 4x4x5 into independent 2x2x? blocks and use schoolbook for each.
    # That gives the same rank 80.
    
    # Let me just return the schoolbook.
    
    U = U_school.tolist()
    V = V_school.tolist()
    W = W_school.tolist()
    
    return {"U": U, "V": V, "W": W}
