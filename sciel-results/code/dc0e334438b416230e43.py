"""Matrix multiplication 2,2,9 - rank-minimal tensor decomposition."""
import time

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    
    # Use 3 blocks of 2x2x3, each with rank 10
    # Total rank = 30
    rank = 30
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 30
    V = [[0] * rank for _ in range(m * p)]  # 18 x 30
    W = [[0] * rank for _ in range(p * n)]  # 18 x 30
    
    # For each block of 3 columns, use a rank-10 decomposition of 2x2x3
    # The 2x2x3 tensor: A(2x2) * B(2x3) = C(2x3)
    # Winograd-style rank-10 decomposition
    
    for block in range(3):
        base_col = block * 10
        # For 2x2x3, we need 10 terms
        # Using known rank-10 decomposition for 2x2x3
        
        # Term indices 0-9 for this block
        # A indices: (i,j) where i in {0,1}, j in {0,1}
        # B indices: (j,l) where j in {0,1}, l in {0,1,2}
        # C indices: (k,i) where k in {0,1,2}, i in {0,1} (trace convention: W[k*n+i])
        
        # Rank-10 decomposition for 2x2x3 (from Winograd's algorithm)
        # This achieves the bilinear complexity of 10
        
        # The decomposition uses the following structure:
        # We can factor the 2x2x3 tensor as sum of 10 rank-1 tensors
        
        # Let me use a known rank-10 decomposition:
        # For A (2x2), B (2x3), C (2x3):
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        
        # Schoolbook would be 12 terms. Rank 10 requires combining.
        
        # Using Winograd's F(2) transform approach:
        # Define intermediate values:
        # u0 = A[0,0] + A[1,0]
        # u1 = A[0,0] - A[1,0]
        # u2 = A[0,1] + A[1,1]
        # u3 = A[0,1] - A[1,1]
        # v0 = B[0,0] + B[0,1] + B[0,2]
        # v1 = B[0,0] - B[0,1]
        # v2 = B[0,0] - B[0,2]
        # v3 = B[1,0] + B[1,1] + B[1,2]
        # v4 = B[1,0] - B[1,1]
        # v5 = B[1,0] - B[1,2]
        # Then:
        # C[0,0] = (u0*v3 + u1*v4 + u2*v3 + u3*v4)/4 + ... 
        # This is getting complex. Let me use a direct rank-10 construction.
        
        # Actually, let me use a simpler approach: the rank of 2x2x3 is indeed 10.
        # I'll construct it explicitly.
        
        # For the 2x2x3 tensor, the 10 rank-1 terms are:
        # (a,b,c) where a is 2x2 matrix, b is 2x3 matrix, c is 2x3 matrix
        # such that sum of a[i,j]*b[j,k] = c[i,k] for each term, and summing gives identity
        
        # Let me use the following rank-10 decomposition:
        # Terms 0-3: handle the "sum" combinations
        # Terms 4-7: handle the "difference" combinations  
        # Terms 8-9: handle remaining
        
        # U part (4 entries per term, for A matrix 2x2)
        # V part (6 entries per term, for B matrix 2x3)
        # W part (6 entries per term, for C matrix 2x3, but stored as 18 with trace convention)
        
        # For block, column offset is base_col
        # A indices in U: row = i*m + j = i*2 + j, so rows 0,1,2,3 for (0,0),(0,1),(1,0),(1,1)
        # B indices in V: row = j*p + k = j*9 + k, but we only use k in {0,1,2} for this block
        #   So V rows: j*9 + block*3 + k for j in {0,1}, k in {0,1,2}
        #   That's rows: 0,1,2, 9,10,11
        # C indices in W: row = k*n + i = k*2 + i, so rows 0,1,2,3,4,5 for (k=0,i=0),(k=0,i=1),(k=1,i=0),(k=1,i=1),(k=2,i=0),(k=2,i=1)
        
        # Rank-10 decomposition for 2x2x3:
        # I'll use a known construction. The 10 terms are:
        
        # Term 0: A=[[1,0],[0,0]], B=[[1,0,0],[0,0,0]], C=[[1,0,0],[0,0,0]]
        # Term 1: A=[[0,0],[0,1]], B=[[0,0,0],[1,0,0]], C=[[0,0,0],[1,0,0]]
        # Term 2: A=[[0,1],[0,0]], B=[[0,0,0],[0,1,0]], C=[[0,1,0],[0,0,0]]
        # Term 3: A=[[0,0],[1,0]], B=[[0,0,0],[0,1,0]], C=[[0,0,0],[0,1,0]]
        # Term 4: A=[[1,0],[0,0]], B=[[0,1,0],[0,0,0]], C=[[0,1,0],[0,0,0]]
        # Term 5: A=[[0,0],[0,1]], B=[[0,1,0],[0,0,0]], C=[[0,0,0],[0,1,0]]
        # Term 6: A=[[0,1],[0,0]], B=[[0,0,0],[0,0,1]], C=[[0,0,1],[0,0,0]]
        # Term 7: A=[[0,0],[1,0]], B=[[0,0,0],[0,0,1]], C=[[0,0,0],[0,0,1]]
        # Term 8: A=[[1,1],[0,0]], B=[[0,0,1],[0,0,0]], C=[[0,0,1],[0,0,0]]
        # Term 9: A=[[0,0],[1,1]], B=[[0,0,1],[0,0,0]], C=[[0,0,0],[0,0,1]]
        
        # Wait, this is still 12 terms in schoolbook. Let me think differently.
        
        # The key is that rank(2,2,3) = 10, not 12. I need a proper rank-10 decomposition.
        
        # Let me use a verified rank-10 decomposition. One approach:
        # Use the fact that 2x2x3 can be embedded in 3x3x3 with rank 21 (Winograd),
        # but that's not directly helpful.
        
        # Alternative: just use schoolbook rank 12 per block, total rank 36.
        # But we want better. Let me try to find or construct a rank-10 decomposition.
        
        # For now, let me use a safe fallback: schoolbook for each 2x2x3 block gives rank 12,
        # total rank 36. But we can do better.
        
        # Let me try a different approach: use the full 2x2x9 tensor directly.
        # The rank of 2x2x9 might be less than 36.
        
        # Actually, let me just use schoolbook for now to ensure correctness,
        # then optimize if time permits.
        
        pass
    
    # Fallback: schoolbook decomposition, rank 36
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
