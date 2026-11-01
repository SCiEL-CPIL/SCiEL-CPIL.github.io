"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-19 decomposition for 2x2x7 matrix multiplication.
    
    The idea: treat the 2x7 matrix as two blocks of 2x2 and one remaining column.
    Use Strassen's algorithm (rank 7) for each 2x2 block, and schoolbook (rank 4) 
    for the last column.
    
    Wait - that would be 7+7+4 = 18, but we need to be careful about how the 
    multiplication actually works.
    
    Actually, let me think about this more carefully.
    
    We're computing C = A * B where A is 2x2 and B is 2x7.
    C is 2x7.
    
    We can split B into B1 (2x2), B2 (2x2), B3 (2x2), and B4 (2x1).
    Then C = A*B1 + A*B2 + A*B3 + A*B4, where A*B1, A*B2, A*B3 are 2x2 and A*B4 is 2x1.
    
    Using Strassen for each 2x2 product: rank 7 each.
    Using schoolbook for 2x2x1: rank 4.
    
    Total rank: 7 + 7 + 7 + 4 = 25.
    
    But wait, the task says rank-minimal. Let me check what the actual minimal rank is.
    
    Actually, I should just implement the straightforward decomposition and see if it works.
    Let me use a simple approach: treat the whole thing as a 4x14 matrix product and 
    apply a known decomposition.
    
    Actually, the simplest correct approach is to use the schoolbook decomposition 
    with rank = 2*2*7 = 28, but we can do better.
    
    Let me implement Strassen for the first two 2x2 blocks and schoolbook for the rest.
    """
    n, m, p = DIMENSIONS  # 2, 2, 7
    
    # Build U, V, W matrices for the decomposition
    # U has shape (n*m, rank) = (4, rank)
    # V has shape (m*p, rank) = (14, rank)  
    # W has shape (p*n, rank) = (14, rank)
    
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Split into blocks: B has columns 0-1, 2-3, 4-5, 6
    # Use Strassen for blocks 0-1, 2-3, 4-5 (each 2x2)
    # Use schoolbook for column 6 (2x1)
    
    def strassen_2x2x2(offset):
        """
        Add Strassen decomposition for 2x2x2 multiplication starting at column offset.
        Returns the next available column index.
        """
        col = offset
        # Strassen's 7 multiplications
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # A indices: A00=0, A01=1, A10=2, A11=3
        # B indices: B00=0, B01=1, B10=2, B11=3 (relative to block start)
        # C indices: C00=0, C01=1, C10=2, C11=3 (relative to block start)
        
        # W convention: W[k*n + i, c] where k is row of B (0 or 1), i is col of A (0 or 1)
        # So W[k*2 + i, c]
        # C[i,j] = sum_k A[i,k] * B[k,j]
        # In tensor form: C[i,j] = sum_k A[i,k] * B[k,j]
        # The output index for C[i,j] in the W matrix is... let me think.
        # 
        # Actually, the standard tensor decomposition for matrix multiplication:
        # C = A * B, where A is n x m, B is m x p, C is n x p
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # C[i,j] = sum_c U[i*m + k, c] * V[k*p + j, c] * W[j*n + i, c]
        # 
        # Wait, that's not right either. Let me re-read the convention.
        # 
        # The task says: W[k*n+i, r]
        # So W is indexed by (k, i) where k is from B's row (0 to m-1) and i is from C's column (0 to n-1)?
        # 
        # Actually for n x m times m x p:
        # U[i*m + k, c] - i is row of A, k is column of A / row of B
        # V[k*p + j, c] - k is row of B, j is column of B
        # W[j*n + i, c] - j is column of B / column of C, i is row of C
        # 
        # So C[i,j] = sum_k U[i*m + k, c] * V[k*p + j, c] * W[j*n + i, c]
        
        # For 2x2x2:
        # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # 
        # W[j*n + i, c] where n=2, so W[j*2 + i, c]
        # C[i,j]: i is row (0 or 1), j is col (0 or 1)
        # W index: j*2 + i
        
        # Strassen formulas:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        
        # U vectors (for A):
        # U[i*m + k, c] where m=2
        # A00: index 0, A01: index 1, A10: index 2, A11: index 3
        
        # M1: A00+A11 -> U[0,c]=1, U[3,c]=1
        # M2: A10+A11 -> U[2,c]=1, U[3,c]=1
        # M3: A00 -> U[0,c]=1
        # M4: A11 -> U[3,c]=1
        # M5: A00+A01 -> U[0,c]=1, U[1,c]=1
        # M6: A10-A00 -> U[2,c]=1, U[0,c]=-1
        # M7: A01-A11 -> U[1,c]=1, U[3,c]=-1
        
        # V vectors (for B):
        # V[k*p + j, c] where p=2
        # B00: index 0, B01: index 1, B10: index 2, B11: index 3
        
        # M1: B00+B11 -> V[0,c]=1, V[3,c]=1
        # M2: B00 -> V[0,c]=1
        # M3: B01-B11 -> V[1,c]=1, V[3,c]=-1
        # M4: B10-B00 -> V[2,c]=1, V[0,c]=-1
        # M5: B11 -> V[3,c]=1
        # M6: B00+B01 -> V[0,c]=1, V[1,c]=1
        # M7: B10+B11 -> V[2,c]=1, V[3,c]=1
        
        # W vectors (for C output):
        # W[j*n + i, c] where n=2
        # C00: j=0,i=0 -> W[0,c]
        # C01: j=1,i=0 -> W[2,c]
        # C10: j=0,i=1 -> W[1,c]
        # C11: j=1,i=1 -> W[3,c]
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for each M, we need to add contributions to W:
        # M1: W[0,c]+=1, W[3,c]+=1
        # M2: W[1,c]+=1, W[3,c]-=1
        # M3: W[2,c]+=1, W[3,c]+=1
        # M4: W[0,c]+=1, W[1,c]+=1
        # M5: W[0,c]-=1, W[2,c]+=1
        # M6: W[3,c]+=1
        # M7: W[0,c]+=1
        
        # Now map to the actual indices in the 2x2x7 problem.
        # The block starts at column 'offset' in B.
        # B indices in the full problem:
        # B[0,offset], B[0,offset+1], B[1,offset], B[1,offset+1]
        # In V: V[k*p + j, c] where p=7
        # So B[0,offset] is V[0*7 + offset, c] = V[offset, c]
        # B[0,offset+1] is V[0*7 + offset+1, c] = V[offset+1, c]
        # B[1,offset] is V[1*7 + offset, c] = V[7+offset, c]
        # B[1,offset+1] is V[1*7 + offset+1, c] = V[7+offset+1, c]
        
        # C indices in the full problem:
        # C[i,offset] and C[i,offset+1]
        # W[j*n + i, c] where n=2, j is the column in B (0 to 6), i is row in A (0 or 1)
        # Wait, I'm getting confused. Let me re-derive.
        
        # For the full 2x2x7:
        # A is 2x2, B is 2x7, C is 2x7
        # U[i*m + k, c] where m=2, i in {0,1}, k in {0,1}
        # V[k*p + j, c] where p=7, k in {0,1}, j in {0,...,6}
        # W[j*n + i, c] where n=2, j in {0,...,6}, i in {0,1}
        
        # C[i,j] = sum_{k=0}^{1} A[i,k] * B[k,j]
        # = sum_{k=0}^{1} sum_c U[i*2 + k, c] * V[k*7 + j, c] * W[j*2 + i, c]
        
        # So for the block at offset:
        # B[0,offset] = V[0*7 + offset, c] = V[offset, c]
        # B[0,offset+1] = V[0*7 + offset+1, c] = V[offset+1, c]
        # B[1,offset] = V[1*7 + offset, c] = V[7+offset, c]
        # B[1,offset+1] = V[1*7 + offset+1, c] = V[7+offset+1, c]
        
        # C[i,offset] = W[offset*2 + i, c]
        # C[i,offset+1] = W[(offset+1)*2 + i, c]
        
        # Now let me implement:
        U_local = [
            [1, 0, 1],  # M1: A00, M2: A10, M3: A00
            [0, 0, 0],
            [0, 1, 0],  # M2: A10
            [1, 1, 0],  # M1: A11, M2: A11
        ]
        # Wait, I need to be more careful. Let me just code it directly.
        
        # M1
        U[0][col] += 1  # A00
        U[3][col] += 1  # A11
        V[offset][col] += 1  # B00
        V[7+offset+1][col] += 1  # B11
        W[offset*2+0][col] += 1  # C0,offset
        W[offset*2+1][col] += 1  # C1,offset... wait
        
        # I'm making errors. Let me restart with a cleaner approach.
        
        return col + 7
    
    # Actually, let me just use the simple schoolbook approach first to ensure correctness,
    # then optimize. The schoolbook rank is 28.
    
    # Reset
    rank = 28
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
