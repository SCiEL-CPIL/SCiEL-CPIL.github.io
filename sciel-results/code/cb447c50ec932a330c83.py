"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]
import time
import numpy as np
from itertools import product
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # The multiplication tensor T[i,j,k] = delta for (i in [0,n), j in [0,m), k in [0,p))
    # We need U (n*m x rank), V (m*p x rank), W (p*n x rank) such that:
    # For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c] ... 
    # Actually the convention is: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor: T[i,j,k] = 1 if this term contributes, else 0
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Schoolbook: rank = n*m*p = 60
    # Strassen for 2x2x2: rank 7 instead of 8
    # For 2x3x10, we can tile into 2x2x2 blocks with some 2x2x1 or 1x2x2 blocks
    
    # Strategy: Use Strassen-based construction
    # Break the 3x10 part into blocks and use Strassen where possible
    
    # Let's build a Strassen-based decomposition:
    # The 2x3x10 multiplication can be seen as:
    # A (2x3) * B (3x10) = C (2x10)
    # 
    # We can partition B into 5 blocks of 3x2, and use Strassen-like 
    # 2x2x2 decompositions for each pair of columns.
    
    # Actually, let's think more carefully.
    # A is 2x3, B is 3x10.
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    # For each pair of output columns (k, k+1), we compute:
    # C[i, k]   = A[i,0]*B[0,k]   + A[i,1]*B[1,k]   + A[i,2]*B[2,k]
    # C[i, k+1] = A[i,0]*B[0,k+1] + A[i,1]*B[1,k+1] + A[i,2]*B[2,k+1]
    
    # This is a 2x3x2 problem for each column pair.
    # We can use a Strassen-like approach for the 2x2 part of A.
    
    # Let's define Strassen for 2x2x2:
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
    
    # For our 2x3x2 problem, A has 3 columns not 2.
    # Let me partition A's columns: A = [A_left | A_right] where A_left is 2x2, A_right is 2x1
    # And B's rows: B = [B_top; B_bottom] where B_top is 2x2, B_bottom is 1x2
    
    # C = A_left * B_top + A_right * B_bottom
    
    # A_left * B_top is a 2x2x2 problem -> Strassen rank 7
    # A_right * B_bottom is a 2x1x2 problem -> rank 2 (schoolbook)
    # Total per column pair: 7 + 2 = 9
    
    # For 5 column pairs: 5 * 9 = 45? That seems too good. Let me verify.
    
    # Wait, I need to be more careful. Let me reconsider.
    
    # Actually, let me just implement a direct search for small rank reductions.
    
    # First, let's establish the schoolbook rank 60 as baseline.
    rank_schoolbook = n * m * p
    U_sb = [[0]*rank_schoolbook for _ in range(n*m)]
    V_sb = [[0]*rank_schoolbook for _ in range(m*p)]
    W_sb = [[0]*rank_schoolbook for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    def verify(U, V, W):
        """Verify that U, V, W form a valid decomposition."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        # Check: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c] should equal 
        # the tensor value, and W[k*n+i,c] provides the output coefficient
        # Actually the condition is:
        # The multiplication tensor is T[i,j,k] = 1 (all entries are 1 in the 
        # sense that every (i,j,k) triple contributes)
        # We need: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # Wait no. The tensor decomposition means:
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # where T[i,j,k] = 1 for all valid i,j,k (since it's the multiplication tensor)
        
        # Actually, I think the convention is different. Let me re-read.
        # "exactly decompose the matrix multiplication tensor"
        # The multiplication tensor for (n,m,p) has entries T[i,j,k] = 1 if the 
        # product term A[i,j]*B[j,k] contributes to C[i,k], which is always true.
        # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        
        # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Let me build a Strassen-based construction.
    # Partition: A (2x3) into A1 (2x2) + A2 (2x1)
    # B (3x10) into B1 (2x10) + B2 (1x10)
    # C = A1*B1 + A2*B2
    # A1*B1 is 2x2x10, A2*B2 is 2x1x10
    
    # For 2x2x10: partition B1 into 5 blocks of 2x2, use Strassen on each: 5*7 = 35
    # For 2x1x10: schoolbook rank = 2*1*10 = 20
    # Total: 35 + 20 = 55
    
    # That gives rank 55! Let me implement this.
    
    def build_strassen_2x2x2_block(A_rows, B_cols, offset_u, offset_v, offset_w, U, V, W):
        """
        Add Strassen 2x2x2 block to U, V, W.
        A_rows: list of 2 row indices in A (for U indexing)
        B_cols: list of 2 column indices in B (for V indexing)
        offset_u, offset_v, offset_w: column offsets in U, V, W
        """
        # A is 2x2, B is 2x2, C is 2x2
        # U indices: A[0,0], A[0,1], A[1,0], A[1,1] -> rows 0,1,2,3 in the 2x2 block
        # But in our global indexing, A[i,j] maps to U[i*m+j, c]
        # So for A rows 0,1 and cols 0,1: U rows are 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4
        # Wait, I'm confusing myself. Let me be more explicit.
        
        pass  # This is getting complicated, let me do it directly.
    
    # Let me just directly construct the rank-55 decomposition.
    # 
    # A (2x3): rows i=0,1; cols j=0,1,2
    # B (3x10): rows j=0,1,2; cols k=0,...,9
    # C (2x10): rows i=0,1; cols k=0,...,9
    #
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # Split: C[i,k] = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + A[i,2]*B[2,k]
    #              = C1[i,k] + C2[i,k]
    #
    # C1: 2x2x10 multiplication (A1 is 2x2, B1 is 2x10)
    # C2: 2x1x10 multiplication (A2 is 2x1, B2 is 1x10)
    #
    # For C1 (2x2x10): split B1 into 5 blocks of 2x2.
    #   For each block b (cols 2b, 2b+1): use Strassen 2x2x2, rank 7
    #   Total: 5 * 7 = 35
    #
    # For C2 (2x1x10): schoolbook, rank = 2*1*10 = 20
    #   C2[i,k] = A[i,2] * B[2,k]
    #   For each i,k: one term. U[i*3+2, c]=1, V[2*10+k, c]=1, W[k*2+i, c]=1
    #
    # Total rank: 35 + 20 = 55
    
    rank_55 = 55
    U = [[0]*rank_55 for _ in range(n*m)]  # 6 x 55
    V = [[0]*rank_55 for _ in range(m*p)]  # 30 x 55
    W = [[0]*rank_55 for _ in range(p*n)]  # 20 x 55
    
    col = 0
    
    # Part 1: C1 = A1 * B1 where A1 is 2x2 (cols 0,1), B1 is 2x10 (rows 0,1)
    # Split B1 into 5 blocks of 2x2
    for block in range(5):
        k0 = 2 * block
        k1 = 2 * block + 1
        # Strassen 2x2x2 for A1 (2x2) * B1_block (2x2)
        # A1 rows: i=0,1; cols: j=0,1
        # B1_block rows: j=0,1; cols: k=k0,k1
        
        # U rows for A1: i*m+j = 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4
        # V rows for B1_block: j*p+k = 0*10+k0=10k0, 0*10+k1=10k0+1, 1*10+k0=10+k0, 1*10+k1=10+k0+1
        # W rows for C1_block: k*n+i = k0*2+0=2k0, k0*2+1=2k0+1, k1*2+0=2k1, k1*2+1=2k1+1
        
        u00 = 0    # A[0,0]
        u01 = 1    # A[0,1]
        u10 = 3    # A[1,0]
        u11 = 4    # A[1,1]
        
        v00 = 0*10 + k0    # B[0,k0]
        v01 = 0*10 + k1    # B[0,k1]
        v10 = 1*10 + k0    # B[1,k0]
        v11 = 1*10 + k1    # B[1,k1]
        
        w00 = k0*2 + 0    # C[0,k0]
        w01 = k0*2 + 1    # C[0,k1]
        w10 = k1*2 + 0    # C[1,k0]
        w11 = k1*2 + 1    # C[1,k1]
        
        # Strassen 7 terms:
        # M1 = (A00+A11)(B00+B11)
        U[u00][col] = 1; U[u11][col] = 1
        V[v00][col] = 1; V[v11][col] = 1
        W[w00][col] = 1; W[w01][col] = 1; W[w10][col] = 1; W[w11][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[u10][col] = 1; U[u11][col] = 1
        V[v00][col] = 1
        W[w10][col] = 1; W[w11][col] = 1
        col += 1
        
        # M3 = A00(B10-B11)
        U[u00][col] = 1
        V[v10][col] = 1; V[v11][col] = -1
        W[w01][col] = 1; W[w11][col] = 1
        col += 1
        
        # M4 = A11(B01-B00)
        U[u11][col] = 1
        V[v01][col] = 1; V[v00][col] = -1
        W[w00][col] = 1; W[w10][col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[u00][col] = 1; U[u01][col] = 1
        V[v11][col] = 1
        W[w01][col] = -1; W[w11][col] = -1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[u10][col] = 1; U[u00][col] = -1
        V[v00][col] = 1; V[v01][col] = 1
        W[w00][col] = 1; W[w01][col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[u01][col] = 1; U[u11][col] = -1
        V[v10][col] = 1; V[v11][col] = 1
        W[w00][col] = 1; W[w01][col] = 1
        col += 1
    
    # Part 2: C2 = A2 * B2 where A2 is 2x1 (col 2), B2 is 1x10 (row 2)
    # C2[i,k] = A[i,2] * B[2,k]
    # U row: i*m+2 = i*3+2 -> 2, 5
    # V row: 2*10+k = 20+k
    # W row: k*2+i
    for i in range(2):
        for k in range(10):
            U[i*3+2][col] = 1
            V[2*10+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
    
    assert col == rank_55, f"Expected {rank_55}, got {col}"
    
    # Verify
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fall back to schoolbook
        return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
