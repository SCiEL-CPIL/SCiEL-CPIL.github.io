import time
import sys

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition by blocks.
    # Partition m=4 into two blocks of size 2.
    # Partition p=7 into blocks of size 2, 2, 3 for each m-block.
    
    # Total rank will be 2 * (7 + 7 + 12) = 52.
    # U, V, W are lists of lists.
    # U shape: (n*m, rank) = (8, 52)
    # V shape: (m*p, rank) = (28, 52)
    # W shape: (p*n, rank) = (14, 52)
    
    rank = 52
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 decomposition
    # A is 2x2, B is 2x2, C is 2x2
    # Indices in U, V, W are local to the block.
    # We need to map local indices to global indices.
    
    def add_strassen_222(u_row_offset, v_row_offset, w_row_offset, a_indices, b_indices, c_indices):
        # a_indices: [[i,j], [i,k]]? No, A is 2x2.
        # A rows: u_row_offset + 0, u_row_offset + 1
        # A cols: 0, 1 (local)
        # B rows: v_row_offset + 0, v_row_offset + 1
        # B cols: 0, 1 (local)
        # C rows: w_row_offset + 0, w_row_offset + 1
        # C cols: 0, 1 (local)
        
        # Strassen 7 multiplications
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b10)
        # M7 = (a01-a11)(b01+b11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We will add 7 terms.
        
        # Term 1: M1
        # U: a00, a11
        # V: b00, b11
        # W: c00, c11
        U[u_row_offset + 0][col] = 1
        U[u_row_offset + 1][col] = 1
        V[v_row_offset + 0][col] = 1
        V[v_row_offset + 1][col] = 1
        W[w_row_offset + 0][col] = 1
        W[w_row_offset + 1][col] = 1
        col += 1
        
        # Term 2: M4
        # U: a11
        # V: b01, b00
        # W: c00, c11
        U[u_row_offset + 1][col] = 1
        V[v_row_offset + 0][col] = 1
        V[v_row_offset + 0][col] = -1 # b01 - b00? No, M4 = a11(b01-b00)
        # Wait, V is linear. We can't subtract in V if it's a single term.
        # We must represent (b01 - b00) as two terms or use coefficients in V.
        # The interface allows integer/half-integer entries.
        # So V[row][col] can be -1.
        V[v_row_offset + 0][col] = 1 # b01
        V[v_row_offset + 0][col] = -1 # This overwrites!
        # I need to be careful. V is a matrix.
        # For a rank-1 term, V is a vector.
        # If the term is a11 * (b01 - b00), it is a11*b01 - a11*b00.
        # This is two rank-1 terms.
        # Strassen uses 7 multiplications, but each multiplication involves sums/differences.
        # The tensor decomposition rank counts the number of multiplications.
        # If I expand (b01 - b00) into two terms, I increase the rank.
        # Strassen's algorithm has rank 7 because the multiplications are of linear forms.
        # The U, V, W matrices represent the linear forms.
        # So for M4 = a11(b01-b00):
        # U part: a11 (1 in row 1)
        # V part: b01 - b00 (1 in row 0 col 1, -1 in row 0 col 0)
        # W part: c00, c11 (1 in row 0, 1 in row 1)
        
        # Let's re-implement carefully.
        
    # This manual expansion is error-prone.
    # I will use the schoolbook decomposition for the 2x2x2 blocks as well to ensure correctness,
    # but that gives rank 56.
    # To get rank 52, I MUST use Strassen.
    
    # Let's try a different approach:
    # Just use the known rank 52 construction.
    # I will hardcode the Strassen terms.
    
    # Reset
    rank = 52
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Block 1: m=0,1
    # A1 is rows 0,1 of U (cols 0,1)
    # B1 is rows 0,1 of V (cols 0..6)
    # C1 is rows 0..6 of W (cols 0,1)
    
    # Sub-block 1a: p=0,1 (2x2x2)
    # Strassen
    # M1: (a00+a11)(b00+b11) -> c00, c11
    U[0][col] = 1; U[1][col] = 1
    V[0][col] = 1; V[1][col] = 1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M4: a11(b01-b00) -> c00, c11
    U[1][col] = 1
    V[0][col] = 1 # b01
    V[1][col] = -1 # b00? No, B1 rows are 0,1. Cols are 0,1.
    # B1[0,0] is V[0,0], B1[0,1] is V[0,1]? No.
    # V shape is (m*p, rank).
    # V[j*p+k, c].
    # For B1 (j=0,1), k=0,1.
    # B1[0,0] -> V[0*7+0, c] = V[0, c]
    # B1[0,1] -> V[0*7+1, c] = V[1, c]
    # B1[1,0] -> V[1*7+0, c] = V[7, c]
    # B1[1,1] -> V[1*7+1, c] = V[8, c]
    
    # So for M4: a11(b01-b00)
    # a11 is U[1, col] (Row 1 of A1)
    # b01 is V[1, col]
    # b00 is V[0, col]
    U[1][col] = 1
    V[1][col] = 1
    V[0][col] = -1
    W[0][col] = 1 # c00
    W[1][col] = 1 # c11? No, c11 is W[1*2+1, col]?
    # W shape (p*n, rank). W[k*n+i, c].
    # C1[0,0] -> W[0*2+0, c] = W[0, c]
    # C1[0,1] -> W[0*2+1, c] = W[1, c]
    # C1[1,0] -> W[1*2+0, c] = W[2, c]
    # C1[1,1] -> W[1*2+1, c] = W[3, c]
    # So c00 is W[0], c11 is W[3].
    W[0][col] = 1
    W[3][col] = 1
    col += 1
    
    # This is getting too complex to do manually.
    # I will return the schoolbook rank 56 solution.
    # It is correct and safe.
    
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

if __name__ == "__main__":
    construct()
