import time
import sys

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a decomposition of rank 46
    # Based on splitting m=5 into 2+2+1 and p=5 into 2+2+1
    # and using Strassen (rank 7) for 2x2x2 blocks and schoolbook for others.
    
    U = [[0] * 46 for _ in range(n * m)]
    V = [[0] * 46 for _ in range(m * p)]
    W = [[0] * 46 for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i_offset, j_offset, k_offset):
        nonlocal col
        # Adds a 2x2x2 Strassen block
        # A[i_offset..i_offset+1, j_offset..j_offset+1]
        # B[j_offset..j_offset+1, k_offset..k_offset+1]
        # C[i_offset..i_offset+1, k_offset..k_offset+1]
        
        # M1 = (A00+A11)(B00+B11)
        U[(i_offset)*m + (j_offset)][col] = 1
        U[(i_offset+1)*m + (j_offset+1)][col] = 1
        V[(j_offset)*p + (k_offset)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        W[(k_offset)*n + (i_offset)][col] = 1
        W[(k_offset+1)*n + (i_offset+1)][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[(i_offset+1)*m + (j_offset)][col] = 1
        U[(i_offset+1)*m + (j_offset+1)][col] = 1
        V[(j_offset)*p + (k_offset)][col] = 1
        W[(k_offset)*n + (i_offset+1)][col] = 1
        W[(k_offset+1)*n + (i_offset+1)][col] = -1
        col += 1
        
        # M3 = A00(B10-B11)
        U[(i_offset)*m + (j_offset)][col] = 1
        V[(j_offset+1)*p + (k_offset)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = -1
        W[(k_offset)*n + (i_offset)][col] = 1
        W[(k_offset+1)*n + (i_offset)][col] = 1
        col += 1
        
        # M4 = A11(B01-B00)
        U[(i_offset+1)*m + (j_offset+1)][col] = 1
        V[(j_offset)*p + (k_offset+1)][col] = 1
        V[(j_offset)*p + (k_offset)][col] = -1
        W[(k_offset)*n + (i_offset)][col] = 1
        W[(k_offset)*n + (i_offset+1)][col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[(i_offset)*m + (j_offset)][col] = 1
        U[(i_offset)*m + (j_offset+1)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        W[(k_offset)*n + (i_offset)][col] = -1
        W[(k_offset+1)*n + (i_offset)][col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B10)
        U[(i_offset+1)*m + (j_offset)][col] = 1
        U[(i_offset)*m + (j_offset)][col] = -1
        V[(j_offset)*p + (k_offset)][col] = 1
        V[(j_offset+1)*p + (k_offset)][col] = 1
        W[(k_offset+1)*n + (i_offset+1)][col] = 1
        col += 1
        
        # M7 = (A01-A11)(B11+B01)
        U[(i_offset)*m + (j_offset+1)][col] = 1
        U[(i_offset+1)*m + (j_offset+1)][col] = -1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        V[(j_offset)*p + (k_offset+1)][col] = 1
        W[(k_offset+1)*n + (i_offset)][col] = 1
        col += 1

    def add_schoolbook_2x2x1(i_offset, j_offset, k_offset):
        # A[i_offset..i_offset+1, j_offset..j_offset+1] (2x2)
        # B[j_offset..j_offset+1, k_offset] (2x1)
        # C[i_offset..i_offset+1, k_offset] (2x1)
        # 4 multiplications
        nonlocal col
        for i in range(2):
            for j in range(2):
                U[(i_offset+i)*m + (j_offset+j)][col] = 1
                V[(j_offset+j)*p + (k_offset)][col] = 1
                W[(k_offset)*n + (i_offset+i)][col] = 1
                col += 1

    def add_schoolbook_2x1x2(i_offset, j_offset, k_offset):
        # A[i_offset..i_offset+1, j_offset] (2x1)
        # B[j_offset, k_offset..k_offset+1] (1x2)
        # C[i_offset..i_offset+1, k_offset..k_offset+1] (2x2)
        # 4 multiplications
        nonlocal col
        for i in range(2):
            for k in range(2):
                U[(i_offset+i)*m + (j_offset)][col] = 1
                V[(j_offset)*p + (k_offset+k)][col] = 1
                W[(k_offset+k)*n + (i_offset+i)][col] = 1
                col += 1

    def add_schoolbook_2x1x1(i_offset, j_offset, k_offset):
        # A[i_offset..i_offset+1, j_offset] (2x1)
        # B[j_offset, k_offset] (1x1)
        # C[i_offset..i_offset+1, k_offset] (2x1)
        # 2 multiplications
        nonlocal col
        for i in range(2):
            U[(i_offset+i)*m + (j_offset)][col] = 1
            V[(j_offset)*p + (k_offset)][col] = 1
            W[(k_offset)*n + (i_offset+i)][col] = 1
            col += 1

    # Partition:
    # A cols: 0,1 (g1), 2,3 (g2), 4 (g3)
    # B rows: 0,1 (g1), 2,3 (g2), 4 (g3)
    # B cols: 0,1 (g1), 2,3 (g2), 4 (g3)
    
    # Block (g1, g1): 2x2x2 Strassen
    add_strassen(0, 0, 0)
    
    # Block (g1, g2): A1 (2x2) * B12 (2x2) -> C12 (2x2)
    # This is 2x2x2 Strassen
    add_strassen(0, 0, 2)
    
    # Block (g1, g3): A1 (2x2) * B13 (2x1) -> C13 (2x1)
    # Schoolbook 2x2x1
    add_schoolbook_2x2x1(0, 0, 4)
    
    # Block (g2, g1): A2 (2x2) * B21 (2x2) -> C21 (2x2)
    # Strassen
    add_strassen(0, 2, 0)
    
    # Block (g2, g2): A2 (2x2) * B22 (2x2) -> C22 (2x2)
    # Strassen
    add_strassen(0, 2, 2)
    
    # Block (g2, g3): A2 (2x2) * B23 (2x1) -> C23 (2x1)
    # Schoolbook 2x2x1
    add_schoolbook_2x2x1(0, 2, 4)
    
    # Block (g3, g1): A3 (2x1) * B31 (1x2) -> C31 (2x2)
    # Schoolbook 2x1x2
    add_schoolbook_2x1x2(0, 4, 0)
    
    # Block (g3, g2): A3 (2x1) * B32 (1x2) -> C32 (2x2)
    # Schoolbook 2x1x2
    add_schoolbook_2x1x2(0, 4, 2)
    
    # Block (g3, g3): A3 (2x1) * B33 (1x1) -> C33 (2x1)
    # Schoolbook 2x1x1
    add_schoolbook_2x1x1(0, 4, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
