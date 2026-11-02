import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-50 decomposition for 2x3x9 matrix multiplication.
    
    Strategy:
    1. Split inner dimension m=3 into m1=2 and m2=1.
    2. Block A1 (2x2) * B1 (2x9):
       - Split B1 into four 2x2 blocks (cols 0-7) and one 2x1 block (col 8).
       - Use Strassen (rank 7) for each 2x2 block.
       - Use schoolbook (rank 4) for the 2x1 block.
    3. Block A2 (2x1) * B2 (1x9):
       - Use schoolbook (rank 18).
       
    Total Rank: 4*7 + 4 + 18 = 50.
    
    Index conventions:
      U has shape (n*m, rank) = (6, rank), with U[i*m + j, c]
      V has shape (m*p, rank) = (27, rank), with V[j*p + k, c]
      W has shape (p*n, rank) = (18, rank), with W[k*n + i, c]
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    # We will build the matrices as lists of lists of integers.
    # We don't know the final rank upfront in a static way if we were searching, 
    # but here we know it is 50. However, to be safe and generic, we can append.
    # But for speed and simplicity, let's pre-allocate if we are sure.
    # Rank = 50.
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # shape (6, 50)
    V = [[0] * rank for _ in range(m * p)]  # shape (27, 50)
    W = [[0] * rank for _ in range(p * n)]  # shape (18, 50)
    
    col = 0
    
    # --- Part 1: A[:, 0:2] (2x2) * B[0:2, :] (2x9) ---
    # A indices in U:
    # A[0,0] -> U[0]
    # A[0,1] -> U[1]
    # A[1,0] -> U[3]
    # A[1,1] -> U[4]
    
    # B indices in V:
    # B[0, k] -> V[0*p + k] = V[k]
    # B[1, k] -> V[1*p + k] = V[9+k]
    
    # C indices in W:
    # C[i, k] -> W[k*n + i]
    
    # Define Strassen factors for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Output reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Loop over 4 blocks of 2 columns (k=0..7)
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # B indices
        v_b00 = k0       # B[0, k0]
        v_b01 = k1       # B[0, k1]
        v_b10 = 9 + k0   # B[1, k0]
        v_b11 = 9 + k1   # B[1, k1]
        
        # A indices
        u_a00 = 0
        u_a01 = 1
        u_a10 = 3
        u_a11 = 4
        
        # C indices
        w_c00 = k0 * n + 0
        w_c01 = k1 * n + 0
        w_c10 = k0 * n + 1
        w_c11 = k1 * n + 1
        
        # M1: (A00+A11)(B00+B11)
        U[u_a00][col] += 1
        U[u_a11][col] += 1
        V[v_b00][col] += 1
        V[v_b11][col] += 1
        W[w_c00][col] += 1
        W[w_c11][col] += 1
        col += 1
        
        # M2: (A10+A11)B00
        U[u_a10][col] += 1
        U[u_a11][col] += 1
        V[v_b00][col] += 1
        W[w_c10][col] += 1
        W[w_c11][col] -= 1
        col += 1
        
        # M3: A00(B10-B11)
        U[u_a00][col] += 1
        V[v_b10][col] += 1
        V[v_b11][col] -= 1
        W[w_c01][col] += 1
        W[w_c11][col] += 1
        col += 1
        
        # M4: A11(B01-B00)
        U[u_a11][col] += 1
        V[v_b01][col] += 1
        V[v_b00][col] -= 1
        W[w_c00][col] += 1
        W[w_c10][col] += 1
        col += 1
        
        # M5: (A00+A01)B11
        U[u_a00][col] += 1
        U[u_a01][col] += 1
        V[v_b11][col] += 1
        W[w_c00][col] -= 1
        W[w_c01][col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[u_a10][col] += 1
        U[u_a00][col] -= 1
        V[v_b00][col] += 1
        V[v_b01][col] += 1
        W[w_c11][col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[u_a01][col] += 1
        U[u_a11][col] -= 1
        V[v_b10][col] += 1
        V[v_b11][col] += 1
        W[w_c00][col] += 1
        col += 1
        
        # End of 2x2 block, col advanced by 7
    
    # One 2x1 block (column 8)
    k8 = 8
    v_b08 = k8       # B[0, 8]
    v_b18 = 9 + k8   # B[1, 8]
    
    w_c08 = k8 * n + 0
    w_c18 = k8 * n + 1
    
    # Schoolbook for 2x1:
    # P1 = A00 * B08
    U[0][col] += 1
    V[v_b08][col] += 1
    W[w_c08][col] += 1
    col += 1
    
    # P2 = A01 * B18
    U[1][col] += 1
    V[v_b18][col] += 1
    W[w_c08][col] += 1
    col += 1
    
    # P3 = A10 * B08
    U[3][col] += 1
    V[v_b08][col] += 1
    W[w_c18][col] += 1
    col += 1
    
    # P4 = A11 * B18
    U[4][col] += 1
    V[v_b18][col] += 1
    W[w_c18][col] += 1
    col += 1
    
    # --- Part 2: A[:, 2:3] (2x1) * B[2:3, :] (1x9) ---
    # A indices in U:
    # A[0,2] -> U[2]
    # A[1,2] -> U[5]
    
    # B indices in V:
    # B[2, k] -> V[2*p + k] = V[18+k]
    
    # C indices in W:
    # C[0, k] -> W[k*n + 0]
    # C[1, k] -> W[k*n + 1]
    
    for k in range(9):
        v_b2k = 18 + k
        
        # Q1 = A02 * B2k
        U[2][col] += 1
        V[v_b2k][col] += 1
        W[k * n + 0][col] += 1
        col += 1
        
        # Q2 = A12 * B2k
        U[5][col] += 1
        V[v_b2k][col] += 1
        W[k * n + 1][col] += 1
        col += 1
        
    # Safety check
    elapsed = time.time() - start_time
    if elapsed > 115:
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
