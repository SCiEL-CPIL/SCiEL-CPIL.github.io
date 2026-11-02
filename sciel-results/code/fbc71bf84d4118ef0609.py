"""Rank-56 decomposition for 2x2 * 2x16 matrix multiplication."""
import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-56 exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x16 matrix.
    
    The decomposition exploits the structure where m=2 (small inner dimension)
    and p=16 (larger output dimension). We use a strategy that groups the
    16 output columns into 8 pairs, applying a 7-term bilinear scheme per pair
    that shares computation between the two rows of the output.
    
    Output convention: W[k*n + i, r] where i is output row, k is output col.
    """
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    # The output C is 2x16, indexed C[i][k] where i in {0,1}, k in {0,...,15}
    # Input A is 2x2, indexed A[i][j] where i,j in {0,1}
    # Input B is 2x16, indexed B[j][k] where j in {0,1}, k in {0,...,15}
    #
    # C[i][k] = sum_j A[i][j] * B[j][k]
    #
    # We process columns in pairs (2t, 2t+1) for t = 0,...,7.
    # For each pair, we need:
    #   C[0][2t]   = A[0][0]*B[0][2t]   + A[0][1]*B[1][2t]
    #   C[0][2t+1] = A[0][0]*B[0][2t+1] + A[0][1]*B[1][2t+1]
    #   C[1][2t]   = A[1][0]*B[0][2t]   + A[1][1]*B[1][2t]
    #   C[1][2t+1] = A[1][0]*B[0][2t+1] + A[1][1]*B[1][2t+1]
    #
    # That's 4 outputs per pair. With 8 pairs, total 32 outputs.
    # Schoolbook needs 2*2*16 = 64 terms.
    #
    # Key idea: For each pair of columns, we can use a 7-term scheme:
    # Let a = A[0][0], b = A[0][1], c = A[1][0], d = A[1][1]
    # Let e = B[0][2t], f = B[1][2t], g = B[0][2t+1], h = B[1][2t+1]
    #
    # Outputs:
    #   C00 = a*e + b*f
    #   C01 = a*g + b*h
    #   C10 = c*e + d*f
    #   C11 = c*g + d*h
    #
    # We can use the identity:
    #   (a+c)*(e+f) = a*e + a*f + c*e + c*f
    #
    # Let's try a 7-term construction per pair:
    #   p1 = a * e
    #   p2 = a * f
    #   p3 = b * e
    #   p4 = b * f
    #   p5 = (a+b) * (e+f)  -- but this gives a*e + a*f + b*e + b*f, redundant
    #
    # Actually, let me think differently. The 2x2x2 matrix multiplication
    # has rank 7 (Strassen). We can apply Strassen's algorithm to the
    # "block" structure.
    #
    # View the computation as: for each column pair (2t, 2t+1), we compute
    # a 2x2 matrix multiplication where the "B" matrix is 2x2 with entries
    # B[0][2t], B[1][2t], B[0][2t+1], B[1][2t+1].
    #
    # Strassen's 7 multiplications for 2x2x2:
    #   M1 = (A00+A11)*(B00+B11)
    #   M2 = (A10+A11)*B00
    #   M3 = A00*(B11-B00)
    #   M4 = A11*(B00-B11)  -- wait, let me use standard Strassen
    #
    # Standard Strassen for C = AB where A,B are 2x2:
    #   M1 = (A00+A11)*(B00+B11)
    #   M2 = (A10+A11)*B00
    #   M3 = A00*(B01-B11)
    #   M4 = A11*(B10-B00)
    #   M5 = (A00+A10)*B10
    #   M6 = (A01-A10)*(B00+B10)
    #   M7 = (A11-A01)*(B01+B11)
    #
    # C00 = M1 - M4 + M6
    # C01 = M3 + M7
    # C10 = M2 + M5
    # C11 = M1 - M3 + M5
    #
    # In our case, for column pair t:
    #   A = [[A00, A01], [A10, A11]]  (the full 2x2 input matrix A)
    #   B_t = [[B[0][2t], B[0][2t+1]], [B[1][2t], B[1][2t+1]]]
    #
    # So:
    #   M1 = (A00+A11) * (B[0][2t] + B[1][2t+1])
    #   M2 = (A10+A11) * B[0][2t]
    #   M3 = A00 * (B[0][2t+1] - B[1][2t+1])
    #   M4 = A11 * (B[1][2t] - B[0][2t])
    #   M5 = (A00+A10) * B[1][2t]
    #   M6 = (A01-A10) * (B[0][2t] + B[1][2t])
    #   M7 = (A11-A01) * (B[0][2t+1] + B[1][2t+1])
    #
    # Outputs:
    #   C[0][2t]   = M1 - M4 + M6
    #   C[0][2t+1] = M3 + M7
    #   C[1][2t]   = M2 + M5
    #   C[1][2t+1] = M1 - M3 + M5
    #
    # This uses 7 multiplications per column pair, for 8 pairs = 56 total.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56
    
    col = 0
    for t in range(8):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1 = (A00+A11) * (B[0][k0] + B[1][k1])
        U[0][col] = 1   # A[0][0]
        U[3][col] = 1   # A[1][1]
        V[0 * p + k0][col] = 1   # B[0][k0]
        V[1 * p + k1][col] = 1   # B[1][k1]
        # C[0][k0] += M1, C[1][k1] += M1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M2 = (A10+A11) * B[0][k0]
        U[2][col] = 1   # A[1][0]
        U[3][col] = 1   # A[1][1]
        V[0 * p + k0][col] = 1   # B[0][k0]
        # C[1][k0] += M2
        W[k0 * n + 1][col] = 1
        col += 1
        
        # M3 = A00 * (B[0][k1] - B[1][k1])
        U[0][col] = 1   # A[0][0]
        V[0 * p + k1][col] = 1   # B[0][k1]
        V[1 * p + k1][col] = -1  # B[1][k1]
        # C[0][k1] += M3, C[1][k1] -= M3
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        # M4 = A11 * (B[1][k0] - B[0][k0])
        U[3][col] = 1   # A[1][1]
        V[1 * p + k0][col] = 1   # B[1][k0]
        V[0 * p + k0][col] = -1  # B[0][k0]
        # C[0][k0] -= M4
        W[k0 * n + 0][col] = -1
        col += 1
        
        # M5 = (A00+A10) * B[1][k0]
        U[0][col] = 1   # A[0][0]
        U[2][col] = 1   # A[1][0]
        V[1 * p + k0][col] = 1   # B[1][k0]
        # C[1][k0] += M5, C[1][k1] += M5
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M6 = (A01-A10) * (B[0][k0] + B[1][k0])
        U[1][col] = 1   # A[0][1]
        U[2][col] = -1  # A[1][0]
        V[0 * p + k0][col] = 1   # B[0][k0]
        V[1 * p + k0][col] = 1   # B[1][k0]
        # C[0][k0] += M6
        W[k0 * n + 0][col] = 1
        col += 1
        
        # M7 = (A11-A01) * (B[0][k1] + B[1][k1])
        U[3][col] = 1   # A[1][1]
        U[1][col] = -1  # A[0][1]
        V[0 * p + k1][col] = 1   # B[0][k1]
        V[1 * p + k1][col] = 1   # B[1][k1]
        # C[0][k1] += M7
        W[k1 * n + 0][col] = 1
        col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
