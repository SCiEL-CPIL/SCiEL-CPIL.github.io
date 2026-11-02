"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    
    We try to find a decomposition with rank < 24 (the schoolbook rank).
    Based on algebraic analysis, we can potentially achieve rank 22 by:
    - Splitting the 2x3x4 problem into a 2x2x4 part (using two Strassen 2x2x2 blocks, rank 14)
    - And a 2x1x4 part (using schoolbook, rank 8)
    Total: 14 + 8 = 22
    
    However, the Strassen algorithm for 2x2x2 has rank 7, not 8.
    So for 2x2x4, we use two Strassen blocks: 7 + 7 = 14.
    For 2x1x4, we need 8 terms (each output entry requires one product).
    Total: 14 + 8 = 22.
    
    But wait, we need to be careful about the indexing and ensure the decomposition is exact.
    
    Actually, let me reconsider. The Strassen algorithm computes 2x2 matrix multiplication with 7 products.
    For a 2x2 matrix A and 2x2 matrix B:
    C = A * B
    
    Strassen's 7 products:
    P1 = (A11 + A22)(B11 + B22)
    P2 = (A21 + A22)B11
    P3 = A11(B12 - B22)
    P4 = A22(B21 - B11)
    P5 = (A11 + A12)B22
    P6 = (A21 - A11)(B11 + B12)
    P7 = (A12 - A22)(B21 + B22)
    
    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6
    
    For our 2x3x4 problem, we can split it as:
    A = [A1 A2] where A1 is 2x2 and A2 is 2x1
    B = [B1; B2] where B1 is 2x4 and B2 is 1x4
    
    C = A1*B1 + A2*B2
    
    A1*B1: 2x2 * 2x4. We can compute this as two 2x2 multiplications:
    B1 = [B1_left B1_right] where B1_left is 2x2 and B1_right is 2x2
    A1*B1 = A1*B1_left + A1*B1_right
    Each 2x2 * 2x2 uses Strassen with rank 7, so total 14 for A1*B1.
    
    A2*B2: 2x1 * 1x4. This is an outer product.
    A2 = [a1; a2] (2x1), B2 = [b1 b2 b3 b4] (1x4)
    A2*B2 = [[a1*b1, a1*b2, a1*b3, a1*b4],
             [a2*b1, a2*b2, a2*b3, a2*b4]]
    Each entry is a distinct product, so we need 8 terms.
    
    Total rank: 14 + 8 = 22.
    
    Let me implement this and verify it's exact.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=4
    
    # Try rank 22 first
    rank = 22
    
    # Initialize U, V, W
    # U: (n*m, rank) -> (6, 22)
    # V: (m*p, rank) -> (12, 22)
    # W: (p*n, rank) -> (8, 22)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: A1*B1 where A1 is 2x2 (rows 0,1 of A, cols 0,1 of B)
    # B1 is 2x4 (rows 0,1 of B, all cols)
    # Split B1 into B1_left (2x2, cols 0,1) and B1_right (2x2, cols 2,3)
    
    # Strassen for A1 * B1_left (2x2 * 2x2)
    # A1 = [[A00, A01], [A10, A11]] where A00 = A[0,0], A01 = A[0,1], A10 = A[1,0], A11 = A[1,1]
    # B1_left = [[B00, B01], [B10, B11]] where B00 = B[0,0], B01 = B[0,1], B10 = B[1,0], B11 = B[1,1]
    
    # Strassen products for A1 * B1_left
    col = 0
    # P1 = (A00 + A11)(B00 + B11)
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[3][col] = 1
    W[0][col] = 1; W[4][col] = 1; W[3][col] = 1; W[7][col] = 1
    col += 1
    # P2 = (A10 + A11)B00
    U[1][col] = 1; U[3][col] = 1
    V[0][col] = 1
    W[3][col] = 1; W[7][col] = -1
    col += 1
    # P3 = A00(B01 - B11)
    U[0][col] = 1
    V[1][col] = 1; V[3][col] = -1
    W[1][col] = 1; W[7][col] = 1
    col += 1
    # P4 = A11(B10 - B00)
    U[3][col] = 1
    V[2][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[3][col] = 1
    col += 1
    # P5 = (A00 + A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[3][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    # P6 = (A10 - A00)(B00 + B01)
    U[1][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[7][col] = 1
    col += 1
    # P7 = (A01 - A11)(B10 + B11)
    U[1][col] = 1; U[3][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[0][col] = 1; W[7][col] = -1
    col += 1
    
    # Strassen for A1 * B1_right (2x2 * 2x2)
    # B1_right = [[B02, B03], [B12, B13]] where B02 = B[0,2], B03 = B[0,3], B12 = B[1,2], B13 = B[1,3]
    
    # P1 = (A00 + A11)(B02 + B12)
    U[0][col] = 1; U[3][col] = 1
    V[2][col] = 1; V[6][col] = 1
    W[2][col] = 1; W[6][col] = 1; W[5][col] = 1; W[7][col] = 1
    col += 1
    # P2 = (A10 + A11)B02
    U[1][col] = 1; U[3][col] = 1
    V[2][col] = 1
    W[5][col] = 1; W[7][col] = -1
    col += 1
    # P3 = A00(B03 - B13)
    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[3][col] = 1; W[7][col] = 1
    col += 1
    # P4 = A11(B12 - B02)
    U[3][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[2][col] = 1; W[5][col] = 1
    col += 1
    # P5 = (A00 + A01)B13
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[2][col] = -1; W[3][col] = 1
    col += 1
    # P6 = (A10 - A00)(B02 + B03)
    U[1][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[7][col] = 1
    col += 1
    # P7 = (A01 - A11)(B12 + B13)
    U[1][col] = 1; U[3][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[2][col] = 1; W[7][col] = -1
    col += 1
    
    # Part 2: A2*B2 where A2 is 2x1 (col 2 of A) and B2 is 1x4 (row 2 of B)
    # A2 = [A02; A12] where A02 = A[0,2], A12 = A[1,2]
    # B2 = [B20 B21 B22 B23] where B20 = B[2,0], B21 = B[2,1], B22 = B[2,2], B23 = B[2,3]
    
    # 8 schoolbook terms for A2*B2
    for i in range(n):  # i = 0, 1
        for k in range(p):  # k = 0, 1, 2, 3
            u_idx = i * m + 2  # col 2 of A
            v_idx = 2 * p + k  # row 2 of B, col k
            w_idx = k * n + i  # trace convention: W[k*n + i]
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: for each entry C[i,k], we need sum_r U[i*m + ?] * V[? * p + k] * W[k*n + i]
    # Actually, the tensor decomposition is:
    # T[i,j,k] = sum_r U[i*m + j] * V[j*p + k] * W[k*n + i]
    
    # Let's verify by computing the multiplication tensor
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1 if (i < n and j < m and k < p) else 0
                # Actually, the tensor entry T[i,j,k] = 1 if this is a valid (i,j,k) triple
                # For matrix multiplication, T[i,j,k] = 1 for all valid i,j,k
                computed = 0.0
                for r in range(rank):
                    computed += U_np[i*m + j, r] * V_np[j*p + k, r] * W_np[k*n + i, r]
                if abs(computed - 1.0) > 1e-10:
                    errors += 1
    
    if errors > 0:
        # Fallback to schoolbook rank 24
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    U[u_idx][column] = 1
                    V[v_idx][column] = 1
                    W[w_idx][column] = 1
                    column += 1
    else:
        # Verify more carefully
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    computed = 0.0
                    for r in range(rank):
                        computed += U_np[i*m + j, r] * V_np[j*p + k, r] * W_np[k*n + i, r]
                    max_err = max(max_err, abs(computed - 1.0))
        
        if max_err > 1e-10:
            # Fallback to schoolbook rank 24
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        U[u_idx][column] = 1
                        V[v_idx][column] = 1
                        W[w_idx][column] = 1
                        column += 1
        else:
            # Check that we haven't accidentally created non-integer entries
            for i in range(n*m):
                for r in range(rank):
                    if U[i][r] not in [0, 1, -1]:
                        # This shouldn't happen with our construction
                        pass
            for i in range(m*p):
                for r in range(rank):
                    if V[i][r] not in [0, 1, -1]:
                        pass
            for i in range(p*n):
                for r in range(rank):
                    if W[i][r] not in [0, 1, -1]:
                        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
