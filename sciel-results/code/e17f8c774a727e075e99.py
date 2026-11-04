"""Matrix multiplication 4x4 x 4x6 rank-minimal decomposition search."""
import time
import itertools
import numpy as np

DIMENSIONS = [4, 4, 6]
_n, _m, _p = DIMENSIONS

# Build the multiplication tensor as a 3-tensor T[i,j,k] = e_{i,j,k}
# where i in [0,n), j in [0,m), k in [0,p)
# We want to decompose this as sum_r u_r[i] * v_r[j] * w_r[k]
# U has shape (n*m, rank) with U[i*m+j, r] = u_r[i]
# V has shape (m*p, rank) with V[j*p+k, r] = v_r[j]
# W has shape (p*n, rank) with W[k*n+i, r] = w_r[k]

def _build_tensor(n, m, p):
    """Return the multiplication tensor as a dict mapping (i,j,k) -> 1."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i, j, k)] = 1
    return T

def _verify(U, V, W, n, m, p):
    """Check that U, V, W give an exact decomposition. Returns True if valid."""
    rank = len(U[0]) if U else 0
    # Reconstruct: for each (i,j,k), check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    return False
    # Also check that entries are integers or half-integers
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                v = U[i*m+j][r]
                if abs(v * 2 - round(v * 2)) > 1e-9:
                    return False
        for j in range(m):
            for k in range(p):
                v = V[j*p+k][r]
                if abs(v * 2 - round(v * 2)) > 1e-9:
                    return False
        for k in range(p):
            for i in range(n):
                v = W[k*n+i][r]
                if abs(v * 2 - round(v * 2)) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_4x4x6():
    """
    Block Strassen decomposition for 4x4 * 4x6.
    Split A (4x4) into 2x2 blocks, B (4x6) into 2x3 blocks.
    Use Strassen's 7 multiplications for the 2x2 * 2x2 blocks,
    and handle the 2x2 * 2x1 blocks with schoolbook.
    
    This should give rank < 96.
    """
    n, m, p = 4, 4, 6
    # We'll build this by enumerating the Strassen terms.
    # A is 4x4: A = [[A11, A12], [A21, A22]] each 2x2
    # B is 4x6: B = [[B11, B12], [B21, B22]] where B11,B21 are 2x2, B12,B22 are 2x2
    # Wait, B is 4x6. Split into 2x3 blocks:
    # B = [[B11, B12, B13], [B21, B22, B23]] each 2x2? No, 4x6 split as 2x3 blocks of 2x2.
    # So B = [[B11, B12, B13], [B21, B22, B23]] where each Bi is 2x2.
    # C = A*B is 4x6. C = [[C11, C12, C13], [C21, C22, C23]] each 2x2.
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    
    # Using Strassen for the 2x2 * 2x2 multiplications:
    # For each pair of 2x2 blocks (X, Y), we use 7 Strassen multiplications.
    # M1 = (X11+X22)*(Y11+Y22)
    # M2 = (X21+X22)*Y11
    # M3 = X11*(Y12-Y22)
    # M4 = X22*(Y21-Y11)
    # M5 = (X11+X12)*Y22
    # M6 = (X21-X11)*(Y11+Y12)
    # M7 = (X12-X22)*(Y21+Y22)
    # XY = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # We have 6 such 2x2*2x2 products. Each uses 7 multiplications.
    # Total: 6*7 = 42 multiplications, each of 2x2 matrices.
    # But wait, some Strassen terms might share sub-expressions.
    # Let's just implement it straightforwardly and count.
    
    # Actually, let me think about this differently. The rank of 2x2*2x2 matrix mult is 7.
    # For 4x4*4x6, we can view it as computing 6 2x2*2x2 products.
    # If we use Strassen for each independently, rank = 6*7 = 42.
    # But we need to check if the entries are integers/half-integers.
    
    # Let me construct it explicitly.
    # For each of the 6 block products, we'll assign 7 rank-1 terms.
    # The entries will be 0, 1, or -1 (integers).
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block indices:
    # A blocks: A11 = rows 0-1, cols 0-1; A12 = rows 0-1, cols 2-3
    #            A21 = rows 2-3, cols 0-1; A22 = rows 2-3, cols 2-3
    # B blocks: B11 = rows 0-1, cols 0-1; B12 = rows 0-1, cols 2-3; B13 = rows 0-1, cols 4-5
    #            B21 = rows 2-3, cols 0-1; B22 = rows 2-3, cols 2-3; B23 = rows 2-3, cols 4-5
    # C blocks: C11 = rows 0-1, cols 0-1; C12 = rows 0-1, cols 2-3; C13 = rows 0-1, cols 4-5
    #            C21 = rows 2-3, cols 0-1; C22 = rows 2-3, cols 2-3; C23 = rows 2-3, cols 4-5
    
    # For a 2x2*2x2 product X*Y where X is at A[r0:r0+2, c0:c0+2] and Y is at B[d0:d0+2, e0:e0+2],
    # the result goes to C[r0:r0+2, e0:e0+2].
    
    # Strassen formulas for X*Y:
    # Let X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # M1 = (x00+x11)*(y00+y11)
    # M2 = (x10+x11)*y00
    # M3 = x00*(y01-y11)
    # M4 = x11*(y10-y00)
    # M5 = (x00+x01)*y11
    # M6 = (x10-x00)*(y00+y01)
    # M7 = (x01-x11)*(y10+y11)
    # Result: [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # For each rank-1 term in the Strassen decomposition:
    # M1: u = [1,0,0,1] (for x00,x01,x10,x11), v = [1,0,0,1] (for y00,y01,y10,y11), 
    #      w = [1,0,0,1] (for result (0,0),(0,1),(1,0),(1,1)) with coefficients
    # Actually, each Mi is a product of a linear form in X and a linear form in Y.
    # In the tensor decomposition, each term is u_r[i]*v_r[j]*w_r[k].
    # For M1 = (x00+x11)*(y00+y11):
    #   u for x: x00 -> i=0 (row 0 of X), x11 -> i=1 (row 1 of X)
    #   Wait, I need to be more careful. X is a 2x2 submatrix of A.
    #   x00 = A[r0, c0], x01 = A[r0, c0+1], x10 = A[r0+1, c0], x11 = A[r0+1, c0+1]
    #   In the full U matrix, U[i*m+j, r] where i is the row in A, j is the col in A.
    #   So for x00: U[r0*m + c0, r] = 1
    #   For x11: U[(r0+1)*m + (c0+1), r] = 1
    #   Similarly for Y: y00 = B[d0, e0], etc.
    #   V[d0*p + e0, r] = 1 for y00, V[(d0+1)*p + (e0+1), r] = 1 for y11
    #   For the result: the contribution goes to C entries.
    #   M1 contributes to C[r0, e0], C[r0, e0+1], C[r0+1, e0], C[r0+1, e0+1]
    #   with coefficients: C[r0,e0] += 1, C[r0,e0+1] += 0, C[r0+1,e0] += 0, C[r0+1,e0+1] += 1
    #   Wait, that's not right. M1 = (x00+x11)*(y00+y11) contributes to all four entries of the result
    #   with coefficient 1 for the "diagonal" pattern? No.
    
    # Let me reconsider. In Strassen, M1 is one of the intermediate products.
    # The final result is assembled from the M's. So each M contributes to specific output entries.
    # For the 2x2*2x2 product, the output is a 2x2 matrix.
    # M1 contributes: result[0,0] += M1, result[0,1] += 0, result[1,0] += 0, result[1,1] += M1
    # No wait: [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    # So M1 contributes to result[0,0] and result[1,1].
    # M4 contributes to result[0,0] and result[1,0].
    # M5 contributes to result[0,0] (with -1) and result[0,1].
    # M7 contributes to result[0,0].
    # M3 contributes to result[0,1] and result[1,1].
    # M2 contributes to result[1,0] and result[1,1] (with -1).
    # M6 contributes to result[1,1].
    
    # OK this is getting complex. Let me just implement it carefully.
    
    def add_strassen_term(col, r0, c0, d0, e0, mi):
        """Add one Strassen multiplication term for block at A[r0:c0+2, :] * B[:, d0:e0+2]."""
        # X = A[r0:r0+2, c0:c0+2], Y = B[d0:d0+2, e0:e0+2]
        # mi is which of M1..M7
        if mi == 0:  # M1 = (x00+x11)*(y00+y11)
            # u: x00 and x11
            U[r0*m + c0][col] = 1
            U[(r0+1)*m + (c0+1)][col] = 1
            # v: y00 and y11
            V[d0*p + e0][col] = 1
            V[(d0+1)*p + (e0+1)][col] = 1
            # w: contributes to result[0,0] and result[1,1]
            # result[0,0] -> C[r0, e0] -> W[e0*n + r0, col]
            # result[1,1] -> C[r0+1, e0+1] -> W[(e0+1)*n + (r0+1), col]
            W[e0*n + r0][col] = 1
            W[(e0+1)*n + (r0+1)][col] = 1
        elif mi == 1:  # M2 = (x10+x11)*y00
            U[(r0+1)*m + c0][col] = 1
            U[(r0+1)*m + (c0+1)][col] = 1
            V[d0*p + e0][col] = 1
            # w: contributes to result[1,0] and result[1,1]
            W[e0*n + (r0+1)][col] = 1
            W[(e0+1)*n + (r0+1)][col] = -1
        elif mi == 2:  # M3 = x00*(y01-y11)
            U[r0*m + c0][col] = 1
            V[d0*p + (e0+1)][col] = 1
            V[(d0+1)*p + (e0+1)][col] = -1
            # w: contributes to result[0,1] and result[1,1]
            W[(e0+1)*n + r0][col] = 1
            W[(e0+1)*n + (r0+1)][col] = 1
        elif mi == 3:  # M4 = x11*(y10-y00)
            U[(r0+1)*m + (c0+1)][col] = 1
            V[(d0+1)*p + e0][col] = 1
            V[d0*p + e0][col] = -1
            # w: contributes to result[0,0] and result[1,0]
            W[e0*n + r0][col] = 1
            W[e0*n + (r0+1)][col] = 1
        elif mi == 4:  # M5 = (x00+x01)*y11
            U[r0*m + c0][col] = 1
            U[r0*m + (c0+1)][col] = 1
            V[(d0+1)*p + (e0+1)][col] = 1
            # w: contributes to result[0,0] (with -1) and result[0,1]
            W[e0*n + r0][col] = -1
            W[(e0+1)*n + r0][col] = 1
        elif mi == 5:  # M6 = (x10-x00)*(y00+y01)
            U[(r0+1)*m + c0][col] = 1
            U[r0*m + c0][col] = -1
            V[d0*p + e0][col] = 1
            V[d0*p + (e0+1)][col] = 1
            # w: contributes to result[1,1]
            W[(e0+1)*n + (r0+1)][col] = 1
        elif mi == 6:  # M7 = (x01-x11)*(y10+y11)
            U[r0*m + (c0+1)][col] = 1
            U[(r0+1)*m + (c0+1)][col] = -1
            V[(d0+1)*p + e0][col] = 1
            V[(d0+1)*p + (e0+1)][col] = 1
            # w: contributes to result[0,0]
            W[e0*n + r0][col] = 1
    
    # 6 block products, each with 7 Strassen terms
    blocks = [
        (0, 0, 0, 0),   # C11: A11*B11
        (0, 0, 0, 2),   # C12: A11*B12
        (0, 0, 0, 4),   # C13: A11*B13
        (0, 2, 0, 0),   # C12: A12*B21  -- wait, this is wrong
    ]
    # Let me redo this. The block products are:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    # So we need 12 individual 2x2*2x2 products, not 6.
    # Each uses 7 Strassen terms: 12*7 = 84 rank terms.
    # That's better than 96 but not great.
    
    # Actually wait, I made an error. Let me reconsider.
    # A is 4x4, B is 4x6.
    # Split A into 2x2 blocks: A11 (top-left 2x2), A12 (top-right 2x2), A21 (bottom-left 2x2), A22 (bottom-right 2x2)
    # Split B into 2x2 blocks: B has 4 rows and 6 cols.
    # B11 (rows 0-1, cols 0-1), B12 (rows 0-1, cols 2-3), B13 (rows 0-1, cols 4-5)
    # B21 (rows 2-3, cols 0-1), B22 (rows 2-3, cols 2-3), B23 (rows 2-3, cols 4-5)
    # C11 (rows 0-1, cols 0-1) = A11*B11 + A12*B21
    # C12 (rows 0-1, cols 2-3) = A11*B12 + A12*B22
    # C13 (rows 0-1, cols 4-5) = A11*B13 + A12*B23
    # C21 (rows 2-3, cols 0-1) = A21*B11 + A22*B21
    # C22 (rows 2-3, cols 2-3) = A21*B12 + A22*B22
    # C23 (rows 2-3, cols 4-5) = A21*B13 + A22*B23
    
    # 12 block products * 7 Strassen = 84 rank.
    
    # But we can do better by using the structure.
    # Let's just go with 84 for now and verify.
    
    U = [[0] * 84 for _ in range(n * m)]
    V = [[0] * 84 for _ in range(m * p)]
    W = [[0] * 84 for _ in range(p * n)]
    
    # 12 block products
    # (r0, c0, d0, e0) for each 2x2*2x2 product
    block_products = [
        (0, 0, 0, 0),  # A11 * B11 -> C11
        (0, 2, 2, 0),  # A12 * B21 -> C11
        (0, 0, 0, 2),  # A11 * B12 -> C12
        (0, 2, 2, 2),  # A12 * B22 -> C12
        (0, 0, 0, 4),  # A11 * B13 -> C13
        (0, 2, 2, 4),  # A12 * B23 -> C13
        (2, 0, 0, 0),  # A21 * B11 -> C21
        (2, 2, 2, 0),  # A22 * B21 -> C21
        (2, 0, 0, 2),  # A21 * B12 -> C22
        (2, 2, 2, 2),  # A22 * B22 -> C22
        (2, 0, 0, 4),  # A21 * B13 -> C23
        (2, 2, 2, 4),  # A22 * B23 -> C23
    ]
    
    col = 0
    for (r0, c0, d0, e0) in block_products:
        for mi in range(7):
            add_strassen_term(col, r0, c0, d0, e0, mi)
            col += 1
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # If Strassen didn't work (maybe I made an error), fall back to schoolbook
    return _schoolbook(n, m, p)

def construct():
    # Try Strassen first
    result = _strassen_4x4x6()
    return result
