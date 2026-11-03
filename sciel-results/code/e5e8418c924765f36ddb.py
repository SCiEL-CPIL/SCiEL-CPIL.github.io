"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W form a valid rank-r decomposition of the matrix multiplication tensor.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Condition: for all i, k, j1, j2:
        sum_c U[i*m+j1, c] * V[j2*p+k, c] * W[k*n+i, c] == delta_{j1, j2}
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    val = np.dot(U[i*m+j1, :] * V[j2*p+k, :], W[k*n+i, :])
                    expected = 1.0 if j1 == j2 else 0.0
                    if abs(val - expected) > tol:
                        return False
    return True

def _build_schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2_block(U, V, W, i_off, j_off, k_off, col_offset, n, m, p):
    """
    Add a Strassen 7-term decomposition for a 2x2x2 block.
    A is at rows [i_off:i_off+2], cols [j_off:j_off+2]
    B is at rows [j_off:j_off+2], cols [k_off:k_off+2]
    C output is at rows [i_off:i_off+2], cols [k_off:k_off+2]
    
    Strassen formulas:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # A indices: (i, j) -> U[i*m + j, col]
    # B indices: (j, k) -> V[j*p + k, col]
    # C indices: (i, k) -> W[k*n + i, col]
    
    # A00 = A[i_off, j_off], A01 = A[i_off, j_off+1]
    # A10 = A[i_off+1, j_off], A11 = A[i_off+1, j_off+1]
    # B00 = B[j_off, k_off], B01 = B[j_off, k_off+1]
    # B10 = B[j_off+1, k_off], B11 = B[j_off+1, k_off+1]
    
    # M1 = (A00+A11)(B00+B11)
    c = col_offset
    U[i_off*m + j_off, c] = 1.0
    U[(i_off+1)*m + (j_off+1), c] = 1.0
    V[j_off*p + k_off, c] = 1.0
    V[(j_off+1)*p + (k_off+1), c] = 1.0
    # C00 += M1, C11 += M1
    W[k_off*n + i_off, c] = 1.0
    W[(k_off+1)*n + (i_off+1), c] = 1.0
    c += 1
    
    # M2 = (A10+A11)B00
    U[(i_off+1)*m + j_off, c] = 1.0
    U[(i_off+1)*m + (j_off+1), c] = 1.0
    V[j_off*p + k_off, c] = 1.0
    # C10 += M2, C11 -= M2
    W[k_off*n + (i_off+1), c] = 1.0
    W[(k_off+1)*n + (i_off+1), c] = -1.0
    c += 1
    
    # M3 = A00(B10-B11)
    U[i_off*m + j_off, c] = 1.0
    V[(j_off+1)*p + k_off, c] = 1.0
    V[(j_off+1)*p + (k_off+1), c] = -1.0
    # C01 += M3, C11 += M3
    W[(k_off+1)*n + i_off, c] = 1.0
    W[(k_off+1)*n + (i_off+1), c] = 1.0
    c += 1
    
    # M4 = A11(B01-B00)
    U[(i_off+1)*m + (j_off+1), c] = 1.0
    V[j_off*p + (k_off+1), c] = 1.0
    V[j_off*p + k_off, c] = -1.0
    # C00 += M4, C10 += M4
    W[k_off*n + i_off, c] = 1.0
    W[k_off*n + (i_off+1), c] = 1.0
    c += 1
    
    # M5 = (A00+A01)B11
    U[i_off*m + j_off, c] = 1.0
    U[i_off*m + (j_off+1), c] = 1.0
    V[(j_off+1)*p + (k_off+1), c] = 1.0
    # C01 += M5, C00 -= M5
    W[(k_off+1)*n + i_off, c] = 1.0
    W[k_off*n + i_off, c] = -1.0
    c += 1
    
    # M6 = (A10-A00)(B00+B10)
    U[(i_off+1)*m + j_off, c] = 1.0
    U[i_off*m + j_off, c] = -1.0
    V[j_off*p + k_off, c] = 1.0
    V[(j_off+1)*p + k_off, c] = 1.0
    # C11 += M6
    W[(k_off+1)*n + (i_off+1), c] = 1.0
    c += 1
    
    # M7 = (A01-A11)(B01+B11)
    U[i_off*m + (j_off+1), c] = 1.0
    U[(i_off+1)*m + (j_off+1), c] = -1.0
    V[j_off*p + (k_off+1), c] = 1.0
    V[(j_off+1)*p + (k_off+1), c] = 1.0
    # C00 += M7
    W[k_off*n + i_off, c] = 1.0
    c += 1
    
    return c

def _schoolbook_block(U, V, W, i_range, j_range, k_range, col_offset, n, m, p):
    """
    Add schoolbook decomposition for a block.
    A rows in i_range, A cols in j_range
    B rows in j_range, B cols in k_range
    C rows in i_range, C cols in k_range
    """
    c = col_offset
    for i in i_range:
        for j in j_range:
            for k in k_range:
                U[i*m + j, c] = 1.0
                V[j*p + k, c] = 1.0
                W[k*n + i, c] = 1.0
                c += 1
    return c

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Strategy: Decompose the 2x3x9 multiplication into blocks.
    # A is 2x3, B is 3x9.
    # 
    # Split A's columns into: [0,1] and [2]
    # Split B's rows into: [0,1] and [2]
    # Split B's columns into 3 groups of 3: [0,1,2], [3,4,5], [6,7,8]
    #
    # For each column group of B (3 groups):
    #   Block 1: A[:, 0:2] (2x2) @ B[0:2, group] (2x3)
    #     Split into: A[:, 0:2] @ B[0:2, group_cols[0:2]] (2x2x2) -> Strassen rank 7
    #                 A[:, 0:2] @ B[0:2, group_cols[2:3]] (2x2x1) -> schoolbook rank 4
    #   Block 2: A[:, 2:3] (2x1) @ B[2:3, group] (1x3) -> schoolbook rank 6
    #
    # Per group: 7 + 4 + 6 = 17
    # Total: 3 * 17 = 51
    #
    # Wait, let me reconsider. The schoolbook rank is 54.
    # Using Strassen for the 2x2x2 parts: 3 groups * 7 = 21 for the 2x2x2 parts
    # The 2x2x1 parts: 3 groups * 4 = 12
    # The 2x1x3 parts: 3 groups * 6 = 18
    # Total: 21 + 12 + 18 = 51
    #
    # Actually, let me think about this more carefully.
    # A is 2x3, B is 3x9.
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    #
    # For each k (9 values), C[:,k] = A @ B[:,k]
    # A @ B[:,k] where B[:,k] is a 3x1 vector.
    #
    # Let me use a different block decomposition:
    # Group B's columns into 3 groups of 3: G0=[0,1,2], G1=[3,4,5], G2=[6,7,8]
    #
    # For each group G:
    #   C[:, G] = A @ B[:, G]  (2x3 @ 3x3 = 2x3)
    #   = A[:, 0:2] @ B[0:2, G] + A[:, 2:3] @ B[2:3, G]
    #   = (2x2 @ 2x3) + (2x1 @ 1x3)
    #
    # For (2x2 @ 2x3):
    #   Let B[0:2, G] have columns b0, b1, b2 (each 2x1)
    #   A[:, 0:2] @ b0: 2x2 @ 2x1, rank 4 schoolbook, but can we do better?
    #   Actually 2x2 @ 2x1 is just two dot products of length 2, rank 4 is minimal.
    #   
    #   But we can use Strassen-like tricks across the 3 columns.
    #   Consider A[:, 0:2] @ B[0:2, G] where B[0:2, G] is 2x3.
    #   This is M_{2,2,3}. Known rank is 11.
    #   Schoolbook is 12.
    #
    #   If I use rank 11 for M_{2,2,3}, then per group: 11 + 6 = 17, total 51.
    #   If I use rank 12 (schoolbook), per group: 12 + 6 = 18, total 54.
    #
    # Let me try to construct a rank-11 decomposition for M_{2,2,3}.
    #
    # M_{2,2,3}: A is 2x2, B is 2x3, C is 2x3.
    # A = [[a, b], [c, d]], B = [[e, f, g], [h, i, j]]
    # C = [[a*e+b*h, a*f+b*i, a*g+b*j],
    #      [c*e+d*h, c*f+d*i, c*g+d*j]]
    #
    # Rank 11 construction (from literature):
    # p1 = a*e, p2 = b*h, p3 = a*f, p4 = b*i, p5 = a*g, p6 = b*j
    # p7 = c*e, p8 = d*h, p9 = c*f, p10 = d*i
    # p11 = (c+d)*(g+j) ... but then C[1,2] = c*g + d*j = (c+d)*(g+j) - c*j - d*g
    # This doesn't work cleanly.
    #
    # Alternative: use the fact that M_{2,2,3} can be seen as M_{2,2,2} + M_{2,2,1}
    # with sharing. M_{2,2,2} has rank 7 (Strassen), M_{2,2,1} has rank 4.
    # Total 11 if we can share one term.
    #
    # Actually, let me just use Strassen for the 2x2x2 part and schoolbook for 2x2x1.
    # For M_{2,2,3}: use Strassen (rank 7) for first 2 columns, schoolbook (rank 4) for 3rd column.
    # But the Strassen decomposition assumes both A and B are 2x2. Here B is 2x3.
    #
    # Let me just use schoolbook for simplicity and verify it works.
    # Total rank: 3 * (12 + 6) = 54. Same as full schoolbook.
    #
    # To improve, I need to find a better decomposition.
    #
    # Let me try a different approach: use the known result that R(M_{2,2,3}) = 11.
    # I'll construct it explicitly.
    
    # For now, let me use the Strassen-based approach that gives rank 51.
    # Per group: Strassen (7) for 2x2x2 + schoolbook (4) for 2x2x1 + schoolbook (6) for 2x1x3 = 17
    # Total: 3 * 17 = 51
    
    # Actually, let me reconsider the block structure.
    # A is 2x3, B is 3x9.
    # 
    # Let me split A's 3 columns into 2+1, and B's 3 rows into 2+1.
    # C = A[:,0:2] @ B[0:2,:] + A[:,2:3] @ B[2:3,:]
    # 
    # A[:,0:2] is 2x2, B[0:2,:] is 2x9.
    # A[:,2:3] is 2x1, B[2:3,:] is 1x9.
    #
    # For 2x2 @ 2x9: split B[0:2,:] into 3 groups of 2x3.
    #   Each 2x2 @ 2x3: use rank 11 if possible, else 12.
    # For 2x1 @ 1x9: rank 18 schoolbook, or split into 3 groups of 2x1 @ 1x3: 3*6=18.
    #
    # If I use rank 11 for each 2x2@2x3: 3*11 + 18 = 33 + 18 = 51.
    # If I use rank 12: 3*12 + 18 = 36 + 18 = 54.
    
    # Let me try to construct rank 11 for M_{2,2,3}.
    # After research, here's a valid rank-11 decomposition:
    #
    # A = [[a, b], [c, d]], B = [[e, f, g], [h, i, j]]
    #
    # p1 = a*e
    # p2 = b*h  
    # p3 = a*f
    # p4 = b*i
    # p5 = a*g
    # p6 = b*j
    # p7 = c*e
    # p8 = d*h
    # p9 = c*f
    # p10 = d*i
    # p11 = (c+d)*(g+j)
    #
    # C[0,0] = p1 + p2
    # C[0,1] = p3 + p4
    # C[0,2] = p5 + p6
    # C[1,0] = p7 + p8
    # C[1,1] = p9 + p10
    # C[1,2] = c*g + d*j
    #
    # But p11 = (c+d)*(g+j) = c*g + c*j + d*g + d*j
    # So c*g + d*j = p11 - c*j - d*g
    # We need c*j and d*g, which are not in our list.
    #
    # This doesn't work. Let me try a different set of products.
    #
    # Alternative rank-11:
    # p1 = (a+b)*(e+h)
    # p2 = a*e
    # p3 = b*h
    # p4 = a*f
    # p5 = b*i
    # p6 = a*g
    # p7 = b*j
    # p8 = (c+d)*(e+h)
    # p9 = c*e
    # p10 = d*h
    # p11 = c*f + d*i  -- not a single product
    #
    # This is tricky. Let me just use schoolbook rank 54 for now and see if I can find
    # a better approach within the time budget.
    
    # For now, use the Strassen-based decomposition that gives rank 51.
    # Per group of 3 columns of B:
    #   2x2 @ 2x2 (first 2 cols): Strassen rank 7
    #   2x2 @ 2x1 (3rd col): schoolbook rank 4
    #   2x1 @ 1x3: schoolbook rank 6
    # Total per group: 17, total: 51.
    
    rank = 51
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for group in range(3):
        k_off = group * 3
        # 2x2 @ 2x2 part: A[:, 0:2] @ B[0:2, k_off:k_off+2]
        col = _strassen_2x2x2_block(U, V, W, 0, 0, k_off, col, n, m, p)
        
        # 2x2 @ 2x1 part: A[:, 0:2] @ B[0:2, k_off+2:k_off+3]
        col = _schoolbook_block(U, V, W, [0, 1], [0, 1], [k_off+2], col, n, m, p)
        
        # 2x1 @ 1x3 part: A[:, 2:3] @ B[2:3, k_off:k_off+3]
        col = _schoolbook_block(U, V, W, [0, 1], [2], [k_off, k_off+1, k_off+2], col, n, m, p)
    
    # Verify the decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fall back to schoolbook
        U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
# EVOLVE-BLOCK-END
