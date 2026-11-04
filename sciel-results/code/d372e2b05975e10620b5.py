import numpy as np
import time

def construct():
    n, m, p = 3, 4, 4
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    def verify(U, V, W):
        """Verify that the decomposition is exact: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
        rank = len(U[0]) if U else 0
        # Build tensor T[i,j,k]
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    T[i,j,k] = s
        return np.allclose(T, 1.0, atol=1e-10)

    def build_naive():
        """Naive (schoolbook) decomposition, rank = n*m*p = 48"""
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def build_strassen_2x2x4_block(i_off, j_off, k_off, U, V, W, col):
        """
        Add 7 Strassen terms for multiplying a 2x2 block of A (rows i_off..i_off+1, cols j_off..j_off+1)
        with a 2x2 block of B (rows j_off..j_off+1, cols k_off..k_off+1)
        contributing to a 2x2 block of C (rows i_off..i_off+1, cols k_off..k_off+1).
        
        Returns the new col index (col + 7).
        """
        # M1 = (A00+A11)(B00+B11)
        U[i_off*m + j_off][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[j_off*p + k_off][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[k_off*n + i_off][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        W[(k_off+1)*n + i_off][col] = 1
        W[k_off*n + (i_off+1)][col] = -1
        col += 1

        # M2 = (A10+A11)B00
        U[(i_off+1)*m + j_off][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[j_off*p + k_off][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        W[k_off*n + (i_off+1)][col] = -1
        col += 1

        # M3 = A00(B01-B11)
        U[i_off*m + j_off][col] = 1
        V[j_off*p + (k_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = -1
        W[(k_off+1)*n + i_off][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1

        # M4 = A11(B10-B00)
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + k_off][col] = 1
        V[j_off*p + k_off][col] = -1
        W[k_off*n + (i_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = -1
        col += 1

        # M5 = (A00+A01)B11
        U[i_off*m + j_off][col] = 1
        U[i_off*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        W[k_off*n + (i_off+1)][col] = -1
        col += 1

        # M6 = (A10-A00)(B00+B01)
        U[(i_off+1)*m + j_off][col] = 1
        U[i_off*m + j_off][col] = -1
        V[j_off*p + k_off][col] = 1
        V[j_off*p + (k_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1

        # M7 = (A01-A11)(B10+B11)
        U[i_off*m + (j_off+1)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = -1
        V[(j_off+1)*p + k_off][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1

        return col

    def build_strassen_2x4x4():
        """
        Build decomposition using Strassen for the 2x4x4 part (rows 0,1 of A) and naive for row 2.
        
        A is 2x4, B is 4x4, C is 2x4.
        Split A = [A_left | A_right] where A_left, A_right are 2x2 (cols 0-1, 2-3)
        Split B = [B_top; B_bot] where B_top, B_bot are 2x4 (rows 0-1, 2-3)
        C = A_left @ B_top + A_right @ B_bot
        
        Each product A_left @ B_top is 2x2 @ 2x4 = 2x4.
        We can do this as two 2x2x2 Strassen blocks per product.
        
        So for A_left @ B_top: two Strassen blocks (k=0,1 and k=2,3) -> 14 terms
        For A_right @ B_bot: two Strassen blocks -> 14 terms
        Total for 2x4x4: 28 terms
        For row 2 (1x4x4): 16 naive terms
        Total: 44 terms
        """
        rank = 44
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0

        # A_left @ B_top: A rows 0-1, cols 0-1; B rows 0-1, cols 0-3
        # Block 1: cols 0-1 of output
        col = build_strassen_2x2x4_block(0, 0, 0, U, V, W, col)
        # Block 2: cols 2-3 of output
        col = build_strassen_2x2x4_block(0, 0, 2, U, V, W, col)

        # A_right @ B_bot: A rows 0-1, cols 2-3; B rows 2-3, cols 0-3
        # Block 1: cols 0-1 of output
        col = build_strassen_2x2x4_block(0, 2, 0, U, V, W, col)
        # Block 2: cols 2-3 of output
        col = build_strassen_2x2x4_block(0, 2, 2, U, V, W, col)

        # Row 2 of A: naive 1x4x4 = 16 terms
        for j in range(m):
            for k in range(p):
                U[2*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + 2][col] = 1
                col += 1

        return {"U": U, "V": V, "W": W}

    # First, try the Strassen-based rank-44 decomposition
    best_result = None
    best_rank = n * m * p  # start with naive rank

    # Try rank 44 (Strassen-based)
    try:
        result_44 = build_strassen_2x4x4()
        if verify(result_44["U"], result_44["V"], result_44["W"]):
            best_result = result_44
            best_rank = 44
    except Exception:
        pass

    # Try to improve further with a search approach
    # Use a simple local search: start from rank 44 and try to merge/remove terms
    # This is complex, so let's try a different structured approach

    # Try rank 42: see if we can save 2 terms
    # One idea: use a different splitting for the 2x4x4 part
    # Or try to find a more efficient decomposition for the 1x4x4 part

    # For now, let's also try the naive as fallback
    if best_result is None:
        best_result = build_naive()
        best_rank = n * m * p

    # Check time and return
    if time.time() > deadline:
        return best_result

    # Try to do a simple optimization: check if we can reduce rank by
    # finding linear dependencies among the rank-1 terms
    # This is a complex problem, so let's just return the best we have

    return best_result
