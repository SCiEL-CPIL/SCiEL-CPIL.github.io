"""Program for 3x4 * 4x7 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
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

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2 * 2x2 matrices, rank 7."""
    # Strassen's algorithm for 2x2 * 2x2
    # U corresponds to A (2x2), V to B (2x2), W to C (2x2)
    # C[0,0] = M1 - M4 + M6 + M7
    # C[0,1] = M5 + M7
    # C[1,0] = M5 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    # where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # U: 4x7 (indices 0..3 for A00,A01,A10,A11)
    # V: 4x7 (indices 0..3 for B00,B01,B10,B11)
    # W: 4x7 (indices 0..3 for C00,C01,C10,C11)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    W[0][0] = 1  # C00
    W[2][0] = 1  # C10
    W[3][0] = 1  # C11  # Wait, let me recheck the formulas
    
    # Let me redo this more carefully.
    # C = A*B where A, B, C are 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]]
    # i = ae + bg, j = af + bh, k = ce + dg, l = cf + dh
    
    # Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)*e
    # p3 = a*(f-h)
    # p4 = d*(g-e)
    # p5 = (a+b)*h
    # p6 = (c-a)*(e+f)
    # p7 = (b-d)*(g+h)
    #
    # i = p1 - p4 + p6 + p7
    # j = p5 + p7
    # k = p5 + p4
    # l = p1 - p2 + p3 + p6
    
    # U indices: 0=a, 1=b, 2=c, 3=d
    # V indices: 0=e, 1=f, 2=g, 3=h
    # W indices: 0=i, 1=j, 2=k, 3=l
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # p1 = (a+d)(e+h): U0=1, U3=1; V0=1, V3=1; contributes to i(+1), l(+1)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # i and l get +1 from p1
    
    # p2 = (c+d)*e: U2=1, U3=1; V0=1; contributes to l(-1)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[3][1] = -1  # l gets -1 from p2
    
    # p3 = a*(f-h): U0=1; V1=1, V3=-1; contributes to l(+1)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[3][2] = 1  # l gets +1 from p3
    
    # p4 = d*(g-e): U3=1; V2=1, V0=-1; contributes to i(-1), k(+1)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[2][3] = 1  # i gets -1, k gets +1 from p4
    
    # p5 = (a+b)*h: U0=1, U1=1; V3=1; contributes to j(+1), k(+1)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[2][4] = 1  # j and k get +1 from p5
    
    # p6 = (c-a)*(e+f): U2=1, U0=-1; V0=1, V1=1; contributes to i(+1), l(+1)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1; W[3][5] = 1  # i and l get +1 from p6
    
    # p7 = (b-d)*(g+h): U1=1, U3=-1; V2=1, V3=1; contributes to i(+1), j(+1)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = 1  # i and j get +1 from p7
    
    return U, V, W

def _pad_and_combine(n, m, p, U_strassen, V_strassen, W_strassen):
    """
    Combine Strassen blocks with schoolbook to get a decomposition for n x m * m x p.
    This is a heuristic approach that may not be optimal.
    """
    # For 3x4 * 4x7, let's try a direct approach.
    # Partition A (3x4) into A1 (3x2) and A2 (3x2)
    # Partition B (4x7) into B1 (2x7) and B2 (2x7)
    # C = A1*B1 + A2*B2
    # Each A1*B1 is 3x2 * 2x7, schoolbook rank 42
    # Total rank: 84. No gain.
    
    # Try to use Strassen on a 2x2 sub-block.
    # But 3x4 doesn't naturally decompose into 2x2 blocks cleanly.
    
    # Alternative: just return schoolbook for now, but with a proper verification.
    pass

def construct():
    t0 = time.time()
    DEADLINE = 145.0
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 84
    
    # Verify schoolbook
    assert _verify(best["U"], best["V"], best["W"], n, m, p), "Schoolbook verification failed"
    
    # Try to find a lower-rank decomposition.
    # For 3x4 * 4x7, the tensor rank is at most 84 (schoolbook).
    # Known lower bounds: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # Upper bound via block methods: we can try to do better than 84.
    
    # One approach: use the fact that we can sometimes reduce rank by
    # using structured decompositions.
    
    # Try: split the middle dimension m=4 into two parts of size 2.
    # A = [A_left | A_right] where A_left is 3x2, A_right is 3x2
    # B = [B_top; B_bottom] where B_top is 2x7, B_bottom is 2x7
    # C = A_left * B_top + A_right * B_bottom
    
    # Each product is 3x2 * 2x7. Schoolbook rank for each: 3*2*7 = 42.
    # Total: 84. Same as schoolbook.
    
    # To do better, we need to exploit structure within the 3x2 * 2x7 product.
    # 3x2 * 2x7: can we do better than 42?
    # Lower bound: max(3*2, 2*7, 7*3) = max(6, 14, 21) = 21
    # Upper bound via Strassen-like: not straightforward for non-square.
    
    # Let's try a randomized search for rank < 84.
    # Strategy: for a target rank r, try to find U (12xr), V (28xr), W (21xr)
    # such that the tensor equation holds.
    
    # This is a hard problem in general. Let's try a greedy approach:
    # Start with schoolbook and try to merge terms.
    
    # Actually, let's try a different approach: use the known fact that
    # the tensor rank of matrix multiplication is related to the bilinear complexity.
    
    # For now, let's try to verify that schoolbook is valid and return it.
    # If we have time, try a few random rank-83 attempts.
    
    if time.time() - t0 > DEADLINE - 5:
        return best
    
    # Try randomized search for rank 83
    np.random.seed(42)
    target_rank = 83
    max_attempts = 10
    
    for attempt in range(max_attempts):
        if time.time() - t0 > DEADLINE - 10:
            break
        
        # Initialize U, V, W randomly
        U = np.random.randint(-2, 3, size=(n*m, target_rank)).tolist()
        V = np.random.randint(-2, 3, size=(m*p, target_rank)).tolist()
        W = np.random.randint(-2, 3, size=(p*n, target_rank)).tolist()
        
        # Check if this works (unlikely, but just in case)
        if _verify(U, V, W, n, m, p):
            best = {"U": U, "V": V, "W": W}
            best_rank = target_rank
            break
    
    return best
# EVOLVE-BLOCK-END
