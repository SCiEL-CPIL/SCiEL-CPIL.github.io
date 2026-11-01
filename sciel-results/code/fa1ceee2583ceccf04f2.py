"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x5 matrix.
    
    Strategy: Use Strassen's algorithm for a 2x2 sub-block and schoolbook for the rest.
    This gives rank 7 + 12 = 19, but let's try to do better.
    
    Actually, let me construct a rank-10 decomposition if possible, or the best I can.
    """
    n, m, p = DIMENSIONS  # 2, 2, 5
    
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(4, 10, 10) = 10
    # Schoolbook: 20
    
    # Let me try to construct rank 10.
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # One approach: if we can find 10 rank-1 tensors that sum to the all-ones tensor.
    # 
    # Actually, for the all-ones tensor of size 2x2x5, the rank is known to be 10?
    # Let me think... The all-ones tensor of size a x b x c has rank... 
    # For 2x2x2, the all-ones tensor has rank 7 (Strassen shows it's at most 7, 
    # and it's known to be exactly 7).
    # For 2x2x5, it's not immediately obvious.
    
    # Let me just do a clean implementation with the best rank I can construct.
    # I'll use Strassen for the 2x2 part (columns 0,1) and schoolbook for columns 2,3,4.
    # Rank = 7 + 12 = 19.
    
    # Actually, let me try a smarter approach.
    # For 2x2 * 2x5, I can view it as:
    # C = A * B where A is 2x2, B is 2x5
    # 
    # Let me use the following construction:
    # Split B into B1 (2x2) and B2 (2x3)
    # Use Strassen for A*B1 (rank 7)
    # For A*B2 (2x2 * 2x3), use schoolbook (rank 12)
    # Total: 19
    
    # Or, I can try to find a better algorithm for 2x2 * 2x3.
    # 2x2 * 2x3: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i=0,1 and k=0,1,2
    # That's 6 outputs, each a sum of 2 products. Schoolbook is 12.
    # Can we do better? The tensor is 2x2x3 with all ones.
    # Rank of 2x2x3 all-ones tensor... 
    
    # Let me just go with rank 19 for now and make sure it's correct.
    
    rank = 19
    U = [[Fraction(0)] * rank for _ in range(n * m)]  # (4, 19)
    V = [[Fraction(0)] * rank for _ in range(m * p)]  # (10, 19)
    W = [[Fraction(0)] * rank for _ in range(p * n)]  # (10, 19)
    
    def add_term(term_idx, U_coeffs, V_coeffs, W_coeffs):
        for row, coeff in U_coeffs:
            U[row][term_idx] += coeff
        for row, coeff in V_coeffs:
            V[row][term_idx] += coeff
        for row, coeff in W_coeffs:
            W[row][term_idx] += coeff
    
    # Strassen for 2x2 * 2x2 (columns 0,1 of B)
    # A[i,j] -> U[i*2+j], B[j,k] -> V[j*5+k], C[i,k] -> W[k*2+i]
    
    # p0 = (a+d)(e+h)
    add_term(0,
             [(0, 1), (3, 1)],   # a, d
             [(0, 1), (6, 1)],   # e, h
             [(0, 1), (3, 1)])   # C[0,0], C[1,1]
    
    # p1 = (c+d)h
    add_term(1,
             [(2, 1), (3, 1)],   # c, d
             [(6, 1)],           # h
             [(1, 1), (3, -1)])  # C[1,0], C[1,1]
    
    # p2 = a(f-h)
    add_term(2,
             [(0, 1)],           # a
             [(1, 1), (6, -1)],  # f, -h
             [(2, 1), (3, 1)])   # C[0,1], C[1,1]
    
    # p3 = d(g-e)
    add_term(3,
             [(3, 1)],           # d
             [(5, 1), (0, -1)],  # g, -e
             [(0, 1)])           # C[0,0]
    
    # p4 = (a+b)e
    add_term(4,
             [(0, 1), (1, 1)],   # a, b
             [(0, 1)],           # e
             [(2, 1), (1, 1)])   # C[0,1], C[1,0]
    
    # p5 = (c+d)(f-h)
    add_term(5,
             [(2, 1), (3, 1)],   # c, d
             [(1, 1), (6, -1)],  # f, -h
             [(0, -1), (3, 1)])  # C[0,0], C[1,1]
    
    # p6 = (a-c)(e+g)
    add_term(6,
             [(0, 1), (2, -1)],  # a, -c
             [(0, 1), (5, 1)],   # e, g
             [(0, 1)])           # C[0,0]
    
    # Schoolbook for 2x2 * 2x3 (columns 2,3,4 of B)
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    for k in range(2, 5):
        for i in range(2):
            for j in range(2):
                term_idx = 7 + (k - 2) * 4 + i * 2 + j
                U_idx = i * 2 + j
                V_idx = j * 5 + k
                W_idx = k * 2 + i
                U[U_idx][term_idx] = Fraction(1)
                V[V_idx][term_idx] = Fraction(1)
                W[W_idx][term_idx] = Fraction(1)
    
    # Verify
    def verify(U, V, W, n, m, p, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p, rank):
        # Fallback to schoolbook rank 20
        rank = n * m * p
        U = [[Fraction(0)] * rank for _ in range(n * m)]
        V = [[Fraction(0)] * rank for _ in range(m * p)]
        W = [[Fraction(0)] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = Fraction(1)
                    V[j * p + k][col] = Fraction(1)
                    W[k * n + i][col] = Fraction(1)
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
