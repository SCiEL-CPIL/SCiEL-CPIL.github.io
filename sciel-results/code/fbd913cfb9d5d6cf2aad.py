"""Matrix multiplication 4x4x4 via Strassen's algorithm (rank 49)."""
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Return the rank-49 Strassen decomposition for 4x4 matrix multiplication.
    
    The decomposition is built by applying Strassen's 2x2 identity recursively.
    We use the standard 7-product Strassen algorithm:
      M1 = (A11+A22)(B11+B22)
      M2 = (A21+A22)B11
      M3 = A11(B12-B22)
      M4 = A22(B21-B11)
      M5 = (A11+A12)B22
      M6 = (A21-A11)(B11+B12)
      M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    For 4x4 matrices, we split into 2x2 blocks of 2x2 submatrices, then apply
    Strassen again to each 2x2 subproblem. This gives rank 7*7 = 49.
    
    The tensor indices follow:
      U[i*m + j, c] where i is row of A, j is col of A
      V[j*p + k, c] where j is row of B, k is col of B  
      W[k*n + i, c] where k is row of C (col of B), i is col of C (row of A)
    
    Wait, let me re-read the convention:
      U: (n*m, rank) - input A has shape (n, m), so index is i*m + j
      V: (m*p, rank) - input B has shape (m, p), so index is j*p + k
      W: (p*n, rank) - output C has shape (p, n)?? 
    
    Actually re-reading: "output factor follows the official trace convention W[k*n+i,r]"
    
    So W[k*n + i, r] where k is the row index of C and i is the column index of C.
    C has shape (n, p) typically for A(n,m) * B(m,p) = C(n,p).
    
    But W has shape (p*n, rank). So the index k*n+i means k goes from 0 to n-1 and i from 0 to p-1?
    Or k from 0 to p-1 and i from 0 to n-1?
    
    The convention W[k*n+i, r] with W having p*n rows means k ranges over p values and i over n values.
    So C is indexed as C[k, i] where k is row (0..p-1) and i is col (0..n-1).
    
    Hmm, that's unusual. Let me think about this differently.
    
    For matrix multiplication A * B = C:
      A is (n, m), B is (m, p), C is (n, p)
      C[i, j] = sum_k A[i, k] * B[k, j]
    
    The tensor T[i, j, k] = A[i, k] * B[k, j] contributes to C[i, j].
    
    In the decomposition:
      U[i*m + k, c] * V[k*p + j, c] * W[?, c]
    
    The task says W[k*n + i, r]. If C is (n, p), then C has n rows and p cols.
    W has p*n entries. So W[row * n + col] where row in 0..p-1, col in 0..n-1?
    That would make C transposed or something.
    
    Actually, I think the convention might be that C is stored as (p, n) or the indices are swapped.
    Let me just follow the schoolbook example:
    
    In the schoolbook:
      U[i * m + j][column] = 1  # A[i, j]
      V[j * p + k][column] = 1  # B[j, k]
      W[k * n + i][column] = 1  # C[k, i]?? 
    
    Wait, that's weird. In standard matrix mult, C[i, k] = sum_j A[i,j] * B[j,k].
    So C has indices (i, k) where i is row (0..n-1) and k is col (0..p-1).
    
    But W[k*n + i] suggests C is indexed as C[k, i] with k being row and i being col.
    So C is (p, n)? That means C = B^T * A^T or something?
    
    No wait. Let me re-examine. The schoolbook has:
      for i in range(n):
        for j in range(m):
          for k in range(p):
            U[i*m+j][c] = 1
            V[j*p+k][c] = 1
            W[k*n+i][c] = 1
    
    So the product is: sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    = sum over (i,j,k) where c corresponds to (i,j,k)
    
    This gives: for fixed (i, j, k), the term is A[i,j] * B[j,k] * W[k*n+i]
    
    So C[k*n+i] gets contribution A[i,j]*B[j,k]. If we think of C as having index (k, i) in row-major with n columns, then C[k, i] = sum_j A[i,j] * B[j,k].
    
    This is C = B^T * A^T! Or equivalently, C^T = A * B.
    
    So the output matrix is actually (A*B)^T, or the indices are swapped. The "official trace convention" W[k*n+i] means the output is C^T where C = A*B, i.e., the output has shape (p, n) and output[k, i] = (A*B)[i, k].
    
    OK so I need to be careful. Let me define:
      A: (n, m)
      B: (m, p)
      Output: (p, n) where output[k, i] = sum_j A[i, j] * B[j, k] = (A*B)[i, k]
    
    So output = (A*B)^T.
    
    Now for Strassen, I need to adapt. Let me just build the 49-term decomposition directly.
    
    For 2x2 matrices:
      A = [[a, b], [c, d]], B = [[e, f], [g, h]]
      C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
      C^T = [[ae+bg, ce+dg], [af+bh, cf+dh]]
    
    Strassen for C^T:
      The 7 products:
        M1 = (a+d)(e+h)
        M2 = (c+d)e
        M3 = a(f-h)
        M4 = d(g-e)
        M5 = (a+b)h
        M6 = (c-a)(e+f)
        M7 = (b-d)(g+h)
      
      C11 = M1 + M4 - M5 + M7
      C12 = M3 + M5
      C21 = M2 + M4
      C22 = M1 - M2 + M3 + M6
    
    For C^T:
      C^T[0,0] = C11 = M1 + M4 - M5 + M7
      C^T[0,1] = C21 = M2 + M4
      C^T[1,0] = C12 = M3 + M5
      C^T[1,1] = C22 = M1 - M2 + M3 + M6
    
    Now for 4x4, I split A into 2x2 blocks of 2x2 matrices:
      A = [[A00, A01], [A10, A11]] where each block is 2x2
      B = [[B00, B01], [B10, B11]] where each block is 2x2
      
      C = A*B = [[C00, C01], [C10, C11]]
      
      C^T = [[C00^T, C10^T], [C01^T, C11^T]]
    
    Apply Strassen at the top level:
      M1 = (A00+A11)(B00+B11)
      M2 = (A10+A11)B00
      M3 = A00(B01-B11)
      M4 = A11(B10-B00)
      M5 = (A00+A01)B11
      M6 = (A10-A00)(B00+B01)
      M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    C^T = [[C00^T, C10^T], [C01^T, C11^T]]
    
    Each of these 2x2 block multiplications (like M1 = (A00+A11)(B00+B11)) is itself a 2x2 matrix product, which I decompose using Strassen's 7 products.
    
    So total rank = 7 (top-level) * 7 (bottom-level) = 49.
    
    Let me construct this explicitly. I'll represent each factor matrix entry as a linear combination.
    
    Actually, let me just build it by enumerating all 49 terms.
    
    Top-level Strassen products (7 of them), each is a 2x2 matrix product (7 sub-products each) = 49 total.
    
    For each top-level product, I have a 2x2 matrix X and 2x2 matrix Y, and I compute X*Y using Strassen.
    
    Let me define the indices:
      A has indices (i, j) with i in 0..3, j in 0..3
      B has indices (j, k) with j in 0..3, k in 0..3
      Output has indices (k, i) with k in 0..3, i in 0..3
    
    Block structure:
      A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
      B00 = B[0:2, 0:2], B01 = B[0:2, 2:4], B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    
    Top-level:
      T1 = (A00+A11) * (B00+B11)
      T2 = (A10+A11) * B00
      T3 = A00 * (B01-B11)
      T4 = A11 * (B10-B00)
      T5 = (A00+A01) * B11
      T6 = (A10-A00) * (B00+B01)
      T7 = (A01-A11) * (B10+B11)
    
    C00 = T1 + T4 - T5 + T7
    C01 = T3 + T5
    C10 = T2 + T4
    C11 = T1 - T2 + T3 + T6
    
    C^T[0:2, 0:2] = C00^T
    C^T[0:2, 2:4] = C10^T
    C^T[2:4, 0:2] = C01^T
    C^T[2:4, 2:4] = C11^T
    
    Now for each T_s (s=1..7), I compute a 2x2 * 2x2 product using Strassen.
    
    Let me write a helper that, given a 2x2 matrix X and 2x2 matrix Y (where each entry is a linear combination of A or B entries), returns the 7 Strassen products and how they combine to give X*Y^T (since we want the transpose).
    
    Actually, let me just directly construct the U, V, W matrices by building up the linear combinations.
    
    I'll use a dictionary-based approach where each "symbol" is a tuple representing which A or B entry it corresponds to, and I track coefficients.
    
    Let me define:
      A[i][j] as symbol ('A', i, j)
      B[i][j] as symbol ('B', i, j)
    
    Each entry of U, V, W will be an integer (0, 1, or -1 typically for Strassen).
    
    Let me build this step by step.
    
    For a 2x2 * 2x2 Strassen decomposition where the output is (X*Y)^T:
      X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
      
      7 products:
        P1 = (x00+x11)(y00+y11)
        P2 = (x10+x11)y00
        P3 = x00(y01-y11)
        P4 = x11(y10-y00)
        P5 = (x00+x01)y11
        P6 = (x10-x00)(y00+y01)
        P7 = (x01-x11)(y10+y11)
      
      (X*Y)^T[0,0] = P1 + P4 - P5 + P7
      (X*Y)^T[0,1] = P2 + P4
      (X*Y)^T[1,0] = P3 + P5
      (X*Y)^T[1,1] = P1 - P2 + P3 + P6
    
    Now, each x_ij and y_ij is itself a linear combination of A or B entries.
    
    Let me represent linear combinations as dicts: {symbol: coeff}.
    
    def add_dict(d1, d2):
        result = dict(d1)
        for k, v in d2.items():
            result[k] = result.get(k, 0) + v
        # remove zeros
        return {k: v for k, v in result.items() if v != 0}
    
    def sub_dict(d1, d2):
        return add_dict(d1, {k: -v for k, v in d2.items()})
    
    def scale_dict(d, s):
        return {k: v * s for k, v in d.items() if v * s != 0}
    
    Let me just hardcode the construction. This is getting complex, so let me be systematic.
    
    A entries: A[i][j] for i,j in 0..3
    B entries: B[i][j] for i,j in 0..3
    
    Block definitions:
      A00 = [[A[0][0], A[0][1]], [A[1][0], A[1][1]]]
      A01 = [[A[0][2], A[0][3]], [A[1][2], A[1][3]]]
      A10 = [[A[2][0], A[2][1]], [A[3][0], A[3][1]]]
      A11 = [[A[2][2], A[2][3]], [A[3][2], A[3][3]]]
      
      B00 = [[B[0][0], B[0][1]], [B[1][0], B[1][1]]]
      B01 = [[B[0][2], B[0][3]], [B[1][2], B[1][3]]]
      B10 = [[B[2][0], B[2][1]], [B[3][0], B[3][1]]]
      B11 = [[B[2][2], B[2][3]], [B[3][2], B[3][3]]]
    
    Top-level combinations:
      S1A = A00 + A11  (2x2 matrix)
      S1B = B00 + B11  (2x2 matrix)
      T1 = S1A * S1B
      
      S2A = A10 + A11
      S2B = B00
      T2 = S2A * S2B
      
      S3A = A00
      S3B = B01 - B11
      T3 = S3A * S3B
      
      S4A = A11
      S4B = B10 - B00
      T4 = S4A * S4B
      
      S5A = A00 + A01
      S5B = B11
      T5 = S5A * S5B
      
      S6A = A10 - A00
      S6B = B00 + B01
      T6 = S6A * S6B
      
      S7A = A01 - A11
      S7B = B10 + B11
      T7 = S7A * S7B
    
    For each T_s, I apply 2x2 Strassen to get 7 scalar products, each of which is a product of a linear combination of A entries and a linear combination of B entries.
    
    Total: 7 * 7 = 49 scalar products.
    
    Let me now write code to build this.
    """
    # Helper functions for linear combinations
    def add_dc(d1, d2):
        r = dict(d1)
        for k, v in d2.items():
            r[k] = r.get(k, 0) + v
        return {k: v for k, v in r.items() if v != 0}
    
    def sub_dc(d1, d2):
        return add_dc(d1, {k: -v for k, v in d2.items()})
    
    def mat_add(M1, M2):
        return [[add_dc(M1[i][j], M2[i][j]) for j in range(2)] for i in range(2)]
    
    def mat_sub(M1, M2):
        return [[sub_dc(M1[i][j], M2[i][j]) for j in range(2)] for i in range(2)]
    
    # Define A and B entries as symbols
    def A_sym(i, j):
        return {('A', i, j): 1}
    
    def B_sym(i, j):
        return {('B', i, j): 1}
    
    # Define blocks
    A00 = [[A_sym(0,0), A_sym(0,1)], [A_sym(1,0), A_sym(1,1)]]
    A01 = [[A_sym(0,2), A_sym(0,3)], [A_sym(1,2), A_sym(1,3)]]
    A10 = [[A_sym(2,0), A_sym(2,1)], [A_sym(3,0), A_sym(3,1)]]
    A11 = [[A_sym(2,2), A_sym(2,3)], [A_sym(3,2), A_sym(3,3)]]
    
    B00 = [[B_sym(0,0), B_sym(0,1)], [B_sym(1,0), B_sym(1,1)]]
    B01 = [[B_sym(0,2), B_sym(0,3)], [B_sym(1,2), B_sym(1,3)]]
    B10 = [[B_sym(2,0), B_sym(2,1)], [B_sym(3,0), B_sym(3,1)]]
    B11 = [[B_sym(2,2), B_sym(2,3)], [B_sym(3,2), B_sym(3,3)]]
    
    # Top-level combinations
    S1A = mat_add(A00, A11)
    S1B = mat_add(B00, B11)
    S2A = mat_add(A10, A11)
    S2B = B00
    S3A = A00
    S3B = mat_sub(B01, B11)
    S4A = A11
    S4B = mat_sub(B10, B00)
    S5A = mat_add(A00, A01)
    S5B = B11
    S6A = mat_sub(A10, A00)
    S6B = mat_add(B00, B01)
    S7A = mat_sub(A01, A11)
    S7B = mat_add(B10, B11)
    
    top_products = [
        (S1A, S1B),  # T1
        (S2A, S2B),  # T2
        (S3A, S3B),  # T3
        (S4A, S4B),  # T4
        (S5A, S5B),  # T5
        (S6A, S6B),  # T6
        (S7A, S7B),  # T7
    ]
    
    # For each 2x2 * 2x2 product, apply Strassen to get 7 scalar products
    # Each scalar product is (lin_comb_A) * (lin_comb_B)
    # We'll collect all 49 products
    
    all_products = []  # list of (A_comb, B_comb)
    
    for X, Y in top_products:
        # Strassen 7 products for X*Y (we want (X*Y)^T in the output)
        # P1 = (x00+x11)(y00+y11)
        # P2 = (x10+x11)y00
        # P3 = x00(y01-y11)
        # P4 = x11(y10-y00)
        # P5 = (x00+x01)y11
        # P6 = (x10-x00)(y00+y01)
        # P7 = (x01-x11)(y10+y11)
        
        P1 = (add_dc(X[0][0], X[1][1]), add_dc(Y[0][0], Y[1][1]))
        P2 = (add_dc(X[1][0], X[1][1]), Y[0][0])
        P3 = (X[0][0], sub_dc(Y[0][1], Y[1][1]))
        P4 = (X[1][1], sub_dc(Y[1][0], Y[0][0]))
        P5 = (add_dc(X[0][0], X[0][1]), Y[1][1])
        P6 = (sub_dc(X[1][0], X[0][0]), add_dc(Y[0][0], Y[0][1]))
        P7 = (sub_dc(X[0][1], X[1][1]), add_dc(Y[1][0], Y[1][1]))
        
        all_products.extend([P1, P2, P3, P4, P5, P6, P7])
    
    rank = len(all_products)  # should be 49
    
    # Now build U, V, W
    # U[i*m+j, c] = coefficient of A[i,j] in all_products[c][0]
    # V[j*p+k, c] = coefficient of B[j,k] in all_products[c][1]
    # W[k*n+i, c] = coefficient in the output combination
    
    n, m, p = 4, 4, 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    
    for c in range(rank):
        A_comb, B_comb = all_products[c]
        for (typ, i, j), coeff in A_comb.items():
            if typ == 'A':
                U[i * m + j][c] += coeff
        for (typ, i, j), coeff in B_comb.items():
            if typ == 'B':
                V[i * p + j][c] += coeff
    
    # Now build W
    # We need to track how the 7 Strassen products combine at each level
    # 
    # Top-level output:
    #   C00 = T1 + T4 - T5 + T7
    #   C01 = T3 + T5
    #   C10 = T2 + T4
    #   C11 = T1 - T2 + T3 + T6
    #
    # C^T[0:2, 0:2] = C00^T
    # C^T[0:2, 2:4] = C10^T
    # C^T[2:4, 0:2] = C01^T
    # C^T[2:4, 2:4] = C11^T
    #
    # For each T_s, which is a 2x2 matrix (result of X*Y), its transpose is:
    #   (X*Y)^T[0,0] = P1 + P4 - P5 + P7
    #   (X*Y)^T[0,1] = P2 + P4
    #   (X*Y)^T[1,0] = P3 + P5
    #   (X*Y)^T[1,1] = P1 - P2 + P3 + P6
    #
    # So for T_s with sub-products indexed at base = s*7:
    #   P1 at base+0, P2 at base+1, P3 at base+2, P4 at base+3, P5 at base+4, P6 at base+5, P7 at base+6
    #
    # (T_s)^T[0,0] = P1 + P4 - P5 + P7 = (base+0) + (base+3) - (base+4) + (base+6)
    # (T_s)^T[0,1] = P2 + P4 = (base+1) + (base+3)
    # (T_s)^T[1,0] = P3 + P5 = (base+2) + (base+4)
    # (T_s)^T[1,1] = P1 - P2 + P3 + P6 = (base+0) - (base+1) + (base+2) + (base+5)
    
    W = [[0] * rank for _ in range(p * n)]
    
    # Top-level block contributions to C^T
    # C^T is (4,4) with blocks:
    #   C^T[0:2, 0:2] = C00^T = (T1+T4-T5+T7)^T = T1^T + T4^T - T5^T + T7^T
    #   C^T[0:2, 2:4] = C10^T = (T2+T4)^T = T2^T + T4^T
    #   C^T[2:4, 0:2] = C01^T = (T3+T5)^T = T3^T + T5^T
    #   C^T[2:4, 2:4] = C11^T = (T1-T2+T3+T6)^T = T1^T - T2^T + T3^T + T6^T
    
    # For each block position in C^T, and each sub-position within the 2x2 block,
    # we add contributions from the relevant T_s^T entries.
    
    # C^T row k (0..3), col i (0..3): W[k*n + i, c]
    
    # Block (br, bc) in C^T, where br in 0..1, bc in 0..1
    # Block size 2x2, so within block, local row lr in 0..1, local col lc in 0..1
    # Global row k = br*2 + lr, global col i = bc*2 + lc
    
    # For each (br, bc), determine which T_s^T entries contribute and with what signs
    # And for each T_s^T entry (lr, lc), determine which sub-products contribute
    
    # T_s^T contributions to block (br, bc):
    # br=0, bc=0: C00^T = T1^T + T4^T - T5^T + T7^T
    # br=0, bc=1: C10^T = T2^T + T4^T
    # br=1, bc=0: C01^T = T3^T + T5^T
    # br=1, bc=1: C11^T = T1^T - T2^T + T3^T + T6^T
    
    block_contribs = {
        (0, 0): [(0, 1), (3, 1), (4, -1), (6, 1)],  # T1, T4, -T5, T7
        (0, 1): [(1, 1), (3, 1)],                     # T2, T4
        (1, 0): [(2, 1), (4, 1)],                     # T3, T5
        (1, 1): [(0, 1), (1, -1), (2, 1), (5, 1)],   # T1, -T2, T3, T6
    }
    
    # T_s^T[lr, lc] sub-product contributions:
    # [0,0]: P1+P4-P5+P7 -> offsets [0, 3, -4, 6] with coeffs [1, 1, -1, 1]
    # [0,1]: P2+P4 -> offsets [1, 3] with coeffs [1, 1]
    # [1,0]: P3+P5 -> offsets [2, 4] with coeffs [1, 1]
    # [1,1]: P1-P2+P3+P6 -> offsets [0, -1, 2, 5] with coeffs [1, -1, 1, 1]
    
    sub_contribs = {
        (0, 0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0, 1): [(1, 1), (3, 1)],
        (1, 0): [(2, 1), (4, 1)],
        (1, 1): [(0, 1), (1, -1), (2, 1), (5, 1)],
    }
    
    for br in range(2):
        for bc in range(2):
            for lr in range(2):
                for lc in range(2):
                    k = br * 2 + lr  # row in C^T
                    i = bc * 2 + lc  # col in C^T
                    for (s, sign_top) in block_contribs[(br, bc)]:
                        for (offset, sign_sub) in sub_contribs[(lr, lc)]:
                            c = s * 7 + offset
                            W[k * n + i][c] += sign_top * sign_sub
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
