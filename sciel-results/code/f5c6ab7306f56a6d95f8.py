"""
Rank-49 decomposition for 4x4 matrix multiplication via recursive Strassen.

Approach:
- Partition 4x4 matrices into 2x2 blocks of 2x2 submatrices.
- Apply Strassen's 7-multiplication algorithm at the block level.
- Each block multiplication is itself a 2x2 matrix multiplication,
  decomposed using Strassen's 7-multiplication algorithm at the scalar level.
- Total rank = 7 * 7 = 49.

The tensor decomposition is constructed by carefully tracking how each
scalar-level Strassen term maps to the original matrix elements and
the output matrix elements.
"""

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # all 4
    
    rank = 49
    U = [[0] * rank for _ in range(16)]  # 16 x 49
    V = [[0] * rank for _ in range(16)]  # 16 x 49
    W = [[0] * rank for _ in range(16)]  # 16 x 49
    
    # Indexing conventions:
    # U row i*4+j corresponds to A[i,j]
    # V row j*4+k corresponds to B[j,k]
    # W row k*4+i corresponds to C[i,k]
    
    # Block structure:
    # A11: rows 0,1 cols 0,1 -> U rows [0,1,4,5]
    # A12: rows 0,1 cols 2,3 -> U rows [2,3,6,7]
    # A21: rows 2,3 cols 0,1 -> U rows [8,9,12,13]
    # A22: rows 2,3 cols 2,3 -> U rows [10,11,14,15]
    
    # B11: rows 0,1 cols 0,1 -> V rows [0,1,4,5]
    # B12: rows 0,1 cols 2,3 -> V rows [2,3,6,7]
    # B21: rows 2,3 cols 0,1 -> V rows [8,9,12,13]
    # B22: rows 2,3 cols 2,3 -> V rows [10,11,14,15]
    
    # C11: rows 0,1 cols 0,1 -> W rows [0,1,4,5] (k*4+i for k in {0,1}, i in {0,1})
    # C12: rows 0,1 cols 2,3 -> W rows [2,3,6,7]
    # C21: rows 2,3 cols 0,1 -> W rows [8,9,12,13]
    # C22: rows 2,3 cols 2,3 -> W rows [10,11,14,15]
    
    # Strassen 2x2 decomposition terms:
    # For X*Y where X=[[a,b],[c,d]], Y=[[e,f],[g,h]]:
    # t1 = (a+d)(e+h)
    # t2 = (c+d)e
    # t3 = a(f-h)
    # t4 = d(g-e)
    # t5 = (a+b)h
    # t6 = (c-a)(e+f)
    # t7 = (b-d)(g+h)
    #
    # Result:
    # p11 = t1 + t4 - t5 + t7
    # p12 = t3 + t5
    # p21 = t2 + t4
    # p22 = t1 - t2 + t3 + t6
    
    # For each term, define:
    # - x_coeffs: coefficients for [a, b, c, d] (positions 0,1,2,3 in 2x2)
    # - y_coeffs: coefficients for [e, f, g, h]
    # - w_coeffs: coefficients for [p11, p12, p21, p22]
    
    strassen_terms = [
        # (x_coeffs, y_coeffs, w_coeffs)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # t1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # t2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # t3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),   # t4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # t5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # t6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # t7
    ]
    
    # Block-level Strassen:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each block term, we need to define:
    # - How each of the 4 elements of X (the A-side 2x2) is expressed as a linear combo of A elements
    # - How each of the 4 elements of Y (the B-side 2x2) is expressed as a linear combo of B elements
    # - How each of the 4 elements of the result Z maps to C elements with coefficients
    
    # X element positions: [0,0], [0,1], [1,0], [1,1] -> indices 0,1,2,3
    # For each element, we store a list of (A_row, coeff) pairs
    
    # Define the 7 block terms' X and Y element definitions
    
    # Helper to get A element row index
    def a_row(i, j):
        return i * 4 + j
    
    def b_row(j, k):
        return j * 4 + k
    
    def c_row(i, k):
        return k * 4 + i
    
    # Block term definitions:
    # Each entry is a list of 4 elements, each element is a list of (row, coeff)
    
    block_defs = []
    
    # M1: X = A11+A22, Y = B11+B22
    # X[0][0] = A[0,0]+A[2,2], X[0][1] = A[0,1]+A[2,3]
    # X[1][0] = A[1,0]+A[3,2], X[1][1] = A[1,1]+A[3,3]
    # Y[0][0] = B[0,0]+B[2,2], Y[0][1] = B[0,1]+B[2,3]
    # Y[1][0] = B[1,0]+B[3,2], Y[1][1] = B[1,1]+B[3,3]
    # W: C11 += 1, C22 += 1
    X1 = [
        [(a_row(0,0), 1), (a_row(2,2), 1)],
        [(a_row(0,1), 1), (a_row(2,3), 1)],
        [(a_row(1,0), 1), (a_row(3,2), 1)],
        [(a_row(1,1), 1), (a_row(3,3), 1)]
    ]
    Y1 = [
        [(b_row(0,0), 1), (b_row(2,2), 1)],
        [(b_row(0,1), 1), (b_row(2,3), 1)],
        [(b_row(1,0), 1), (b_row(3,2), 1)],
        [(b_row(1,1), 1), (b_row(3,3), 1)]
    ]
    W1 = [
        [(c_row(0,0), 1), (c_row(2,2), 1)],  # p11
        [(c_row(0,1), 1), (c_row(2,3), 1)],  # p12
        [(c_row(1,0), 1), (c_row(3,2), 1)],  # p21
        [(c_row(1,1), 1), (c_row(3,3), 1)]   # p22
    ]
    block_defs.append((X1, Y1, W1))
    
    # M2: X = A21+A22, Y = B11
    # X[0][0] = A[2,0]+A[2,2], X[0][1] = A[2,1]+A[2,3]
    # X[1][0] = A[3,0]+A[3,2], X[1][1] = A[3,1]+A[3,3]
    # Y = B11
    # W: C21 += 1, C22 += -1
    X2 = [
        [(a_row(2,0), 1), (a_row(2,2), 1)],
        [(a_row(2,1), 1), (a_row(2,3), 1)],
        [(a_row(3,0), 1), (a_row(3,2), 1)],
        [(a_row(3,1), 1), (a_row(3,3), 1)]
    ]
    Y2 = [
        [(b_row(0,0), 1)],
        [(b_row(0,1), 1)],
        [(b_row(1,0), 1)],
        [(b_row(1,1), 1)]
    ]
    W2 = [
        [(c_row(2,0), 1), (c_row(2,2), -1)],
        [(c_row(2,1), 1), (c_row(2,3), -1)],
        [(c_row(3,0), 1), (c_row(3,2), -1)],
        [(c_row(3,1), 1), (c_row(3,3), -1)]
    ]
    block_defs.append((X2, Y2, W2))
    
    # M3: X = A11, Y = B12-B22
    # X = A11
    # Y[0][0] = B[0,2]-B[2,2], Y[0][1] = B[0,3]-B[2,3]
    # Y[1][0] = B[1,2]-B[3,2], Y[1][1] = B[1,3]-B[3,3]
    # W: C12 += 1, C22 += 1
    X3 = [
        [(a_row(0,0), 1)],
        [(a_row(0,1), 1)],
        [(a_row(1,0), 1)],
        [(a_row(1,1), 1)]
    ]
    Y3 = [
        [(b_row(0,2), 1), (b_row(2,2), -1)],
        [(b_row(0,3), 1), (b_row(2,3), -1)],
        [(b_row(1,2), 1), (b_row(3,2), -1)],
        [(b_row(1,3), 1), (b_row(3,3), -1)]
    ]
    W3 = [
        [(c_row(0,2), 1), (c_row(2,2), 1)],
        [(c_row(0,3), 1), (c_row(2,3), 1)],
        [(c_row(1,2), 1), (c_row(3,2), 1)],
        [(c_row(1,3), 1), (c_row(3,3), 1)]
    ]
    block_defs.append((X3, Y3, W3))
    
    # M4: X = A22, Y = B21-B11
    # X = A22
    # Y[0][0] = B[2,0]-B[0,0], Y[0][1] = B[2,1]-B[0,1]
    # Y[1][0] = B[3,0]-B[1,0], Y[1][1] = B[3,1]-B[1,1]
    # W: C11 += 1, C21 += 1
    X4 = [
        [(a_row(2,2), 1)],
        [(a_row(2,3), 1)],
        [(a_row(3,2), 1)],
        [(a_row(3,3), 1)]
    ]
    Y4 = [
        [(b_row(2,0), 1), (b_row(0,0), -1)],
        [(b_row(2,1), 1), (b_row(0,1), -1)],
        [(b_row(3,0), 1), (b_row(1,0), -1)],
        [(b_row(3,1), 1), (b_row(1,1), -1)]
    ]
    W4 = [
        [(c_row(0,0), 1), (c_row(2,0), 1)],
        [(c_row(0,1), 1), (c_row(2,1), 1)],
        [(c_row(1,0), 1), (c_row(3,0), 1)],
        [(c_row(1,1), 1), (c_row(3,1), 1)]
    ]
    block_defs.append((X4, Y4, W4))
    
    # M5: X = A11+A12, Y = B22
    # X[0][0] = A[0,0]+A[0,2], X[0][1] = A[0,1]+A[0,3]
    # X[1][0] = A[1,0]+A[1,2], X[1][1] = A[1,1]+A[1,3]
    # Y = B22
    # W: C11 += -1, C12 += 1
    X5 = [
        [(a_row(0,0), 1), (a_row(0,2), 1)],
        [(a_row(0,1), 1), (a_row(0,3), 1)],
        [(a_row(1,0), 1), (a_row(1,2), 1)],
        [(a_row(1,1), 1), (a_row(1,3), 1)]
    ]
    Y5 = [
        [(b_row(2,2), 1)],
        [(b_row(2,3), 1)],
        [(b_row(3,2), 1)],
        [(b_row(3,3), 1)]
    ]
    W5 = [
        [(c_row(0,0), -1), (c_row(0,2), 1)],
        [(c_row(0,1), -1), (c_row(0,3), 1)],
        [(c_row(1,0), -1), (c_row(1,2), 1)],
        [(c_row(1,1), -1), (c_row(1,3), 1)]
    ]
    block_defs.append((X5, Y5, W5))
    
    # M6: X = A21-A11, Y = B11+B12
    # X[0][0] = A[2,0]-A[0,0], X[0][1] = A[2,1]-A[0,1]
    # X[1][0] = A[3,0]-A[1,0], X[1][1] = A[3,1]-A[1,1]
    # Y[0][0] = B[0,0]+B[0,2], Y[0][1] = B[0,1]+B[0,3]
    # Y[1][0] = B[1,0]+B[1,2], Y[1][1] = B[1,1]+B[1,3]
    # W: C22 += 1
    X6 = [
        [(a_row(2,0), 1), (a_row(0,0), -1)],
        [(a_row(2,1), 1), (a_row(0,1), -1)],
        [(a_row(3,0), 1), (a_row(1,0), -1)],
        [(a_row(3,1), 1), (a_row(1,1), -1)]
    ]
    Y6 = [
        [(b_row(0,0), 1), (b_row(0,2), 1)],
        [(b_row(0,1), 1), (b_row(0,3), 1)],
        [(b_row(1,0), 1), (b_row(1,2), 1)],
        [(b_row(1,1), 1), (b_row(1,3), 1)]
    ]
    W6 = [
        [(c_row(2,2), 1)],
        [(c_row(2,3), 1)],
        [(c_row(3,2), 1)],
        [(c_row(3,3), 1)]
    ]
    block_defs.append((X6, Y6, W6))
    
    # M7: X = A12-A22, Y = B21+B22
    # X[0][0] = A[0,2]-A[2,2], X[0][1] = A[0,3]-A[2,3]
    # X[1][0] = A[1,2]-A[3,2], X[1][1] = A[1,3]-A[3,3]
    # Y[0][0] = B[2,0]+B[2,2], Y[0][1] = B[2,1]+B[2,3]
    # Y[1][0] = B[3,0]+B[3,2], Y[1][1] = B[3,1]+B[3,3]
    # W: C11 += 1
    X7 = [
        [(a_row(0,2), 1), (a_row(2,2), -1)],
        [(a_row(0,3), 1), (a_row(2,3), -1)],
        [(a_row(1,2), 1), (a_row(3,2), -1)],
        [(a_row(1,3), 1), (a_row(3,3), -1)]
    ]
    Y7 = [
        [(b_row(2,0), 1), (b_row(2,2), 1)],
        [(b_row(2,1), 1), (b_row(2,3), 1)],
        [(b_row(3,0), 1), (b_row(3,2), 1)],
        [(b_row(3,1), 1), (b_row(3,3), 1)]
    ]
    W7 = [
        [(c_row(0,0), 1)],
        [(c_row(0,1), 1)],
        [(c_row(1,0), 1)],
        [(c_row(1,1), 1)]
    ]
    block_defs.append((X7, Y7, W7))
    
    # Now for each block term (0-6) and each Strassen term (0-6), add the contribution
    for bi in range(7):
        X_def, Y_def, W_def = block_defs[bi]
        base_col = bi * 7
        
        for si in range(7):
            col = base_col + si
            x_coeffs, y_coeffs, w_coeffs = strassen_terms[si]
            
            # For this Strassen term, we have:
            # scalar_product = (sum_i x_coeffs[i] * X_elem[i]) * (sum_j y_coeffs[j] * Y_elem[j])
            # This contributes w_coeffs[k] to Z_elem[k] for each k
            
            # Since each X_elem[i] is a sum of A elements and each Y_elem[j] is a sum of B elements,
            # the product expands to a sum of products of individual A and B elements.
            # Each such product is a rank-1 term in the tensor decomposition.
            
            # For each pair (i, j) where x_coeffs[i] != 0 and y_coeffs[j] != 0:
            for i in range(4):
                if x_coeffs[i] == 0:
                    continue
                for j in range(4):
                    if y_coeffs[j] == 0:
                        continue
                    
                    # X_elem[i] = sum of (a_row, a_coeff) pairs
                    # Y_elem[j] = sum of (b_row, b_coeff) pairs
                    # The product X_elem[i] * Y_elem[j] = sum over all (a_row, b_row) pairs
                    # of (a_coeff * b_coeff) * A[a_row] * B[b_row]
                    
                    for a_idx, a_coeff in X_def[i]:
                        for b_idx, b_coeff in Y_def[j]:
                            # This product contributes to the scalar term with coefficient
                            # x_coeffs[i] * y_coeffs[j] * a_coeff * b_coeff
                            scalar_coeff = x_coeffs[i] * y_coeffs[j] * a_coeff * b_coeff
                            
                            # This scalar term contributes to Z elements with w_coeffs
                            for k in range(4):
                                if w_coeffs[k] == 0:
                                    continue
                                # Z_elem[k] gets contribution scalar_coeff * w_coeffs[k]
                                # Z_elem[k] maps to C elements via W_def[k]
                                for c_idx, c_coeff in W_def[k]:
                                    total_coeff = scalar_coeff * w_coeffs[k] * c_coeff
                                    if total_coeff != 0:
                                        U[a_idx][col] += total_coeff
                                        V[b_idx][col] += total_coeff
                                        W[c_idx][col] += total_coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
