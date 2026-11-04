"""Hybrid rank-88 decomposition for 3x4 * 4x8 matrix multiplication.

Strategy: partition the 3x4 matrix into a 2x4 top block (A_top) and a 1x4
bottom row (A_bot).  Multiply A_top (2x4) by B (4x8) using a Strassen-style
7-product decomposition, and multiply A_bot (1x4) by B with 4 schoolbook
products.  Total rank = 7 + 4 = 88.

Index conventions (trace convention for W):
  U row  = i*m + j          (i in [0,n), j in [0,m))
  V row  = j*p + k          (j in [0,m), k in [0,p))
  W row  = k*n + i          (k in [0,p), i in [0,n))
A[i,j] * B[j,k] -> C[i,k], and W[k*n+i, c] accumulates the contribution.
"""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U = [[0] * 88 for _ in range(n * m)]
    V = [[0] * 88 for _ in range(m * p)]
    W = [[0] * 88 for _ in range(p * n)]

    # --- Strassen block for the top 2 rows (i = 0, 1), 4x4 sub-block ---
    # Splits: A_top[0..1, 0..3] into 2x2 blocks (a11,a12,a21,a22)
    #         B[:, 0..3] into 2x2 blocks (b11,b12,b21,b22)
    # Strassen products (7) with standard signs:
    #   P1 = (a11+a22)(b11+b22)
    #   P2 = (a21+a22)b11
    #   P3 = a11(b12-b22)
    #   P4 = a22(b21-b11)
    #   P5 = (a11+a12)b22
    #   P6 = (a21-a11)(b11+b12)
    #   P7 = (a12-a22)(b21+b22)
    # Output:
    #   c11 = P1+P4-P5+P7
    #   c12 = P3+P5
    #   c21 = P2+P4
    #   c22 = P1-P2+P3+P6
    #
    # We express each product as a sum of (U-row, V-row) pairs with coefficients,
    # and each output entry as a sum of products with coefficients.

    # Define the 7 Strassen products as lists of (coeff, U_row, V_row)
    # U_row = i*4 + j, V_row = j*8 + k
    # Product s is computed by: sum over (U[i,j] * V[j,k]) for listed (i,j,k)
    strassen_products = [
        # P1 = (a11+a22)(b11+b22)
        [(1, 0, 0, 0), (1, 0, 1, 1), (1, 1, 0, 0), (1, 1, 1, 1)],
        # P2 = (a21+a22)b11
        [(1, 1, 0, 0), (1, 1, 1, 0)],
        # P3 = a11(b12-b22)
        [(1, 0, 0, 1), (-1, 0, 1, 1)],
        # P4 = a22(b21-b11)
        [(1, 1, 1, 1), (-1, 1, 0, 1)],
        # P5 = (a11+a12)b22
        [(1, 0, 1, 1), (1, 0, 0, 1)],
        # P6 = (a21-a11)(b11+b12)
        [(1, 1, 0, 0), (1, 1, 0, 1), (-1, 0, 0, 0), (-1, 0, 0, 1)],
        # P7 = (a12-a22)(b21+b22)
        [(1, 0, 1, 1), (1, 0, 1, 0), (-1, 1, 1, 1), (-1, 1, 1, 0)],
    ]

    # Output combinations: c[i_out, k_out] = sum of (coeff, product_index)
    # i_out in {0,1}, k_out in {0,1,2,3}
    # c11 (i=0,k=0): P1+P4-P5+P7
    # c12 (i=0,k=1): P3+P5
    # c21 (i=1,k=0): P2+P4
    # c22 (i=1,k=1): P1-P2+P3+P6
    output_combos = {
        (0, 0): [(1, 0), (1, 3), (-1, 4), (1, 6)],
        (0, 1): [(1, 2), (1, 4)],
        (1, 0): [(1, 1), (1, 3)],
        (1, 1): [(1, 0), (-1, 1), (1, 2), (1, 5)],
    }

    col = 0
    # Build Strassen columns
    for s in range(7):
        # U and V entries
        for (coeff, i, j, k) in strassen_products[s]:
            U[i * m + j][col] += coeff
            V[j * p + k][col] += coeff
        # W entries: for each output (i_out, k_out), add coeff * (product contribution)
        # W[k_out * n + i_out, col] += coeff
        for (i_out, k_out), terms in output_combos.items():
            for (coeff, prod_idx) in terms:
                if prod_idx == s:
                    W[k_out * n + i_out][col] += coeff
        col += 1

    # --- Schoolbook block for the bottom row (i = 2), 4x8 ---
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
