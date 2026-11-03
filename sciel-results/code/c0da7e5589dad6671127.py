"""Matrix multiplication 2x4*4x8 rank-minimal decomposition."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    # We want to compute C = A @ B where A is n x m (2x4) and B is m x p (4x8)
    # Output C is n x p (2x8)
    # Tensor decomposition: C[i,k] = sum_j A[i,j] * B[j,k]
    # We can view this as a bilinear map.
    # The schoolbook rank is n*m*p = 2*4*8 = 64.
    #
    # Strategy: Pad to 4x4 * 4x4 and use Strassen rank 32?
    # Actually, let's think about the structure.
    # A is 2x4. We can split A into two 2x2 blocks: A = [A1, A2] where A1, A2 are 2x2.
    # B is 4x8. We can split B into four 2x2 blocks vertically? No, B is 4x8.
    # B = [B1; B2] where B1 is 2x8 and B2 is 2x8.
    # Then C = A1 @ B1 + A2 @ B2.
    # Each term is (2x2) @ (2x8).
    # A (2x2) @ B (2x8) can be computed by splitting B into four 2x2 blocks: B = [B11, B12, B13, B14]
    # Then A @ B = [A@B11, A@B12, A@B13, A@B14].
    # Each A@Bij is a 2x2 matrix product.
    # So total we have 2 (from A1, A2) * 4 (from B blocks) = 8 independent 2x2 matrix multiplications.
    # The rank of 2x2 matrix multiplication is known to be 7 (Strassen).
    # So total rank = 8 * 7 = 56.
    #
    # Let's verify this decomposition.
    # A is 2x4. Let A = [A_left, A_right] where A_left, A_right are 2x2.
    # B is 4x8. Let B = [B_top; B_bottom] where B_top, B_bottom are 2x8.
    # C = A @ B = A_left @ B_top + A_right @ B_bottom.
    # Now B_top is 2x8. Split into four 2x2 blocks: B_top = [B11, B12, B13, B14].
    # Similarly B_bottom = [B21, B22, B23, B24].
    # C = A_left @ [B11, B12, B13, B14] + A_right @ [B21, B22, B23, B24]
    #   = [A_left@B11 + A_right@B21, A_left@B12 + A_right@B22, A_left@B13 + A_right@B23, A_left@B14 + A_right@B24]
    # Each entry is a sum of two 2x2 matrix products.
    # We can compute each 2x2 matrix product using Strassen's algorithm (rank 7).
    # So for each of the 4 output blocks (each 2x2), we need:
    #   A_left @ B1j + A_right @ B2j
    # This is two separate 2x2 multiplications, each of rank 7, so total rank 14 per output block.
    # 4 output blocks * 14 = 56.
    #
    # Wait, is it possible to do better?
    # The tensor is M_{2,4,8}. The rank is known to be at most 56 via this decomposition.
    # Is the rank exactly 56? Or can it be lower?
    # The rank of M_{n,m,p} is at least max(n*m, m*p, n*p) = max(8, 32, 16) = 32.
    # Also, it's known that rank(M_{n,m,p}) <= n*m*p.
    # For 2x2x2, rank is 7.
    # For 2x4x8, we can use the block structure.
    #
    # Let's implement the rank-56 decomposition.
    #
    # Strassen's algorithm for 2x2 matrix multiplication:
    # Given X, Y (2x2 matrices), compute Z = X @ Y.
    # X = [[x11, x12], [x21, x22]], Y = [[y11, y12], [y21, y22]]
    # Z = [[z11, z12], [z21, z22]]
    #
    # Strassen's 7 products:
    # M1 = (x11 + x22) * (y11 + y22)
    # M2 = (x21 + x22) * y11
    # M3 = x11 * (y12 - y22)
    # M4 = x22 * (y21 - y11)
    # M5 = (x11 + x12) * y22
    # M6 = (x21 - x11) * (y11 + y12)
    # M7 = (x12 - x22) * (y21 + y22)
    #
    # z11 = M1 + M4 - M5 + M7
    # z12 = M3 + M5
    # z21 = M2 + M4
    # z22 = M1 - M2 + M3 + M6
    #
    # Now, we need to map this to the tensor decomposition format.
    # The tensor T has indices (i, j, k) where i in {0,1} (row of A), j in {0,1,2,3} (col of A / row of B), k in {0,...,7} (col of B).
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] appears in C[i,k].
    #
    # In our decomposition, we have 8 independent 2x2 multiplications.
    # Let's label them by (a, b) where a in {0,1} (which A block: left or right) and b in {0,1,2,3} (which B block column).
    # For a=0 (A_left), the A block is A[0:2, 0:2] and the B block is B_top[0:2, b*2:(b+1)*2].
    # For a=1 (A_right), the A block is A[0:2, 2:4] and the B block is B_bottom[0:2, b*2:(b+1)*2].
    #
    # Wait, let me re-index.
    # A is 2x4. Rows 0,1. Cols 0,1,2,3.
    # A_left = A[:, 0:2], A_right = A[:, 2:4].
    # B is 4x8. Rows 0,1,2,3. Cols 0..7.
    # B_top = B[0:2, :], B_bottom = B[2:4, :].
    # B_top has rows 0,1. B_bottom has rows 2,3.
    #
    # C = A_left @ B_top + A_right @ B_bottom.
    #
    # For the (a, b) subproblem:
    #   X = A block (2x2), Y = B block (2x2).
    #   For a=0: X = A[:, 0:2] (rows 0,1 of A, cols 0,1), Y = B[0:2, b*2:(b+1)*2] (rows 0,1 of B, cols b*2..b*2+1).
    #   For a=1: X = A[:, 2:4] (rows 0,1 of A, cols 2,3), Y = B[2:4, b*2:(b+1)*2] (rows 2,3 of B, cols b*2..b*2+1).
    #
    # The result Z = X @ Y contributes to C[:, b*2:(b+1)*2].
    # Specifically, Z[0,0] -> C[0, b*2], Z[0,1] -> C[0, b*2+1], Z[1,0] -> C[1, b*2], Z[1,1] -> C[1, b*2+1].
    #
    # Now, for each (a, b), we use Strassen's 7 products.
    # Total rank = 8 * 7 = 56.
    #
    # Let's build the U, V, W matrices.
    # U has shape (n*m, rank) = (8, 56).
    # V has shape (m*p, rank) = (32, 56).
    # W has shape (p*n, rank) = (16, 56).
    #
    # The convention is W[k*n + i, r] where i is the row of C (0,1), k is the col of C (0..7).
    # So W index is k*2 + i.
    #
    # U index is i*m + j where i is row of A (0,1), j is col of A (0..3).
    # V index is j*p + k where j is row of B (0..3), k is col of B (0..7).
    #
    # For each (a, b) and each Strassen product s (0..6), we have a rank-1 term.
    # The Strassen product s computes a linear combination of X entries times a linear combination of Y entries.
    #
    # Let's define the Strassen coefficients.
    # For X = [[x11, x12], [x21, x22]], Y = [[y11, y12], [y21, y22]]:
    #
    # M1 = (x11 + x22) * (y11 + y22)
    #   X coeffs: x11:1, x22:1. Y coeffs: y11:1, y22:1.
    #   Contributes to: z11 (+1), z22 (+1).
    #
    # M2 = (x21 + x22) * y11
    #   X coeffs: x21:1, x22:1. Y coeffs: y11:1.
    #   Contributes to: z21 (+1), z22 (-1).
    #
    # M3 = x11 * (y12 - y22)
    #   X coeffs: x11:1. Y coeffs: y12:1, y22:-1.
    #   Contributes to: z11 (-1), z12 (+1), z22 (+1).
    #
    # M4 = x22 * (y21 - y11)
    #   X coeffs: x22:1. Y coeffs: y21:1, y11:-1.
    #   Contributes to: z11 (+1), z21 (+1).
    #
    # M5 = (x11 + x12) * y22
    #   X coeffs: x11:1, x12:1. Y coeffs: y22:1.
    #   Contributes to: z11 (-1), z12 (+1).
    #
    # M6 = (x21 - x11) * (y11 + y12)
    #   X coeffs: x21:1, x11:-1. Y coeffs: y11:1, y12:1.
    #   Contributes to: z22 (+1).
    #
    # M7 = (x12 - x22) * (y21 + y22)
    #   X coeffs: x12:1, x22:-1. Y coeffs: y21:1, y22:1.
    #   Contributes to: z11 (+1).
    #
    # Let's double-check the Strassen formulas:
    # z11 = M1 + M4 - M5 + M7
    # z12 = M3 + M5
    # z21 = M2 + M4
    # z22 = M1 - M2 + M3 + M6
    #
    # Yes, that matches.
    #
    # Now, for each (a, b), we map the X and Y entries to the global A and B indices.
    #
    # For a=0 (A_left):
    #   X[0,0] = A[0,0], X[0,1] = A[0,1], X[1,0] = A[1,0], X[1,1] = A[1,1]
    #   U indices: i*m + j.
    #   X[0,0] -> U[0*4+0] = U[0]
    #   X[0,1] -> U[0*4+1] = U[1]
    #   X[1,0] -> U[1*4+0] = U[4]
    #   X[1,1] -> U[1*4+1] = U[5]
    #
    # For a=1 (A_right):
    #   X[0,0] = A[0,2], X[0,1] = A[0,3], X[1,0] = A[1,2], X[1,1] = A[1,3]
    #   U indices:
    #   X[0,0] -> U[0*4+2] = U[2]
    #   X[0,1] -> U[0*4+3] = U[3]
    #   X[1,0] -> U[1*4+2] = U[6]
    #   X[1,1] -> U[1*4+3] = U[7]
    #
    # For b=0,1,2,3:
    #   For a=0 (B_top):
    #     Y[0,0] = B[0, b*2], Y[0,1] = B[0, b*2+1], Y[1,0] = B[1, b*2], Y[1,1] = B[1, b*2+1]
    #     V indices: j*p + k.
    #     Y[0,0] -> V[0*8 + b*2]
    #     Y[0,1] -> V[0*8 + b*2+1]
    #     Y[1,0] -> V[1*8 + b*2]
    #     Y[1,1] -> V[1*8 + b*2+1]
    #
    #   For a=1 (B_bottom):
    #     Y[0,0] = B[2, b*2], Y[0,1] = B[2, b*2+1], Y[1,0] = B[3, b*2], Y[1,1] = B[3, b*2+1]
    #     V indices:
    #     Y[0,0] -> V[2*8 + b*2]
    #     Y[0,1] -> V[2*8 + b*2+1]
    #     Y[1,0] -> V[3*8 + b*2]
    #     Y[1,1] -> V[3*8 + b*2+1]
    #
    # The output Z[0,0] -> C[0, b*2] -> W index: k*n + i = b*2*2 + 0 = b*4
    # Z[0,1] -> C[0, b*2+1] -> W index: (b*2+1)*2 + 0 = b*4 + 2
    # Z[1,0] -> C[1, b*2] -> W index: b*2*2 + 1 = b*4 + 1
    # Z[1,1] -> C[1, b*2+1] -> W index: (b*2+1)*2 + 1 = b*4 + 3
    #
    # Now, let's build the U, V, W matrices.
    rank = 56
    U = np.zeros((8, rank))
    V = np.zeros((32, rank))
    W = np.zeros((16, rank))
    #
    # Define the Strassen products.
    # Each product is a tuple: (x_coeffs, y_coeffs, z_coeffs)
    # x_coeffs: dict mapping (row, col) of X to coefficient
    # y_coeffs: dict mapping (row, col) of Y to coefficient
    # z_coeffs: dict mapping (row, col) of Z to coefficient
    #
    strassen_products = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): -1, (0,1): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    #
    col = 0
    for a in range(2):  # 0: A_left, 1: A_right
        for b in range(4):  # 0,1,2,3
            for s in range(7):
                x_coeffs, y_coeffs, z_coeffs = strassen_products[s]
                # Fill U
                for (xi, xj), coeff in x_coeffs.items():
                    if a == 0:
                        # X[0,0]->U[0], X[0,1]->U[1], X[1,0]->U[4], X[1,1]->U[5]
                        u_idx = xi * 4 + xj
                    else:
                        # X[0,0]->U[2], X[0,1]->U[3], X[1,0]->U[6], X[1,1]->U[7]
                        u_idx = xi * 4 + (xj + 2)
                    U[u_idx, col] += coeff
                # Fill V
                for (yi, yj), coeff in y_coeffs.items():
                    if a == 0:
                        # Y[0,0]->V[0*8+b*2], Y[0,1]->V[0*8+b*2+1], Y[1,0]->V[1*8+b*2], Y[1,1]->V[1*8+b*2+1]
                        v_idx = yi * 8 + (b * 2 + yj)
                    else:
                        # Y[0,0]->V[2*8+b*2], etc.
                        v_idx = (yi + 2) * 8 + (b * 2 + yj)
                    V[v_idx, col] += coeff
                # Fill W
                for (zi, zj), coeff in z_coeffs.items():
                    # Z[0,0]->W[b*4], Z[0,1]->W[b*4+2], Z[1,0]->W[b*4+1], Z[1,1]->W[b*4+3]
                    w_idx = b * 4 + zj * 2 + zi
                    W[w_idx, col] += coeff
                col += 1
    #
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    #
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
