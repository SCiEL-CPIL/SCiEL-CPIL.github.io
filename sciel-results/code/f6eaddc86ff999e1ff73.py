"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-28 exact tensor decomposition for 2x4 * 4x4 matrix multiplication
    using Strassen's algorithm on 2x2 blocks.

    A is 2x4, split as A1 = A[:, 0:2], A2 = A[:, 2:4] (each 2x2).
    B is 4x4, split into 2x2 blocks:
        B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
        B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    C = A1*B11 + A2*B21  (cols 0:2)
      + A1*B12 + A2*B22  (cols 2:4)

    Each 2x2x2x2 sub-problem uses Strassen's 7 multiplications.
    Total rank = 4 * 7 = 28.

    Trace convention: W[k*n + i, r] corresponds to C[i, k].
    """
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28

    # Strassen's 7 products for 2x2 * 2x2:
    # Given X (2x2) and Y (2x2):
    #   X = [[x00, x01], [x10, x11]]
    #   Y = [[y00, y01], [y10, y11]]
    # M1 = (x00 + x11)(y00 + y11)
    # M2 = (x10 + x11) * y00
    # M3 = x00 * (y10 - y11)
    # M4 = x11 * (y01 - y00)
    # M5 = (x00 + x01) * y11
    # M6 = (x10 - x00) * (y00 + y10)
    # M7 = (x01 - x11) * (y10 + y11)
    #
    # Z = X*Y:
    #   z00 = M1 + M4 - M5 + M7
    #   z01 = M3 + M5
    #   z10 = M2 + M4
    #   z11 = M1 - M2 + M3 + M6

    # For each sub-problem, we need to map the Strassen vectors to global indices.
    # Sub-problem s has:
    #   X rows: i in {0,1} -> A row i
    #   X cols: offset_x + j, j in {0,1} -> A col (offset_x + j)
    #   Y rows: offset_y + a, a in {0,1} -> B row (offset_y + a)
    #   Y cols: offset_z + b, b in {0,1} -> B col (offset_z + b)
    #   Output Z rows: i in {0,1} -> C row i
    #   Output Z cols: offset_z + b, b in {0,1} -> C col (offset_z + b)

    # The four sub-problems:
    # SP1: A1*B11 -> C[:, 0:2].  offset_x=0, offset_y=0, offset_z=0
    # SP2: A2*B21 -> C[:, 0:2].  offset_x=2, offset_y=2, offset_z=0
    # SP3: A1*B12 -> C[:, 2:4].  offset_x=0, offset_y=0, offset_z=2
    # SP4: A2*B22 -> C[:, 2:4].  offset_x=2, offset_y=2, offset_z=2

    sub_problems = [
        (0, 0, 0),  # SP1
        (2, 2, 0),  # SP2
        (0, 0, 2),  # SP3
        (2, 2, 2),  # SP4
    ]

    # Strassen vectors for X (2x2, flattened row-major: [x00, x01, x10, x11])
    # and Y (2x2, flattened row-major: [y00, y01, y10, y11])
    # Each M_k has u vector (for X) and v vector (for Y), and w contributions to Z.
    #
    # M1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
    # M2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
    # M3: u=[1,0,0,0], v=[0,0,1,-1], w=[1,1,0,0]
    # M4: u=[0,0,0,1], v=[0,1,-1,0], w=[1,0,0,1]
    # M5: u=[1,1,0,0], v=[0,0,0,1], w=[-1,1,0,0]
    # M6: u=[-1,0,1,0], v=[1,0,1,0], w=[0,0,0,1]
    # M7: u=[0,1,0,-1], v=[0,0,1,1], w=[1,0,0,0]

    strassen = [
        # (u_vec, v_vec, w_vec) where w_vec is [w00, w01, w10, w11]
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),   # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [1, 1, 0, 0]),  # M3
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 0, 1]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
        ([-1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # M7
    ]

    col = 0
    for (ox, oy, oz) in sub_problems:
        for (u_vec, v_vec, w_vec) in strassen:
            # Fill U: A[i, ox+j] for i,j in {0,1}
            for j in range(2):
                for i in range(2):
                    # u_vec is [x00, x01, x10, x11] = [u[0,0], u[0,1], u[1,0], u[1,1]]
                    # Index into u_vec: i*2 + j
                    val = u_vec[i * 2 + j]
                    if val != 0:
                        U[i * m + (ox + j)][col] = val

            # Fill V: B[oy+a, oz+b] for a,b in {0,1}
            for b in range(2):
                for a in range(2):
                    val = v_vec[a * 2 + b]
                    if val != 0:
                        V[(oy + a) * p + (oz + b)][col] = val

            # Fill W: C[i, oz+b] -> W[(oz+b)*n + i, col]
            for b in range(2):
                for i in range(2):
                    val = w_vec[i * 2 + b]
                    if val != 0:
                        W[(oz + b) * n + i][col] = val

            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
