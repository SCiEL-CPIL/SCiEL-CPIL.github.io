"""
Matrix multiplication 2x3x8 decomposition.

Strategy: Split the 3x8 multiplication into four 3x2 blocks (columns 0-1, 2-3, 4-5, 6-7).
For each 3x2 block, split A (2x3) into:
  - A12 = A[:, 0:2] (2x2) -> use Strassen (rank 7) with the 2x2 block
  - A3 = A[:, 2] (2x1) -> use direct multiplication (rank 4: 2 rows * 2 cols)

Total rank = 4 * (7 + 4) = 44.

Indexing convention:
  U rows: i*m + j where i in [0,1], j in [0,2] -> i*3 + j
  V rows: j*p + k where j in [0,2], k in [0,7] -> j*8 + k
  W rows: k*n + i where k in [0,7], i in [0,1] -> k*2 + i

Strassen for 2x2x2:
  M1 = (a00 + a11)(b00 + b11)
  M2 = (a10 + a11)b00
  M3 = a00(b10 - b11)
  M4 = a11(b01 - b00)
  M5 = (a00 + a01)b11
  M6 = (a10 - a00)(b00 + b01)
  M7 = (a01 - a10)(b10 + b11)
  c00 = M1 + M4 - M5 + M7
  c01 = M3 + M5
  c10 = M2 + M4
  c11 = M1 - M2 + M3 + M6
"""
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    # Rank = 4 blocks * (7 Strassen + 4 direct) = 44
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44

    # Strassen coefficients: 7 products
    # For A = [[a00, a01], [a10, a11]] and B = [[b00, b01], [b10, b11]]
    # Each product r corresponds to:
    #   U coefficients on A entries, V coefficients on B entries
    #   W coefficients on output C entries
    #
    # We encode each rank-1 term as a triple (u_vec, v_vec, w_vec)
    # where u_vec has 4 entries (a00, a01, a10, a11),
    #       v_vec has 4 entries (b00, b01, b10, b11),
    #       w_vec has 4 entries (c00, c01, c10, c11)

    strassen_terms = [
        # (u_vec, v_vec, w_vec)
        # M1 = (a00+a11)(b00+b11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # contributes to c00 and c11
        # Wait, we need to split each Strassen product into rank-1 terms properly.
        # Actually, each M_r is a single product of a linear form in A and a linear form in B.
        # The output c_ij is a linear combination of M_r's.
        # So each rank-1 term for the whole 2x2x2 multiplication is:
        #   For each r in 1..7: the product (linear_form_A_r) * (linear_form_B_r)
        #   Then c_ij = sum_r alpha_{ij,r} * M_r
        # This gives 7 rank-1 terms where:
        #   U vector = linear_form_A_r (on the 4 A entries)
        #   V vector = linear_form_B_r (on the 4 B entries)
        #   W vector = (alpha_{00,r}, alpha_{01,r}, alpha_{10,r}, alpha_{11,r})
    ]

    # Strassen rank-1 terms:
    # M1 = (a00+a11)(b00+b11): u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1] (c00+=1, c11+=1)
    # M2 = (a10+a11)(b00):     u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]  (c10+=1)
    # M3 = a00(b10-b11):       u=[1,0,0,0], v=[0,0,1,-1], w=[0,1,0,1] (c01+=1, c11+=1)
    # M4 = a11(b01-b00):       u=[0,0,0,1], v=[-1,1,0,0], w=[1,0,1,0] (c00+=1, c10+=1)
    # M5 = (a00+a01)(b11):     u=[1,1,0,0], v=[0,0,0,1], w=[-1,1,0,0] (c00-=1, c01+=1)
    # M6 = (a10-a00)(b00+b01): u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,0,1] (c11+=1)
    # M7 = (a01-a10)(b10+b11): u=[0,1,-1,0], v=[0,0,1,1], w=[1,0,0,0] (c00+=1)

    strassen_r1 = [
        ([1, 0, 0, 1],   [1, 0, 0, 1],   [1, 0, 0, 1]),
        ([0, 0, 1, 1],   [1, 0, 0, 0],   [0, 0, 1, 0]),
        ([1, 0, 0, 0],   [0, 0, 1, -1],  [0, 1, 0, 1]),
        ([0, 0, 0, 1],   [-1, 1, 0, 0],  [1, 0, 1, 0]),
        ([1, 1, 0, 0],   [0, 0, 0, 1],   [-1, 1, 0, 0]),
        ([-1, 0, 1, 0],  [1, 1, 0, 0],   [0, 0, 0, 1]),
        ([0, 1, -1, 0],  [0, 0, 1, 1],   [1, 0, 0, 0]),
    ]

    # Direct multiplication for A3 (2x1) * B3 (1x2): 4 rank-1 terms
    # C = [[a00*b00, a00*b01], [a10*b00, a10*b01]]
    direct_r1 = [
        # c00 = a00*b00
        ([1, 0], [1, 0], [1, 0, 0, 0]),
        # c01 = a00*b01
        ([1, 0], [0, 1], [0, 1, 0, 0]),
        # c10 = a10*b00
        ([0, 1], [1, 0], [0, 0, 1, 0]),
        # c11 = a10*b01
        ([0, 1], [0, 1], [0, 0, 0, 1]),
    ]

    # For each of the 4 column blocks (cols 0-1, 2-3, 4-5, 6-7):
    col_blocks = [0, 2, 4, 6]
    col = 0

    for cb in col_blocks:
        # Strassen part: A12 (cols 0,1 of A) * B_block (rows 0,1; cols cb, cb+1)
        for (u_vec, v_vec, w_vec) in strassen_r1:
            # u_vec is on A entries (a00, a01, a10, a11) = (A[0,0], A[0,1], A[1,0], A[1,1])
            # Map to U rows: i*3+j
            # a00 -> U[0*3+0] = U[0]
            # a01 -> U[0*3+1] = U[1]
            # a10 -> U[1*3+0] = U[3]
            # a11 -> U[1*3+1] = U[4]
            U[0][col] = u_vec[0]
            U[1][col] = u_vec[1]
            U[3][col] = u_vec[2]
            U[4][col] = u_vec[3]

            # v_vec is on B entries (b00, b01, b10, b11) = (B[0,cb], B[0,cb+1], B[1,cb], B[1,cb+1])
            # Map to V rows: j*8+k
            # b00 -> V[0*8+cb] = V[cb]
            # b01 -> V[0*8+cb+1] = V[cb+1]
            # b10 -> V[1*8+cb] = V[8+cb]
            # b11 -> V[1*8+cb+1] = V[8+cb+1]
            V[cb][col] = v_vec[0]
            V[cb+1][col] = v_vec[1]
            V[8+cb][col] = v_vec[2]
            V[8+cb+1][col] = v_vec[3]

            # w_vec is on C entries (c00, c01, c10, c11)
            # c00 -> C[0, cb] -> W row = cb*2 + 0
            # c01 -> C[0, cb+1] -> W row = (cb+1)*2 + 0
            # c10 -> C[1, cb] -> W row = cb*2 + 1
            # c11 -> C[1, cb+1] -> W row = (cb+1)*2 + 1
            W[cb*2 + 0][col] = w_vec[0]
            W[(cb+1)*2 + 0][col] = w_vec[1]
            W[cb*2 + 1][col] = w_vec[2]
            W[(cb+1)*2 + 1][col] = w_vec[3]

            col += 1

        # Direct part: A3 (col 2 of A) * B_row2 (row 2 of B, cols cb, cb+1)
        for (u_vec, v_vec, w_vec) in direct_r1:
            # u_vec on A entries (a00, a10) = (A[0,2], A[1,2])
            U[0*3+2][col] = u_vec[0]  # U[2]
            U[1*3+2][col] = u_vec[1]  # U[5]

            # v_vec on B entries (b00, b01) = (B[2,cb], B[2,cb+1])
            V[2*8+cb][col] = v_vec[0]      # V[16+cb]
            V[2*8+cb+1][col] = v_vec[1]    # V[16+cb+1]

            # w_vec on C entries (c00, c01, c10, c11)
            W[cb*2 + 0][col] = w_vec[0]
            W[(cb+1)*2 + 0][col] = w_vec[1]
            W[cb*2 + 1][col] = w_vec[2]
            W[(cb+1)*2 + 1][col] = w_vec[3]

            col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}

# Quick self-test if run directly
if __name__ == "__main__":
    import numpy as np
    res = construct()
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    # Verify: T[i,j,k] = sum_r U[i*3+j, r] * V[j*8+k, r] * W[k*2+i, r]
    # True tensor: T[i,j,k] = 1 if valid indices else 0 (matrix mult tensor)
    # Actually the matrix multiplication tensor T[i,j,k] = A[i,j]*B[j,k]
    # Check that for random A, B, the reconstruction matches
    A = np.random.randn(2, 3)
    B = np.random.randn(3, 8)
    C_true = A @ B
    C_rec = np.zeros((2, 8))
    for i in range(2):
        for k in range(8):
            s = 0.0
            for r in range(U.shape[1]):
                for j in range(3):
                    s += U[i*3+j, r] * V[j*8+k, r] * W[k*2+i, r]
            C_rec[i, k] = s
    print("Max error:", np.max(np.abs(C_true - C_rec)))
    print("Rank:", U.shape[1])
