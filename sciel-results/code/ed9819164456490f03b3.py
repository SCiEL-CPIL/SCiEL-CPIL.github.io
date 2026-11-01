"""Matrix multiplication 2,2,14 — rank-minimal exact decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = t0 + 145  # safety margin before 150 s kill

    n, m, p = DIMENSIONS
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # W[k*n+i, c]  (official trace convention)

    # Strategy: exploit the 2x2 left matrix via Strassen on 2x2 blocks.
    # Split the right 2x14 matrix into 7 blocks of 2x2.
    # Strassen gives 7 multiplications per 2x2 block pair.
    # So we need 7 * 7 = 49 scalar multiplications.
    #
    # For each of the 7 blocks b (covering columns 2b, 2b+1 of the right matrix):
    #   A is 2x2, B_b is 2x2
    #   Strassen:
    #     M1 = (A00+A11)*(B00+B11)
    #     M2 = (A10+A11)*B00
    #     M3 = A00*(B01-B11)
    #     M4 = A11*(B10-B00)
    #     M5 = (A00+A01)*B11
    #     M6 = (A10-A00)*(B00+B01)
    #     M7 = (A01-A11)*(B10+B11)
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #
    # Total rank = 7 blocks * 7 Strassen factors = 49

    num_blocks = p // 2  # 7
    factors_per_block = 7
    rank = num_blocks * factors_per_block  # 49

    U = [[0.0] * rank for _ in range(n * m)]   # 4 rows
    V = [[0.0] * rank for _ in range(m * p)]   # 28 rows
    W = [[0.0] * rank for _ in range(p * n)]   # 28 rows

    col = 0
    for b in range(num_blocks):
        # For block b, the right matrix columns are 2b and 2b+1
        # V rows: row j*p + k for j in {0,1}, k in {2b, 2b+1}
        # i.e., rows 0*14+2b, 0*14+2b+1, 1*14+2b, 1*14+2b+1

        # Strassen factor 1: M1 = (A00+A11)*(B00+B11)
        # U: A00 at row 0, A11 at row 3
        # V: B00 at row 2b, B11 at row 14+2b+1
        U[0][col] = 1; U[3][col] = 1
        V[2*b][col] = 1; V[14+2*b+1][col] = 1
        # W: contributes to C00 (+1) and C11 (+1)
        # C00 is at output index k*n+i for k=0,i=0 -> row 0
        # C11 is at output index k*n+i for k=1,i=1 -> row 1*2+1 = 3
        W[0][col] = 1
        W[3][col] = 1
        col += 1

        # Strassen factor 2: M2 = (A10+A11)*B00
        # U: A10 at row 2, A11 at row 3
        # V: B00 at row 2b
        U[2][col] = 1; U[3][col] = 1
        V[2*b][col] = 1
        # W: contributes to C10 (+1) and C11 (-1)
        # C10 is at k*n+i for k=0,i=1 -> row 1
        # C11 is at row 3
        W[1][col] = 1
        W[3][col] = -1
        col += 1

        # Strassen factor 3: M3 = A00*(B01-B11)
        # U: A00 at row 0
        # V: B01 at row 2b+1, B11 at row 14+2b+1
        U[0][col] = 1
        V[2*b+1][col] = 1; V[14+2*b+1][col] = -1
        # W: contributes to C01 (+1) and C11 (+1)
        # C01 is at k*n+i for k=1,i=0 -> row 2
        # C11 is at row 3
        W[2][col] = 1
        W[3][col] = 1
        col += 1

        # Strassen factor 4: M4 = A11*(B10-B00)
        # U: A11 at row 3
        # V: B10 at row 14+2b, B00 at row 2b
        U[3][col] = 1
        V[14+2*b][col] = 1; V[2*b][col] = -1
        # W: contributes to C00 (+1) and C10 (+1)
        # C00 is at row 0, C10 is at row 1
        W[0][col] = 1
        W[1][col] = 1
        col += 1

        # Strassen factor 5: M5 = (A00+A01)*B11
        # U: A00 at row 0, A01 at row 1
        # V: B11 at row 14+2b+1
        U[0][col] = 1; U[1][col] = 1
        V[14+2*b+1][col] = 1
        # W: contributes to C00 (-1) and C01 (+1)
        # C00 is at row 0, C01 is at row 2
        W[0][col] = -1
        W[2][col] = 1
        col += 1

        # Strassen factor 6: M6 = (A10-A00)*(B00+B01)
        # U: A10 at row 2, A00 at row 0
        # V: B00 at row 2b, B01 at row 2b+1
        U[2][col] = 1; U[0][col] = -1
        V[2*b][col] = 1; V[2*b+1][col] = 1
        # W: contributes to C11 (+1)
        W[3][col] = 1
        col += 1

        # Strassen factor 7: M7 = (A01-A11)*(B10+B11)
        # U: A01 at row 1, A11 at row 3
        # V: B10 at row 14+2b, B11 at row 14+2b+1
        U[1][col] = 1; U[3][col] = -1
        V[14+2*b][col] = 1; V[14+2*b+1][col] = 1
        # W: contributes to C00 (+1)
        W[0][col] = 1
        col += 1

    # Verify with random test
    if time.time() < DEADLINE - 5:
        try:
            A = np.random.randint(-5, 6, size=(n, m)).astype(float)
            B = np.random.randint(-5, 6, size=(m, p)).astype(float)
            C_expected = A @ B

            U_arr = np.array(U)
            V_arr = np.array(V)
            W_arr = np.array(W)

            # Reconstruct C: C[i,k] = sum_c U[i,k,c] * V[*,c] * W[k,n+i,c]
            # U has shape (n*m, rank): U[i*m + j, c] = coefficient for A[i,j]
            # V has shape (m*p, rank): V[j*p + k, c] = coefficient for B[j,k]
            # W has shape (p*n, rank): W[k*n + i, c] = coefficient for C[i,k]

            C_reconstructed = np.zeros((n, p), dtype=float)
            for c in range(rank):
                # Extract A_part: n x m
                A_part = np.zeros((n, m))
                for i in range(n):
                    for j in range(m):
                        A_part[i, j] = U_arr[i * m + j, c]
                # Extract B_part: m x p
                B_part = np.zeros((m, p))
                for j in range(m):
                    for k in range(p):
                        B_part[j, k] = V_arr[j * p + k, c]
                # Product
                P = A_part @ B_part
                # Add to C_reconstructed
                for k in range(p):
                    for i in range(n):
                        C_reconstructed[i, k] += W_arr[k * n + i, c] * P[i, k]

            if not np.allclose(C_reconstructed, C_expected, atol=1e-10):
                # Fallback to schoolbook
                rank_fb = n * m * p
                U_fb = [[0] * rank_fb for _ in range(n * m)]
                V_fb = [[0] * rank_fb for _ in range(m * p)]
                W_fb = [[0] * rank_fb for _ in range(p * n)]
                column = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_fb[i * m + j][column] = 1
                            V_fb[j * p + k][column] = 1
                            W_fb[k * n + i][column] = 1
                            column += 1
                return {"U": U_fb, "V": V_fb, "W": W_fb}
        except Exception:
            pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
