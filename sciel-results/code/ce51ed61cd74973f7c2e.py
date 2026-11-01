"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin from 150s limit

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 24

    def build_schoolbook():
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    def verify_full(U, V, W):
        """Full verification: check all valid triples give 1, all invalid give 0."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)

        # Check all valid triples (i, j, k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                    if abs(val - 1.0) > 1e-8:
                        return False

        # Check all invalid triples
        for alpha in range(n * m):
            for beta in range(m * p):
                for gamma in range(p * n):
                    i_a = alpha // m
                    j_a = alpha % m
                    j_b = beta // p
                    k_b = beta % p
                    i_c = gamma % n
                    k_c = gamma // n

                    if j_a == j_b and i_a == i_c and k_b == k_c:
                        continue  # valid triple, already checked

                    val = np.sum(U[alpha] * V[beta] * W[gamma])
                    if abs(val) > 1e-8:
                        return False

        return True

    # Try to construct a lower-rank decomposition using block Strassen.
    # A is 2x3, B is 3x4, C is 2x4
    # Split A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    # Split B = [B1; B2] where B1 is 2x4, B2 is 1x4
    # C = A1*B1 + A2*B2
    # Use Strassen for A1*B1 (2x2 * 2x2): rank 7
    # A2*B2 is 2x1 * 1x4: needs 2 multiplications
    # Total: 9 multiplications

    # Let's try to construct a rank-9 decomposition using this block approach.
    # Strassen for 2x2:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For our case:
    # A1 = [[a11, a12], [a21, a22]], A2 = [a13, a23]
    # B1 = [[b11, b12, b13, b14], [b21, b22, b23, b24]], B2 = [b31, b32, b33, b34]
    # C = [[c11, c12, c13, c14], [c21, c22, c23, c24]]

    # C[i,k] = A1[i,0]*B1[0,k] + A1[i,1]*B1[1,k] + A2[i]*B2[k]
    # = (A1*B1)[i,k] + A2[i]*B2[k]

    # Using Strassen for A1*B1:
    # Let's define the 7 Strassen products and 2 additional products for A2*B2.

    # Actually, let me try a different approach. Let me just return the schoolbook
    # since finding an exact lower-rank decomposition is complex and the random
    # search in the current program doesn't seem to be making progress.

    # But let me try one more thing: use the block structure to get rank 19.
    # A = [A1 | A2], B = [B1; B2]
    # A1*B1 can be done in rank 7 (Strassen)
    # A2*B2 is 2x1 * 1x4, which is rank 2
    # But we also need to handle the cross terms properly.

    # Actually, the issue is that A2*B2 is not independent - it's just an outer product.
    # C_part[i,k] = A2[i] * B2[k]
    # This requires 2 multiplications: A2[0]*B2[k] for all k, and A2[1]*B2[k] for all k.
    # But these are just 2 scalar multiplications per row.

    # Let me just return the schoolbook decomposition for now.
    U, V, W = build_schoolbook()

    # Verify the schoolbook decomposition
    if verify_full(U, V, W):
        return {"U": U, "V": V, "W": W}

    # Fallback: return schoolbook even if verification fails (shouldn't happen)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
