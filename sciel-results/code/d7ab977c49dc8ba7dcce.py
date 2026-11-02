"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    WFT-based decomposition for 2x4x7 matrix multiplication.
    
    Strategy: Split the middle dimension m=4 into two blocks of size 2.
    C = A1*B1 + A2*B2 where A1,A2 are 2x2 and B1,B2 are 2x7.
    
    For each (2x2)*(2x7) subproblem, we use a WFT-style decomposition over Z_2 x Z_7.
    Since m_sub=2 <= n_sub=2 and m_sub=2 <= p_sub=7, the WFT applies.
    
    However, WFT over Z_2 x Z_7 involves 7th roots of unity (complex numbers).
    To get integer/half-integer entries, we use the fact that the 7th cyclotomic
    polynomial has degree 6, and we can work in the ring Z[omega_7]. But the task
    requires rational entries.
    
    Alternative: Use the standard rank-28 decomposition per subproblem (schoolbook for 2x2x7),
    giving total rank 56. This matches the seed.
    
    To improve, we exploit the n=2 structure: for each column k of B and each row i of A,
    we can combine the two rows using sum/difference tricks to reduce rank.
    
    Specifically, for (2xm)*(mxp):
    Let S[j] = A[0,j] + A[1,j] and D[j] = A[0,j] - A[1,j]
    Then:
      C[0,k] = (sum_j S[j]*B[j,k] + sum_j D[j]*B[j,k]) / 2
      C[1,k] = (sum_j S[j]*B[j,k] - sum_j D[j]*B[j,k]) / 2
    
    This still requires 2*m*p terms for the sums, so no improvement.
    
    We'll use schoolbook rank 56 as a correct baseline.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    rank = n * m * p  # 56
    U = [[0] * rank for _ in range(n * m)]    # shape (8, 56)
    V = [[0] * rank for _ in range(m * p)]    # shape (28, 56)
    W = [[0] * rank for _ in range(p * n)]    # shape (14, 56)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
