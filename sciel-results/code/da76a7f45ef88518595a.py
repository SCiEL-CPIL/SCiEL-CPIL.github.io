import numpy as np
import time

def construct():
    """
    Matrix multiplication 2x5 by 5x5: rank-38 decomposition.
    
    Decomposition:
    - Part 1: 2x4x5 multiplication (inner dim 0..3), rank 36
      * 4 blocks of 2x2x2 using Strassen (rank 7 each) = 28
      * 2 blocks of 2x2x1 using schoolbook (rank 4 each) = 8
    - Part 2: 2x1x5 multiplication (inner dim 4), rank 2
      * 2 terms: A[0,4]*B[4,k] for C[0,k], A[1,4]*B[4,k] for C[1,k]
    
    Total rank = 38
    """
    rank = 38
    
    U = [[0] * rank for _ in range(10)]   # 10 x 38 (A: 2x5)
    V = [[0] * rank for _ in range(25)]   # 25 x 38 (B: 5x5)
    W = [[0] * rank for _ in range(10)]   # 10 x 38 (C: 5x2, stored as W[k*2+i])
    
    col = 0
    
    def add_strassen(A00, A01, A10, A11, B00, B01, B10, B11, C00, C01, C10, C11):
        """Add Strassen 2x2x2 decomposition (rank 7)."""
        nonlocal col
        # M1 = (A00+A11)(B00+B11) -> C00(+), C11(+)
        U[A00][col] += 1; U[A11][col] += 1
        V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; W[C11][col] += 1
        col += 1
        # M2 = (A10+A11)B00 -> C10(+), C11(-)
        U[A10][col] += 1; U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1; W[C11][col] += -1
        col += 1
        # M3 = A00(B01-B11) -> C01(+), C11(+)
        U[A00][col] += 1
        V[B01][col] += 1; V[B11][col] += -1
        W[C01][col] += 1; W[C11][col] += 1
        col += 1
        # M4 = A11(B10-B00) -> C00(+), C10(+)
        U[A11][col] += 1
        V[B10][col] += 1; V[B00][col] += -1
        W[C00][col] += 1; W[C10][col] += 1
        col += 1
        # M5 = (A00+A01)B11 -> C00(-), C01(+)
        U[A00][col] += 1; U[A01][col] += 1
        V[B11][col] += 1
        W[C00][col] += -1; W[C01][col] += 1
        col += 1
        # M6 = (A10-A00)(B00+B01) -> C11(+)
        U[A10][col] += 1; U[A00][col] += -1
        V[B00][col] += 1; V[B01][col] += 1
        W[C11][col] += 1
        col += 1
        # M7 = (A01-A11)(B10+B11) -> C00(+)
        U[A01][col] += 1; U[A11][col] += -1
        V[B10][col] += 1; V[B11][col] += 1
        W[C00][col] += 1
        col += 1
    
    def add_rank4_2x2x1(A00, A01, A10, A11, B0, B1, C0, C1):
        """Add schoolbook 2x2x1 decomposition (rank 4)."""
        nonlocal col
        # C0 = A00*B0 + A01*B1
        U[A00][col] += 1; V[B0][col] += 1; W[C0][col] += 1
        col += 1
        U[A01][col] += 1; V[B1][col] += 1; W[C0][col] += 1
        col += 1
        # C1 = A10*B0 + A11*B1
        U[A10][col] += 1; V[B0][col] += 1; W[C1][col] += 1
        col += 1
        U[A11][col] += 1; V[B1][col] += 1; W[C1][col] += 1
        col += 1
    
    def add_rank2_2x1x5(A0, A1, B0, B1, B2, B3, B4,
                        C00, C01, C02, C03, C04,
                        C10, C11, C12, C13, C14):
        """Add 2x1x5 outer product decomposition (rank 2)."""
        nonlocal col
        # Term 1: A[0,4] * B[4,:] -> C[0,:]
        U[A0][col] += 1
        V[B0][col] += 1; V[B1][col] += 1; V[B2][col] += 1; V[B3][col] += 1; V[B4][col] += 1
        W[C00][col] += 1; W[C01][col] += 1; W[C02][col] += 1; W[C03][col] += 1; W[C04][col] += 1
        col += 1
        # Term 2: A[1,4] * B[4,:] -> C[1,:]
        U[A1][col] += 1
        V[B0][col] += 1; V[B1][col] += 1; V[B2][col] += 1; V[B3][col] += 1; V[B4][col] += 1
        W[C10][col] += 1; W[C11][col] += 1; W[C12][col] += 1; W[C13][col] += 1; W[C14][col] += 1
        col += 1
    
    # Index helpers
    def u_idx(i, j): return i * 5 + j       # A[i,j], i in {0,1}, j in {0..4}
    def v_idx(j, k): return j * 5 + k       # B[j,k], j in {0..4}, k in {0..4}
    def w_idx(i, k): return k * 2 + i       # C[i,k] stored as W[k*2+i]
    
    # Part 1: 2x4x5 multiplication (inner dim 0..3)
    # Output columns {0,1}: 2x2x2 blocks
    # A[:,{0,1}] * B[{0,1},{0,1}]
    add_strassen(
        u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1),
        v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1),
        w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1)
    )
    # A[:,{2,3}] * B[{2,3},{0,1}]
    add_strassen(
        u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3),
        v_idx(2,0), v_idx(2,1), v_idx(3,0), v_idx(3,1),
        w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1)
    )
    
    # Output columns {2,3}: 2x2x2 blocks
    # A[:,{0,1}] * B[{0,1},{2,3}]
    add_strassen(
        u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1),
        v_idx(0,2), v_idx(0,3), v_idx(1,2), v_idx(1,3),
        w_idx(0,2), w_idx(0,3), w_idx(1,2), w_idx(1,3)
    )
    # A[:,{2,3}] * B[{2,3},{2,3}]
    add_strassen(
        u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3),
        v_idx(2,2), v_idx(2,3), v_idx(3,2), v_idx(3,3),
        w_idx(0,2), w_idx(0,3), w_idx(1,2), w_idx(1,3)
    )
    
    # Output column {4}: 2x2x1 blocks
    # A[:,{0,1}] * B[{0,1},{4}]
    add_rank4_2x2x1(
        u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1),
        v_idx(0,4), v_idx(1,4),
        w_idx(0,4), w_idx(1,4)
    )
    # A[:,{2,3}] * B[{2,3},{4}]
    add_rank4_2x2x1(
        u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3),
        v_idx(2,4), v_idx(3,4),
        w_idx(0,4), w_idx(1,4)
    )
    
    # Part 2: 2x1x5 multiplication (inner dim 4)
    add_rank2_2x1x5(
        u_idx(0,4), u_idx(1,4),
        v_idx(4,0), v_idx(4,1), v_idx(4,2), v_idx(4,3), v_idx(4,4),
        w_idx(0,0), w_idx(0,1), w_idx(0,2), w_idx(0,3), w_idx(0,4),
        w_idx(1,0), w_idx(1,1), w_idx(1,2), w_idx(1,3), w_idx(1,4)
    )
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
