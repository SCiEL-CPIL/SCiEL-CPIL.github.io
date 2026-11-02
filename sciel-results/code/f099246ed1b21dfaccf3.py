"""Rank-97 decomposition for 4x4 by 4x7 matrix multiplication using Strassen Kronecker + schoolbook tail."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    # Strategy:
    # Split output: p = 4 + 3
    # Part 1: 4x4 * 4x4 -> use Kronecker of two Strassen 2x2x2 (rank 7*7 = 49)
    # Part 2: 4x4 * 4x3 -> schoolbook (rank 4*4*3 = 48)
    # Total rank = 49 + 48 = 97
    #
    # Strassen 2x2x2 decomposition (rank 7):
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    # For 2x2x2, the factor matrices (u, v, w) for each of the 7 terms:
    # u: (4, 7) - A factor: rows are (i,j) pairs
    # v: (4, 7) - B factor: rows are (j,k) pairs  
    # w: (4, 7) - C factor: rows are (i,k) pairs
    # Using row-major: A index = i*2+j, B index = j*2+k, C index = i*2+k

    # Strassen terms for 2x2x2:
    # Term r: u_r[i*2+j], v_r[j*2+k], w_r[i*2+k]
    strassen_u = []
    strassen_v = []
    strassen_w = []

    # M1 = (A00+A11)*(B00+B11)
    # u: A00=1, A11=1; v: B00=1, B11=1; w: C00=1, C11=1
    strassen_u.append([1,0,0,1])  # A00,A01,A10,A11
    strassen_v.append([1,0,0,1])  # B00,B01,B10,B11
    strassen_w.append([1,0,0,1])  # C00,C01,C10,C11

    # M2 = (A10+A11)*B00
    strassen_u.append([0,0,1,1])
    strassen_v.append([1,0,0,0])
    strassen_w.append([0,0,1,0])  # C10

    # M3 = A00*(B10-B11)
    strassen_u.append([1,0,0,0])
    strassen_v.append([0,0,1,-1])
    strassen_w.append([0,1,0,0])  # C01

    # M4 = A11*(B01-B10)
    strassen_u.append([0,0,0,1])
    strassen_v.append([0,1,-1,0])
    strassen_w.append([1,0,0,1])  # C00, C11

    # M5 = (A00+A01)*B11
    strassen_u.append([1,1,0,0])
    strassen_v.append([0,0,0,1])
    strassen_w.append([1,0,0,0])  # C00

    # M6 = (A10-A00)*(B00+B10)
    strassen_u.append([1,0,-1,0])
    strassen_v.append([1,0,1,0])
    strassen_w.append([0,0,0,1])  # C11

    # M7 = (A01-A11)*(B01+B11)
    strassen_u.append([0,1,0,-1])
    strassen_v.append([0,1,0,1])
    strassen_w.append([1,0,0,0])  # C00

    # Convert to Fraction
    strassen_u = [[Fraction(x) for x in row] for row in strassen_u]
    strassen_v = [[Fraction(x) for x in row] for row in strassen_v]
    strassen_w = [[Fraction(x) for x in row] for row in strassen_w]

    # Now build the 4x4x4 decomposition via Kronecker product of two Strassen 2x2x2
    # 4x4x4: A is 4x4, B is 4x4, C is 4x4
    # Index A: i*4+j where i = i1*2+i2, j = j1*2+j2
    # Index B: j*4+k where j = j1*2+j2, k = k1*2+k2
    # Index C: i*4+k where i = i1*2+i2, k = k1*2+k2
    #
    # Kronecker: rank = 7*7 = 49
    # For term (r1, r2) where r1,r2 in 0..6:
    # u[i1*2+i2, j1*2+j2] = strassen_u[r1][i1*2+j2] * strassen_u[r2][i2*2+j1]
    # Wait, need to be careful with index ordering.
    #
    # Let's define: A_{i,j} where i = 2*i1 + i2, j = 2*j1 + j2
    # B_{j,k} where j = 2*j1 + j2, k = 2*k1 + k2
    # C_{i,k} where i = 2*i1 + i2, k = 2*k1 + k2
    #
    # The Kronecker product decomposition:
    # For each pair (r1, r2) in {0..6} x {0..6}:
    # u_{i,j} = u1_{i1,j1} * u2_{i2,j2}  where i=2*i1+i2, j=2*j1+j2
    # v_{j,k} = v1_{j1,k1} * v2_{j2,k2}  where j=2*j1+j2, k=2*k1+k2
    # w_{i,k} = w1_{i1,k1} * w2_{i2,k2}  where i=2*i1+i2, k=2*k1+k2

    U_444 = []  # Will be list of 49 columns, each of length 16
    V_444 = []
    W_444 = []

    for r1 in range(7):
        for r2 in range(7):
            u_col = [Fraction(0)] * 16
            v_col = [Fraction(0)] * 16
            w_col = [Fraction(0)] * 16

            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            i = 2*i1 + i2
                            j = 2*j1 + j2
                            u_col[i*4+j] = strassen_u[r1][i1*2+j1] * strassen_u[r2][i2*2+j2]

            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            j = 2*j1 + j2
                            k = 2*k1 + k2
                            v_col[j*4+k] = strassen_v[r1][j1*2+k1] * strassen_v[r2][j2*2+k2]

            for i1 in range(2):
                for i2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            i = 2*i1 + i2
                            k = 2*k1 + k2
                            w_col[i*4+k] = strassen_w[r1][i1*2+k1] * strassen_w[r2][i2*2+k2]

            U_444.append(u_col)
            V_444.append(v_col)
            W_444.append(w_col)

    # Part 2: 4x4 * 4x3 schoolbook, rank = 4*4*3 = 48
    # A is 4x4, B_tail is 4x3 (columns 4,5,6 of the full 4x7 B)
    # C_tail is 4x3
    # Schoolbook: for each (i,j,k) in {0..3}x{0..3}x{0..2}:
    # u[i*4+j] = 1, v[j*4+k] = 1, w[i*3+k] = 1
    # Note: C index in the full problem is k*n+i where k is column (0..6) and i is row (0..3)
    # For the tail, k is 4,5,6. So w index = k*4+i

    U_tail = []
    V_tail = []
    W_tail = []

    for i in range(4):
        for j in range(4):
            for k in range(3):  # k = 0,1,2 corresponds to output columns 4,5,6
                u_col = [Fraction(0)] * 16
                v_col = [Fraction(0)] * 16
                w_col = [Fraction(0)] * 28  # p*n = 7*4 = 28

                u_col[i*4+j] = Fraction(1)
                v_col[j*4+(4+k)] = Fraction(1)  # B index: j*4 + k_full where k_full = 4+k
                w_col[(4+k)*4+i] = Fraction(1)  # W index: k_full*n + i

                U_tail.append(u_col)
                V_tail.append(v_col)
                W_tail.append(w_col)

    # Now combine: total rank = 49 + 48 = 97
    # U: (16, 97), V: (28, 97), W: (28, 97)
    #
    # Wait, V dimension: m*p = 4*7 = 28. Correct.
    # W dimension: p*n = 7*4 = 28. Correct.
    # U dimension: n*m = 4*4 = 16. Correct.

    # For the 4x4x4 part, the V indices only use columns 0-3 of B (since it's 4x4 * 4x4)
    # But in the full problem, B is 4x7, so V has 28 rows.
    # The 4x4x4 part uses B columns 0-3, so v_col[j*4+k] for k in 0..3.
    # The tail part uses B columns 4-6.

    # Let me rebuild with proper dimensions:
    # U: 16 rows (n*m = 4*4)
    # V: 28 rows (m*p = 4*7)
    # W: 28 rows (p*n = 7*4)

    U_full = []
    V_full = []
    W_full = []

    # Part 1: 4x4x4 Strassen Kronecker (49 terms)
    for r1 in range(7):
        for r2 in range(7):
            u_col = [Fraction(0)] * 16
            v_col = [Fraction(0)] * 28
            w_col = [Fraction(0)] * 28

            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            i = 2*i1 + i2
                            j = 2*j1 + j2
                            u_col[i*4+j] = strassen_u[r1][i1*2+j1] * strassen_u[r2][i2*2+j2]

            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            j = 2*j1 + j2
                            k = 2*k1 + k2  # k in 0..3 (first 4 columns of B)
                            v_col[j*7+k] = strassen_v[r1][j1*2+k1] * strassen_v[r2][j2*2+k2]

            for i1 in range(2):
                for i2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            i = 2*i1 + i2
                            k = 2*k1 + k2  # k in 0..3 (first 4 columns of C)
                            w_col[k*4+i] = strassen_w[r1][i1*2+k1] * strassen_w[r2][i2*2+k2]

            U_full.append(u_col)
            V_full.append(v_col)
            W_full.append(w_col)

    # Part 2: 4x4x3 schoolbook (48 terms)
    for i in range(4):
        for j in range(4):
            for kt in range(3):  # kt = 0,1,2 -> output columns 4,5,6
                u_col = [Fraction(0)] * 16
                v_col = [Fraction(0)] * 28
                w_col = [Fraction(0)] * 28

                u_col[i*4+j] = Fraction(1)
                v_col[j*7+(4+kt)] = Fraction(1)
                w_col[(4+kt)*4+i] = Fraction(1)

                U_full.append(u_col)
                V_full.append(v_col)
                W_full.append(w_col)

    total_rank = len(U_full)  # Should be 97

    # Convert to lists of lists for output
    U_out = [list(row) for row in U_full]
    V_out = [list(row) for row in V_full]
    W_out = [list(row) for row in W_full]

    # Verify quickly with random test
    if time.time() < deadline - 10:
        try:
            U_np = np.array(U_full, dtype=float)
            V_np = np.array(V_full, dtype=float)
            W_np = np.array(W_full, dtype=float)

            # Test: for random A (4x4) and B (4x7), check C = A*B
            np.random.seed(42)
            A_test = np.random.randn(4, 4)
            B_test = np.random.randn(4, 7)
            C_expected = A_test @ B_test

            # Reconstruct C from decomposition
            # C[i,k] = sum_r U[i*4+j, r] * V[j*7+k, r] * W[k*4+i, r]
            # More precisely: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # Wait, that's not right. The decomposition is:
            # C_{i,k} = sum_r (sum_j U_{i*m+j, r} * A_{i,j}) * (sum_j V_{j*p+k, r} * B_{j,k}) * W_{k*n+i, r}
            # No, the standard tensor decomposition is:
            # T_{i,j,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{i,k,r}
            # where T_{i,j,k} is the multiplication tensor.
            # C_{i,k} = sum_{j} A_{i,j} B_{j,k} = sum_r (sum_j U_{i*m+j,r} A_{i,j}) (sum_j V_{j*p+k,r} B_{j,k}) W_{k*n+i,r}
            
            # Let's verify: C_{i,k} = sum_r W_{k*n+i, r} * (sum_j U_{i*m+j, r} * A_{i,j}) * (sum_j V_{j*p+k, r} * B_{j,k})
            C_computed = np.zeros((4, 7))
            for i in range(4):
                for k in range(7):
                    val = 0.0
                    for r in range(total_rank):
                        a_part = 0.0
                        for j in range(4):
                            a_part += U_np[i*4+j, r] * A_test[i, j]
                        b_part = 0.0
                        for j in range(4):
                            b_part += V_np[j*7+k, r] * B_test[j, k]
                        val += W_np[k*4+i, r] * a_part * b_part
                    C_computed[i, k] = val

            error = np.max(np.abs(C_computed - C_expected))
            if error > 1e-10:
                # Fallback to schoolbook if verification fails
                rank_school = n * m * p
                U_school = [[Fraction(0)] * rank_school for _ in range(n * m)]
                V_school = [[Fraction(0)] * rank_school for _ in range(m * p)]
                W_school = [[Fraction(0)] * rank_school for _ in range(p * n)]
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_school[i*m+j][col] = Fraction(1)
                            V_school[j*p+k][col] = Fraction(1)
                            W_school[k*n+i][col] = Fraction(1)
                            col += 1
                return {"U": U_school, "V": V_school, "W": W_school}
        except Exception:
            # Fallback to schoolbook
            rank_school = n * m * p
            U_school = [[Fraction(0)] * rank_school for _ in range(n * m)]
            V_school = [[Fraction(0)] * rank_school for _ in range(m * p)]
            W_school = [[Fraction(0)] * rank_school for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_school[i*m+j][col] = Fraction(1)
                        V_school[j*p+k][col] = Fraction(1)
                        W_school[k*n+i][col] = Fraction(1)
                        col += 1
            return {"U": U_school, "V": V_school, "W": W_school}

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
