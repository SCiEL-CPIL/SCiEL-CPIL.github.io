"""Matrix multiplication tensor decomposition for 2x3 * 3x8."""
import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = 2, 3, 8
    t0 = time.time()
    deadline = t0 + 120.0  # safety margin for 150s kill

    # We'll build a rank-36 decomposition using Strassen-based sub-decompositions.
    # Structure: split B into two 3x4 halves. For each half:
    #   2x3 * 3x4 = (2x2 * 2x4) + (2x1 * 1x4)
    #   2x2 * 2x4 = two Strassen 2x2*2x2 (rank 7 each) = rank 14
    #   2x1 * 1x4 = rank 4
    #   Total per half: 18, total: 36

    rank = 36
    U = [[0.0] * rank for _ in range(6)]
    V = [[0.0] * rank for _ in range(24)]
    W = [[0.0] * rank for _ in range(16)]

    def add_strassen_2x2x2(base, a_r0, a_r1, b_c0, b_c1, b_k0):
        """Strassen 2x2*2x2. A rows {a_r0,a_r1}, cols {b_c0,b_c1};
        B rows {b_c0,b_c1}, cols {b_k0,b_k0+1}; C[i,k] for i in {a_r0,a_r1}, k in {b_k0,b_k0+1}.
        W index: k*n+i = k*2+i."""
        # M1 = (A00+A11)*(B00+B11)
        U[a_r0*3+b_c0][base] += 1; V[b_c0*8+b_k0][base] += 1
        U[a_r1*3+b_c1][base] += 1; V[b_c1*8+b_k0+1][base] += 1
        W[b_k0*2+a_r0][base] += 1
        W[(b_k0+1)*2+a_r1][base] += 1
        # M2 = (A10+A11)*B00
        U[a_r1*3+b_c0][base+1] += 1; V[b_c0*8+b_k0][base+1] += 1
        U[a_r1*3+b_c1][base+1] += 1
        W[b_k0*2+a_r1][base+1] += 1
        # M3 = A00*(B10-B00)
        U[a_r0*3+b_c0][base+2] += 1; V[b_c1*8+b_k0][base+2] += 1
        V[b_c0*8+b_k0][base+2] += -1
        W[b_k0*2+a_r0][base+2] += -1
        W[b_k0*2+a_r1][base+2] += 1
        # M4 = A11*(B01+B11)
        U[a_r1*3+b_c1][base+3] += 1; V[b_c0*8+b_k0+1][base+3] += 1
        V[b_c1*8+b_k0+1][base+3] += 1
        W[b_k0*2+a_r0][base+3] += 1
        W[(b_k0+1)*2+a_r0][base+3] += 1
        # M5 = (A00+A01)*B11
        U[a_r0*3+b_c0][base+4] += 1; V[b_c1*8+b_k0+1][base+4] += 1
        U[a_r0*3+b_c1][base+4] += 1
        W[(b_k0+1)*2+a_r0][base+4] += 1
        # M6 = (A10-A00)*(B01-B11)
        U[a_r1*3+b_c0][base+5] += 1; V[b_c0*8+b_k0+1][base+5] += 1
        U[a_r0*3+b_c0][base+5] += -1; V[b_c1*8+b_k0+1][base+5] += -1
        W[(b_k0+1)*2+a_r1][base+5] += 1
        # M7 = (A01-A11)*(B00+B10)
        U[a_r0*3+b_c1][base+6] += 1; V[b_c0*8+b_k0][base+6] += 1
        U[a_r1*3+b_c1][base+6] += -1; V[b_c1*8+b_k0][base+6] += 1
        W[b_k0*2+a_r0][base+6] += -1
        W[(b_k0+1)*2+a_r1][base+6] += -1

    for half in range(2):
        bk = half * 4
        base = half * 18
        add_strassen_2x2x2(base, 0, 1, 0, 1, bk)
        add_strassen_2x2x2(base + 7, 0, 1, 0, 1, bk + 2)
        for dk in range(4):
            c = base + 14 + dk
            U[0*3+2][c] += 1
            U[1*3+2][c] += 1
            V[2*8+bk+dk][c] += 1
            W[(bk+dk)*2+0][c] += 1
            W[(bk+dk)*2+1][c] += 1

    # Verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    def verify(U_arr, V_arr, W_arr, n, m, p, rank, num_trials=5):
        for _ in range(num_trials):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for c in range(rank):
                u_mat = U_arr[:, c].reshape(n, m)
                v_mat = V_arr[:, c].reshape(m, p)
                prod = u_mat @ v_mat
                w_vec = W_arr[:, c]
                for i in range(n):
                    for k in range(p):
                        C_computed[i, k] += w_vec[k*n+i] * prod[i, k]
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True

    if verify(U_np, V_np, W_np, n, m, p, rank):
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank 48
    rank2 = n * m * p
    U2 = [[0] * rank2 for _ in range(n * m)]
    V2 = [[0] * rank2 for _ in range(m * p)]
    W2 = [[0] * rank2 for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U2[i*m+j][col] = 1
                V2[j*p+k][col] = 1
                W2[k*n+i][col] = 1
                col += 1
    return {"U": U2, "V": V2, "W": W2}
