import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Strassen 2x2 coefficients
    a_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0
        [[0, 0], [1, 1]],   # s=1
        [[1, 0], [0, 0]],   # s=2
        [[0, 0], [0, 1]],   # s=3
        [[1, 1], [0, 0]],   # s=4
        [[-1, 0], [1, 0]],  # s=5
        [[0, 1], [0, -1]],  # s=6
    ], dtype=int)

    b_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0
        [[1, 0], [0, 0]],   # s=1
        [[0, 1], [0, -1]],  # s=2
        [[-1, 0], [1, 0]],  # s=3
        [[0, 0], [0, 1]],   # s=4
        [[1, 1], [0, 0]],   # s=5
        [[0, 0], [1, 1]],   # s=6
    ], dtype=int)

    c_coeffs = np.array([
        [[1, 0], [0, 1]],   # s=0
        [[0, 0], [1, -1]],  # s=1
        [[0, 1], [0, 1]],   # s=2
        [[1, 0], [1, 0]],   # s=3
        [[-1, 1], [0, 0]],  # s=4
        [[0, 0], [0, 1]],   # s=5
        [[1, 0], [0, 0]],   # s=6
    ], dtype=int)

    # Build U, V, W directly via index arithmetic
    # U has shape (16, 49): U[i*4+j, 7*s+t] = a_s[i//2][j//2] * a_t[i%2][j%2]
    # V has shape (16, 49): V[j*4+k, 7*s+t] = b_s[j//2][k//2] * b_t[j%2][k%2]
    # W has shape (16, 49): W[k*4+i, 7*s+t] = c_s[i//2][k//2] * c_t[i%2][k%2]

    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)

    for s in range(7):
        for t in range(7):
            r = 7 * s + t
            for i in range(4):
                for j in range(4):
                    ib, ii = i // 2, i % 2
                    jb, jj = j // 2, j % 2
                    U[i * 4 + j, r] = a_coeffs[s][ib][jb] * a_coeffs[t][ii][jj]

            for j in range(4):
                for k in range(4):
                    jb, jj = j // 2, j % 2
                    kb, kk = k // 2, k % 2
                    V[j * 4 + k, r] = b_coeffs[s][jb][kb] * b_coeffs[t][jj][kk]

            for i in range(4):
                for k in range(4):
                    ib, ii = i // 2, i % 2
                    kb, kk = k // 2, k % 2
                    # W index: k*n + i = k*4 + i
                    W[k * 4 + i, r] = c_coeffs[s][ib][kb] * c_coeffs[t][ii][kk]

    # Verify: check tensor identity on a random instance
    # C[i,k] should equal sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] summed over j
    # Actually the tensor identity is: C[i,k] = sum_{j,r} U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where U is indexed by (i,j), V by (j,k), W by (k,i)
    # Let's verify with random matrices
    np.random.seed(42)
    A_test = np.random.randint(-5, 5, (4, 4))
    B_test = np.random.randint(-5, 5, (4, 4))
    C_expected = A_test @ B_test

    C_computed = np.zeros((4, 4), dtype=int)
    for i in range(4):
        for k in range(4):
            total = 0
            for j in range(4):
                for r in range(49):
                    total += U[i * 4 + j, r] * V[j * 4 + k, r] * W[k * 4 + i, r]
            C_computed[i, k] = total

    # The decomposition should satisfy: C[i,k] = sum_{j,r} U[i,j,r]*V[j,k,r]*W[k,i,r]
    # But actually the standard tensor decomposition is:
    # C_{ik} = sum_r (sum_j U_{ij}^{(r)} A_{ij}) * (sum_j V_{jk}^{(r)} B_{jk}) * W_{ki}^{(r)}
    # Wait, that's not right either. Let me think about this more carefully.
    
    # The tensor for matrix multiplication is T_{ik} = sum_j A_{ij} B_{jk}
    # A rank-r decomposition means: T = sum_{r=1}^{rank} u_r v_r w_r
    # where u_r, v_r, w_r are vectors in R^{n*m}, R^{m*p}, R^{p*n} respectively
    # T_{i,j,k} = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]  (using trace convention for W)
    
    # So for each (i,j,k): T[i,j,k] = delta_{j,k}... no.
    # T[i,j,k] = 1 if j==k... no, T is the multiplication tensor:
    # T[i,j,k] is the coefficient of A_{i,j} * B_{j,k} in C_{i,k}
    # So T[i,j,k] = 1 for all valid (i,j,k), and C_{i,k} = sum_j T[i,j,k] * A_{i,j} * B_{j,k}
    
    # Rank decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, W should be indexed to match. The trace convention W[k*n+i, r] means W is indexed by (k,i).
    
    # So the identity is: for all i,j,k in {0..3}:
    # 1 = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
    
    # Let's check this identity instead
    identity_holds = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                total = 0
                for r in range(49):
                    total += U[i * 4 + j, r] * V[j * 4 + k, r] * W[k * 4 + i, r]
                if total != 1:
                    identity_holds = False
                    print(f"Identity fails at ({i},{j},{k}): total={total}")
                    break
        if not identity_holds:
            break
    if not identity_holds:
        print("WARNING: Tensor identity does not hold!")

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
