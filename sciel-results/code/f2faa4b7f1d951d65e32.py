import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    start_time = time.time()
    deadline = 130  # Leave safety margin

    # Strassen-based decomposition: rank 56
    # View A (2x4) as 2x2 blocks of 2x2
    # View B (4x8) as 2x4 blocks of 2x2
    # Each Strassen "slot" s has 8 rank-1 terms (for the 2x2 sub-blocks)
    # Total: 7 slots * 8 = 56

    # Strassen formulas for M = P * Q where P, Q are 2x2:
    # S1 = Q[1,1] - Q[0,1]  (col 0 = indices 0,1; col 1 = indices 2,3)
    # S2 = P[0,0] + P[0,1]
    # S3 = P[1,0] + P[1,1]
    # S4 = Q[0,0] - Q[1,0]
    # S5 = (P[0,0]+P[1,1]) - S2
    # S6 = (P[1,1]+Q[0,0]) - S1  (actually this is for the M formula)
    # M1 = P[0,0]*S1
    # M2 = S2*Q[1,0]
    # M3 = S3*Q[0,1]
    # M4 = P[1,1]*S4
    # M5 = S5*(Q[0,0]+Q[1,1])
    # M6 = S6*(P[1,0]-P[1,1])  -- wait let me be more careful
    
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    # For our problem, we have 2x2 blocks.
    # A has blocks A00, A01 (row block 0) and A10, A11 (row block 1), each 2x2
    # B has blocks B00, B01, B02, B03 (col blocks) in row block 0, and B10,... in row block 1
    # Actually, B is 4x8. Let's partition B into 2x2 blocks:
    # B = [[B00, B01, B02, B03],
    #      [B10, B11, B12, B13]]
    # where each Bij is 2x2.

    # C = A @ B is 2x8. Partition C into 2x2 blocks:
    # C = [[C00, C01, C02, C03],
    #      [C10, C11, C12, C13]]

    # The multiplication C = AB can be viewed as: for each column block k (k=0..3):
    # C[:, k*2:(k+1)*2] = A @ B[:, k*2:(k+1)*2]
    # But that's not how Strassen works directly on the whole thing.

    # Better approach: Use the tensor decomposition directly.
    # The rank of M_{2,4,8} is known to be at most 56 via Strassen-like methods.
    # Let me just verify that rank 56 works with a random search, and if not, 
    # use the schoolbook rank 64.

    def get_schoolbook():
        rank = n * m * p  # 64
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, r] = 1.0
                    V[j * p + k, r] = 1.0
                    W[k * n + i, r] = 1.0
                    r += 1
        return U, V, W

    def compute_error(U, V, W, A, B):
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        X = np.einsum('ijr,ij->ir', U_reshaped, A)
        Y = np.einsum('jkr,jk->kr', V_reshaped, B)
        C_recon = np.einsum('ir,kr,ki->ik', X, Y, W_reshaped)
        C_true = A @ B
        return np.sum((C_recon - C_true) ** 2)

    def verify_decomposition(U, V, W, num_tests=20, seed=42):
        rng = np.random.RandomState(seed)
        max_err = 0
        for _ in range(num_tests):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            err = compute_error(U, V, W, A, B)
            max_err = max(max_err, err)
        return max_err

    # Try to find rank 56 decomposition via random initialization + gradient descent
    best_rank = n * m * p
    best_U, best_V, best_W = get_schoolbook()

    target_rank = 56
    if time.time() < deadline - 20:
        for trial in range(10):
            if time.time() > deadline - 10:
                break
            rng = np.random.RandomState(target_rank * 100 + trial)
            U = rng.randn(n * m, target_rank)
            V = rng.randn(m * p, target_rank)
            W = rng.randn(p * n, target_rank)

            A_test = np.random.RandomState(0).randn(n, m)
            B_test = np.random.RandomState(1).randn(m, p)

            # Gradient descent
            lr = 0.005
            for step in range(200):
                if time.time() > deadline - 10:
                    break

                U_reshaped = U.reshape(n, m, -1)
                V_reshaped = V.reshape(m, p, -1)
                W_reshaped = W.reshape(p, n, -1)

                X = np.einsum('ijr,ij->ir', U_reshaped, A_test)
                Y = np.einsum('jkr,jk->kr', V_reshaped, B_test)
                C_recon = np.einsum('ir,kr,ki->ik', X, Y, W_reshaped)
                C_true = A_test @ B_test
                diff = C_recon - C_true

                Z = np.einsum('ik,kr,ki->ir', diff, Y, W_reshaped)
                dU = 2 * np.einsum('ij,ir->ijr', A_test, Z).reshape(n * m, target_rank)

                Z2 = np.einsum('ik,ir,ki->kr', diff, X, W_reshaped)
                dV = 2 * np.einsum('jk,kr->jkr', B_test, Z2).reshape(m * p, target_rank)

                dW_reshaped = 2 * np.einsum('ik,ir,kr->ki', diff, X, Y)
                dW = dW_reshaped.reshape(p * n, target_rank)

                U -= lr * dU
                V -= lr * dV
                W -= lr * dW

                err = compute_error(U, V, W, A_test, B_test)
                if err < 1e-10:
                    break

            max_verify_err = verify_decomposition(U, V, W, num_tests=20)
            if max_verify_err < 1e-6:
                best_rank = target_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break

    # If rank 56 didn't work, try rank 60
    if best_rank == n * m * p and time.time() < deadline - 15:
        for target_rank in [60, 58]:
            if time.time() > deadline - 10:
                break
            for trial in range(5):
                if time.time() > deadline - 10:
                    break
                rng = np.random.RandomState(target_rank * 1000 + trial)
                U = rng.randn(n * m, target_rank)
                V = rng.randn(m * p, target_rank)
                W = rng.randn(p * n, target_rank)

                A_test = np.random.RandomState(0).randn(n, m)
                B_test = np.random.RandomState(1).randn(m, p)

                lr = 0.005
                for step in range(200):
                    if time.time() > deadline - 10:
                        break
                    U_reshaped = U.reshape(n, m, -1)
                    V_reshaped = V.reshape(m, p, -1)
                    W_reshaped = W.reshape(p, n, -1)
                    X = np.einsum('ijr,ij->ir', U_reshaped, A_test)
                    Y = np.einsum('jkr,jk->kr', V_reshaped, B_test)
                    C_recon = np.einsum('ir,kr,ki->ik', X, Y, W_reshaped)
                    C_true = A_test @ B_test
                    diff = C_recon - C_true
                    Z = np.einsum('ik,kr,ki->ir', diff, Y, W_reshaped)
                    dU = 2 * np.einsum('ij,ir->ijr', A_test, Z).reshape(n * m, target_rank)
                    Z2 = np.einsum('ik,ir,ki->kr', diff, X, W_reshaped)
                    dV = 2 * np.einsum('jk,kr->jkr', B_test, Z2).reshape(m * p, target_rank)
                    dW_reshaped = 2 * np.einsum('ik,ir,kr->ki', diff, X, Y)
                    dW = dW_reshaped.reshape(p * n, target_rank)
                    U -= lr * dU
                    V -= lr * dV
                    W -= lr * dW
                    err = compute_error(U, V, W, A_test, B_test)
                    if err < 1e-10:
                        break

                max_verify_err = verify_decomposition(U, V, W, num_tests=20)
                if max_verify_err < 1e-6:
                    best_rank = target_rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
            if best_rank < n * m * p:
                break

    if best_rank < n * m * p:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
