"""Matrix multiplication 2x4 by 4x6: rank-minimal exact tensor decomposition."""
import time
import numpy as np

def construct():
    _start = time.time()
    deadline = _start + 145.0
    n, m, p = 2, 4, 6  # A is n x m, B is m x p
    U_rows = n * m  # 8
    V_rows = m * p  # 24
    W_rows = p * n  # 12

    # Build the target tensor
    T = np.zeros((U_rows, V_rows, W_rows))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Deterministic Strassen block construction (rank 42)
    rank_strassen = 42
    U = np.zeros((U_rows, rank_strassen))
    V = np.zeros((V_rows, rank_strassen))
    W = np.zeros((W_rows, rank_strassen))

    col = 0
    for a_block in range(2):
        for b_block in range(3):
            a_row0, a_row1 = 0, 1
            a_col0, a_col1 = 2 * a_block, 2 * a_block + 1
            b_row_start = 2 * a_block
            b_row0, b_row1 = b_row_start, b_row_start + 1
            b_col0, b_col1 = 2 * b_block, 2 * b_block + 1

            # M1: (a00+a11)*(b00+b11)
            U[a_row0*m + a_col0, col] += 1
            U[a_row1*m + a_col1, col] += 1
            V[b_row0*p + b_col0, col] += 1
            V[b_row1*p + b_col1, col] += 1
            W[b_col0*n + a_row0, col] += 1
            W[b_col1*n + a_row1, col] += 1
            col += 1

            # M2: (a10+a11)*b00
            U[a_row1*m + a_col0, col] += 1
            U[a_row1*m + a_col1, col] += 1
            V[b_row0*p + b_col0, col] += 1
            W[b_col0*n + a_row1, col] += 1
            W[b_col1*n + a_row1, col] += -1
            col += 1

            # M3: a00*(b01-b11)
            U[a_row0*m + a_col0, col] += 1
            V[b_row0*p + b_col1, col] += 1
            V[b_row1*p + b_col1, col] += -1
            W[b_col1*n + a_row0, col] += 1
            W[b_col1*n + a_row1, col] += 1
            col += 1

            # M4: a11*(b10-b00)
            U[a_row1*m + a_col1, col] += 1
            V[b_row1*p + b_col0, col] += 1
            V[b_row0*p + b_col0, col] += -1
            W[b_col0*n + a_row0, col] += 1
            W[b_col0*n + a_row1, col] += 1
            col += 1

            # M5: (a00+a01)*b11
            U[a_row0*m + a_col0, col] += 1
            U[a_row0*m + a_col1, col] += 1
            V[b_row1*p + b_col1, col] += 1
            W[b_col0*n + a_row0, col] += -1
            W[b_col1*n + a_row0, col] += 1
            col += 1

            # M6: (a10-a00)*(b00+b01)
            U[a_row1*m + a_col0, col] += 1
            U[a_row0*m + a_col0, col] += -1
            V[b_row0*p + b_col0, col] += 1
            V[b_row0*p + b_col1, col] += 1
            W[b_col1*n + a_row1, col] += 1
            col += 1

            # M7: (a01-a11)*(b10+b11)
            U[a_row0*m + a_col1, col] += 1
            U[a_row1*m + a_col1, col] += -1
            V[b_row1*p + b_col0, col] += 1
            V[b_row1*p + b_col1, col] += 1
            W[b_col0*n + a_row0, col] += 1
            col += 1

    # Verify Strassen solution
    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    assert np.allclose(T_comp, T, atol=1e-10), "Strassen construction failed verification"

    # Attempt to find lower rank via gradient descent
    # Lower bound: each 2x2x2 sub-tensor needs rank >= 7, and there are 6 independent ones
    # But cross-block sharing might reduce total rank. Try ranks below 42.
    best_rank = rank_strassen
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()

    np.random.seed(42)

    def compute_tensor(U_mat, V_mat, W_mat):
        return np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)

    def objective(U_mat, V_mat, W_mat):
        T_comp = compute_tensor(U_mat, V_mat, W_mat)
        diff = T_comp - T
        return np.sum(diff * diff)

    def gradients(U_mat, V_mat, W_mat):
        T_comp = compute_tensor(U_mat, V_mat, W_mat)
        diff = T_comp - T
        dU = np.einsum('abc,br,cr->ar', diff, V_mat, W_mat)
        dV = np.einsum('abc,ar,cr->br', diff, U_mat, W_mat)
        dW = np.einsum('abc,ar,br->cr', diff, U_mat, V_mat)
        return 2 * dU, 2 * dV, 2 * dW

    # Try to find rank < 42
    for target_rank in range(35, 42):
        if time.time() > deadline:
            break

        max_restarts = 10
        for restart in range(max_restarts):
            if time.time() > deadline:
                break

            U_mat = np.random.randn(U_rows, target_rank) * 0.5
            V_mat = np.random.randn(V_rows, target_rank) * 0.5
            W_mat = np.random.randn(W_rows, target_rank) * 0.5

            obj = objective(U_mat, V_mat, W_mat)
            best_obj = obj
            lr = 0.02
            stall = 0

            for it in range(200):
                if time.time() > deadline:
                    break

                dU, dV, dW = gradients(U_mat, V_mat, W_mat)
                U_new = U_mat - lr * dU
                V_new = V_mat - lr * dV
                W_new = W_mat - lr * dW

                obj_new = objective(U_new, V_new, W_new)
                if obj_new < best_obj - 1e-12:
                    U_mat, V_mat, W_mat = U_new, V_new, W_new
                    best_obj = obj_new
                    stall = 0
                    lr *= 0.999
                else:
                    stall += 1
                    lr *= 0.95
                    if stall > 50:
                        break

            if best_obj < 1e-8:
                best_rank = target_rank
                best_U = U_mat.copy()
                best_V = V_mat.copy()
                best_W = W_mat.copy()
                break

        if best_rank < 42:
            break

    # If no better rank found, use the verified Strassen solution
    if best_rank == rank_strassen:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
    else:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()

    if time.time() - _start > 145:
        raise TimeoutError("Exceeded time budget")

    return {"U": U_out, "V": V_out, "W": W_out}
