import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 138.0

    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10

    # Build target tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def validate(U, V, W, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)

    def snap_to_half(arr):
        return np.round(arr * 2) / 2

    def snap_to_int(arr):
        return np.round(arr)

    def try_snap_validate(U, V, W):
        U_i = snap_to_int(U)
        V_i = snap_to_int(V)
        W_i = snap_to_int(W)
        if validate(U_i, V_i, W_i):
            return True, U_i, V_i, W_i
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        if validate(U_s, V_s, W_s):
            return True, U_s, V_s, W_s
        return False, None, None, None

    def build_rank46():
        """Rank-46 decomposition using Strassen on 2x2 blocks + border terms."""
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))

        def add_strassen(j_off, k_off, col_start):
            c = col_start
            # P1 = (A00 + A11)(B00 + B11)
            U[0 * m + j_off, c] += 1
            U[1 * m + (j_off + 1), c] += 1
            V[j_off * p + k_off, c] += 1
            V[(j_off + 1) * p + (k_off + 1), c] += 1
            W[k_off * n + 0, c] += 1
            W[(k_off + 1) * n + 1, c] += 1
            c += 1
            # P2 = (A10 + A11) B00
            U[1 * m + j_off, c] += 1
            U[1 * m + (j_off + 1), c] += 1
            V[j_off * p + k_off, c] += 1
            W[k_off * n + 1, c] += 1
            W[(k_off + 1) * n + 1, c] -= 1
            c += 1
            # P3 = A00 (B01 - B11)
            U[0 * m + j_off, c] += 1
            V[j_off * p + (k_off + 1), c] += 1
            V[(j_off + 1) * p + (k_off + 1), c] -= 1
            W[(k_off + 1) * n + 0, c] += 1
            W[(k_off + 1) * n + 1, c] += 1
            c += 1
            # P4 = A11 (B10 - B00)
            U[1 * m + (j_off + 1), c] += 1
            V[(j_off + 1) * p + k_off, c] += 1
            V[j_off * p + k_off, c] -= 1
            W[k_off * n + 0, c] += 1
            W[k_off * n + 1, c] += 1
            c += 1
            # P5 = (A00 + A01) B11
            U[0 * m + j_off, c] += 1
            U[0 * m + (j_off + 1), c] += 1
            V[(j_off + 1) * p + (k_off + 1), c] += 1
            W[k_off * n + 0, c] -= 1
            W[(k_off + 1) * n + 0, c] += 1
            c += 1
            # P6 = (A10 - A00)(B00 + B01)
            U[1 * m + j_off, c] += 1
            U[0 * m + j_off, c] -= 1
            V[j_off * p + k_off, c] += 1
            V[j_off * p + (k_off + 1), c] += 1
            W[(k_off + 1) * n + 1, c] += 1
            c += 1
            # P7 = (A01 - A11)(B10 + B11)
            U[0 * m + (j_off + 1), c] += 1
            U[1 * m + (j_off + 1), c] -= 1
            V[(j_off + 1) * p + k_off, c] += 1
            V[(j_off + 1) * p + (k_off + 1), c] += 1
            W[k_off * n + 0, c] += 1
            return c

        col = add_strassen(0, 0, 0)
        col = add_strassen(0, 2, col)
        col = add_strassen(2, 0, col)
        col = add_strassen(2, 2, col)

        # Border: row 4 of A times columns 0-3 of B
        for i in range(2):
            for k in range(4):
                U[i * m + 4, col] = 1
                V[4 * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
        # Border: rows 0-3 of A times column 4 of B
        for i in range(2):
            for j in range(4):
                U[i * m + j, col] = 1
                V[j * p + 4, col] = 1
                W[4 * n + i, col] = 1
                col += 1
        # Corner: A[*,4] * B[4,*]
        for i in range(2):
            U[i * m + 4, col] = 1
            V[4 * p + 4, col] = 1
            W[4 * n + i, col] = 1
            col += 1

        return U, V, W

    def adam_optimize(U_init, V_init, W_init, target_rank, max_iter=300, lr=0.05,
                      beta1=0.9, beta2=0.999, eps=1e-8, lr_decay=0.998):
        """Adam optimizer with early stopping and best tracking."""
        R = target_rank
        U = U_init[:, :R].copy()
        V = V_init[:, :R].copy()
        W = W_init[:, :R].copy()

        mU = np.zeros_like(U)
        vU = np.zeros_like(U)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)

        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        current_lr = lr

        for it in range(max_iter):
            if time.time() > deadline - 12:
                break

            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            loss = np.sum(diff * diff)

            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            if loss < 1e-12:
                break

            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)

            t = it + 1
            mU = beta1 * mU + (1 - beta1) * gradU
            vU = beta2 * vU + (1 - beta2) * (gradU ** 2)
            U -= current_lr * (mU / (1 - beta1**t)) / (np.sqrt(vU / (1 - beta2**t)) + eps)

            mV = beta1 * mV + (1 - beta1) * gradV
            vV = beta2 * vV + (1 - beta2) * (gradV ** 2)
            V -= current_lr * (mV / (1 - beta1**t)) / (np.sqrt(vV / (1 - beta2**t)) + eps)

            mW = beta1 * mW + (1 - beta1) * gradW
            vW = beta2 * vW + (1 - beta2) * (gradW ** 2)
            W -= current_lr * (mW / (1 - beta1**t)) / (np.sqrt(vW / (1 - beta2**t)) + eps)

            current_lr *= lr_decay

        return best_U, best_V, best_W, best_loss

    def newton_polish(U, V, W, max_iter=50, tol=1e-14):
        U = U.copy()
        V = V.copy()
        W = W.copy()
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            loss = np.sum(diff * diff)
            if loss < tol:
                break
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            grad_norm = np.sqrt(np.sum(gradU**2) + np.sum(gradV**2) + np.sum(gradW**2))
            if grad_norm < 1e-15:
                break
            lr = 0.001 * min(1.0, 1e-5 / grad_norm)
            U -= lr * gradU
            V -= lr * gradV
            W -= lr * gradW
        return U, V, W

    # Build rank-46 base
    U46, V46, W46 = build_rank46()
    if not validate(U46, V46, W46):
        # Fallback: standard rank 50
        rank = 50
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = None

    # Strategy 1: Try rank 45 by zeroing one column and optimizing
    np.random.seed(42)
    col_scores = []
    for c in range(46):
        nz = np.count_nonzero(U46[:, c]) + np.count_nonzero(V46[:, c]) + np.count_nonzero(W46[:, c])
        col_scores.append(nz)
    col_order = np.argsort(col_scores)

    for attempt in range(min(12, 46)):
        if time.time() > deadline - 35:
            break
        col_to_zero = col_order[attempt]
        mask = [i for i in range(46) if i != col_to_zero]
        U_test = U46[:, mask].copy()
        V_test = V46[:, mask].copy()
        W_test = W46[:, mask].copy()

        for seed in [42, 123, 777]:
            if time.time() > deadline - 30:
                break
            np.random.seed(seed + attempt * 100)
            for noise in [0.01, 0.05, 0.1]:
                if time.time() > deadline - 28:
                    break
                U_try = U_test + np.random.randn(*U_test.shape) * noise
                V_try = V_test + np.random.randn(*V_test.shape) * noise
                W_try = W_test + np.random.randn(*W_test.shape) * noise

                U_opt, V_opt, W_opt, loss = adam_optimize(U_try, V_try, W_try, 45, max_iter=150, lr=0.04)
                if loss < 1e-6:
                    U2, V2, W2 = newton_polish(U_opt, V_opt, W_opt)
                    valid, Us, Vs, Ws = try_snap_validate(U2, V2, W2)
                    if valid:
                        best_result = (Us, Vs, Ws)
                        break
                if best_result is not None:
                    break
            if best_result is not None:
                break
        if best_result is not None:
            break

    # Strategy 2: Try rank 44 by zeroing two columns
    if best_result is None and time.time() < deadline - 40:
        pairs = []
        for i in range(46):
            for j in range(i+1, 46):
                ov = (np.count_nonzero(U46[:, i]) + np.count_nonzero(V46[:, i]) + np.count_nonzero(W46[:, i]) +
                      np.count_nonzero(U46[:, j]) + np.count_nonzero(V46[:, j]) + np.count_nonzero(W46[:, j]))
                pairs.append((ov, i, j))
        pairs.sort()

        for idx in range(min(10, len(pairs))):
            if time.time() > deadline - 35:
                break
            _, c1, c2 = pairs[idx]
            mask = [i for i in range(46) if i not in (c1, c2)]
            U_test = U46[:, mask].copy()
            V_test = V46[:, mask].copy()
            W_test = W46[:, mask].copy()

            for seed in [42, 123]:
                if time.time() > deadline - 32:
                    break
                np.random.seed(seed + idx * 20)
                for noise in [0.02, 0.08]:
                    if time.time() > deadline - 30:
                        break
                    U_try = U_test + np.random.randn(*U_test.shape) * noise
                    V_try = V_test + np.random.randn(*V_test.shape) * noise
                    W_try = W_test + np.random.randn(*W_test.shape) * noise

                    U_opt, V_opt, W_opt, loss = adam_optimize(U_try, V_try, W_try, 44, max_iter=120, lr=0.035)
                    if loss < 1e-6:
                        U2, V2, W2 = newton_polish(U_opt, V_opt, W_opt)
                        valid, Us, Vs, Ws = try_snap_validate(U2, V2, W2)
                        if valid:
                            best_result = (Us, Vs, Ws)
                            break
                    if best_result is not None:
                        break
                if best_result is not None:
                    break
            if best_result is not None:
                break

    # Strategy 3: Direct rank 45 search from random init
    if best_result is None and time.time() < deadline - 30:
        for seed in [42, 123, 777, 999]:
            if time.time() > deadline - 25:
                break
            np.random.seed(seed)
            U_try = np.random.randn(U_dim, 45) * 0.3
            V_try = np.random.randn(V_dim, 45) * 0.3
            W_try = np.random.randn(W_dim, 45) * 0.3
            U_opt, V_opt, W_opt, loss = adam_optimize(U_try, V_try, W_try, 45, max_iter=200, lr=0.03)
            if loss < 1e-6:
                U2, V2, W2 = newton_polish(U_opt, V_opt, W_opt)
                valid, Us, Vs, Ws = try_snap_validate(U2, V2, W2)
                if valid:
                    best_result = (Us, Vs, Ws)
                    break

    if best_result is not None:
        U_r, V_r, W_r = best_result
        if validate(U_r, V_r, W_r):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    # Return rank 46
    return {"U": U46.tolist(), "V": V46.tolist(), "W": W46.tolist()}
# EVOLVE-BLOCK-END
