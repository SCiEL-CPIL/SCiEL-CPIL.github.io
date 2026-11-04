import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill

    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28

    # Build target tensor
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    schoolbook_rank = n * m * p  # 112

    # Precompute index arrays for efficient verification
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx

    def verify(U, V, W, r):
        """Check if U, V, W form a valid tensor decomposition. Returns max error."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)

        u_vals = Ua[u_idx, :]  # (n*m*p, r)
        v_vals = Va[v_idx, :]  # (n*m*p, r)
        w_vals = Wa[w_idx, :]  # (n*m*p, r)

        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)

        return np.max(np.abs(result - 1.0))

    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = make_schoolbook()
    best_rank = schoolbook_rank

    # Strategy 1: Structural decomposition with Strassen for 4x4x4 + schoolbook for 4x4x3
    # This gives rank 49 + 48 = 97
    U_444, V_444_full, W_444_full = _build_strassen_444()

    U_right_school = np.zeros((16, 48))
    V_right_school = np.zeros((28, 48))
    W_right_school = np.zeros((28, 48))

    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U_right_school[i*4+j, col] = 1
                V_right_school[j*7+k, col] = 1
                W_right_school[k*4+i, col] = 1
                col += 1

    U_total = np.hstack([U_444, U_right_school])
    V_total = np.hstack([V_444_full, V_right_school])
    W_total = np.hstack([W_444_full, W_right_school])

    recon_full = np.einsum('ak,bk,ck->abc', U_total, V_total, W_total)
    if np.allclose(recon_full, T, atol=1e-10):
        best_rank = 97
        best_result = {
            "U": U_total.tolist(),
            "V": V_total.tolist(),
            "W": W_total.tolist()
        }

    # Strategy 2: Try to improve the 4x4x3 part using block-based approach
    # 4x4x3 = 2x(2x4x3) = 2x(2x(2x2x3))
    # Use 2x2x3 decomposition (rank 7) for each 2x2x3 block
    # This could potentially give rank 49 + 2*2*7 = 49 + 28 = 77
    # But we need to be careful about the structure

    if time.time() < deadline - 20.0:
        right_result = _improve_4x4x3(deadline - 15.0)
        if right_result is not None:
            r_right = len(right_result["U"][0])
            U_r = np.array(right_result["U"]).T
            V_r = np.array(right_result["V"]).T
            W_r = np.array(right_result["W"]).T

            U_combined = np.hstack([U_444, U_r])
            V_combined = np.hstack([V_444_full, V_r])
            W_combined = np.hstack([W_444_full, W_r])

            recon = np.einsum('ak,bk,ck->abc', U_combined, V_combined, W_combined)
            if np.allclose(recon, T, atol=1e-10):
                total_rank = 49 + r_right
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_result = {
                        "U": U_combined.tolist(),
                        "V": V_combined.tolist(),
                        "W": W_combined.tolist()
                    }

    # Strategy 3: Try gradient-based refinement on the full tensor
    # Start from the current best and try to reduce rank
    if time.time() < deadline - 10.0 and best_rank > 80:
        refined = _try_refinement(T, dim_u, dim_v, dim_w, deadline - 5.0, best_rank)
        if refined is not None:
            r_ref = len(refined["U"][0])
            if r_ref < best_rank:
                best_rank = r_ref
                best_result = refined

    if best_rank < schoolbook_rank:
        return best_result

    return make_schoolbook()


def _build_strassen_444():
    """Build Strassen decomposition for 4x4x4 using 2x2x2 Strassen blocks."""
    U_2, V_2, W_2 = _strassen_2x2x2()

    U_444 = np.zeros((16, 49))
    V_444_std = np.zeros((16, 49))
    W_444_std = np.zeros((16, 49))

    for r1 in range(7):
        for r2 in range(7):
            idx = r1 * 7 + r2
            U_444[:, idx] = np.kron(U_2[:, r1], U_2[:, r2])
            V_444_std[:, idx] = np.kron(V_2[:, r1], V_2[:, r2])
            W_444_std[:, idx] = np.kron(W_2[:, r1], W_2[:, r2])

    # Embed into full tensor dimensions
    V_444_full = np.zeros((28, 49))
    for j in range(4):
        for k in range(4):
            V_444_full[j*7+k, :] = V_444_std[j*4+k, :]

    W_444_full = np.zeros((28, 49))
    for k in range(4):
        for i in range(4):
            W_444_full[k*4+i, :] = W_444_std[k*4+i, :]

    return U_444, V_444_full, W_444_full


def _improve_4x4x3(deadline):
    """Try to find a lower-rank decomposition for the 4x4x3 subproblem."""
    # The 4x4x3 tensor: A(4x4) * B(4x3) = C(4x3)
    # Structure: for each k in {4,5,6}, we have A * B[:,k] = C[:,k]
    # This is 3 independent 4x4x1 problems, but they share the same A

    # Schoolbook: rank 4*4*3 = 48
    # Can we do better?

    # Approach 1: Greedy peeling with structured initialization
    T_right = np.zeros((16, 28, 28), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                a = i * 4 + j
                b = j * 7 + k
                c = k * 4 + i
                T_right[a, b, c] = 1

    # Try greedy peeling
    result = _greedy_peel_structured(T_right, 16, 28, 28, deadline, 48)
    if result is not None:
        r = len(result["U"][0])
        if r < 48:
            return result

    # Approach 2: Try a structured decomposition
    # Split into 2x2 blocks
    # 4x4x3 = [2x2 blocks] * [2x3 blocks]
    # For each 2x2x3 block, use a rank-7 decomposition

    # 2x2x3 Strassen-like decomposition
    # M1 = (a00+a11)(b00+b01)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b00)
    # M4 = a11(b00-b01)
    # M5 = (a00+a01)b01
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b01-b00)

    U_223 = np.zeros((4, 7))
    V_223 = np.zeros((6, 7))
    W_223 = np.zeros((6, 7))

    # M1 = (a00+a11)(b00+b01)
    U_223[0, 0] = 1; U_223[3, 0] = 1
    V_223[0, 0] = 1; V_223[2, 0] = 1
    W_223[0, 0] = 1; W_223[4, 0] = 1

    # M2 = (a10+a11)b00
    U_223[2, 1] = 1; U_223[3, 1] = 1
    V_223[0, 1] = 1
    W_223[2, 1] = 1; W_223[6, 1] = -1

    # M3 = a00(b01-b00)
    U_223[0, 2] = 1
    V_223[2, 2] = 1; V_223[0, 2] = -1
    W_223[1, 2] = 1; W_223[5, 2] = 1

    # M4 = a11(b00-b01)
    U_223[3, 3] = 1
    V_223[0, 3] = 1; V_223[2, 3] = -1
    W_223[0, 3] = 1; W_223[4, 3] = 1

    # M5 = (a00+a01)b01
    U_223[0, 4] = 1; U_223[1, 4] = 1
    V_223[2, 4] = 1
    W_223[0, 4] = -1; W_223[1, 4] = 1

    # M6 = (a10-a00)(b00+b01)
    U_223[2, 5] = 1; U_223[0, 5] = -1
    V_223[0, 5] = 1; V_223[2, 5] = 1
    W_223[3, 5] = 1

    # M7 = (a01-a11)(b01-b00)
    U_223[1, 6] = 1; U_223[3, 6] = -1
    V_223[2, 6] = 1; V_223[0, 6] = -1
    W_223[0, 6] = 1

    # Verify 2x2x3 decomposition
    T_223 = np.zeros((4, 6, 6))
    for i in range(2):
        for j in range(2):
            for k in range(3):
                T_223[i*2+j, j*3+k, k*2+i] = 1

    recon_223 = np.einsum('ak,bk,ck->abc', U_223, V_223, W_223)
    if not np.allclose(recon_223, T_223, atol=1e-10):
        # 2x2x3 decomposition is invalid, fall back to greedy
        return None

    # Now build 4x4x3 from 2x2x3 blocks
    # 4x4 A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # 4x3 B = [[B00, B01, B02], [B10, B11, B12]] where each Bj is 2x1
    # 4x3 C = [[C00, C01, C02], [C10, C11, C12]] where each Cj is 2x1

    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12

    # Each C block is 2x1, computed from 2x2 * 2x1
    # 2x2 * 2x1 = (2x2x1) which has rank 4 (schoolbook) or less

    # Actually, 2x2x1 is just 2x2 matrix times 2x1 vector
    # This is rank 4 (schoolbook): c0 = a00*b0 + a01*b1, c1 = a10*b0 + a11*b1

    # But we can use the 2x2x3 structure differently
    # Let's think of B as 2x(3x1) = 2x3, and use 2x2x3 blocks

    # Actually, let's just use the 2x2x3 decomposition directly
    # For the top half: A_top (2x4) * B_top (4x3) = C_top (2x3)
    # A_top = [A00, A01], B_top = [B00, B01, B02; B10, B11, B12]
    # This is 2x(4x3) = 2x4x3, which we can decompose

    # 2x4x3 = 2x(2x2x3) + 2x(2x2x3)
    # Use 2x2x3 Strassen for each part

    # Top half: A_top * B_top
    # A_top = [A00, A01] (2x4)
    # B_top = [B00, B01, B02; B10, B11, B12] (4x3)
    # C_top = A00 * [B00, B01, B02] + A01 * [B10, B11, B12]

    # A00 * [B00, B01, B02] is 2x2 * 2x3 = 2x2x3 (rank 7)
    # A01 * [B10, B11, B12] is 2x2 * 2x3 = 2x2x3 (rank 7)
    # Total for top half: rank 14

    # Similarly for bottom half: rank 14
    # Total: rank 28

    # Let's build this
    U_443 = np.zeros((16, 28))
    V_443 = np.zeros((28, 28))
    W_443 = np.zeros((28, 28))

    # Top half: rows 0-3 of U (i=0,1), columns 0-3 of V (j=0,1,2)
    # A00 * [B00, B01, B02]: U indices 0,3 (i=0,1 in 2x2), V indices 0,3,6 (j=0, k=0,1,2)
    # W indices: k*4+i for k=0,1,2 and i=0,1 -> 0,1,4,5,8,9

    # Term 1: A00 * B_top_left (2x2 * 2x3)
    for t in range(7):
        U_443[0, t] = U_223[0, t]  # i=0
        U_443[1, t] = U_223[1, t]  # i=1
        U_443[4, t] = U_223[2, t]  # i=2
        U_443[5, t] = U_223[3, t]  # i=3

        V_443[0, t] = V_223[0, t]  # j=0, k=0
        V_443[1, t] = V_223[1, t]  # j=0, k=1
        V_443[2, t] = V_223[2, t]  # j=0, k=2
        V_443[7, t] = V_223[3, t]  # j=1, k=0
        V_443[8, t] = V_223[4, t]  # j=1, k=1
        V_443[9, t] = V_223[5, t]  # j=1, k=2

        W_443[0, t] = W_223[0, t]  # k=0, i=0
        W_443[1, t] = W_223[1, t]  # k=0, i=1
        W_443[4, t] = W_223[2, t]  # k=1, i=0
        W_443[5, t] = W_223[3, t]  # k=1, i=1
        W_443[8, t] = W_223[4, t]  # k=2, i=0
        W_443[9, t] = W_223[5, t]  # k=2, i=1

    # Term 2: A01 * B_top_right (2x2 * 2x3)
    # A01 is rows 0-1, cols 2-3 of A
    # B_top_right is rows 2-3, cols 0-2 of B
    for t in range(7):
        U_443[2, 7+t] = U_223[0, t]  # i=0, j=2
        U_443[3, 7+t] = U_223[1, t]  # i=0, j=3
        U_443[6, 7+t] = U_223[2, t]  # i=1, j=2
        U_443[7, 7+t] = U_223[3, t]  # i=1, j=3

        V_443[14, 7+t] = V_223[0, t]  # j=2, k=0
        V_443[15, 7+t] = V_223[1, t]  # j=2, k=1
        V_443[16, 7+t] = V_223[2, t]  # j=2, k=2
        V_443[21, 7+t] = V_223[3, t]  # j=3, k=0
        V_443[22, 7+t] = V_223[4, t]  # j=3, k=1
        V_443[23, 7+t] = V_223[5, t]  # j=3, k=2

        W_443[0, 7+t] = W_223[0, t]  # k=0, i=0
        W_443[1, 7+t] = W_223[1, t]  # k=0, i=1
        W_443[4, 7+t] = W_223[2, t]  # k=1, i=0
        W_443[5, 7+t] = W_223[3, t]  # k=1, i=1
        W_443[8, 7+t] = W_223[4, t]  # k=2, i=0
        W_443[9, 7+t] = W_223[5, t]  # k=2, i=1

    # Bottom half: rows 4-7 of U (i=2,3), columns 4-7 of V (j=2,3)
    # A10 * [B00, B01, B02]: U indices 8,11 (i=2,3), V indices 0,3,6
    for t in range(7):
        U_443[8, 14+t] = U_223[0, t]  # i=2, j=0
        U_443[9, 14+t] = U_223[1, t]  # i=2, j=1
        U_443[12, 14+t] = U_223[2, t]  # i=3, j=0
        U_443[13, 14+t] = U_223[3, t]  # i=3, j=1

        V_443[0, 14+t] = V_223[0, t]  # j=0, k=0
        V_443[1, 14+t] = V_223[1, t]  # j=0, k=1
        V_443[2, 14+t] = V_223[2, t]  # j=0, k=2
        V_443[7, 14+t] = V_223[3, t]  # j=1, k=0
        V_443[8, 14+t] = V_223[4, t]  # j=1, k=1
        V_443[9, 14+t] = V_223[5, t]  # j=1, k=2

        W_443[12, 14+t] = W_223[0, t]  # k=0, i=2
        W_443[13, 14+t] = W_223[1, t]  # k=0, i=3
        W_443[16, 14+t] = W_223[2, t]  # k=1, i=2
        W_443[17, 14+t] = W_223[3, t]  # k=1, i=3
        W_443[20, 14+t] = W_223[4, t]  # k=2, i=2
        W_443[21, 14+t] = W_223[5, t]  # k=2, i=3

    # Term 4: A11 * [B10, B11, B12]
    for t in range(7):
        U_443[10, 21+t] = U_223[0, t]  # i=2, j=2
        U_443[11, 21+t] = U_223[1, t]  # i=2, j=3
        U_443[14, 21+t] = U_223[2, t]  # i=3, j=2
        U_443[15, 21+t] = U_223[3, t]  # i=3, j=3

        V_443[14, 21+t] = V_223[0, t]  # j=2, k=0
        V_443[15, 21+t] = V_223[1, t]  # j=2, k=1
        V_443[16, 21+t] = V_223[2, t]  # j=2, k=2
        V_443[21, 21+t] = V_223[3, t]  # j=3, k=0
        V_443[22, 21+t] = V_223[4, t]  # j=3, k=1
        V_443[23, 21+t] = V_223[5, t]  # j=3, k=2

        W_443[12, 21+t] = W_223[0, t]  # k=0, i=2
        W_443[13, 21+t] = W_223[1, t]  # k=0, i=3
        W_443[16, 21+t] = W_223[2, t]  # k=1, i=2
        W_443[17, 21+t] = W_223[3, t]  # k=1, i=3
        W_443[20, 21+t] = W_223[4, t]  # k=2, i=2
        W_443[21, 21+t] = W_223[5, t]  # k=2, i=3

    # Verify the 4x4x3 decomposition
    T_right = np.zeros((16, 28, 28), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                a = i * 4 + j
                b = j * 7 + k
                c = k * 4 + i
                T_right[a, b, c] = 1

    recon_443 = np.einsum('ak,bk,ck->abc', U_443, V_443, W_443)
    if np.allclose(recon_443, T_right, atol=1e-10):
        return {
            "U": U_443.tolist(),
            "V": V_443.tolist(),
            "W": W_443.tolist()
        }

    return None


def _try_refinement(T, dim_u, dim_v, dim_w, deadline, current_rank):
    """Try gradient-based refinement to reduce rank."""
    # Start from a random initialization with rank slightly less than current
    target_rank = current_rank - 5

    for trial in range(10):
        if time.time() > deadline:
            break

        np.random.seed(trial * 1000 + 42)

        U = np.random.randn(dim_u, target_rank) * 0.5
        V = np.random.randn(dim_v, target_rank) * 0.5
        W = np.random.randn(dim_w, target_rank) * 0.5

        # Gradient descent
        for it in range(100):
            if time.time() > deadline - 5:
                break

            # Compute error
            recon = np.einsum('ak,bk,ck->abc', U, V, W)
            error = recon - T
            err_norm = np.linalg.norm(error)

            if err_norm < 1e-8:
                break

            # Compute gradients
            # grad_U[a, k] = sum_{b,c} error[a,b,c] * V[b,k] * W[c,k]
            grad_U = np.einsum('abc,bk,ck->ak', error, V, W)
            grad_V = np.einsum('abc,ak,ck->bk', error, U, W)
            grad_W = np.einsum('abc,ak,bk->ck', error, U, V)

            # Adaptive learning rate
            lr = 0.1 / (1.0 + it * 0.1)

            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W

        # Verify
        recon = np.einsum('ak,bk,ck->abc', U, V, W)
        if np.allclose(recon, T, atol=1e-6):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    return None


def _greedy_peel_structured(T, dim_u, dim_v, dim_w, deadline, max_terms):
    """Greedy rank-1 peeling optimized for the 4x4x3 sub-problem."""
    R = T.copy()
    U_list = []
    V_list = []
    W_list = []

    for term_idx in range(max_terms):
        if time.time() > deadline:
            break
        if not np.any(R):
            break

        best_score = 0
        best_u, best_v, best_w = None, None, None

        # Multiple restarts with different initializations
        n_restarts = 12
        for restart in range(n_restarts):
            if time.time() > deadline:
                break

            if restart == 0:
                # Sign-based initialization
                cu = np.sign(np.sum(R, axis=(1, 2)))
                cv = np.sign(np.sum(R, axis=(0, 2)))
                cw = np.sign(np.sum(R, axis=(0, 1)))
            elif restart == 1:
                # Random +/-1
                cu = np.random.choice([-1, 1], size=dim_u).astype(np.float64)
                cv = np.random.choice([-1, 1], size=dim_v).astype(np.float64)
                cw = np.random.choice([-1, 1], size=dim_w).astype(np.float64)
            elif restart == 2:
                # Random with zeros
                cu = np.random.choice([-1, 0, 1], size=dim_u).astype(np.float64)
                cv = np.random.choice([-1, 0, 1], size=dim_v).astype(np.float64)
                cw = np.random.choice([-1, 0, 1], size=dim_w).astype(np.float64)
            else:
                # Random Gaussian
                cu = np.random.randn(dim_u)
                cv = np.random.randn(dim_v)
                cw = np.random.randn(dim_w)

            # Alternating minimization with more iterations
            n_iters = 20
            for it in range(n_iters):
                if time.time() > deadline:
                    break

                M_u = np.einsum('abc,b,c->a', R, cv, cw)
                new_u = np.sign(M_u)

                M_v = np.einsum('abc,a,c->b', R, new_u, cw)
                new_v = np.sign(M_v)

                M_w = np.einsum('abc,a,b->c', R, new_u, new_v)
                new_w = np.sign(M_w)

                cu, cv, cw = new_u, new_v, new_w

            score = np.einsum('abc,a,b,c', R, cu, cv, cw)
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = cu.copy(), cv.copy(), cw.copy()

        if best_score <= 0 or best_u is None:
            break

        R -= np.einsum('a,b,c->abc', best_u, best_v, best_w)
        U_list.append(best_u)
        V_list.append(best_v)
        W_list.append(best_w)

        if np.allclose(R, 0, atol=1e-10):
            break

    if np.allclose(R, 0, atol=1e-10) and len(U_list) > 0:
        rank = len(U_list)
        U = np.array(U_list).T
        V = np.array(V_list).T
        W = np.array(W_list).T

        reconstructed = np.einsum('ak,bk,ck->abc', U, V, W)
        if np.allclose(reconstructed, T, atol=1e-10):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    return None


def _strassen_2x2x2():
    """Return Strassen decomposition of 2x2x2 matrix multiplication.

    Returns U, V, W each of shape (4, 7) such that
    sum_k U[a,k]*V[b,k]*W[c,k] = T[a,b,c]
    where T[i*2+j, j*2+k, k*2+i] = 1
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)

    # M1 = (a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (a21+a22)b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1

    # M3 = a11(b12-b22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1

    # M4 = a22(b21-b11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1

    # M5 = (a11+a12)b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1

    # M6 = (a21-a11)(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # M7 = (a12-a22)(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

# EVOLVE-BLOCK-END
