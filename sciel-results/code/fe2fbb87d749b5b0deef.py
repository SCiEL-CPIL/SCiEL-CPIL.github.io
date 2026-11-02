import time
import numpy as np
import random
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 7]
WALL_DEADLINE = 145.0
SAFETY_MARGIN = 5.0


def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7

    def elapsed():
        return time.time() - start_time

    def check_valid(U, V, W, tol=1e-8):
        """Check if the decomposition is valid."""
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def residual_norm(U, V, W):
        """Compute the residual norm."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        r = len(U_arr[0])
        total_sq = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                    total_sq += (s - 1.0) ** 2
        return np.sqrt(total_sq)

    def build_schoolbook():
        """Standard schoolbook decomposition with rank n*m*p."""
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    def build_strassen_2x2_2x2():
        """Strassen's rank-7 decomposition for 2x2x2."""
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        # p1 = (a11+a22)(b11+b22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        # p2 = (a21+a22)b11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        # p3 = a11(b12-b22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        # p4 = a22(b21-b11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        # p5 = (a11+a12)b22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        # p6 = (a21-a11)(b11+b12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        # p7 = (a12-a22)(b21+b22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        return U, V, W

    def kronecker_decompose_4x4x4():
        """Kronecker product of Strassen for 4x4x4, rank 49."""
        U2, V2, W2 = build_strassen_2x2_2x2()
        U4 = [[0] * 49 for _ in range(16)]
        V4 = [[0] * 49 for _ in range(16)]
        W4 = [[0] * 49 for _ in range(16)]
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                val = U2[i1 * 2 + j1][r1] * U2[i2 * 2 + j2][r2]
                                if val != 0:
                                    U4[i * 4 + j][r] += val
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                j = j1 * 2 + j2
                                k = k1 * 2 + k2
                                val = V2[j1 * 2 + k1][r1] * V2[j2 * 2 + k2][r2]
                                if val != 0:
                                    V4[j * 4 + k][r] += val
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                k = k1 * 2 + k2
                                i = i1 * 2 + i2
                                val = W2[k1 * 2 + i1][r1] * W2[k2 * 2 + i2][r2]
                                if val != 0:
                                    W4[k * 4 + i][r] += val
        return U4, V4, W4

    def build_4x4x7_decomposition():
        """Build 4x4x7 decomposition from Kronecker 4x4x4 + schoolbook for extra columns."""
        U4, V4, W4 = kronecker_decompose_4x4x4()
        r4 = len(U4[0])
        r_total = r4 + 48
        U = [[0] * r_total for _ in range(16)]
        V = [[0] * r_total for _ in range(28)]
        W = [[0] * r_total for _ in range(28)]
        for i in range(16):
            for r in range(r4):
                U[i][r] = U4[i][r]
        for j in range(4):
            for k in range(4):
                for r in range(r4):
                    V[j * 7 + k][r] = V4[j * 4 + k][r]
        for k in range(4):
            for i in range(4):
                for r in range(r4):
                    W[k * 4 + i][r] = W4[k * 4 + i][r]
        col_offset = r4
        for i in range(4):
            for j in range(4):
                for k in range(4, 7):
                    U[i * 4 + j][col_offset] = 1
                    V[j * 7 + k][col_offset] = 1
                    W[k * 4 + i][col_offset] = 1
                    col_offset += 1
        return U, V, W

    def tensors_to_vec(U, V, W):
        """Flatten the three factor matrices into one vector."""
        r = len(U[0])
        u = np.array(U, dtype=float).flatten()
        v = np.array(V, dtype=float).flatten()
        w = np.array(W, dtype=float).flatten()
        return np.concatenate([u, v, w])

    def vec_to_tensors(x, r):
        """Unflatten a vector into three factor matrices."""
        u_len = n * m * r
        v_len = m * p * r
        U = x[:u_len].reshape(n * m, r)
        V = x[u_len:u_len + v_len].reshape(m * p, r)
        W = x[u_len + v_len:].reshape(p * n, r)
        return U, V, W

    def build_jacobian(U, V, W):
        """
        Build the Jacobian of the residual vector w.r.t. the parameter vector x=(U,V,W).
        Residual for (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1 = 0
        """
        r = len(U[0])
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        u_len = n * m * r
        v_len = m * p * r
        J = np.zeros((n * m * p, u_len + v_len + p * n * r))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = (i * m + j) * p + k
                    for c in range(r):
                        uv = V[j * p + k, c] * W[k * n + i, c]
                        vw = U[i * m + j, c] * W[k * n + i, c]
                        wu = U[i * m + j, c] * V[j * p + k, c]
                        J[row, i * m * r + j * r + c] = uv
                        J[row, u_len + j * p * r + k * r + c] = vw
                        J[row, u_len + v_len + k * n * r + i * r + c] = wu
        return J

    def residual_vec(U, V, W):
        """Compute the residual vector."""
        r = len(U[0])
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        res = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    res[(i * m + j) * p + k] = s - 1.0
        return res

    def random_merge(U, V, W, n_merge, rng, tol=1e-9):
        """
        Merge n_merge columns via a random linear combination, then
        solve a least-squares correction to restore validity.
        """
        r = len(U[0])
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()

        new_r = r - n_merge
        cols = list(range(r))
        rng.shuffle(cols)
        kept = cols[:new_r]
        dropped = cols[new_r:]

        M = np.zeros((new_r, r))
        for t, kc in enumerate(kept):
            M[t, kc] = 1.0
            for dc in dropped:
                M[t, dc] = rng.uniform(-1.0, 1.0)

        U_new = U @ M.T
        V_new = V @ M.T
        W_new = W @ M.T

        res = residual_vec(U_new, V_new, W_new)
        J = build_jacobian(U_new, V_new, W_new)
        try:
            delta, *_ = np.linalg.lstsq(J, -res, rcond=None)
        except Exception:
            return None
        x_new = tensors_to_vec(U_new, V_new, W_new) + delta
        U2, V2, W2 = vec_to_tensors(x_new, new_r)

        if not check_valid(U2.tolist(), V2.tolist(), W2.tolist(), tol=1e-8):
            return None
        return U2.tolist(), V2.tolist(), W2.tolist(), new_r

    def targeted_elimination(U, V, W, target_rank, rng, tol=1e-8, max_rounds=20):
        """
        Targeted iterative column elimination: at each step, find the column
        that can be most effectively eliminated (lowest impact on validity),
        then merge it with a random linear combination.
        """
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()
        cur_rank = len(U[0])

        for round_num in range(max_rounds):
            if cur_rank <= target_rank:
                break
            if elapsed() > WALL_DEADLINE - SAFETY_MARGIN:
                break

            # Find the column with the smallest "impact" on the decomposition
            # This is approximated by the norm of the column's contribution
            impacts = np.zeros(cur_rank)
            for c in range(cur_rank):
                # Compute how much this column contributes to the residual
                # by measuring the norm of the slice it affects
                impact = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_val = U[i * m + j, c]
                            v_val = V[j * p + k, c]
                            w_val = W[k * n + i, c]
                            impact += u_val * u_val * v_val * v_val * w_val * w_val
                impacts[c] = impact

            # Sort by impact (lowest first)
            sorted_cols = np.argsort(impacts)
            
            # Try eliminating the lowest-impact columns
            eliminated = False
            for idx in sorted_cols[:5]:  # Try top 5 candidates
                if cur_rank - 1 < target_rank:
                    break
                
                c = int(idx)
                # Create a random linear combination to absorb column c into another
                other_cols = [cc for cc in range(cur_rank) if cc != c]
                if not other_cols:
                    break
                
                # Pick a random other column to merge into
                target_col = rng.choice(other_cols)
                
                # Build the merge matrix: new column = old column + alpha * eliminated column
                M = np.zeros((cur_rank - 1, cur_rank))
                new_col_idx = 0
                for old_col in range(cur_rank):
                    if old_col == c:
                        continue
                    M[new_col_idx, old_col] = 1.0
                    if old_col == target_col:
                        # Add the eliminated column with a random coefficient
                        M[new_col_idx, c] = rng.uniform(-0.5, 0.5)
                    new_col_idx += 1
                
                U_new = U @ M.T
                V_new = V @ M.T
                W_new = W @ M.T
                
                # Solve for correction
                res = residual_vec(U_new, V_new, W_new)
                J = build_jacobian(U_new, V_new, W_new)
                try:
                    delta, *_ = np.linalg.lstsq(J, -res, rcond=None)
                except Exception:
                    continue
                
                x_new = tensors_to_vec(U_new, V_new, W_new) + delta
                U2, V2, W2 = vec_to_tensors(x_new, cur_rank - 1)
                
                if check_valid(U2.tolist(), V2.tolist(), W2.tolist(), tol=1e-8):
                    U, V, W = U2, V2, W2
                    cur_rank -= 1
                    eliminated = True
                    break

            if not eliminated:
                break

        return U.tolist(), V.tolist(), W.tolist(), cur_rank

    def polish_decomposition(U, V, W, tol=1e-10, max_iter=100):
        """
        Polish a decomposition using Newton's method to drive residuals to machine precision.
        """
        r = len(U[0])
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()

        for iteration in range(max_iter):
            res = residual_vec(U, V, W)
            res_norm = np.linalg.norm(res)
            if res_norm < tol:
                break

            J = build_jacobian(U, V, W)
            try:
                delta, *_ = np.linalg.lstsq(J, -res, rcond=None)
            except Exception:
                return None

            alpha = 1.0
            for _ in range(20):
                x_trial = tensors_to_vec(U, V, W) + alpha * delta
                U_t, V_t, W_t = vec_to_tensors(x_trial, r)
                res_trial = residual_vec(U_t, V_t, W_t)
                if np.linalg.norm(res_trial) < res_norm:
                    break
                alpha *= 0.5

            x_new = tensors_to_vec(U, V, W) + alpha * delta
            U, V, W = vec_to_tensors(x_new, r)

        if not check_valid(U.tolist(), V.tolist(), W.tolist(), tol=tol):
            return None
        return U, V, W

    # ---- Main flow ----
    U_school, V_school, W_school = build_schoolbook()
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p

    rng = random.Random(12345)

    # Step 1: Build Kronecker + schoolbook base (rank 97)
    if elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            U_kron, V_kron, W_kron = build_4x4x7_decomposition()
            r_kron = len(U_kron[0])
            if check_valid(U_kron, V_kron, W_kron):
                if r_kron < best_rank:
                    best_result = {"U": U_kron, "V": V_kron, "W": W_kron}
                    best_rank = r_kron
        except Exception:
            pass

    # Step 2: Apply targeted elimination to reduce rank
    if elapsed() < WALL_DEADLINE - SAFETY_MARGIN and best_rank > 50:
        try:
            cur_U = np.array(best_result["U"], dtype=float)
            cur_V = np.array(best_result["V"], dtype=float)
            cur_W = np.array(best_result["W"], dtype=float)
            cur_rank = len(cur_U[0])

            # Target rank: aim for 60-70
            target_rank = 65
            U_new, V_new, W_new, new_rank = targeted_elimination(
                cur_U, cur_V, cur_W, target_rank, rng, tol=1e-8, max_rounds=100
            )
            if new_rank < best_rank:
                best_result = {"U": U_new, "V": V_new, "W": W_new}
                best_rank = new_rank
        except Exception:
            pass

    # Step 3: Apply random merging to further reduce rank
    if elapsed() < WALL_DEADLINE - SAFETY_MARGIN and best_rank > 40:
        try:
            cur_U = np.array(best_result["U"], dtype=float)
            cur_V = np.array(best_result["V"], dtype=float)
            cur_W = np.array(best_result["W"], dtype=float)
            cur_rank = len(cur_U[0])

            max_attempts = 300
            attempts = 0
            target = 40
            while cur_rank > target and attempts < max_attempts and elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
                merged = False
                for n_merge in (2, 1):
                    if cur_rank - n_merge < 1:
                        continue
                    res = random_merge(cur_U, cur_V, cur_W, n_merge, rng, tol=1e-9)
                    attempts += 1
                    if res is not None:
                        cur_U, cur_V, cur_W, cur_rank = res
                        merged = True
                        if cur_rank < best_rank:
                            best_rank = cur_rank
                            best_result = {
                                "U": cur_U.tolist(),
                                "V": cur_V.tolist(),
                                "W": cur_W.tolist(),
                            }
                        break
                    if elapsed() > WALL_DEADLINE - SAFETY_MARGIN:
                        break
                if not merged:
                    break
        except Exception:
            pass

    # Step 4: Final polish pass
    if elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            cur_U = np.array(best_result["U"], dtype=float)
            cur_V = np.array(best_result["V"], dtype=float)
            cur_W = np.array(best_result["W"], dtype=float)
            polished = polish_decomposition(cur_U, cur_V, cur_W, tol=1e-12, max_iter=50)
            if polished is not None:
                best_result = {
                    "U": polished[0].tolist(),
                    "V": polished[1].tolist(),
                    "W": polished[2].tolist(),
                }
        except Exception:
            pass

    return best_result
