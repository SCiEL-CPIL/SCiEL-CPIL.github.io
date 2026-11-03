"""Matrix multiplication 4x4x9 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin before 120s limit

    n, m, p = DIMENSIONS
    # Shapes
    U_rows = n * m  # 16
    V_rows = m * p  # 36
    W_rows = p * n  # 36

    # Build the target tensor structure
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))

    triple_set = set(triples)

    # Precompute target tensors for all three modes
    T_U = np.zeros((U_rows, V_rows, W_rows))
    T_V = np.zeros((V_rows, U_rows, W_rows))
    T_W = np.zeros((W_rows, U_rows, V_rows))
    for (a, b, c) in triples:
        T_U[a, b, c] = 1.0
        T_V[b, a, c] = 1.0
        T_W[c, a, b] = 1.0

    T_U_flat = T_U.reshape(U_rows, V_rows * W_rows)
    T_V_flat = T_V.reshape(V_rows, U_rows * W_rows)
    T_W_flat = T_W.reshape(W_rows, U_rows * V_rows)

    # Schoolbook solution (rank = n*m*p = 144)
    rank_school = n * m * p
    U_school = np.zeros((U_rows, rank_school))
    V_school = np.zeros((V_rows, rank_school))
    W_school = np.zeros((W_rows, rank_school))
    for col, (a, b, c) in enumerate(triples):
        U_school[a, col] = 1
        V_school[b, col] = 1
        W_school[c, col] = 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    def check_decomposition_exact(U, V, W, tol=1e-9):
        """Check if the decomposition is exact for all entries."""
        try:
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        except:
            return False
        diff = np.abs(T_recon - T_U)
        if np.max(diff) > tol:
            return False
        return True

    def check_decomposition_fast(U, V, W, tol=1e-6):
        """Faster check using sampling for intermediate steps."""
        # Check all triples
        for (a, b, c) in triples:
            val = np.dot(U[a, :] * V[b, :], W[c, :])
            if abs(val - 1.0) > tol:
                return False

        # Sample non-triples
        rng = np.random.RandomState(42)
        num_checks = 3000
        for _ in range(num_checks):
            a = rng.randint(0, U_rows)
            b = rng.randint(0, V_rows)
            c = rng.randint(0, W_rows)
            if (a, b, c) in triple_set:
                continue
            val = np.dot(U[a, :] * V[b, :], W[c, :])
            if abs(val) > tol:
                return False
        return True

    def try_round_and_check(U, V, W, scale=2):
        """Try rounding to half-integers and verify exactness."""
        U_r = np.round(U * scale) / scale
        V_r = np.round(V * scale) / scale
        W_r = np.round(W * scale) / scale
        if check_decomposition_exact(U_r, V_r, W_r, tol=1e-9):
            return U_r, V_r, W_r
        return None

    def als_decompose(target_rank, max_iters=100, tol=1e-10, seed=42):
        """Alternating Least Squares for tensor decomposition."""
        if time.time() > deadline - 10:
            return None

        rng = np.random.RandomState(seed)
        # Initialize with small random values
        U = rng.randn(U_rows, target_rank) * 0.01
        V = rng.randn(V_rows, target_rank) * 0.01
        W = rng.randn(W_rows, target_rank) * 0.01

        prev_err = float('inf')

        for it in range(max_iters):
            if time.time() > deadline - 10:
                break

            # Update U
            B_tensor = np.einsum('br,cr->bcr', V, W)  # (V_rows, W_rows, R)
            B = B_tensor.transpose(2, 0, 1).reshape(target_rank, V_rows * W_rows)

            G = B @ B.T
            G += 1e-10 * np.eye(target_rank)
            try:
                G_inv = np.linalg.inv(G)
            except:
                G_inv = np.linalg.pinv(G)

            BT = B @ T_U_flat.T  # (R, U_rows)
            U = (G_inv @ BT).T

            # Update V
            C_tensor = np.einsum('ar,cr->acr', U, W)  # (U_rows, W_rows, R)
            C = C_tensor.transpose(2, 0, 1).reshape(target_rank, U_rows * W_rows)

            Gv = C @ C.T
            Gv += 1e-10 * np.eye(target_rank)
            try:
                Gv_inv = np.linalg.inv(Gv)
            except:
                Gv_inv = np.linalg.pinv(Gv)

            CT = C @ T_V_flat.T
            V = (Gv_inv @ CT).T

            # Update W
            D_tensor = np.einsum('ar,br->abr', U, V)  # (U_rows, V_rows, R)
            D = D_tensor.transpose(2, 0, 1).reshape(target_rank, U_rows * V_rows)

            Gw = D @ D.T
            Gw += 1e-10 * np.eye(target_rank)
            try:
                Gw_inv = np.linalg.inv(Gw)
            except:
                Gw_inv = np.linalg.pinv(Gw)

            DT = D @ T_W_flat.T
            W = (Gw_inv @ DT).T

            # Check convergence periodically
            if it % 5 == 0:
                err = 0.0
                for (a, b, c) in triples:
                    val = np.dot(U[a, :] * V[b, :], W[c, :])
                    err += (val - 1.0) ** 2
                rng_check = np.random.RandomState(it)
                for _ in range(200):
                    a = rng_check.randint(0, U_rows)
                    b = rng_check.randint(0, V_rows)
                    c = rng_check.randint(0, W_rows)
                    if (a, b, c) in triple_set:
                        continue
                    val = np.dot(U[a, :] * V[b, :], W[c, :])
                    err += val ** 2

                if err < tol:
                    break
                if abs(prev_err - err) < 1e-14:
                    break
                prev_err = err

        return U, V, W

    def polish(U, V, W, target_rank, max_iters=200):
        """High-precision polish pass with tighter tolerance."""
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break

            # Update U with tighter regularization
            B_tensor = np.einsum('br,cr->bcr', V, W)
            B = B_tensor.transpose(2, 0, 1).reshape(target_rank, V_rows * W_rows)
            G = B @ B.T
            G += 1e-12 * np.eye(target_rank)
            try:
                G_inv = np.linalg.inv(G)
            except:
                G_inv = np.linalg.pinv(G)
            BT = B @ T_U_flat.T
            U_new = (G_inv @ BT).T

            # Update V
            C_tensor = np.einsum('ar,cr->acr', U_new, W)
            C = C_tensor.transpose(2, 0, 1).reshape(target_rank, U_rows * W_rows)
            Gv = C @ C.T
            Gv += 1e-12 * np.eye(target_rank)
            try:
                Gv_inv = np.linalg.inv(Gv)
            except:
                Gv_inv = np.linalg.pinv(Gv)
            CT = C @ T_V_flat.T
            V_new = (Gv_inv @ CT).T

            # Update W
            D_tensor = np.einsum('ar,br->abr', U_new, V_new)
            D = D_tensor.transpose(2, 0, 1).reshape(target_rank, U_rows * V_rows)
            Gw = D @ D.T
            Gw += 1e-12 * np.eye(target_rank)
            try:
                Gw_inv = np.linalg.inv(Gw)
            except:
                Gw_inv = np.linalg.pinv(Gw)
            DT = D @ T_W_flat.T
            W_new = (Gw_inv @ DT).T

            # Check change
            change = np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V)) + np.max(np.abs(W_new - W))
            U, V, W = U_new, V_new, W_new
            if change < 1e-14:
                break

        return U, V, W

    # Try to find lower rank decompositions
    # Start from a reasonable lower bound
    ranks_to_try = list(range(120, 35, -2))

    for target_rank in ranks_to_try:
        if time.time() > deadline - 20:
            break

        found = False
        # Try multiple seeds
        for seed in range(8):
            if time.time() > deadline - 20:
                break

            result = als_decompose(target_rank, max_iters=80, seed=seed)
            if result is None:
                continue

            U, V, W = result

            # Fast check first
            if check_decomposition_fast(U, V, W, tol=1e-5):
                # Try to round to half-integers
                rounded = try_round_and_check(U, V, W)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = target_rank
                    found = True
                    break
                else:
                    # Try polish then round
                    U_p, V_p, W_p = polish(U, V, W, target_rank, max_iters=100)
                    rounded = try_round_and_check(U_p, V_p, W_p)
                    if rounded is not None:
                        U_r, V_r, W_r = rounded
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = target_rank
                        found = True
                        break

        if found:
            break

    # If we found a better rank, return it
    if best_rank < rank_school:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
