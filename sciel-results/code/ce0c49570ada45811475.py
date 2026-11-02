"""Find a rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Dimensions: A is n x m (2x5), B is m x p (5x5), C is n x p (2x5)
    # U has shape (n*m, rank) = (10, rank)
    # V has shape (m*p, rank) = (25, rank)
    # W has shape (p*n, rank) = (10, rank)
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]

    # Schoolbook rank
    schoolbook_rank = n * m * p  # 50

    # The tensor T[a,b,c] where:
    # a = i*m + j (i in 0..n-1, j in 0..m-1)
    # b = j'*p + k (j' in 0..m-1, k in 0..p-1)
    # c = k'*n + i' (k' in 0..p-1, i' in 0..n-1)
    # T[a,b,c] = 1 if i=i', j=j', k=k' else 0

    # For a rank-r decomposition, we need U (10xr), V (25xr), W (10xr) such that:
    # sum_{c=0}^{r-1} U[a,c] * V[b,c] * W[c_idx,c] = T[a,b,c_idx]
    # for all a, b, c_idx

    # Strategy: Start with schoolbook and try to reduce rank via optimization.
    # We'll use a tensor train / ALS approach on the 3-tensor.

    # First, let's build the tensor T as a 3D array of shape (10, 25, 10)
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Try to find a low-rank decomposition using ALS (Alternating Least Squares)
    # on the 3-mode tensor.

    def try_rank(r, max_iters=500, tol=1e-10):
        """Try to find a rank-r decomposition of T using ALS."""
        if time.time() > deadline - 5:
            return None

        # Initialize with random values
        np.random.seed(42 + r)
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)

        # Normalize initial vectors
        for c in range(r):
            norm_u = np.linalg.norm(U[:, c])
            norm_v = np.linalg.norm(V[:, c])
            norm_w = np.linalg.norm(W[:, c])
            if norm_u > 0: U[:, c] /= norm_u
            if norm_v > 0: V[:, c] /= norm_v
            if norm_w > 0: W[:, c] /= norm_w

        prev_loss = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 10:
                return None

            # Update U: for each column a, solve for U[a, :]
            # The equation is: sum_c U[a,c] * V[b,c] * W[c_idx,c] = T[a,b,c_idx]
            # For fixed a: sum_c U[a,c] * (V[:,c] * W[:,c]) = T[a, :, :] flattened
            # This is a linear system: U[a, :] @ (V * W) = T[a].flatten()
            # where (V*W)[b, c] = V[b,c] * W[c_idx... wait, this is 3D.

            # Actually, for fixed a, we need:
            # sum_c U[a,c] * V[b,c] * W[c_idx,c] = T[a,b,c_idx] for all b, c_idx
            # Let M[c] be a vector of length (m*p)*(p*n) = 25*10 = 250
            # M[c][b*(p*n) + c_idx] = V[b,c] * W[c_idx,c]
            # Then U[a, :] @ M = T[a, :, :].flatten()
            # So U[a, :] = (M^T M)^{-1} M^T T[a].flatten()

            # This is expensive. Let's use a different approach.
            # Instead, let's use the fact that for each (b, c_idx), we have:
            # sum_c U[a,c] * V[b,c] * W[c_idx,c] = T[a,b,c_idx]
            # Group by (b, c_idx): for each (b, c_idx), this is a linear equation in U[a,:].

            # Better approach: use mode-n unfolding.
            # T is (10, 25, 10). For mode-1 unfolding (fix first index a):
            # T_1[a, (b, c_idx)] = T[a, b, c_idx]
            # We want T_1 ≈ U * (V ⊗ W)^T where V⊗W is (25*10) x r
            # (V⊗W)[(b, c_idx), c] = V[b, c] * W[c_idx, c]

            # Let's compute the Khatri-Rao product.
            # KR(V, W) has shape (25*10, r) where KR(V,W)[b*10 + c_idx, c] = V[b,c] * W[c_idx,c]

            # Update U: T_1 ≈ U @ KR(V,W).T
            # U = T_1 @ pinv(KR(V,W).T) = T_1 @ pinv(KR(V,W))

            KR_VW = np.zeros((m*p * p*n, r))
            for b in range(m*p):
                for c_idx in range(p*n):
                    for c in range(r):
                        KR_VW[b * p*n + c_idx, c] = V[b, c] * W[c_idx, c]

            # U = T_1 @ pinv(KR_VW)
            # T_1 has shape (10, 250)
            T_1 = T.reshape(n*m, m*p * p*n)
            try:
                U_new = T_1 @ np.linalg.pinv(KR_VW)
            except np.linalg.LinAlgError:
                return None
            U = U_new

            # Update V: T_2 ≈ V @ KR(U, W).T
            # T_2[b, (a, c_idx)] = T[a, b, c_idx]
            # KR(U, W)[(a, c_idx), c] = U[a, c] * W[c_idx, c]
            KR_UW = np.zeros((n*m * p*n, r))
            for a in range(n*m):
                for c_idx in range(p*n):
                    for c in range(r):
                        KR_UW[a * p*n + c_idx, c] = U[a, c] * W[c_idx, c]

            T_2 = np.zeros((m*p, n*m * p*n))
            for b in range(m*p):
                for a in range(n*m):
                    for c_idx in range(p*n):
                        T_2[b, a * p*n + c_idx] = T[a, b, c_idx]

            try:
                V_new = T_2 @ np.linalg.pinv(KR_UW)
            except np.linalg.LinAlgError:
                return None
            V = V_new

            # Update W: T_3 ≈ W @ KR(U, V).T
            # T_3[c_idx, (a, b)] = T[a, b, c_idx]
            # KR(U, V)[(a, b), c] = U[a, c] * V[b, c]
            KR_UV = np.zeros((n*m * m*p, r))
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(r):
                        KR_UV[a * m*p + b, c] = U[a, c] * V[b, c]

            T_3 = np.zeros((p*n, n*m * m*p))
            for c_idx in range(p*n):
                for a in range(n*m):
                    for b in range(m*p):
                        T_3[c_idx, a * m*p + b] = T[a, b, c_idx]

            try:
                W_new = T_3 @ np.linalg.pinv(KR_UV)
            except np.linalg.LinAlgError:
                return None
            W = W_new

            # Compute loss
            # Loss = sum_{a,b,c_idx} (T[a,b,c_idx] - sum_c U[a,c]V[b,c]W[c_idx,c])^2
            # = ||T - sum_c U[:,c] V[:,c] W[:,c]^T||_F^2
            # Let's compute this efficiently.
            # For each c, the contribution is U[:,c] V[:,c] W[:,c]^T
            # This is a 3D tensor. We can compute the Frobenius norm.

            # Actually, let's just check a few entries for speed.
            loss = 0.0
            for a in range(n*m):
                for b in range(m*p):
                    for c_idx in range(p*n):
                        val = 0.0
                        for c in range(r):
                            val += U[a, c] * V[b, c] * W[c_idx, c]
                        diff = T[a, b, c_idx] - val
                        loss += diff * diff

            if abs(prev_loss - loss) < tol:
                break
            prev_loss = loss

        # Verify the decomposition
        max_err = 0.0
        for a in range(n*m):
            for b in range(m*p):
                for c_idx in range(p*n):
                    val = 0.0
                    for c in range(r):
                        val += U[a, c] * V[b, c] * W[c_idx, c]
                    max_err = max(max_err, abs(T[a, b, c_idx] - val))

        if max_err < 1e-6:
            return U, V, W, max_err
        return None

    # Try ranks from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, n*p) = max(10, 25, 10) = 25
    # But we know rank >= 7 for 2x2x2, so for 2x5x5 it's likely higher.
    # Let's try from 35 up to 49.

    best_result = None
    best_rank = schoolbook_rank

    for r in range(35, schoolbook_rank):
        if time.time() > deadline - 20:
            break
        result = try_rank(r, max_iters=200)
        if result is not None:
            U, V, W, err = result
            best_result = (U, V, W)
            best_rank = r
            print(f"Found rank {r} decomposition with max error {err:.2e}")
            break  # Found a valid one, stop

    if best_result is None:
        # Fall back to schoolbook
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    U, V, W = best_result
    # Convert to list of lists with rounded values
    U_list = [[round(float(U[i, j]), 10) for j in range(best_rank)] for i in range(n*m)]
    V_list = [[round(float(V[i, j]), 10) for j in range(best_rank)] for i in range(m*p)]
    W_list = [[round(float(W[i, j]), 10) for j in range(best_rank)] for i in range(p*n)]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
