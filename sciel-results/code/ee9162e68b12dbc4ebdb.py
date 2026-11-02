import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    # Build the multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Fallback: Schoolbook rank 24
    rank_school = n * m * p
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    fallback = schoolbook()

    def verify(U, V, W, tol=1e-8):
        """Check that sum_r U[:,r] V[:,r] W[:,r] == T exactly (within tol)."""
        T_curr = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_curr, T, atol=tol, rtol=0)

    def to_lists(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ------------------------------------------------------------------
    # Strategy 1: Structured "sum of products" decomposition.
    # For C = A B, we can share computations across output positions.
    # A standard trick: for each pair of rows of A, form sums, and for
    # each pair of columns of B, form sums, then combine.
    # This gives rank <= mn + np + mp - n (for square) but for rectangular
    # we use a different bound.
    #
    # Concretely for 2x3 * 3x4:
    #   Let A be 2x3, B be 3x4.
    #   We compute C[0,k] = sum_j A[0,j] B[j,k]
    #   and C[1,k] = sum_j A[1,j] B[j,k]
    #
    # Idea: introduce auxiliary products that share B's columns.
    # For each column k of B, we need both A[0,:].B[:,k] and A[1,:].B[:,k].
    # We can compute:
    #   S0 = A[0,0] + A[1,0]
    #   S1 = A[0,1] + A[1,1]
    #   S2 = A[0,2] + A[1,2]
    #   D0 = A[0,0] - A[1,0]
    #   D1 = A[0,1] - A[1,1]
    #   D2 = A[0,2] - A[1,2]
    # Then:
    #   C[0,k] = (S0*B[0,k] + S1*B[1,k] + S2*B[2,k] + D0*B[0,k] + D1*B[1,k] + D2*B[2,k]) / 2
    #   C[1,k] = (S0*B[0,k] + S1*B[1,k] + S2*B[2,k] - D0*B[0,k] - D1*B[1,k] - D2*B[2,k]) / 2
    #
    # But this still uses 6 products per column k, for 4 columns = 24 products.
    # That's the same as schoolbook. We need a smarter sharing.
    #
    # Better: share across columns of B too.
    # Let's use the "Strassen-like" approach:
    #   For 2x2 * 2x2, Strassen uses 7 products instead of 8.
    #   For 2x3 * 3x4, we can tile and apply Strassen-like tricks.
    #
    # Let me try a direct construction with rank 22.
    # ------------------------------------------------------------------

    # Try a structured construction: rank 22
    # We'll use 22 "slots" and assign coefficients.
    # The idea: for each output (i,k), C[i,k] = sum_j A[i,j] B[j,k].
    # We introduce 22 auxiliary products, each being a product of a linear
    # combination of A entries and a linear combination of B entries.

    # Let's try a known construction. For 2x3 * 3x4, the rank is at most 22.
    # We'll construct it explicitly.

    # Construction:
    # A = [[a00, a01, a02],
    #      [a10, a11, a12]]
    # B = [[b00, b01, b02, b03],
    #      [b10, b11, b12, b13],
    #      [b20, b21, b22, b23]]
    #
    # C[0,0] = a00*b00 + a01*b10 + a02*b20
    # C[0,1] = a00*b01 + a01*b11 + a02*b21
    # C[0,2] = a00*b02 + a01*b12 + a02*b22
    # C[0,3] = a00*b03 + a01*b13 + a02*b23
    # C[1,0] = a10*b00 + a11*b10 + a12*b20
    # C[1,1] = a10*b01 + a11*b11 + a12*b21
    # C[1,2] = a10*b02 + a11*b12 + a12*b22
    # C[1,3] = a10*b03 + a11*b13 + a12*b23
    #
    # We need 24 scalar multiplications in schoolbook.
    # To reduce, we share products.
    #
    # Let's define 22 products:
    # P0 = a00 * b00
    # P1 = a00 * b01
    # P2 = a00 * b02
    # P3 = a00 * b03
    # P4 = a01 * b10
    # P5 = a01 * b11
    # P6 = a01 * b12
    # P7 = a01 * b13
    # P8 = a02 * b20
    # P9 = a02 * b21
    # P10 = a02 * b22
    # P11 = a02 * b23
    # P12 = a10 * b00
    # P13 = a10 * b01
    # P14 = a10 * b02
    # P15 = a10 * b03
    # P16 = a11 * b10
    # P17 = a11 * b11
    # P18 = a11 * b12
    # P19 = a11 * b13
    # P20 = a12 * b20
    # P21 = a12 * b21
    # P22 = a12 * b22
    # P23 = a12 * b23
    #
    # That's 24 products. We need to reduce to 22.
    #
    # Let's try to combine some. For example:
    # P0 + P12 = (a00 + a10) * b00  -- no, that's not right.
    # (a00 + a10) * b00 = a00*b00 + a10*b00 = P0 + P12
    # So if we compute Q0 = (a00 + a10) * b00, we get P0 + P12.
    # But we need P0 and P12 separately for different outputs.
    # C[0,0] needs P0, C[1,0] needs P12.
    # So we can't just use Q0.
    #
    # We need a smarter combination. Let's use the Strassen idea:
    # For 2x2, we compute:
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b10 - b11)
    # M4 = a11 * (b01 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b10)
    # M7 = (a01 - a11) * (b11 + b01)
    # Then C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M1
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 - M6
    #
    # Let's apply this to our 2x3 * 3x4 case by tiling.
    # We can split B into two 3x2 blocks: B1 = B[:, 0:2], B2 = B[:, 2:4].
    # Then C[:, 0:2] = A * B1 and C[:, 2:4] = A * B2.
    # Each is a 2x3 * 3x2 multiplication.
    #
    # For 2x3 * 3x2, the schoolbook rank is 12.
    # Can we do better? Let's try a Strassen-like approach.
    #
    # Actually, let's just try to find a rank-22 decomposition by solving
    # the system of equations directly. We have 24 equations and 26*r unknowns.
    # For r=22, that's 572 unknowns. Very underdetermined.
    #
    # Let's try a random search with the constraint that entries are in
    # {-1, -0.5, 0, 0.5, 1}.
    #
    # But first, let's try the SVD approach to find the minimal rank.
    # ------------------------------------------------------------------

    # Strategy 2: SVD-based rank identification
    # Flatten T into a matrix T_mat of shape (n*m, m*p*p*n) or similar.
    # Actually, for tensor decomposition T = sum_r U_r V_r W_r,
    # we can flatten along one mode.
    #
    # Let's flatten T along the first mode:
    # T_flat has shape (n*m, m*p * p*n)
    # T_flat[a, b*p*n + c] = T[a, b, c]
    #
    # Then T_flat = U * (V ⊗ W)^T, where ⊗ is the Kronecker product.
    # The rank of T_flat is at most the rank of the tensor decomposition.
    # Actually, the rank of T_flat equals the tensor rank if the decomposition
    # is minimal and the vectors are linearly independent.

    T_flat = T.reshape(n*m, m*p * p*n)
    # SVD
    U_svd, s_vals, Vt_svd = np.linalg.svd(T_flat, full_matrices=False)
    # The rank of T_flat is the number of non-zero singular values.
    rank_flat = np.sum(s_vals > 1e-10)
    # This gives a lower bound on the tensor rank.

    # For our tensor, let's see what the rank is.
    # Actually, the tensor rank is at least the rank of any matricization.
    # So rank_flat is a lower bound.

    # Let's also try other matricizations.
    T_flat2 = T.reshape(m*p, n*m * p*n)
    U_svd2, s_vals2, Vt_svd2 = np.linalg.svd(T_flat2, full_matrices=False)
    rank_flat2 = np.sum(s_vals2 > 1e-10)

    T_flat3 = T.reshape(p*n, n*m * m*p)
    U_svd3, s_vals3, Vt_svd3 = np.linalg.svd(T_flat3, full_matrices=False)
    rank_flat3 = np.sum(s_vals3 > 1e-10)

    # The tensor rank is at least max(rank_flat, rank_flat2, rank_flat3).
    lower_bound = max(rank_flat, rank_flat2, rank_flat3)

    # For 2x3 * 3x4, the matricization ranks are:
    # rank_flat = rank of T as (6, 96) matrix
    # rank_flat2 = rank of T as (12, 48) matrix
    # rank_flat3 = rank of T as (8, 72) matrix
    #
    # The tensor rank is at least the maximum of these.
    # For our tensor, I expect the matricization ranks to be 6, 12, and 8
    # respectively, since each matricization has a lot of structure.
    #
    # Actually, let's just compute it.
    #
    # If the lower bound is, say, 12, then we know the tensor rank is at least 12.
    # The schoolbook rank is 24.
    #
    # Let's try to find a decomposition with rank = lower_bound.
    # If that fails, try lower_bound + 1, etc.

    # Strategy 3: Try to find a decomposition with rank = lower_bound
    # using a structured approach.
    #
    # We'll use the SVD to identify the principal components and then
    # try to construct a decomposition with that many rank-one terms.

    # For now, let's try a random search with the constraint that entries
    # are in {-1, -0.5, 0, 0.5, 1}.

    import random
    random.seed(42)

    def try_random_rank(r, max_attempts=5000):
        """Try to find a rank-r decomposition with entries in {-1, -0.5, 0, 0.5, 1}."""
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, r))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, r))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, r))
            if verify(U, V, W):
                return U, V, W
        return None

    # Try ranks from the lower bound up to schoolbook.
    for r in range(int(lower_bound), rank_school):
        if time.time() > deadline:
            break
        result = try_random_rank(r, max_attempts=2000)
        if result is not None:
            U, V, W = result
            return to_lists(U, V, W)

    # Strategy 4: Gradient descent with SVD seeding
    # Use the SVD to seed the gradient descent, which may help it
    # escape the schoolbook solution.

    # Try gradient descent for ranks from lower_bound to schoolbook.
    for r in range(int(lower_bound), rank_school):
        if time.time() > deadline:
            break
        for attempt in range(3):
            if time.time() > deadline:
                break
            # Seed with SVD
            if attempt == 0:
                # Use the SVD principal components as a seed.
                U = U_svd[:, :r] * s_vals[:r]
                V = np.zeros((m*p, r))
                W = np.zeros((p*n, r))
                # This is not quite right, but it gives a non-zero seed.
                # Let's just use random initialization for now.
                U = np.random.randn(n*m, r)
                V = np.random.randn(m*p, r)
                W = np.random.randn(p*n, r)
            else:
                U = np.random.randn(n*m, r)
                V = np.random.randn(m*p, r)
                W = np.random.randn(p*n, r)

            lr = 0.01
            for step in range(300):
                if time.time() > deadline:
                    break
                T_curr = np.einsum('ar,br,cr->abc', U, V, W)
                E = T - T_curr
                err_norm = np.linalg.norm(E)
                if err_norm < 1e-6:
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    if verify(U_snap, V_snap, W_snap):
                        return to_lists(U_snap, V_snap, W_snap)
                    break
                grad_U = np.einsum('abc,br,cr->ar', E, V, W)
                grad_V = np.einsum('abc,ar,cr->br', E, U, W)
                grad_W = np.einsum('abc,ar,br->cr', E, U, V)
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                if step % 20 == 0:
                    U = np.round(U * 2) / 2
                    V = np.round(V * 2) / 2
                    W = np.round(W * 2) / 2

    # Fallback to schoolbook
    return fallback
