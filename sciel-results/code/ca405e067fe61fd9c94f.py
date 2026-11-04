    def batched_als_decomposition(rank, max_steps=500, tol=1e-8, seed=42):
        """
        Use Alternating Least Squares to find a rank-r decomposition.
        """
        if time.time() > deadline:
            return None

        rng = np.random.RandomState(seed)

        # Initialize factors
        U = rng.randn(n * m, rank) * 0.5
        V = rng.randn(m * p, rank) * 0.5
        W = rng.ravel() if False else rng.randn(p * n, rank) * 0.5

        prev_err = np.inf

        for step in range(max_steps):
            if time.time() > deadline - 5:
                return None

            # --- Update U ---
            # Target[a, :, :].flatten() = M_U @ U[a, :]
            # M_U[:, r] = (V[:, r] * W[:, r].T).flatten()
            M_U = np.zeros((m * p * p * n, rank))
            for r in range(rank):
                M_U[:, r] = (V[:, r] * W[:, r].T).flatten()

            T_flat = target.reshape(n * m, m * p * p * n)
            # Solve M_U @ U.T ≈ T_flat.T
            try:
                U.T = np.linalg.lstsq(M_U, T_flat.T, rcond=None)[0]
            except:
                pass

            # --- Update V ---
            # Target[:, b, :].flatten() = M_V @ V[b, :]
            # M_V[:, r] = (U[:, r] * W[:, r].T).flatten()
            M_V = np.zeros((n * m * p * n, rank))
            for r in range(rank):
                M_V[:, r] = (U[:, r] * W[:, r].T).flatten()

            # T_flat_v[b, :] = target[:, b, :].flatten()
            # We want M_V @ V.T ≈ T_flat_v.T
            T_flat_v = np.zeros((m * p, n * m * p * n))
            for b in range(m * p):
                T_flat_v[b, :] = target[:, b, :].flatten()
            
            try:
                V.T = np.linalg.lstsq(M_V, T_flat_v.T, rcond=None)[0]
            except:
                pass

            # --- Update W ---
            # Target[:, :, c].flatten() = M_W @ W[c, :]
            # M_W[:, r] = (U[:, r] * V[:, r].T).flatten()
            # Note: Target[:, :, c] is (n*m, m*p). Flatten size n*m*m*p.
            M_W = np.zeros((n * m * m * p, rank))
            for r in range(rank):
                M_W[:, r] = (U[:, r] * V[:, r].T).flatten()

            T_flat_w = np.zeros((p * n, n * m * m * p))
            for c in range(p * n):
                T_flat_w[c, :] = target[:, :, c].flatten()
            
            try:
                W.T = np.linalg.lstsq(M_W, T_flat_w.T, rcond=None)[0]
            except:
                pass

            # Compute error
            if step % 10 == 0 or step == max_steps - 1:
                err = 0.0
                # Efficient reconstruction check
                recon = np.zeros_like(target)
                for r in range(rank):
                    recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                err = np.linalg.norm(recon - target)

                if err < prev_err - 1e-6:
                    prev_err = err
                elif err < tol:
                    break
                elif prev_err - err < 1e-9 and step > 50:
                    break

        # Check if we found an exact solution
        if verify_decomposition(U, V, W, target, tol=1e-6):
            # Try to round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            if verify_decomposition(U_rounded, V_rounded, W_rounded, target, tol=1e-6):
                return U_rounded, V_rounded, W_rounded

            # Try rounding to integers
            U_rounded = np.round(U)
            V_rounded = np.round(V)
            W_rounded = np.round(W)
            if verify_decomposition(U_rounded, V_rounded, W_rounded, target, tol=1e-6):
                return U_rounded, V_rounded, W_rounded

            # Return as-is if close enough
            return U, V, W

        return None
