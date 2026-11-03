    def try_rank(R, seed_base=42):
        """Try to find a valid decomposition at rank R using multiple L-BFGS-B restarts."""
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None

        # Number of restarts depends on time remaining, but let's start with a fixed number
        # and break if time runs out.
        num_restarts = 20  # Increased from 8
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break

            np.random.seed(seed_base + restart * 137)
            
            # Initialization: Sparse random matrices often work better for exact solutions
            # We try a mix of dense and sparse
            if restart % 2 == 0:
                U = np.random.randn(n * m, R) * 0.5
                V = np.random.randn(m * p, R) * 0.5
                W = np.random.randn(p * n, R) * 0.5
            else:
                # Sparse init: mostly zeros, some 1s or -1s
                U = np.random.choice([-1, 0, 1], size=(n * m, R))
                V = np.random.choice([-1, 0, 1], size=(m * p, R))
                W = np.random.choice([-1, 0, 1], size=(p * n, R))
                # Scale to avoid huge gradients
                U *= 0.5
                V *= 0.5
                W *= 0.5

            # Use L-BFGS-B directly
            # We need to flatten U, V, W
            init_params = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
            
            # Objective function
            def objective(params):
                U_flat = params[:n * m * R]
                V_flat = params[n * m * R : n * m * R + m * p * R]
                W_flat = params[n * m * R + m * p * R :]

                U_mat = U_flat.reshape(n * m, R)
                V_mat = V_flat.reshape(m * p, R)
                W_mat = W_flat.reshape(p * n, R)

                tensor_vals = np.zeros(n * m * p)
                for r in range(R):
                    u_r = U_mat[:, r]
                    v_r = V_mat[:, r]
                    w_r = W_mat[:, r]
                    tensor_vals += u_r[U_idx] * v_r[V_idx] * w_r[W_idx]

                return np.sum((tensor_vals - 1.0) ** 2)

            # Run optimization
            # Set strict tolerances
            result = minimize(objective, init_params, method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12})
            
            if result.fun < best_error:
                best_error = result.fun
                best_U = result.x[:n * m * R].reshape(n * m, R)
                best_V = result.x[n * m * R : n * m * R + m * p * R].reshape(m * p, R)
                best_W = result.x[n * m * R + m * p * R :].reshape(p * n, R)
                
                if best_error < 1e-8:
                    break

        if best_error < 1e-5:
            # Round to half-integers
            U_int = np.round(best_U * 2) / 2
            V_int = np.round(best_V * 2) / 2
            W_int = np.round(best_W * 2) / 2

            # Verify exactness
            tensor_vals = compute_tensor(U_int, V_int, W_int, R)
            if np.all(np.abs(tensor_vals - 1.0) < 1e-9):
                return U_int.tolist(), V_int.tolist(), W_int.tolist()

        return None
