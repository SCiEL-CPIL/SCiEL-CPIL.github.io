<<<<<<< SEARCH
            # Different initialization strategies
            if restart % 5 == 0:
                # Random Gaussian
                U_init = np.random.randn(n * m, target_rank)
                V_init = np.random.randn(m * p, target_rank)
                W_init = np.random.randn(p * n, target_rank)
            elif restart % 5 == 1:
                # Random uniform
                U_init = np.random.rand(n * m, target_rank) * 2 - 1
                V_init = np.random.rand(m * p, target_rank) * 2 - 1
                W_init = np.random.rand(p * n, target_rank) * 2 - 1
            elif restart % 5 == 2:
                # Binary initialization
                U_init = np.random.choice([-1, 1], size=(n * m, target_rank)).astype(float)
                V_init = np.random.choice([-1, 1], size=(m * p, target_rank)).astype(float)
                W_init = np.random.choice([-1, 1], size=(p * n, target_rank)).astype(float)
            elif restart % 5 == 3:
                # Sparse initialization
                U_init = np.zeros((n * m, target_rank))
                V_init = np.zeros((m * p, target_rank))
                W_init = np.zeros((p * n, target_rank))
                for r in range(target_rank):
                    idx_u = np.random.choice(n * m, size=max(1, n * m // 3), replace=False)
                    idx_v = np.random.choice(m * p, size=max(1, m * p // 3), replace=False)
                    idx_w = np.random.choice(p * n, size=max(1, p * n // 3), replace=False)
                    U_init[idx_u, r] = np.random.choice([-1, 1])
                    V_init[idx_v, r] = np.random.choice([-1, 1])
                    W_init[idx_w, r] = np.random.choice([-1, 1])
            else:
                # Structured: use Kronecker-like patterns
                U_init = np.zeros((n * m, target_rank))
                V_init = np.zeros((m * p, target_rank))
                W_init = np.zeros((p * n, target_rank))
                for r in range(target_rank):
                    # Each term activates a random combination
                    i_r = r % n
                    j_r = (r // n) % m
                    k_r = (r // (n * m)) % p
                    U_init[i_r * m + j_r, r] = 1.0
                    V_init[j_r * p + k_r, r] = 1.0
                    W_init[k_r * n + i_r, r] = 1.0
                    # Add some noise
                    if r < target_rank // 2:
                        U_init[np.random.choice(n * m, size=2, replace=False), r] += np.random.randn(2) * 0.1
                        V_init[np.random.choice(m * p, size=2, replace=False), r] += np.random.randn(2) * 0.1
                        W_init[np.random.choice(p * n, size=2, replace=False), r] += np.random.randn(2) * 0.1
=======
            # Different initialization strategies
            if restart % 8 == 0:
                # Random Gaussian
                U_init = np.random.randn(n * m, target_rank)
                V_init = np.random.randn(m * p, target_rank)
                W_init = np.random.randn(p * n, target_rank)
            elif restart % 8 == 1:
                # Random uniform
                U_init = np.random.rand(n * m, target_rank) * 2 - 1
                V_init = np.random.rand(m * p, target_rank) * 2 - 1
                W_init = np.random.rand(p * n, target_rank) * 2 - 1
            elif restart % 8 == 2:
                # Binary initialization
                U_init = np.random.choice([-1, 1], size=(n * m, target_rank)).astype(float)
                V_init = np.random.choice([-1, 1], size=(m * p, target_rank)).astype(float)
                W_init = np.random.choice([-1, 1], size=(p * n, target_rank)).astype(float)
            elif restart % 8 == 3:
                # Sparse initialization
                U_init = np.zeros((n * m, target_rank))
                V_init = np.zeros((m * p, target_rank))
                W_init = np.zeros((p * n, target_rank))
                for r in range(target_rank):
                    idx_u = np.random.choice(n * m, size=max(1, n * m // 3), replace=False)
                    idx_v = np.random.choice(m * p, size=max(1, m * p // 3), replace=False)
                    idx_w = np.random.choice(p * n, size=max(1, p * n // 3), replace=False)
                    U_init[idx_u, r] = np.random.choice([-1, 1])
                    V_init[idx_v, r] = np.random.choice([-1, 1])
                    W_init[idx_w, r] = np.random.choice([-1, 1])
            elif restart % 8 == 4:
                # Structured: use Kronecker-like patterns
                U_init = np.zeros((n * m, target_rank))
                V_init = np.zeros((m * p, target_rank))
                W_init = np.zeros((p * n, target_rank))
                for r in range(target_rank):
                    # Each term activates a random combination
                    i_r = r % n
                    j_r = (r // n) % m
                    k_r = (r // (n * m)) % p
                    U_init[i_r * m + j_r, r] = 1.0
                    V_init[j_r * p + k_r, r] = 1.0
                    W_init[k_r * n + i_r, r] = 1.0
                    # Add some noise
                    if r < target_rank // 2:
                        U_init[np.random.choice(n * m, size=2, replace=False), r] += np.random.randn(2) * 0.1
                        V_init[np.random.choice(m * p, size=2, replace=False), r] += np.random.randn(2) * 0.1
                        W_init[np.random.choice(p * n, size=2, replace=False), r] += np.random.randn(2) * 0.1
            elif restart % 8 == 5:
                # Structured: exploit row/column structure of matrix multiplication
                # For each rank term, pick a subset of rows from A, cols from B
                U_init = np.zeros((n * m, target_rank))
                V_init = np.zeros((m * p, target_rank))
                W_init = np.zeros((p * n, target_rank))
                for r in range(target_rank):
                    # Pick which (i,j) pairs this term contributes to
                    i_r = r % n
                    # This term contributes to all j for row i_r
                    for j in range(m):
                        U_init[i_r * m + j, r] = 1.0
                    # Pick which (j,k) pairs
                    j_r = (r // n) % m
                    for k in range(p):
                        V_init[j_r * p + k, r] = 1.0
                    # Pick which (k,i) pairs
                    k_r = (r // (n * m)) % p
                    for i in range(n):
                        W_init[k_r * n + i, r] = 1.0
            elif restart % 8 == 6:
                # Structured: use the fact that matrix mult can be decomposed
                # with terms that mix indices in specific ways
                U_init = np.zeros((n * m, target_rank))
                V_init = np.zeros((m * p, target_rank))
                W_init = np.zeros((p * n, target_rank))
                for r in range(target_rank):
                    # Use a pattern based on r
                    a = r % n
                    b = (r // n) % m
                    c = (r // (n * m)) % p
                    # Term r: U[i,j,r] = 1 if i==a, V[j,k,r] = 1 if j==b, W[k,i,r] = 1 if k==c
                    U_init[a * m + b, r] = 1.0
                    V_init[b * p + c, r] = 1.0
                    W_init[c * n + a, r] = 1.0
                    # Also add cross terms
                    if r < target_rank // 2:
                        a2 = (a + 1) % n
                        b2 = (b + 1) % m
                        c2 = (c + 1) % p
                        U_init[a2 * m + b2, r] = 0.5
                        V_init[b2 * p + c2, r] = 0.5
                        W_init[c2 * n + a2, r] = 0.5
            else:
                # Structured: use random linear combinations
                U_init = np.zeros((n * m, target_rank))
                V_init = np.zeros((m * p, target_rank))
                W_init = np.zeros((p * n, target_rank))
                for r in range(target_rank):
                    # Random sparse patterns with varying density
                    density = np.random.uniform(0.3, 0.7)
                    U_init[:, r] = (np.random.rand(n * m) < density) * np.random.choice([-1, 1], size=n * m)
                    V_init[:, r] = (np.random.rand(m * p) < density) * np.random.choice([-1, 1], size=m * p)
                    W_init[:, r] = (np.random.rand(p * n) < density) * np.random.choice([-1, 1], size=p * n)
>>>>>>> REPLACE

