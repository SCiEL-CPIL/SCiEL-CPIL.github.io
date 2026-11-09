            if restart % 2 == 0:
                U = rng.uniform(-1, 1, size=(n * m, target_rank))
                V = rng.uniform(-1, 1, size=(m * p, target_rank))
                W = rng.uniform(-1, 1, size=(p * n, target_rank))
            else:
                U = rng.integers(-1, 2, size=(n * m, target_rank)).astype(float)
                V = rng.integers(-1, 2, size=(m * p, target_rank)).astype(float)
                W = rng.integers(-1, 2, size=(p * n, target_rank)).astype(float)
