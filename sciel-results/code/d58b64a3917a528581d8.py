    for target_rank in range(36, rank_school):
        if time.time() > deadline - 20:
            break

        # Try SVD initialization
        U, V, W = svd_init(target_rank)
        for it in range(50):
            if time.time() > deadline - 20:
                break
            # ... ALS updates ...
        
        if check_exact(U, V, W, tol=1e-5):
            rounded = round_and_check(U, V, W)
            if rounded is not None:
                best_U, best_V, best_W = rounded
                best_rank = target_rank
                break

        # Try random initializations
        for seed in range(5):
            # ... random init and ALS ...
