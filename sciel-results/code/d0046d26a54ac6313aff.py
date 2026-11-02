<<<<<<< SEARCH
def _try_rank_search(n, m, p, T, target_rank, deadline, num_trials=50):
    """Try to find a decomposition of target_rank."""
    U18, V18, W18 = _build_rank18(n, m, p)
    U18_a = np.array(U18, dtype=np.float64)
    V18_a = np.array(V18, dtype=np.float64)
    W18_a = np.array(W18, dtype=np.float64)
    
    # Strategy 1: Remove terms from rank-18 and perturb
    if target_rank == 17:
        for term_to_remove in range(18):
            if time.time() > deadline:
                break
            idx = [i for i in range(18) if i != term_to_remove]
            U_init = U18_a[:, idx].copy()
            V_init = V18_a[:, idx].copy()
            W_init = W18_a[:, idx].copy()
            for seed in range(15):
                if time.time() > deadline:
                    break
                np.random.seed(term_to_remove * 100 + seed)
                scale = 0.0005 + 0.001 * (seed % 5)
                U_pert = U_init + np.random.randn(4, 17) * scale
                V_pert = V_init + np.random.randn(10, 17) * scale
                W_pert = W_init + np.random.randn(10, 17) * scale
                result = _als_from_init(U_pert, V_pert, W_pert, T, n, m, p, 17, max_iters=200)
                if result is not None:
                    return result
    
    # Strategy 2: Random initializations
    for seed in range(num_trials):
        if time.time() > deadline:
            break
        np.random.seed(seed * 13 + 7)
        U_init = np.random.randn(4, target_rank) * 0.5
        V_init = np.random.randn(10, target_rank) * 0.5
        W_init = np.random.randn(10, target_rank) * 0.5
        result = _als_from_init(U_init, V_init, W_init, T, n, m, p, target_rank, max_iters=150)
        if result is not None:
            return result
    
    return None
=======
def _try_rank_search(n, m, p, T, target_rank, deadline, num_trials=50):
    """Try to find a decomposition of target_rank."""
    U18, V18, W18 = _build_rank18(n, m, p)
    U18_a = np.array(U18, dtype=np.float64)
    V18_a = np.array(V18, dtype=np.float64)
    W18_a = np.array(W18, dtype=np.float64)
    
    # Strategy 1: SVD projection of rank-18 to target_rank, then perturb and ALS
    if target_rank in [16, 17]:
        # Project the rank-18 tensors to target_rank using SVD
        # This creates a valid rank-target_rank approximation of the rank-18 solution
        # which is close to the target tensor, providing a good starting point for ALS
        for seed in range(20):
            if time.time() > deadline:
                break
            np.random.seed(seed * 31 + 7)
            
            # Add noise to break symmetry and explore different projections
            noise_scale = 0.01
            U_noisy = U18_a + np.random.randn(4, 18) * noise_scale
            V_noisy = V18_a + np.random.randn(10, 18) * noise_scale
            W_noisy = W18_a + np.random.randn(10, 18) * noise_scale
            
            # SVD projection for U
            _, s_u, vh_u = np.linalg.svd(U_noisy, full_matrices=False)
            U_init = U_noisy @ (vh_u[:target_rank].T)
            
            # SVD projection for V
            _, s_v, vh_v = np.linalg.svd(V_noisy, full_matrices=False)
            V_init = V_noisy @ (vh_v[:target_rank].T)
            
            # SVD projection for W
            _, s_w, vh_w = np.linalg.svd(W_noisy, full_matrices=False)
            W_init = W_noisy @ (vh_w[:target_rank].T)
            
            # Perturb further to escape local minima
            pert_scale = 0.05
            U_init += np.random.randn(4, target_rank) * pert_scale
            V_init += np.random.randn(10, target_rank) * pert_scale
            W_init += np.random.randn(10, target_rank) * pert_scale
            
            result = _als_from_init(U_init, V_init, W_init, T, n, m, p, target_rank, max_iters=300)
            if result is not None:
                return result

    # Strategy 2: Random initializations (less likely to succeed but covers edge cases)
    for seed in range(num_trials):
        if time.time() > deadline:
            break
        np.random.seed(seed * 13 + 7)
        U_init = np.random.randn(4, target_rank) * 0.5
        V_init = np.random.randn(10, target_rank) * 0.5
        W_init = np.random.randn(10, target_rank) * 0.5
        result = _als_from_init(U_init, V_init, W_init, T, n, m, p, target_rank, max_iters=150)
        if result is not None:
            return result
    
    return None
>>>>>>> REPLACE

I also want to slightly increase the time budget allocation for the rank 16 search if rank 17 fails, as finding rank 16 is a bigger win. The current code allocates `deadline - 20` for rank 17 and `deadline - 10` for rank 16. This means rank 16 has very little time. I should adjust the deadlines to give more time to rank 16 if rank 17 is not found quickly, or simply increase the total time usage safely.

Actually, the `construct` function logic is:
1. Try rank 17 with deadline `start + 130 - 20 = start + 110`.
2. Try rank 16 with deadline `start + 130 - 10 = start + 120`.

This gives rank 17 up to 110s and rank 16 up to 120s (but it starts after rank 17 finishes, so if rank 17 takes 100s, rank 16 only gets 20s).
Let's optimize the `construct` function to be smarter about time allocation.

