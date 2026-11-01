<<<<<<< SEARCH
def _try_rank17_search(n, m, p, T, deadline):
    """Try to find a rank-17 decomposition with enhanced strategies."""
    U18, V18, W18 = _build_rank18(n, m, p)
    U18_a = np.array(U18, dtype=np.float64)
    V18_a = np.array(V18, dtype=np.float64)
    W18_a = np.array(W18, dtype=np.float64)
    
    # Strategy 1: remove each schoolbook term and perturb with varying scales
    for term_to_remove in [14, 15, 16, 17]:
        if time.time() > deadline:
            break
        
        idx = [i for i in range(18) if i != term_to_remove]
        U_init = U18_a[:, idx].copy()
        V_init = V18_a[:, idx].copy()
        W_init = W18_a[:, idx].copy()
        
        for seed in range(10):
            if time.time() > deadline:
                break
            np.random.seed(term_to_remove * 100 + seed)
            scale = 0.01 + 0.01 * (seed % 5)
            U_pert = U_init + np.random.randn(4, 17) * scale
            V_pert = V_init + np.random.randn(10, 17) * scale
            W_pert = W_init + np.random.randn(10, 17) * scale
            
            result = _try_rank_als_from_init(U_pert, V_pert, W_pert, T, n, m, p, 17, max_iters=200)
            if result is not None:
                return result
    
    # Strategy 2: remove Strassen terms and perturb
    for term_to_remove in range(14):
        if time.time() > deadline:
            break
        
        idx = [i for i in range(18) if i != term_to_remove]
        U_init = U18_a[:, idx].copy()
        V_init = V18_a[:, idx].copy()
        W_init = W18_a[:, idx].copy()
        
        for seed in range(5):
            if time.time() > deadline:
                break
            np.random.seed(term_to_remove * 100 + seed + 50)
            scale = 0.01 + 0.01 * (seed % 5)
            U_pert = U_init + np.random.randn(4, 17) * scale
            V_pert = V_init + np.random.randn(10, 17) * scale
            W_pert = W_init + np.random.randn(10, 17) * scale
            
            result = _try_rank_als_from_init(U_pert, V_pert, W_pert, T, n, m, p, 17, max_iters=200)
            if result is not None:
                return result
    
    # Strategy 3: random initializations with varying scales
    for seed in range(30):
        if time.time() > deadline:
            break
        np.random.seed(seed * 7 + 123)
        scale = 0.3 + 0.2 * (seed % 5)
        U_init = np.random.randn(4, 17) * scale
        V_init = np.random.randn(10, 17) * scale
        W_init = np.random.randn(10, 17) * scale
        
        result = _try_rank_als_from_init(U_init, V_init, W_init, T, n, m, p, 17, max_iters=150)
        if result is not None:
            return result
    
    return None
=======
def _try_rank17_search(n, m, p, T, deadline):
    """Try to find a rank-17 decomposition with enhanced strategies."""
    U18, V18, W18 = _build_rank18(n, m, p)
    U18_a = np.array(U18, dtype=np.float64)
    V18_a = np.array(V18, dtype=np.float64)
    W18_a = np.array(W18, dtype=np.float64)
    
    best_result = None
    
    # Strategy 1: remove each schoolbook term and perturb with varying scales
    for term_to_remove in [14, 15, 16, 17]:
        if time.time() > deadline - 10:
            break
        
        idx = [i for i in range(18) if i != term_to_remove]
        U_init = U18_a[:, idx].copy()
        V_init = V18_a[:, idx].copy()
        W_init = W18_a[:, idx].copy()
        
        for seed in range(15):
            if time.time() > deadline - 5:
                break
            np.random.seed(term_to_remove * 100 + seed)
            # Use geometric scale progression for better coverage
            scale = 0.005 * (2 ** (seed % 4))
            U_pert = U_init + np.random.randn(4, 17) * scale
            V_pert = V_init + np.random.randn(10, 17) * scale
            W_pert = W_init + np.random.randn(10, 17) * scale
            
            result = _try_rank_als_from_init(U_pert, V_pert, W_pert, T, n, m, p, 17, max_iters=150)
            if result is not None:
                return result
    
    # Strategy 2: remove Strassen terms and perturb
    for term_to_remove in range(14):
        if time.time() > deadline - 10:
            break
        
        idx = [i for i in range(18) if i != term_to_remove]
        U_init = U18_a[:, idx].copy()
        V_init = V18_a[:, idx].copy()
        W_init = W18_a[:, idx].copy()
        
        for seed in range(8):
            if time.time() > deadline - 5:
                break
            np.random.seed(term_to_remove * 100 + seed + 50)
            scale = 0.005 * (2 ** (seed % 4))
            U_pert = U_init + np.random.randn(4, 17) * scale
            V_pert = V_init + np.random.randn(10, 17) * scale
            W_pert = W_init + np.random.randn(10, 17) * scale
            
            result = _try_rank_als_from_init(U_pert, V_pert, W_pert, T, n, m, p, 17, max_iters=150)
            if result is not None:
                return result
    
    # Strategy 3: random initializations with varying scales
    for seed in range(40):
        if time.time() > deadline - 5:
            break
        np.random.seed(seed * 7 + 123)
        # Wider range of scales
        scale = 0.1 * (2 ** (seed % 5))
        U_init = np.random.randn(4, 17) * scale
        V_init = np.random.randn(10, 17) * scale
        W_init = np.random.randn(10, 17) * scale
        
        result = _try_rank_als_from_init(U_init, V_init, W_init, T, n, m, p, 17, max_iters=100)
        if result is not None:
            return result
    
    # Strategy 4: Try removing pairs of terms (more aggressive)
    for t1 in range(18):
        for t2 in range(t1+1, 18):
            if time.time() > deadline - 5:
                break
            idx = [i for i in range(18) if i != t1 and i != t2]
            if len(idx) != 16:
                continue
            U_init = U18_a[:, idx].copy()
            V_init = V18_a[:, idx].copy()
            W_init = W18_a[:, idx].copy()
            
            # Pad back to 17 with small random values
            np.random.seed(t1 * 100 + t2)
            U_pad = np.hstack([U_init, np.random.randn(4, 1) * 0.01])
            V_pad = np.hstack([V_init, np.random.randn(10, 1) * 0.01])
            W_pad = np.hstack([W_init, np.random.randn(10, 1) * 0.01])
            
            result = _try_rank_als_from_init(U_pad, V_pad, W_pad, T, n, m, p, 17, max_iters=100)
            if result is not None:
                return result
    
    return None
>>>>>>> REPLACE

