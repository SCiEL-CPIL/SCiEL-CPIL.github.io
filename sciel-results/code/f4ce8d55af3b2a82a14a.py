<<<<<<< SEARCH
def _try_rank_direct(r, n, m, p, time_limit=15.0, seed=0):
    """Direct random search: fix U,V, solve W, verify."""
    rng = np.random.RandomState(seed)
    start = time.time()
    num_ij = n * m
    num_jk = m * p
    num_ki = p * n
    
    for attempt in range(100):
        if time.time() - start > time_limit:
            break
        
        U = rng.randn(num_ij, r)
        V = rng.randn(num_jk, r)
        
        W = _solve_W(U, V, n, m, p)
        if W is None:
            continue
        
        if _verify(W.tolist(), V.tolist(), U.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None
=======
def _try_rank_direct(r, n, m, p, time_limit=15.0, seed=0):
    """Direct random search: fix U,V, solve W, verify."""
    rng = np.random.RandomState(seed)
    start = time.time()
    num_ij = n * m
    num_jk = m * p
    num_ki = p * n
    
    for attempt in range(100):
        if time.time() - start > time_limit:
            break
        
        U = rng.randn(num_ij, r)
        V = rng.randn(num_jk, r)
        
        W = _solve_W(U, V, n, m, p)
        if W is None:
            continue
        
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            # Polish for precision
            U, V, W, max_res = _polish(U, V, W, n, m, p, max_iters=50, tol=1e-12)
            if _verify_strict(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-10):
                return U.tolist(), V.tolist(), W.tolist()
            else:
                return U.tolist(), V.tolist(), W.tolist()
    
    return None
>>>>>>> REPLACE

