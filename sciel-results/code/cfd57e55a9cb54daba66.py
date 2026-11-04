<<<<<<< SEARCH
def construct():
    start = time.time()
    T = _build_tensor(n, m, p)
    
    # Try ranks from 30 upward
    for rank in range(30, 61):
        if time.time() - start > DEADLINE - 10:
            break
        for seed in range(5):
            if time.time() - start > DEADLINE - 10:
                break
            U, V, W, err = _als(n, m, p, rank, max_iters=200, seed=seed)
            if err < 1e-8:
                return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Fall back to trivial
    U, V, W = _trivial(n, m, p)
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
=======
def _check_decomposition(U, V, W, T, tol=1e-8):
    """Correctly check if U, V, W decompose T."""
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.max(np.abs(T_recon - T)) < tol

def construct():
    start = time.time()
    T = _build_tensor(n, m, p)
    
    # Try ranks from 30 upward with multiple seeds and iterations
    for rank in range(30, 61):
        if time.time() - start > DEADLINE - 15:
            break
        # Number of seeds based on available time
        remaining = DEADLINE - 15 - (time.time() - start)
        # Estimate time per ALS run: ~0.05-0.2s for small ranks
        max_seeds = max(3, int(remaining / 0.15))
        for seed in range(max_seeds):
            if time.time() - start > DEADLINE - 15:
                break
            max_iters = 300
            U, V, W, err = _als(n, m, p, rank, max_iters=max_iters, seed=seed, T=T)
            if _check_decomposition(U, V, W, T, tol=1e-8):
                # Final polish with higher precision
                U2, V2, W2, err2 = _als(n, m, p, rank, max_iters=100, seed=seed, tol=1e-12, T=T)
                if _check_decomposition(U2, V2, W2, T, tol=1e-8):
                    return {'U': U2.tolist(), 'V': V2.tolist(), 'W': W2.tolist()}
                return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Fall back to trivial
    U, V, W = _trivial(n, m, p)
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
>>>>>>> REPLACE

