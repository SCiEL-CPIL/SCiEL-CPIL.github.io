<<<<<<< SEARCH
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    deadline = time.time() + 145  # safety margin (evaluator kills at 150s)
    
    T = _build_target_tensor()
    
    # Build rank-77 decomposition
    U77, V77, W77 = _build_rank77()
    
    # Verify
    assert _check(U77, V77, W77, T, atol=1e-6), "Rank-77 verification failed"
    
    best_rank = 77
    best_U, best_V, best_W = U77.copy(), V77.copy(), W77.copy()
    
    # Try a direct 4x4x4 Strassen + schoolbook approach for better initial rank
    # This might give a lower starting point
    try:
        U_alt, V_alt, W_alt = _build_rank64()
        if _check(U_alt, V_alt, W_alt, T, atol=1e-6):
            best_rank = 64
            best_U, best_V, best_W = U_alt.copy(), V_alt.copy(), W_alt.copy()
    except:
        pass
=======
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    deadline = time.time() + 145  # safety margin (evaluator kills at 150s)
    
    T = _build_target_tensor()
    
    # Build rank-77 decomposition
    U77, V77, W77 = _build_rank77()
    
    # Verify
    assert _check(U77, V77, W77, T, atol=1e-6), "Rank-77 verification failed"
    
    best_rank = 77
    best_U, best_V, best_W = U77.copy(), V77.copy(), W77.copy()
>>>>>>> REPLACE

