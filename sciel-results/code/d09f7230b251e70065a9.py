<<<<<<< SEARCH
def _try_rank(n, m, p, rank, num_trials=5):
    """Try multiple random initializations for a given rank."""
    for trial in range(num_trials):
        if time.time() > DEADLINE - 10:
            break
        result = _optimize_decomposition(n, m, p, rank, max_iter=2000, lr=0.005, seed=trial*1000+42)
        if result is not None:
            return result
    return None
=======
def _try_rank(n, m, p, rank, num_trials=5):
    """Try multiple random initializations for a given rank."""
    for trial in range(num_trials):
        if time.time() > DEADLINE - 10:
            break
        # Reduced iterations to ensure fast execution
        result = _optimize_decomposition(n, m, p, rank, max_iter=200, lr=0.01, seed=trial*1000+42)
        if result is not None:
            return result
    return None
>>>>>>> REPLACE
