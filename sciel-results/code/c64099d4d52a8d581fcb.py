    grad_sorted = np.zeros((n, M), dtype=np.float64)
    grad_sorted[0, :] = -active[0, :]
    if n > 2:
        grad_sorted[1:n-1, :] = active[0:n-2, :] - active[1:n-1, :]
    grad_sorted[n-1, :] = active[n-2, :]
