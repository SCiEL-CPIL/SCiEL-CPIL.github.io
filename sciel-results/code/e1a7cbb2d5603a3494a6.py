<<<<<<< SEARCH
def _drop_column(U, V, W, col, keep_cols):
    """Try to drop column `col` by expressing W[:,col] as a combination of
    W[:,keep_cols]. Returns (new_U, new_V, new_W) or None."""
    Wc = W[:, col]
    Wk = W[:, keep_cols]
    # solve Wk @ a = Wc
    try:
        a, res, rank, sv = np.linalg.lstsq(Wk, Wc, rcond=None)
    except Exception:
        return None
    # check it's an exact solution
    if not np.allclose(Wk @ a, Wc, atol=1e-7, rtol=1e-7):
        return None
    new_U = np.zeros((U.shape[0], len(keep_cols)))
    new_V = np.zeros((V.shape[0], len(keep_cols)))
    new_W = np.zeros((W.shape[0], len(keep_cols)))
    for li, kc in enumerate(keep_cols):
        new_U[:, li] = U[:, kc] + a[li] * U[:, col]
        new_V[:, li] = V[:, kc] + a[li] * V[:, col]
        new_W[:, li] = W[:, kc] + a[li] * W[:, col]
    return new_U, new_V, new_W

def _local_improve(U, V, W, time_budget):
    """Attempt to reduce rank via null-space column elimination."""
    start = time.time()
    cur_U, cur_V, cur_W = U.copy(), V.copy(), W.copy()
    cur_cols = list(range(cur_U.shape[1]))
    improved = True
    while improved:
        improved = False
        if time.time() - start > time_budget:
            break
        # find the most "redundant" column: smallest norm or check each
        # For efficiency, iterate columns and attempt drop when its W is
        # in span of others. Use a heuristic ordering.
        for col in list(cur_cols):
            if time.time() - start > time_budget:
                break
            keep_cols = [c for c in cur_cols if c != col]
            if not keep_cols:
                continue
            res = _drop_column(cur_U, cur_V, cur_W, col, keep_cols)
            if res is not None:
                cur_U, cur_V, cur_W = res
                cur_cols = list(range(cur_U.shape[1]))
                improved = True
                break
    return cur_U, cur_V, cur_W, len(cur_cols)
=======
# (replaced by _try_drop and _local_improve_fast defined in construct section)
>>>>>>> REPLACE

