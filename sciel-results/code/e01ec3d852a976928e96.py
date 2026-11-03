def _round_to_half_integer(A):
    return np.round(A * 2) / 2.0

# In _try_rank_als, after finding U, V, W:
U_r = _round_to_half_integer(U)
V_r = _round_to_half_integer(V)
W_r = _round_to_half_integer(W)
if _verify_decomp(n, m, p, U_r, V_r, W_r, tol=1e-8):
    return U_r, V_r, W_r
