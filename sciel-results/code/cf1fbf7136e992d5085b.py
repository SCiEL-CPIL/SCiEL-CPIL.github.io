U_r = U.reshape(n, m, R)
V_r = V.reshape(m, p, R)
W_r = W.reshape(p, n, R) # Note: reshape to (p, n, R)
# Loop:
val = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
if abs(val - 1.0) > 1e-9: return False
