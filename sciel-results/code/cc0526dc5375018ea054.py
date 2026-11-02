T_a = T_np.reshape(n * m, -1) # Shape (6, 54)
# ...
Z = np.zeros((m * p * p * n, target_rank)) # Shape (54, rank)
for b in range(m * p): # 9
    for c in range(p * n): # 6
        Z[b * p * n + c, :] = V[b, :] * W[c, :]
# Z shape is (54, rank).
ZtZ = Z.T @ Z # (rank, rank)
ZtT = Z.T @ T_a # (rank, 6)
U = np.linalg.solve(ZtZ, ZtT).T # (6, rank)
