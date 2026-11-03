I_IDX, J_IDX = np.triu_indices(N, k=1)

def objective(x, beta):
    pts = x.reshape(-1, 2)
    diffs = pts[I_IDX] - pts[J_IDX]
    dists = np.linalg.norm(diffs, axis=1)
    # ...
