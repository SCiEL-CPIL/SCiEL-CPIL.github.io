sums = np.array([0.0])
masks = np.array([0], dtype=np.uint64)
for i, d in enumerate(B_denoms):
    v = 1.0/d
    sums = np.concatenate([sums, sums + v])
    masks = np.concatenate([masks, masks | (1 << i)])
