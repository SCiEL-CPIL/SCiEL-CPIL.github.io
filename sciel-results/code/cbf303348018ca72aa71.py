def equation(x, y, params):
    X = np.column_stack([x, y])
    # Define centers
    centers = np.array([...]) # 10x2
    # Define widths
    sigma = 0.4
    # Compute RBF values
    # dist^2 = ||X - C||^2
    # K = exp(-dist^2 / (2 * sigma^2))
    # return K @ params
