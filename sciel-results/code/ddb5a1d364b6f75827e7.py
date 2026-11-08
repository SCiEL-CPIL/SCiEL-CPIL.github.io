    # 6. U-shape (mass at edges)
    u_shape = np.abs(t)
    inits.append(u_shape / np.sum(u_shape))
    # 7. Extreme U-shape
    u_extreme = np.zeros(N)
    u_extreme[0] = 1
    u_extreme[-1] = 1
    inits.append(u_extreme / np.sum(u_extreme))
