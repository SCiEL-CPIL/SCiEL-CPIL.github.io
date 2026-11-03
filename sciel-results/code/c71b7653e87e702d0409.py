    grad = -np.sum(diff * factor[:, :, np.newaxis], axis=1)
    # diff = X[i] - X[j]
    # term = (X[i]-X[j])/d^2
    # sum = sum (X[i]-X[j])/d^2
    # -sum = sum (X[j]-X[i])/d^2
    # This is the correct gradient.
