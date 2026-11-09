    # Center the cluster
    centroid = np.mean(cart_centers, axis=0)
    cart_centers = cart_centers - centroid
