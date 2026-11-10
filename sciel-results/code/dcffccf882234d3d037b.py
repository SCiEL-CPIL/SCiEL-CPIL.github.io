<<<<<<< SEARCH
    centers_init = np.array(centers_init)
    # Center the grid
    centroid = np.mean(centers_init, axis=0)
    centers_init -= centroid
    
    x0 = centers_init.flatten()
=======
    centers_init = np.array(centers_init)
    # Center the grid
    centroid = np.mean(centers_init, axis=0)
    centers_init -= centroid
    
    # Expand slightly to ensure strict separation > 1e-5
    # Scale by 1.001
    centers_init *= 1.001
    
    x0 = centers_init.flatten()
>>>>>>> REPLACE
