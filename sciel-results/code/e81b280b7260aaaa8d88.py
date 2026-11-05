<<<<<<< SEARCH
    # Initial best result: the seed
    best_coeff = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_score = np.inf
=======
    # Physically-motivated initial condition:
    # c0=0.55 is the standard growth index for flat LCDM
    # c3=-0.5 cancels the leading-order w0 term in gamma expansion
    # Remaining coefficients model subdominant curvature and wa corrections
    best_coeff = np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
    best_score = np.inf
>>>>>>> REPLACE

