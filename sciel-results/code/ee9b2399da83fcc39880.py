    # Calculate final s
    a1 = np.array([1, 0])
    a2 = np.array([0.5, np.sqrt(3)/2])
    a3 = np.array([-0.5, np.sqrt(3)/2])
    
    c_a1 = centers[:, 0]
    c_a2 = centers @ a2
    c_a3 = centers @ a3
    
    s1 = max(np.max(c_a1) + 1, -np.min(c_a1) + 1)
    s2 = max(np.max(c_a2) + 1, -np.min(c_a2) + 1)
    s3 = max(np.max(c_a3) + 1, -np.min(c_a3) + 1)
    s = max(s1, s2, s3)
    
    # Add a small safety margin to ensure strict containment
    # and to handle floating point precision in the evaluator
    s = s * 1.001 + 0.01
